import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from datasets import Dataset

from src import execute
from src.execute import VerificationOutcome
from src.mteb_wrappers import LiveExecReranker, SearchEncoderWrapper


EXAMPLE = "-----Example-----\nInput\nx\nOutput\n1"
PASS, FAIL = VerificationOutcome.PASS, VerificationOutcome.FAIL


class AdaptiveRerankerTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.checkpoint = Path(directory) / "live.jsonl"
        self.query = {"id": ["q"], "text": [EXAMPLE]}
        self.docs = {f"d{i:03}": f"code {i}" for i in range(205)}
        self.encoder = SimpleNamespace(mteb_model_meta=None)
        self.live = LiveExecReranker(self.encoder, self.checkpoint, adapt=True)
        self.live.doc_text = self.docs.copy()
        self.hits = {did: -i / 205 for i, did in enumerate(self.docs)}
        self.parent = self.enterContext(patch.object(SearchEncoderWrapper, "search",
                                                   side_effect=lambda *a, **kw: {"q": self.hits.copy()}))
        self.verify = self.enterContext(patch("src.mteb_wrappers.verify"))

    def test_default_unchanged_and_separate_checkpoints(self):
        default = LiveExecReranker(self.encoder, self.checkpoint)
        self.assertFalse(default.adapt)
        self.assertEqual(default.k, 50)
        self.assertEqual(default.checkpoint, self.checkpoint)
        self.assertEqual(self.live.checkpoint.name, "live.adapt.jsonl")
        with self.assertRaises(ValueError):
            LiveExecReranker(self.encoder, self.checkpoint, k=100, adapt=True)
        self.verify.return_value = [FAIL, PASS, FAIL]
        default.doc_text = self.docs
        self.hits = dict(list(self.hits.items())[:3])
        default.search(self.query, top_k=3)
        self.assertNotIn("strict", self.verify.call_args.kwargs)
        self.assertIn("verification_key", json.loads(self.checkpoint.read_bytes().splitlines()[-1]))
        self.assertFalse(self.live.checkpoint.exists())

    def test_early_stop_and_shallow_final_results(self):
        self.verify.return_value = [PASS if i in (4, 49) else FAIL for i in range(50)]
        result = self.live.search(self.query, top_k=3)["q"]
        self.assertEqual(list(result), ["d004", "d049", "d000"])
        self.assertEqual(sorted(result, key=lambda d: -result[d]), list(result))
        self.assertEqual(self.parent.call_args.kwargs["top_k"], 205)
        self.verify.assert_called_once()
        self.assertTrue(self.verify.call_args.kwargs["detailed"])
        self.assertEqual(self.verify.call_args.args[:3], (list(self.docs.values())[:50], ("x", "1"), 1))

    def test_extensions_ties_cutoff_and_stable_order(self):
        from mteb._evaluators.retrieval_metrics import mrr

        self.hits = {did: 0.4 for did in reversed(self.docs)}
        self.verify.side_effect = [[FAIL] * 50, [PASS if i in (8, 149) else FAIL for i in range(150)]]
        result = self.live.search(self.query, top_k=205)["q"]
        self.assertEqual(list(result)[:4], ["d058", "d199", "d000", "d001"])
        self.assertEqual(list(result)[-5:], [f"d{i:03}" for i in range(200, 205)])
        self.assertEqual(mrr({"q": {"d058": 1}}, {"q": result}, [10])["MRR@10"], [1.0])
        calls = self.verify.call_args_list
        self.assertEqual([len(c.args[0]) for c in calls], [50, 150])
        self.assertEqual(calls[0].args[0] + calls[1].args[0], list(self.docs.values())[:200])
        self.verify.reset_mock()
        self.assertEqual(self.live.search(self.query, top_k=2)["q"], dict(list(result.items())[:2]))
        self.verify.assert_not_called()

    def test_no_fixture_no_launch_and_incomplete_pool_no_suffix(self):
        self.assertEqual(list(self.live.search({"id": ["q"], "text": ["missing example"]}, top_k=4)["q"]),
                         list(self.docs)[:4])
        self.verify.assert_not_called()
        self.assertFalse(self.live.checkpoint.exists())
        self.hits = dict(list(self.hits.items())[:12])
        self.verify.return_value = [FAIL] * 12
        self.assertEqual(list(self.live.search(self.query, top_k=100)["q"]), list(self.hits))
        self.verify.assert_called_once()
        self.verify.reset_mock()
        self.assertEqual(self.live.search(self.query, top_k=0), {"q": {}})
        self.verify.assert_not_called()

    def test_truncated_incompatible_and_inconsistent_checkpoint_records(self):
        self.live.checkpoint.write_bytes(b'{bad\n{\xff')
        self.verify.return_value = [PASS] + [FAIL] * 49
        expected = self.live.search(self.query, top_k=5)
        last = json.loads(self.live.checkpoint.read_bytes().splitlines()[-1])
        self.verify.reset_mock()
        self.assertEqual(self.live.search(self.query, top_k=5), expected)
        self.verify.assert_not_called()
        last["passer_ids"] = []
        with self.live.checkpoint.open("a", encoding="utf-8") as out:
            out.write(json.dumps(last) + "\n")
        self.live.search(self.query, top_k=5)
        self.verify.assert_called_once()
        self.verify.reset_mock()
        self.live.doc_text["d001"] = "changed bytes"
        self.live.search(self.query, top_k=5)
        self.verify.assert_called_once()
        self.verify.reset_mock()
        with patch.dict("os.environ", {"CANDIDATE_TEST_VALUE": "changed"}):
            self.live.search(self.query, top_k=5)
        self.verify.assert_called_once()

    def test_infrastructure_failure_or_cancellation_leaves_no_verdict(self):
        failure = execute.VerificationInfrastructureError("broken runner")
        for error in (failure, KeyboardInterrupt()):
            with self.subTest(error=error):
                self.verify.side_effect = [[FAIL] * 50, error]
                with self.assertRaises(type(error)):
                    self.live.search(self.query, top_k=2)
                self.assertFalse(self.live.checkpoint.exists())
        self.verify.side_effect = [[PASS] + [FAIL] * 49]
        self.assertEqual(list(self.live.search(self.query, top_k=1)["q"]), ["d000"])
        self.assertTrue(self.live.checkpoint.exists())

    def test_checkpoint_never_crosses_query_ids(self):
        self.verify.return_value = [PASS] + [FAIL] * 49
        self.live.search(self.query, top_k=2)
        self.parent.side_effect = None
        self.parent.return_value = {"another-query": self.hits.copy()}
        self.live.search({"id": ["another-query"], "text": [EXAMPLE]}, top_k=2)
        self.assertEqual(self.verify.call_count, 2)

    def test_recoverable_failure_extends_and_promotes_only_passes_without_checkpoint(self):
        err = VerificationOutcome.INFRASTRUCTURE_ERROR
        self.verify.side_effect = [[err] + [FAIL] * 49, [FAIL, PASS] + [FAIL] * 148] * 2
        for _ in range(2):
            result = self.live.search(self.query, top_k=3)["q"]
            self.assertEqual(list(result), ["d051", "d000", "d001"])
            self.assertFalse(self.live.checkpoint.exists())
        self.assertEqual(self.verify.call_count, 4)

    def test_fixed_depth_uses_passes_but_retries_incomplete_batch(self):
        default = LiveExecReranker(self.encoder, self.checkpoint)
        default.doc_text = self.docs
        err = VerificationOutcome.INFRASTRUCTURE_ERROR
        self.verify.return_value = [err, PASS] + [FAIL] * 48
        for _ in range(2):
            result = default.search(self.query, top_k=50)["q"]
            self.assertGreater(result["d001"], result["d000"])
            self.assertEqual(self.checkpoint.read_text(), "")
        self.assertEqual(self.verify.call_count, 2)


class AdaptiveIntegrationTests(unittest.TestCase):
    def test_cli_routes_only_explicit_live_opt_in(self):
        import evaluate
        import mteb

        encoder = SimpleNamespace(device="cpu", cache={}, batch_size=2)
        for flags in (["--mode", "baseline", "--adapt"], ["--mode", "full", "--adapt", "--k", "100"]):
            with (self.subTest(flags=flags), patch.object(sys, "argv", ["evaluate.py", *flags]),
                  patch.object(evaluate, "evaluation_encoder") as load, redirect_stderr(StringIO()),
                  self.assertRaises(SystemExit) as caught):
                evaluate.main()
            self.assertEqual(caught.exception.code, 2)
            load.assert_not_called()
        with (patch.object(sys, "argv", ["evaluate.py", "--mode", "full", "--adapt"]),
              patch.object(evaluate, "report_path") as output,
              patch.object(evaluate, "evaluation_encoder", return_value=encoder),
              patch("src.mteb_wrappers.E5Encoder"), patch("src.mteb_wrappers.LiveExecReranker") as live,
              patch.object(mteb, "get_task"), patch.object(mteb, "evaluate", side_effect=RuntimeError("intercept")),
              redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "intercept")):
            evaluate.main()
        self.assertTrue(live.call_args.kwargs["adapt"])
        self.assertEqual(output.call_args.args[2], "appsretrieval_full_adapt.json")

    def test_real_index_promotes_below_output_cutoff(self):
        ids = [f"d{i:03}" for i in range(55)]
        corpus = Dataset.from_dict({"id": ids, "text": [did for did in ids]})
        queries = Dataset.from_dict({"id": ["q"], "text": [EXAMPLE]})
        encoder = SimpleNamespace(
            mteb_model_meta=None,
            encode=lambda batches, **kw: np.ones((1, 1)) if len(batches.dataset) == 1
            else np.linspace(1, 0.4, len(ids)).reshape(-1, 1),
            similarity=lambda a, b: a @ b.T,
        )
        options = dict(task_metadata=SimpleNamespace(get_modalities=lambda _: ["text"]),
                       hf_split="train", hf_subset="default", encode_kwargs={})
        with tempfile.TemporaryDirectory() as directory:
            live = LiveExecReranker(encoder, Path(directory) / "exec.jsonl", adapt=True)
            live.index(corpus, **options)
            with patch("src.mteb_wrappers.verify",
                       side_effect=lambda codes, *a, **kw: [PASS if c == "d053" else FAIL for c in codes]):
                result = live.search(queries, top_k=2, **options)["q"]
            self.assertEqual(list(result), ["d053", "d000"])

    def test_normal_timeout_fails_but_infrastructure_is_reported(self):
        with patch.object(execute, "_run_candidate", side_effect=subprocess.TimeoutExpired("python", 1)):
            self.assertEqual(execute.verify(["print(1)"], ("x", "1"), workers=1, strict=True), [False])
        cases = [OSError("launch"), TimeoutError("reap"),
                 subprocess.CompletedProcess("python", 0, stdout="1", stderr=None)]
        for case in cases:
            with self.subTest(case=case), patch.object(execute, "_run_candidate", side_effect=[case]):
                with self.assertRaises(execute.VerificationInfrastructureError):
                    execute.verify(["print(1)"], ("x", "1"), workers=1, strict=True)
