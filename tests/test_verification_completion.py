import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import execute


class CandidateExecutionTests(unittest.TestCase):
    def test_allocation_failure_does_not_abort_other_candidates(self):
        allocate = tempfile.mkdtemp
        calls = iter((True, False))

        def allocate_once(*args, **kwargs):
            if next(calls):
                raise OSError("temporary storage unavailable")
            return allocate(*args, **kwargs)

        with patch.object(tempfile, "mkdtemp", side_effect=allocate_once):
            self.assertEqual(
                execute.verify(["print(1)", "print(1)"], ("input", "1"), workers=1),
                [False, True],
            )

    def test_cleans_candidate_files_after_success_and_timeout(self):
        allocate = tempfile.mkdtemp
        write = "from pathlib import Path\nPath('nested').mkdir()\nPath('nested/state').touch()"
        with tempfile.TemporaryDirectory() as parent:
            def in_parent(suffix=None, prefix=None, dir=None):
                return allocate(suffix, prefix, parent)

            with patch.object(tempfile, "mkdtemp", side_effect=in_parent):
                flags = execute.verify(
                    [write + "\nprint(1)", write + "\nprint(1, flush=True)\nimport time\ntime.sleep(5)"],
                    ("input", "1"), timeout=0.75, workers=1,
                )
            self.assertEqual(flags, [True, False])
            self.assertEqual(list(Path(parent).iterdir()), [])

    def test_empty_stdout_requires_success(self):
        self.assertEqual(
            execute.verify(["pass", "raise SystemExit(1)", "print(0)"], ("input", ""), workers=1),
            [True, False, False],
        )

    def test_float_and_mixed_output_stay_exact(self):
        for expected, actual in [("1.00", "1.0"), ("1\nEND", "1 \nEND")]:
            with self.subTest(expected=expected):
                self.assertEqual(
                    execute.verify([f"print({actual!r})"], ("input", expected), workers=1,
                                   allow_trailing_padding=True),
                    [False],
                )


class LiveCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "requires Windows runner fingerprint")
    def test_windows_runner_change_misses_old_checkpoint_key(self):
        from src.mteb_wrappers import LiveExecReranker, SearchEncoderWrapper

        query = {"id": ["q"], "text": ["-----Example-----\nInput\nx\nOutput\n1"]}
        read_bytes = Path.read_bytes

        def previous_source(path):
            contents = read_bytes(path)
            return contents + b"\n# previous runner" if path.name == "_windows_job.py" else contents

        with tempfile.TemporaryDirectory() as directory:
            live = LiveExecReranker.__new__(LiveExecReranker)
            live.checkpoint = Path(directory) / "verdicts.jsonl"
            live.k, live.timeout, live.doc_text = 1, 1, {"a": "print(1)"}
            with patch.object(SearchEncoderWrapper, "search", side_effect=lambda *a, **kw: {"q": {"a": .8}}):
                with (patch.object(Path, "read_bytes", previous_source),
                      patch("src.mteb_wrappers.verify", return_value=[execute.VerificationOutcome.FAIL])):
                    self.assertEqual(live.search(query), {"q": {"a": .8}})
                current = live.search(query)
                self.assertGreater(current["q"]["a"], .8)
                with patch("src.mteb_wrappers.verify", side_effect=AssertionError("cache miss")):
                    self.assertEqual(live.search(query), current)
            records = [json.loads(line) for line in live.checkpoint.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual([record["passer_ids"] for record in records], [[], ["a"]])
            self.assertEqual(records[0].keys(), records[1].keys())
            self.assertNotEqual(records[0]["verification_key"], records[1]["verification_key"])

    def test_legacy_and_interrupted_records_are_ignored_and_new_results_resume(self):
        from datasets import Dataset
        import numpy as np

        from src.mteb_wrappers import LiveExecReranker

        query = Dataset.from_dict({"id": ["q"], "text": ["-----Example-----\nInput\nx\nOutput\n1"]})
        corpus = Dataset.from_dict({"id": ["wrong", "gold"], "text": ["print(1)\nraise EOFError", "print(1)"]})
        encoder = SimpleNamespace(
            mteb_model_meta=None,
            encode=lambda batches, **kwargs: np.array([[1.0]]) if len(batches.dataset) == 1 else np.array([[0.8], [0.3]]),
            similarity=lambda queries, docs: queries @ docs.T,
        )
        metadata = SimpleNamespace(get_modalities=lambda prompt: ["text"])
        options = dict(task_metadata=metadata, hf_split="train", hf_subset="default", encode_kwargs={})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "verdicts.jsonl"
            checkpoint.write_bytes(b'{"qid":"q","passer_ids":["wrong"]}\n{\xff')
            live = LiveExecReranker(encoder, checkpoint, k=2)
            live.index(corpus, **options)
            cold = live.search(query, top_k=2, **options)
            self.assertGreater(cold["q"]["gold"], cold["q"]["wrong"])
            with patch("src.mteb_wrappers.verify", side_effect=AssertionError("cache miss")):
                live.index(corpus, **options)
                self.assertEqual(live.search(query, top_k=2, **options), cold)
            self.assertEqual(json.loads(checkpoint.read_bytes().splitlines()[-1])["passer_ids"], ["gold"])

    def test_resume_keys_cover_source_examples_timeout_policy_and_top_k(self):
        from src.mteb_wrappers import LiveExecReranker, SearchEncoderWrapper

        sample = "-----Example-----\nInput\nx\nOutput\n1"
        query = {"id": ["q"], "text": [sample]}
        hits = {"a": 0.8, "b": 0.3}
        with tempfile.TemporaryDirectory() as directory:
            live = LiveExecReranker.__new__(LiveExecReranker)
            live.checkpoint = Path(directory) / "verdicts.jsonl"
            live.k, live.timeout = 2, 1
            live.doc_text = {"a": "print(0)", "b": "print(1)"}
            with (
                patch.object(SearchEncoderWrapper, "search", side_effect=lambda *a, **kw: {"q": hits.copy()}),
                patch("src.mteb_wrappers.verify", wraps=execute.verify) as verify,
            ):
                live.search(query)
                live.search(query)
                self.assertEqual(verify.call_count, 1)
                live.doc_text["a"] = "print(1)"
                live.search(query)
                query["text"] = [sample.replace("\nx\n", "\ny\n")]
                live.search(query)
                query["text"] = [query["text"][0][:-1] + "2"]
                live.search(query)
                live.timeout = 2
                live.search(query)
                query["text"] = ["-----Output-----\nPrint space-separated integers.\n" + query["text"][0]]
                live.search(query)
                live.k = 1
                live.search(query)
                self.assertEqual(verify.call_count, 7)


class HistoryCacheTests(unittest.TestCase):
    def test_nul_does_not_merge_different_examples_across_versions(self):
        from src.evolution import Entry, History

        history = History.__new__(History)
        source = "print('c' if input() == 'a\\x00b' else 'x')"
        history.entries = [Entry("d", source, ["v1"]), Entry("d", source, ["v2"])]
        history._verdicts = {}
        earlier = ("a\x00b", "c")
        later = ("a", "b\x00c")
        self.assertEqual(history.verify([0], earlier), {0: True})
        with patch("src.evolution.verify", side_effect=AssertionError("cache miss")):
            self.assertEqual(history.verify([1], earlier), {1: True})
        self.assertEqual(history.verify([1], later), {1: False})
        history.entries[1].text = "print('b\\x00c')"
        self.assertEqual(history.verify([1], later), {1: True})


class DemoVerificationTests(unittest.TestCase):
    def test_query_source_and_policy_are_used_without_loading_demo_dataset(self):
        import numpy as np

        from src.evolution import rank_verified

        demo = Path(__file__).resolve().parents[1] / "demo.py"
        tree = ast.parse(demo.read_text(encoding="utf-8"), filename=str(demo))
        search = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "search")
        scope = dict(
            np=np, time=time, SCOPES={"Original": np.array([0, 1])},
            E=SimpleNamespace(encode_queries=lambda _: (np.array([[1.0]]), 0)),
            POOL_EMB=np.array([[0.8], [0.3]]), POOL_TEXT=["print('1 \\n2')", "print('1\\n2')"],
            parse_example=execute.parse_example, allows_trailing_padding=execute.allows_trailing_padding,
            verify=execute.verify, VerificationOutcome=execute.VerificationOutcome,
            rank_verified=rank_verified, _VERDICTS={}, QID_BY_TEXT={}, SHOW=2,
            _card=lambda rank, i, score, verdict, was, gold: f"{i}:{verdict} ",
        )
        exec(compile(ast.Module(body=[search], type_ignores=[]), demo, "exec"), scope)
        run = scope["search"]
        query = "-----Example-----\nInput\nx\nOutput\n1\n2"
        self.assertEqual(run(query, "Original", 2, True)[1], "1:passed 0:failed ")
        scoped = "-----Output-----\nPrint space-separated integers.\n" + query
        self.assertEqual(run(scoped, "Original", 2, True)[1], "0:passed 1:passed ")
        scope["POOL_TEXT"][0] += "\nraise EOFError"
        self.assertEqual(run(scoped, "Original", 2, True)[1], "1:passed 0:failed ")
        with patch.dict(scope, verify=Mock(side_effect=AssertionError("cache miss"))):
            stats, cards = run(scoped, "Original", 2, True)
        self.assertIn("2 reused", stats)
        self.assertEqual(cards, "1:passed 0:failed ")


class EvaluationModeTests(unittest.TestCase):
    def test_live_mode_cannot_return_a_preexisting_mteb_score(self):
        import mteb

        import evaluate

        stop = RuntimeError("evaluation intercepted before loading data")
        fake_encoder = SimpleNamespace(device="cpu", cache={}, batch_size=2)
        with (
            patch.object(sys, "argv", ["evaluate.py", "--mode", "full"]),
            patch.object(evaluate, "evaluation_encoder", return_value=fake_encoder),
            patch("src.mteb_wrappers.E5Encoder"),
            patch("src.mteb_wrappers.LiveExecReranker"),
            patch.object(mteb, "get_task"),
            patch.object(mteb, "evaluate", side_effect=stop) as run,
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "intercepted"):
                evaluate.main()
            self.assertIsNone(run.call_args.kwargs["cache"])
