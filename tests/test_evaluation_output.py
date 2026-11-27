from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate
import evaluate_versions
from src import evaluation_output


class EvaluationOutputTests(unittest.TestCase):
    def setUp(self):
        self.artifacts = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.runs = self.artifacts / "evaluation_runs"
        self.enterContext(patch.object(evaluation_output, "ARTIFACTS", self.artifacts))
        self.enterContext(patch.object(evaluation_output, "RUN_OUTPUTS", self.runs))
        self.enterContext(redirect_stdout(StringIO()))

    def test_default_reports_leave_archives_intact(self):
        import mteb

        cases = [
            (evaluate, ["--mode", "baseline"], "appsretrieval_results.json", "appsretrieval_baseline.json"),
            (evaluate_versions, [], "versions_report.json", "versions_report.json"),
        ]
        for cli, args, historical_name, output_name in cases:
            with self.subTest(cli=cli.__name__):
                historical = self.artifacts / historical_name
                historical.write_bytes(b"archived report")
                fake = SimpleNamespace(device="cpu", cache={}, batch_size=2)
                result = {"scores": {"test": [{"ndcg_at_10": 0.2, "mrr_at_10": 0.1}]}}
                folder = self.artifacts / "versions"
                folder.mkdir(exist_ok=True)
                for name in ("bugs", "tracked"):
                    (folder / f"{name}.json").write_text("[]", encoding="utf-8")
                with (
                    patch.object(sys, "argv", [cli.__name__, *args]),
                    patch.object(evaluate, "evaluation_encoder", return_value=fake),
                    patch("src.mteb_wrappers.E5Encoder"),
                    patch.object(mteb, "get_task"),
                    patch.object(mteb, "evaluate") as scoring,
                    patch.object(evaluate_versions, "VERSIONS_DIR", folder),
                    patch.object(evaluate_versions, "VersionStore"),
                    patch.object(evaluate_versions, "load_apps"),
                    patch.object(evaluate_versions, "evaluation_encoder"),
                    patch.object(evaluate_versions, "History") as history,
                ):
                    scoring.return_value.task_results = [SimpleNamespace(to_dict=lambda: result)]
                    history.return_value.names = []
                    history.return_value.build_seconds = 0
                    evaluate_versions.load_apps.return_value.query_ids = []
                    evaluate_versions.load_apps.return_value.query_texts = []
                    evaluate_versions.evaluation_encoder.return_value.encode_queries.return_value = ([], 0)
                    cli.main()
                self.assertEqual(historical.read_bytes(), b"archived report")
                self.assertIsInstance(json.loads((self.runs / output_name).read_bytes()), dict)

    def test_cli_rejects_protected_and_existing_output_before_evaluation(self):
        existing = self.artifacts.parent / f"{self.artifacts.name}-existing.json"
        self.addCleanup(existing.unlink, missing_ok=True)
        existing.write_bytes(b"existing report")
        cases = [
            (evaluate, ["--mode", "cached"], "baseline_frozen.json"),
            (evaluate_versions, [], "versions_report.json"),
        ]
        for cli, args, filename in cases:
            protected = self.artifacts / filename
            for output in (protected, existing):
                with (
                    self.subTest(cli=cli.__name__, output=output),
                    patch.object(sys, "argv", [cli.__name__, *args, "--output", str(output)]),
                    patch.object(cli, "evaluation_encoder", side_effect=AssertionError("evaluation started")),
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    cli.main()
                self.assertEqual(caught.exception.code, 2)
            self.assertFalse(protected.exists())
        self.assertEqual(existing.read_bytes(), b"existing report")

    def test_report_created_after_path_check_is_preserved(self):
        report = self.runs / "report.json"
        self.runs.mkdir()
        report.write_bytes(b"other run")
        with self.assertRaisesRegex(SystemExit, "already exists"):
            evaluation_output.write_report(report, {"score": 1})
        self.assertEqual(report.read_bytes(), b"other run")

    def test_embedding_cache_reads_archive_and_saves_to_run_directory(self):
        frozen = self.artifacts / "emb_cache.npz"
        frozen.write_bytes(b"archived vectors")
        with patch.object(evaluation_output, "EMB_CACHE", frozen), patch("src.encoder.E5") as encoder:
            evaluation_output.evaluation_encoder()
            encoder.assert_called_once_with(cache_path=frozen)
            run_cache = encoder.return_value.cache.path
            self.assertEqual(run_cache, self.runs / frozen.name)
            self.runs.mkdir()
            run_cache.write_bytes(b"new vectors")
            evaluation_output.evaluation_encoder()
            encoder.assert_called_with(cache_path=run_cache)
        self.assertEqual(frozen.read_bytes(), b"archived vectors")
