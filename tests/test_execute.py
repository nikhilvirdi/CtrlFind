import unittest
from unittest.mock import patch

from src.execute import allows_trailing_padding, parse_example, verify


class VerifyTests(unittest.TestCase):
    example = ("sample", "sample")

    def test_successful_execution_requires_matching_output(self):
        snippets = [
            "print(input())",
            "print('different')",
            "import sys\nprint('diagnostic', file=sys.stderr)\nprint(input())",
        ]
        self.assertEqual(verify(snippets, self.example, workers=1), [True, False, True])

    def test_explicit_nonzero_exit_rejects_matching_output(self):
        snippet = "print(input(), flush=True)\nraise SystemExit(2)"
        self.assertEqual(verify([snippet], self.example, workers=1), [False])

    def test_crash_after_matching_output_is_rejected(self):
        snippet = "print(input(), flush=True)\nraise RuntimeError('crashed')"
        self.assertEqual(verify([snippet], self.example, workers=1), [False])

    def test_timeout_rejects_partial_matching_output(self):
        snippet = "import time\nprint(input(), flush=True)\ntime.sleep(30)"
        self.assertEqual(verify([snippet], self.example, timeout=0.25, workers=1), [False])

    def test_subprocess_launch_error_is_rejected(self):
        with patch("src.execute._run_candidate", side_effect=OSError("cannot start interpreter")):
            self.assertEqual(verify(["print(input())"], self.example, workers=1), [False])


class TrailingPaddingTests(unittest.TestCase):
    declaration = "-----Output-----\nPrint space-separated integers.\n"
    example = ("sample", "1 2\n\n -3  +4\n5")

    def test_declared_integer_rows_accept_trailing_spaces_and_tabs(self):
        snippet = "print('1 2 \\t\\n \\t\\n -3  +4\\t\\n5')"
        padding = allows_trailing_padding(self.declaration)
        self.assertTrue(padding)
        self.assertEqual(verify([snippet], self.example, workers=1, allow_trailing_padding=padding), [True])
        self.assertEqual(verify([snippet], self.example, workers=1), [False])

    def test_leading_spacing_internal_spacing_line_order_and_content_are_preserved(self):
        outputs = [
            "1 2\n\n-3  +4\n5",          # leading indentation
            "1 2\n\n -3 +4\n5",          # internal spacing
            "1 2\n -3  +4\n5",           # missing blank line
            "1 2\n\n5\n -3  +4",         # row order
            "1 2\n\n -3  +7\n5",         # content
            "1 2\u00a0\n\n -3  +4\n5",    # non-ASCII spacing
        ]
        with patch("src.execute.run_snippet", side_effect=outputs):
            self.assertEqual(
                verify([""] * len(outputs), self.example, workers=1, allow_trailing_padding=True),
                [False] * len(outputs),
            )

    def test_fixed_width_and_literal_padding_remain_exact(self):
        snippet = "print('1\\n23')"
        expected = ("sample", "1  \n23")
        for description in (
            "Every line has a fixed width of three characters.",
            "End each line with a space.",
            "Trailing spaces are significant.",
            "Put a blank after the final integer on every row.",
            "The rows encode a visual pattern.",
        ):
            with self.subTest(description=description):
                padding = allows_trailing_padding(self.declaration + description)
                self.assertFalse(padding)
                self.assertEqual(
                    verify([snippet], expected, workers=1, allow_trailing_padding=padding),
                    [False],
                )

    def test_ascii_art_and_mixed_content_remain_exact(self):
        self.assertFalse(allows_trailing_padding(self.declaration + "Render the ASCII art below."))
        snippets = ["print('1  \\n#')", "print('1\\n#')"]
        self.assertEqual(
            verify(snippets, ("sample", "1  \n#"), workers=1, allow_trailing_padding=True),
            [True, False],
        )

    def test_example_text_does_not_grant_padding_permission(self):
        statement = (
            "-----Output-----\nPrint the output exactly.\n"
            "-----Example-----\nInput\nPrint space-separated integers.\nOutput\n1  \n23"
        )
        self.assertIsNotNone(parse_example(statement))
        self.assertFalse(allows_trailing_padding(statement))
        self.assertFalse(allows_trailing_padding("The text '-----Output-----' precedes space-separated integers."))

    def test_crashing_output_is_rejected_with_padding_allowed(self):
        snippet = "print('1 \\n2', flush=True)\nraise RuntimeError('crashed')"
        self.assertEqual(
            verify([snippet], ("sample", "1\n2"), workers=1, allow_trailing_padding=True),
            [False],
        )

    def test_version_verdict_cache_keeps_comparison_modes_separate(self):
        from src.evolution import Entry, History

        history = History.__new__(History)
        history.entries = [Entry("d", "print('1\\n2')")]
        history._verdicts = {}
        example = ("sample", "1 \n2")
        self.assertEqual(history.verify([0], example), {0: False})
        self.assertEqual(history.verify([0], example, allow_trailing_padding=True), {0: True})
        self.assertEqual(history.verify([0], example), {0: False})
