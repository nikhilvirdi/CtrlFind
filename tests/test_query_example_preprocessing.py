"""Contract checks for the research-only query representation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from artifacts.query_example_preprocessing import strip_examples
from src.cache import EmbeddingCache
from src.config import DOC_PREFIX, MODEL_NAME, QUERY_PREFIX
from src.encoder import E5
from src.execute import allows_trailing_padding, parse_example


class ExampleStrippingTests(unittest.TestCase):
    def test_removes_only_labelled_examples_with_multiple_pairs(self):
        prefix = "Task\r\n\r\n-----Input-----\r\nLength n\r\n-----Output-----\r\nSpace-separated integers\r\n"
        sample = "-----Examples-----\r\nInput\r\n4\r\nOutput\r\n12\r\nInput\r\n8\r\nOutput\r\n16\r\n"
        note = "-----Note-----\r\nThis explains both cases and the last constraint.\r\n"
        original = prefix + sample + note
        self.assertEqual(strip_examples(original), prefix + note)
        self.assertEqual(parse_example(original), ("4", "12"))
        self.assertTrue(allows_trailing_padding(original))
        self.assertIsNone(parse_example(strip_examples(original)))

    def test_handles_standalone_colons_and_separate_sample_headings(self):
        prefix = "Task\n-----Output-----\nPrint one line.\n"
        note = "-----Constraints-----\n1 <= n <= 1e5\n"
        for sample in (
            "-----Example-----\nInput:\n2e1 -1.2\nOutput:\nYes\n",
            "-----Sample Input:-----\n2e1 -1.2\n-----Sample Output:-----\nYes\n",
            "-----Example Input-----\n11\n-----Example Output-----\n+1 .5 -1e-4\n",
        ):
            with self.subTest(sample=sample):
                self.assertEqual(strip_examples(prefix + sample + note), prefix + note)

    def test_leaves_ambiguous_examples_and_specifications_intact(self):
        for source in (
            "Task\nInput: 2\nOutput: 4\n",
            "Task\n-----Input-----\n2\n-----Output-----\n4\n",
            "Task\n-----Example-----\nInput\n2\nOutput\n4\nExplanation: numeric reasoning",
            "Task\n-----Example-----\nInput\nalpha beta\nOutput\nalpha\n",
            "Task\n-----Example-----\nInput: 2\nOutput: 4\n",
            "Task\n-----Example-----\nInput\n2\nOutput\n",
            "Task\n-----Example-----\nOutput\n4\nInput\n2\n",
            "Task\n-----Sample Input-----\n2\n-----Output-----\n4\n",
            "Task\n-----Sample Input-----\n2\n-----Example Output-----\n4\n",
            "Task\n-----Example-----\nInput\n1\nOutput\n2\nInput\n3",
        ):
            with self.subTest(source=source):
                self.assertEqual(strip_examples(source), source)

    def test_original_variant_and_document_are_separate_cache_entries(self):
        original = "Task\n-----Example-----\nInput\n2\nOutput\n4"
        variant = strip_examples(original)
        with tempfile.TemporaryDirectory() as directory:
            e5 = E5.__new__(E5)
            e5.cache = EmbeddingCache(Path(directory) / "cache.npz", MODEL_NAME)
            encoded = []

            def encode(texts):
                values = []
                for text in texts:
                    encoded.append(text)
                    values.append([len(encoded)])
                return np.asarray(values, dtype=np.float32)

            e5._encode_raw = encode
            before, before_miss = e5.encode_queries([original])
            after, after_miss = e5.encode_queries([variant])
            docs, doc_miss = e5.encode_docs([variant])
            warm, warm_misses = e5.encode_queries([variant, original])
            self.assertEqual((before_miss, after_miss, doc_miss, warm_misses), (1, 1, 1, 0))
            self.assertEqual(encoded, [QUERY_PREFIX + original, QUERY_PREFIX + variant, DOC_PREFIX + variant])
            np.testing.assert_array_equal(warm, np.concatenate([after, before]))
            self.assertFalse(np.array_equal(after, docs))
