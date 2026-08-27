"""Unit tests for capability evaluation answer extractors and dataset validity."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.gufo import eval_extract


class TestEvalExtract(unittest.TestCase):

    def setUp(self):
        self.quality_dir = Path(__file__).resolve().parent.parent / "quality"

    def test_mcq_extractor_fixtures(self):
        fixtures_path = self.quality_dir / "extractor-fixtures.json"
        self.assertTrue(fixtures_path.exists())
        data = json.loads(fixtures_path.read_text("utf-8"))

        for case in data["cases"]:
            if case["type"] == "mcq":
                extracted = eval_extract.extract_mcq_answer(case["text"])
                self.assertEqual(extracted, case["expected"])
            elif case["type"] == "integer":
                extracted = eval_extract.extract_integer_answer(case["text"])
                self.assertEqual(extracted, case["expected"])
            elif case["type"] == "linespec":
                extracted = eval_extract.extract_linespec_answer(case["text"])
                self.assertEqual(extracted, case["expected"])

    def test_gpqa_dataset_validity(self):
        gpqa_path = self.quality_dir / "gpqa.json"
        self.assertTrue(gpqa_path.exists())
        data = json.loads(gpqa_path.read_text("utf-8"))
        self.assertEqual(len(data["cases"]), 25)
        for case in data["cases"]:
            self.assertIn("id", case)
            self.assertIn("question", case)
            self.assertIn("options", case)
            self.assertIn(case["answer"], ["A", "B", "C", "D"])

    def test_supergpqa_dataset_validity(self):
        supergpqa_path = self.quality_dir / "supergpqa.json"
        self.assertTrue(supergpqa_path.exists())
        data = json.loads(supergpqa_path.read_text("utf-8"))
        self.assertEqual(len(data["cases"]), 25)
        for case in data["cases"]:
            self.assertIn("id", case)
            self.assertIn("question", case)
            self.assertIn("options", case)
            self.assertIn(case["answer"], ["A", "B", "C", "D"])

    def test_aime2025_dataset_validity(self):
        aime_path = self.quality_dir / "aime2025.json"
        self.assertTrue(aime_path.exists())
        data = json.loads(aime_path.read_text("utf-8"))
        self.assertEqual(len(data["cases"]), 25)
        for case in data["cases"]:
            self.assertIn("id", case)
            self.assertIn("question", case)
            self.assertIsInstance(case["answer"], int)

    def test_provenance_manifest_validity(self):
        provenance_path = self.quality_dir / "provenance.json"
        self.assertTrue(provenance_path.exists())
        data = json.loads(provenance_path.read_text("utf-8"))
        self.assertEqual(data["schema"], "gufo.eval-provenance.v1")
        self.assertIn("gpqa", data["suites"])
        self.assertIn("supergpqa", data["suites"])
        self.assertIn("aime2025", data["suites"])


if __name__ == "__main__":
    unittest.main()
