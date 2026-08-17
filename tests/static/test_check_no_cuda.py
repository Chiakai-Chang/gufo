#!/usr/bin/env python3
"""
test_check_no_cuda.py — Unit test suite for anti-CUDA scanner
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCANNER_SCRIPT = REPO_ROOT / "tools" / "check-no-cuda.py"
FIXTURES_DIR = REPO_ROOT / "tests" / "static" / "fixtures"


class TestCheckNoCudaScanner(unittest.TestCase):
    def test_self_test_fixtures(self):
        """Verify the built-in self-test mode against positive & negative fixtures."""
        res = subprocess.run(
            [sys.executable, str(SCANNER_SCRIPT), "--self-test", str(FIXTURES_DIR)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Scanner self-test failed:\n{res.stdout}\n{res.stderr}")
        self.assertIn("All self-tests passed successfully", res.stdout)

    def test_clean_source_tree(self):
        """Verify that the actual project source tree passes with 0 CUDA violations."""
        res = subprocess.run(
            [
                sys.executable,
                str(SCANNER_SCRIPT),
                "--source",
                str(REPO_ROOT / "src"),
                "--source",
                str(REPO_ROOT / "models"),
                "--source",
                str(REPO_ROOT / "CMakeLists.txt"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Source tree scan failed:\n{res.stdout}\n{res.stderr}")
        self.assertIn("PASSED", res.stdout)

    def test_negative_include_detected(self):
        """Verify scanner catches forbidden CUDA header include."""
        res = subprocess.run(
            [
                sys.executable,
                str(SCANNER_SCRIPT),
                "--source",
                str(FIXTURES_DIR / "cuda_include.cpp"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("FORBIDDEN_CUDA_HEADER", res.stdout)

    def test_negative_symbol_detected(self):
        """Verify scanner catches forbidden CUDA API symbol."""
        res = subprocess.run(
            [
                sys.executable,
                str(SCANNER_SCRIPT),
                "--source",
                str(FIXTURES_DIR / "cuda_symbol.cpp"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("FORBIDDEN_CUDA_SYMBOL", res.stdout)

    def test_negative_compile_commands_detected(self):
        """Verify scanner catches CUDA compile flags in compile_commands.json."""
        res = subprocess.run(
            [
                sys.executable,
                str(SCANNER_SCRIPT),
                "--compile-commands",
                str(FIXTURES_DIR / "cuda_link.json"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("FORBIDDEN_CUDA_COMPILE_FLAG", res.stdout)


if __name__ == "__main__":
    unittest.main()
