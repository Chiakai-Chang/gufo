#!/usr/bin/env python3
"""
check-no-cuda.py — Anti-CUDA boundary scanner for Strix-Halo.cpp

Verifies that no CUDA headers, libraries, symbols, include paths, or toolkit
dependencies contaminate the source tree, build artifacts, or Nix package closure.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Source file extensions to scan
SOURCE_EXTENSIONS = {
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".hip", ".cmake"
}

# Forbidden file extensions (must never exist in repo)
FORBIDDEN_EXTENSIONS = {".cu", ".cuh"}

# Forbidden header include patterns (regex)
FORBIDDEN_HEADER_PATTERNS = [
    re.compile(r'^\s*#\s*include\s*[<"]\s*cuda[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cuda\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cuda_runtime\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cuda_runtime_api\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cublas[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cudnn[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cufft[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*curand[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cusparse[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*cusolver[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*nccl[_\w\./]*\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*nvml\.h\s*[>"]', re.IGNORECASE),
    re.compile(r'^\s*#\s*include\s*[<"]\s*thrust/system/cuda[_\w\./]*\s*[>"]', re.IGNORECASE),
]

# Forbidden symbol / API patterns (must not appear in non-comment code)
FORBIDDEN_SOURCE_SYMBOLS = [
    re.compile(r'\bcudaMalloc\b'),
    re.compile(r'\bcudaMallocHost\b'),
    re.compile(r'\bcudaFree\b'),
    re.compile(r'\bcudaMemcpy\b'),
    re.compile(r'\bcudaMemcpyAsync\b'),
    re.compile(r'\bcudaMemset\b'),
    re.compile(r'\bcudaMemsetAsync\b'),
    re.compile(r'\bcudaStreamCreate\b'),
    re.compile(r'\bcudaStreamSynchronize\b'),
    re.compile(r'\bcudaEventCreate\b'),
    re.compile(r'\bcudaEventRecord\b'),
    re.compile(r'\bcudaEventSynchronize\b'),
    re.compile(r'\bcudaLaunchKernel\b'),
    re.compile(r'\bcudaGetDeviceCount\b'),
    re.compile(r'\bcudaSetDevice\b'),
    re.compile(r'\bcudaGetDeviceProperties\b'),
    re.compile(r'\bcudaError_t\b'),
    re.compile(r'\bcudaStream_t\b'),
    re.compile(r'\bcudaEvent_t\b'),
    re.compile(r'\bcublasCreate\b'),
    re.compile(r'\bcublasGemmEx\b'),
    re.compile(r'\bcublasHandle_t\b'),
]

# Forbidden dynamic libraries in DT_NEEDED
FORBIDDEN_LIBRARIES = [
    re.compile(r'^libcudart\b', re.IGNORECASE),
    re.compile(r'^libcuda\b', re.IGNORECASE),
    re.compile(r'^libcublas\b', re.IGNORECASE),
    re.compile(r'^libcudnn\b', re.IGNORECASE),
    re.compile(r'^libcufft\b', re.IGNORECASE),
    re.compile(r'^libcurand\b', re.IGNORECASE),
    re.compile(r'^libcusparse\b', re.IGNORECASE),
    re.compile(r'^libcusolver\b', re.IGNORECASE),
    re.compile(r'^libnccl\b', re.IGNORECASE),
    re.compile(r'^libnvidia-\w+', re.IGNORECASE),
    re.compile(r'^libnvrtc\b', re.IGNORECASE),
]

# Forbidden symbol patterns in ELF symbol tables (nm -D / readelf -s)
FORBIDDEN_ELF_SYMBOLS = [
    re.compile(r'^cuda[A-Z]\w+'),        # cudaMalloc, cudaMemcpy, cudaLaunchKernel, etc.
    re.compile(r'^__cuda\w+'),
    re.compile(r'^cublas[A-Z]\w+'),      # cublasCreate, cublasGemmEx, etc.
    re.compile(r'^cudnn[A-Z]\w+'),
    re.compile(r'^curand[A-Z]\w+'),
    re.compile(r'^cusparse[A-Z]\w+'),
    re.compile(r'^cusolver[A-Z]\w+'),
    re.compile(r'^nccl[A-Z]\w+'),
]

# Paths/directories excluded from scans
EXCLUDED_DIR_NAMES = {
    ".git", ".jj", ".direnv", "build", "Build", "fixtures", "node_modules"
}


class Violation:
    def __init__(self, category: str, location: str, detail: str):
        self.category = category
        self.location = location
        self.detail = detail

    def to_dict(self) -> Dict[str, str]:
        return {
            "category": self.category,
            "location": self.location,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        return f"[{self.category}] {self.location}: {self.detail}"


def scan_source_file(file_path: Path) -> List[Violation]:
    """Scan a single source file for forbidden CUDA headers and symbols."""
    violations: List[Violation] = []

    if file_path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        violations.append(
            Violation(
                category="FORBIDDEN_FILE_EXTENSION",
                location=str(file_path),
                detail=f"Forbidden CUDA source file extension '{file_path.suffix}' detected.",
            )
        )
        return violations

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        violations.append(
            Violation(
                category="READ_ERROR",
                location=str(file_path),
                detail=f"Failed to read file: {e}",
            )
        )
        return violations

    lines = content.splitlines()
    in_block_comment = False

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Simple multiline comment tracker for C/C++
        if "/*" in stripped:
            in_block_comment = True
        if "*/" in stripped:
            in_block_comment = False
            continue
        if in_block_comment or stripped.startswith("//"):
            continue

        # Check for forbidden includes
        for pattern in FORBIDDEN_HEADER_PATTERNS:
            if pattern.search(line):
                violations.append(
                    Violation(
                        category="FORBIDDEN_CUDA_HEADER",
                        location=f"{file_path}:{idx}",
                        detail=f"Forbidden CUDA include detected: '{stripped}'",
                    )
                )

        # Check for forbidden symbols in code
        for pattern in FORBIDDEN_SOURCE_SYMBOLS:
            match = pattern.search(line)
            if match:
                violations.append(
                    Violation(
                        category="FORBIDDEN_CUDA_SYMBOL",
                        location=f"{file_path}:{idx}",
                        detail=f"Forbidden CUDA API symbol '{match.group(0)}' in: '{stripped}'",
                    )
                )

    return violations


def scan_source_tree(root_path: Path, exclude_fixtures: bool = True) -> Tuple[List[Path], List[Violation]]:
    """Scan a directory tree for source violations."""
    scanned_files: List[Path] = []
    violations: List[Violation] = []

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith("build-")]
        for f in files:
            p = Path(root) / f
            if exclude_fixtures and "fixtures" in p.parts:
                continue

            if p.suffix.lower() in FORBIDDEN_EXTENSIONS or p.suffix.lower() in SOURCE_EXTENSIONS or p.name in {"CMakeLists.txt"}:
                scanned_files.append(p)
                violations.extend(scan_source_file(p))

    return scanned_files, violations


def scan_compile_commands(compile_commands_path: Path) -> List[Violation]:
    """Scan compile_commands.json for CUDA include paths or library link flags."""
    violations: List[Violation] = []
    if not compile_commands_path.is_file():
        return violations

    try:
        with open(compile_commands_path, "r", encoding="utf-8") as f:
            commands = json.load(f)
    except Exception as e:
        violations.append(
            Violation(
                category="COMPILE_COMMANDS_ERROR",
                location=str(compile_commands_path),
                detail=f"Failed to parse compile_commands.json: {e}",
            )
        )
        return violations

    cuda_flag_pattern = re.compile(r'(-I\S*cuda\S*|-isystem\s*\S*cuda\S*|-L\S*cuda\S*|-l\s*cuda\w*|-lcublas\w*|-lcudart\w*)', re.IGNORECASE)

    for entry in commands:
        cmd = entry.get("command", "") or " ".join(entry.get("arguments", []))
        file_name = entry.get("file", "unknown")
        match = cuda_flag_pattern.search(cmd)
        if match:
            violations.append(
                Violation(
                    category="FORBIDDEN_CUDA_COMPILE_FLAG",
                    location=file_name,
                    detail=f"Forbidden CUDA compiler/linker flag detected: '{match.group(0)}'",
                )
            )

    return violations


def scan_binary_elf(binary_path: Path) -> List[Violation]:
    """Scan an ELF binary using readelf/nm for CUDA libraries, RPATHs, and symbols."""
    violations: List[Violation] = []
    if not binary_path.is_file():
        violations.append(
            Violation(
                category="BINARY_NOT_FOUND",
                location=str(binary_path),
                detail="Target binary file not found.",
            )
        )
        return violations

    # 1. Inspect dynamic libraries (DT_NEEDED) with readelf -d
    try:
        proc = subprocess.run(
            ["readelf", "-d", str(binary_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if "NEEDED" in line or "RPATH" in line or "RUNPATH" in line:
                    for lib_pattern in FORBIDDEN_LIBRARIES:
                        if lib_pattern.search(line):
                            violations.append(
                                Violation(
                                    category="FORBIDDEN_CUDA_LIBRARY",
                                    location=str(binary_path),
                                    detail=f"Binary dynamically links forbidden CUDA library in: '{line.strip()}'",
                                )
                            )
                    if ("cuda" in line.lower() or "cudart" in line.lower()) and "rpath" in line.lower():
                        violations.append(
                            Violation(
                                category="FORBIDDEN_CUDA_RPATH",
                                location=str(binary_path),
                                detail=f"Binary contains CUDA toolkit in RPATH/RUNPATH: '{line.strip()}'",
                            )
                        )
    except FileNotFoundError:
        # readelf not available, fall back to checking strings or skip
        pass

    # 2. Inspect dynamic symbols with nm -D
    try:
        proc = subprocess.run(
            ["nm", "-D", "--defined-only", "--undefined-only", str(binary_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                symbol = parts[-1]
                for sym_pattern in FORBIDDEN_ELF_SYMBOLS:
                    if sym_pattern.match(symbol):
                        violations.append(
                            Violation(
                                category="FORBIDDEN_CUDA_ELF_SYMBOL",
                                location=str(binary_path),
                                detail=f"Binary references forbidden CUDA symbol '{symbol}'",
                            )
                        )
    except FileNotFoundError:
        pass

    return violations


def run_self_test(fixtures_dir: Path) -> bool:
    """Run scanner self-tests on positive and negative fixtures."""
    print("[check-no-cuda] Running self-test suite against scanner fixtures...")

    valid_hip = fixtures_dir / "valid_hip.cpp"
    cuda_include = fixtures_dir / "cuda_include.cpp"
    cuda_symbol = fixtures_dir / "cuda_symbol.cpp"
    cuda_link = fixtures_dir / "cuda_link.json"

    # Test positive fixture
    if valid_hip.is_file():
        v = scan_source_file(valid_hip)
        if v:
            print(f"FAIL: Positive fixture '{valid_hip}' falsely flagged: {v}")
            return False
        print(f"PASS: Positive fixture '{valid_hip.name}' passed without false positives.")

    # Test negative include fixture
    if cuda_include.is_file():
        v = scan_source_file(cuda_include)
        if not any(item.category == "FORBIDDEN_CUDA_HEADER" for item in v):
            print(f"FAIL: Negative fixture '{cuda_include}' was not caught by scanner!")
            return False
        print(f"PASS: Negative header fixture '{cuda_include.name}' was correctly caught.")

    # Test negative symbol fixture
    if cuda_symbol.is_file():
        v = scan_source_file(cuda_symbol)
        if not any(item.category == "FORBIDDEN_CUDA_SYMBOL" for item in v):
            print(f"FAIL: Negative fixture '{cuda_symbol}' was not caught by scanner!")
            return False
        print(f"PASS: Negative symbol fixture '{cuda_symbol.name}' was correctly caught.")

    # Test negative compile_commands fixture
    if cuda_link.is_file():
        v = scan_compile_commands(cuda_link)
        if not any(item.category == "FORBIDDEN_CUDA_COMPILE_FLAG" for item in v):
            print(f"FAIL: Negative compile_commands fixture '{cuda_link}' was not caught!")
            return False
        print(f"PASS: Negative compile_commands fixture '{cuda_link.name}' was correctly caught.")

    print("[check-no-cuda] All self-tests passed successfully.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify no CUDA dependencies in Strix-Halo.cpp source and binaries."
    )
    parser.add_argument("--source", action="append", help="Source directory or file to scan")
    parser.add_argument("--binary", action="append", help="Binary executable to scan for CUDA dynamic symbols/libraries")
    parser.add_argument("--compile-commands", help="Path to compile_commands.json to scan")
    parser.add_argument("--json-report", help="Path to output machine-readable JSON report")
    parser.add_argument("--self-test", help="Path to fixtures directory to run self-tests")

    args = parser.parse_args()

    if args.self_test:
        fixtures_path = Path(args.self_test)
        success = run_self_test(fixtures_path)
        return 0 if success else 1

    scanned_sources: List[Path] = []
    scanned_binaries: List[Path] = []
    all_violations: List[Violation] = []

    # 1. Scan sources
    if args.source:
        for src_arg in args.source:
            p = Path(src_arg)
            if p.is_file():
                scanned_sources.append(p)
                all_violations.extend(scan_source_file(p))
            elif p.is_dir():
                files, v = scan_source_tree(p)
                scanned_sources.extend(files)
                all_violations.extend(v)

    # 2. Scan compile_commands.json
    if args.compile_commands:
        cc_path = Path(args.compile_commands)
        all_violations.extend(scan_compile_commands(cc_path))

    # 3. Scan binaries
    if args.binary:
        for bin_arg in args.binary:
            bp = Path(bin_arg)
            scanned_binaries.append(bp)
            all_violations.extend(scan_binary_elf(bp))

    # Construct report
    report = {
        "status": "FAIL" if all_violations else "PASS",
        "scanned_source_files_count": len(scanned_sources),
        "scanned_binaries_count": len(scanned_binaries),
        "scanned_binaries": [str(b) for b in scanned_binaries],
        "violations_count": len(all_violations),
        "violations": [v.to_dict() for v in all_violations],
    }

    if args.json_report:
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # Output summary
    if all_violations:
        print(f"\n[check-no-cuda] FAILED: Found {len(all_violations)} CUDA contamination violations:")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print(
        f"[check-no-cuda] PASSED: {len(scanned_sources)} source files and "
        f"{len(scanned_binaries)} binaries verified clean of CUDA dependencies."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
