"""Server lifecycle for Gufo and reference servers."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_READY_TIMEOUT = 3600.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


DROP_CACHES = "sync; echo 3 > /proc/sys/vm/drop_caches"


def drop_file_cache(command: str | None = None) -> None:
    """Drop the page cache so the next launch measures a cold file cache."""
    candidates = [command] if command else [
        DROP_CACHES,
        f"sudo -n sh -c '{DROP_CACHES}'",
        f"doas -n sh -c '{DROP_CACHES}'",
    ]
    for candidate in candidates:
        completed = subprocess.run(candidate, shell=True, capture_output=True, text=True)
        if completed.returncode == 0:
            return
    raise SystemExit(
        "cold-file-cache loading needs privileges to run "
        f"`{DROP_CACHES}`; pass --drop-caches with a command that does it "
        "(for example `doas sh -c '...'`), or skip the loading table"
    )


def rocm_used_gib() -> float | None:
    """Device-visible memory in use (VRAM + GTT) from rocm-smi, or None when unavailable."""
    if shutil.which("rocm-smi") is None:
        return None
    completed = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "gtt", "--json"], capture_output=True, text=True
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    total = 0.0
    found = False
    for card in payload.values():
        if not isinstance(card, dict):
            continue
        for key in ("VRAM Total Used Memory (B)", "GTT Total Used Memory (B)"):
            try:
                total += float(card[key])
                found = True
            except (KeyError, TypeError, ValueError):
                continue
    return total / (1024 ** 3) if found else None


class Server:
    """A benchmark server process bound to a private loopback port."""

    def __init__(self, command: list[str], readiness: str, log_path: Path, env: dict[str, str] | None = None):
        self.command = command
        self.readiness = readiness
        self.log_path = log_path
        self.env = env
        self.port = free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self.ready_seconds: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = DEFAULT_READY_TIMEOUT) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        started = time.perf_counter()
        self.process = subprocess.Popen(
            self.command, stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
        deadline = started + timeout
        url = self.base_url + self.readiness
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.process.returncode} before readiness; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        self.ready_seconds = time.perf_counter() - started
                        return self
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(f"server not ready within {timeout:.0f} s; see {self.log_path}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def wait_process_exit(server: Server) -> None:
    server.stop()
    # Give the driver time to release device memory before the next launch.
    time.sleep(2.0)
