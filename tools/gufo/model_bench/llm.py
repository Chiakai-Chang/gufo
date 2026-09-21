"""LLM category: loading, single-user depth sweeps, concurrency, memory."""

from __future__ import annotations

import datetime as dt
import json
import random
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from gufo.serving_bench import (
    RequestObservation,
    load_prompt_suite,
    load_reference_report,
    run_corpus_benchmark,
    run_request,
)

from .artifacts import artifact_path, load_artifact, merge_rows, new_artifact, public_command, save_artifact
from .config import BenchConfig, TableSpec
from .servers import Server, drop_file_cache, rocm_used_gib, wait_process_exit

MODEL_ALIAS = "bench"
CLIENT_ID = "model-bench"
REQUEST_TIMEOUT = 3600.0

WORDS = (
    "the river bends past a quiet town where old mills once ground wheat for "
    "every family along the valley and children still climb the stone bridge "
    "to watch boats carry timber salt and wool toward distant markets while "
    "farmers mend fences count sheep and argue about rain clouds that gather "
    "over the western hills each autumn evening before the harvest festival "
    "brings music lanterns and long tables of bread cheese apples and cider "
    "shared by neighbours who remember older winters when snow closed the "
    "road for weeks and stories were traded by firelight instead of coins"
).split()


class Session:
    """Shared run state: config, target, server factory and identities."""

    def __init__(
        self,
        config: BenchConfig,
        target: str,
        *,
        gufo_binary: Path,
        reference_binary: str,
        source: dict[str, Any],
        fingerprint: dict[str, Any] | None,
        log_dir: Path,
        document: str,
        todo_only: bool,
        drop_caches: str | None = None,
    ):
        self.config = config
        self.target = target
        self.gufo_binary = gufo_binary
        self.reference_binary = reference_binary
        self.source = source
        self.fingerprint = fingerprint
        self.log_dir = log_dir
        self.document = document
        self.todo_only = todo_only
        self.drop_caches = drop_caches

    @property
    def profile(self) -> str:
        return "gufo" if self.target == "gufo" else self.config.data["reference"]["endpoint_profile"]

    @property
    def cache_prompt(self) -> bool | None:
        return None if self.target == "gufo" else True

    # ----- server commands -------------------------------------------------

    def gufo_command(self, table: TableSpec, *, mode: str | None, context: int, sessions: int, port: int) -> list[str]:
        cfg = self.config
        gguf = cfg.file("gguf", table.variant)
        command = [str(self.gufo_binary), "serve", "--port", str(port), "--sessions", str(sessions),
                   *cfg.data["gufo"].get("serve", []), "llm", "--model", str(gguf),
                   "--context", str(context), "--served-model-name", MODEL_ALIAS,
                   *cfg.data["gufo"].get("llm", [])]
        if "mmproj" in cfg.files:
            command += ["--mmproj", str(cfg.file("mmproj", table.variant))]
        if mode and mode != "ar":
            for part in cfg.speculative["gufo_args"]:
                if part.startswith("{") and part.endswith("}"):
                    part = str(cfg.file(part[1:-1], table.variant))
                command.append(part)
        return command

    def reference_command(self, table: TableSpec, *, context: int, parallel: int, port: int) -> list[str]:
        cfg = self.config
        gguf = cfg.file("gguf", table.variant)
        return [self.reference_binary, "-m", str(gguf), "-c", str(context), "-np", str(parallel),
                "--port", str(port), "--host", "127.0.0.1", "--alias", MODEL_ALIAS,
                *cfg.data["reference"]["args"]]

    def server(self, table: TableSpec, *, mode: str | None, context: int, sessions: int, tag: str) -> Server:
        readiness = self.config.data["gufo"]["readiness"] if self.target == "gufo" else self.config.data["reference"]["readiness"]
        log = self.log_dir / f"{table.id}-{self.target}-{tag}.log"
        placeholder = Server([], readiness, log)
        if self.target == "gufo":
            command = self.gufo_command(table, mode=mode, context=context, sessions=sessions, port=placeholder.port)
        else:
            command = self.reference_command(table, context=context, parallel=sessions, port=placeholder.port)
        placeholder.command = command
        return placeholder

    def reference_version(self) -> str | None:
        if self.target != "reference":
            return None
        if not hasattr(self, "_reference_version"):
            completed = subprocess.run([self.reference_binary, "--version"], capture_output=True, text=True)
            text = (completed.stdout or completed.stderr).strip().splitlines()
            self._reference_version = text[0] if text else None
        return self._reference_version

    def artifact(self, table: TableSpec, *, mode: str | None, command: list[str], notes: list[str]) -> dict[str, Any]:
        version = self.reference_version()
        if version:
            notes = [f"{self.config.reference_name}: {version}", *notes]
        return new_artifact(self.config, table, self.target, mode=mode, command=command,
                            source=self.source, fingerprint=self.fingerprint, notes=notes)

    def store(self, path: Path, artifact: dict[str, Any]) -> None:
        save_artifact(path, merge_rows(load_artifact(path), artifact))
        print(f"artifact: {path}")

    def request(self, base_url: str, prompt: str, max_tokens: int, *, index: int = 0,
                messages: list[dict[str, str]] | None = None) -> RequestObservation:
        return run_request(
            base_url=base_url, model=MODEL_ALIAS, prompt=prompt, max_tokens=max_tokens,
            temperature=float(self.config.data["sampling"]["temperature"]),
            timeout_seconds=REQUEST_TIMEOUT, client_id=CLIENT_ID, concurrency=1,
            repetition=1, request_index=index, endpoint_profile=self.profile,
            cache_prompt=self.cache_prompt, messages=messages,
        )

    def chat_text(self, base_url: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Non-streaming completion text, used to build a reusable conversation prefix."""
        payload = {"model": MODEL_ALIAS, "messages": messages, "max_tokens": max_tokens,
                   "temperature": float(self.config.data["sampling"]["temperature"]), "stream": False}
        if self.cache_prompt is not None:
            payload["cache_prompt"] = self.cache_prompt
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json", "X-Client-ID": CLIENT_ID}, method="POST")
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            reply = json.loads(response.read().decode("utf-8"))
        content = reply["choices"][0]["message"].get("content")
        return content if isinstance(content, str) else ""


# ----- synthetic prompts ---------------------------------------------------


def synthetic_text(seed: int, words: int) -> str:
    """Deterministic prose-like text with a fixed vocabulary."""
    rng = random.Random(seed)
    out: list[str] = []
    count = 0
    while count < words:
        sentence_len = min(rng.randint(6, 14), words - count)
        sentence = [rng.choice(WORDS) for _ in range(sentence_len)]
        sentence[0] = sentence[0].capitalize()
        out.append(" ".join(sentence) + ".")
        count += sentence_len
        if rng.random() < 0.2:
            out.append("\n\n")
    return " ".join(out).replace(" \n\n ", "\n\n").strip()


class Tokenizer:
    """Estimates token counts for synthetic text from server-reported usage."""

    def __init__(self, session: Session, base_url: str):
        self.session = session
        self.base_url = base_url
        overhead = session.request(base_url, "Hi", 1).prompt_tokens - 1
        probe_words = 3000
        probe = session.request(base_url, synthetic_text(7777, probe_words), 1).prompt_tokens
        self.overhead = overhead
        self.ratio = (probe - overhead) / probe_words

    def words_for(self, tokens: int) -> int:
        return max(1, round(tokens / self.ratio))


def _tolerance(target: int, fraction: float) -> int:
    return max(32, int(target * fraction))


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    if len(values) < 2:
        return values[0], None
    return statistics.fmean(values), statistics.stdev(values)


# ----- tables ----------------------------------------------------------------


def _selected_rows(session: Session, table: TableSpec, labels: dict[Any, str]) -> list[Any]:
    """Row keys to measure, honouring --todo against the current document."""
    if not session.todo_only:
        return list(labels)
    from .render import todo_rows

    todo = todo_rows(session.config, session.document, table, session.target)
    if todo is None:
        return list(labels)
    return [key for key, label in labels.items() if label in todo]


def run_loading(session: Session, table: TableSpec) -> None:
    cfg = session.config
    spec = table.spec
    keys = _selected_rows(session, table, {v: cfg.variant_label(v) for v in cfg.variants})
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    rows: dict[str, Any] = {}
    command: list[str] = []
    for variant in keys:
        sub = TableSpec(table.id, table.base, variant, spec)
        cfg.require_files(variant)
        samples: list[float] = []
        for repetition in range(int(spec.get("repetitions", 1))):
            drop_file_cache(session.drop_caches)
            mode = cfg.speculative["mode"] if session.target == "gufo" else None
            server = session.server(sub, mode=mode, context=int(spec["context"]),
                                    sessions=int(spec.get("sessions", 2)), tag=f"{variant}-{repetition}")
            command = server.command
            with server:
                assert server.ready_seconds is not None
                samples.append(server.ready_seconds)
            wait_process_exit(server)
        mean, sd = _mean_sd(samples)
        rows[variant] = {"ready_s": round(mean, 3), "ready_s_sd": None if sd is None else round(sd, 3),
                         "samples": len(samples)}
        print(f"{table.id} {variant}: ready {mean:.2f} s")
    artifact = session.artifact(table, mode=None, command=command,
                                notes=["cold file cache: `echo 3 > /proc/sys/vm/drop_caches` before each launch",
                                       f"readiness = HTTP 200 on {cfg.data['gufo' if session.target == 'gufo' else 'reference']['readiness']}"])
    artifact["rows"] = rows
    session.store(artifact_path(cfg, table, session.target), artifact)


def run_single(session: Session, table: TableSpec) -> None:
    cfg = session.config
    spec = table.spec
    if table.speculative and session.target != "gufo":
        print(f"{table.id}: reference uses the single-ar table ({cfg.speculative['reference']})")
        return
    cfg.require_files(table.variant)
    depths = [int(d) for d in spec["depths"]]
    keys = _selected_rows(session, table, {d: f"{d:,}" for d in depths})
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    prompt_tokens = int(spec["prompt_tokens"])
    output_tokens = int(spec["output_tokens"])
    fraction = float(spec.get("depth_tolerance", 0.005))
    repetitions = int(spec.get("repetitions", 1))
    base_seed = int(spec.get("prefix", {}).get("seed", 1))
    mode = cfg.speculative["mode"] if table.speculative else "ar"

    server = session.server(table, mode=mode, context=int(spec["context"]), sessions=1, tag="single")
    rows: dict[str, Any] = {}
    with server:
        tokenizer = Tokenizer(session, server.base_url)
        # Warm kernels and allocations with an untimed full-size request.
        session.request(server.base_url, synthetic_text(8888, tokenizer.words_for(prompt_tokens)), 16)
        for depth in keys:
            pps: list[float] = []
            tgs: list[float] = []
            accepts: list[float] = []
            counts: list[dict[str, int]] = []
            for repetition in range(repetitions):
                observation = _measure_depth(session, server.base_url, tokenizer, depth=depth,
                                             prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                                             fraction=fraction, seed=base_seed, repetition=repetition)
                if observation.prefill_tokens_per_second is not None:
                    pps.append(observation.prefill_tokens_per_second)
                if observation.decode_tokens_per_second is not None:
                    tgs.append(observation.decode_tokens_per_second)
                if observation.draft_acceptance is not None:
                    accepts.append(observation.draft_acceptance * 100.0)
                counts.append({"cache_n": observation.cached_prompt_tokens,
                               "prompt_n": observation.prefill_tokens,
                               "predicted_n": observation.completion_tokens})
            row: dict[str, Any] = {"counts": counts, "samples": repetitions}
            for name, values in (("pp", pps), ("tg", tgs), ("acceptance", accepts)):
                if values:
                    mean, sd = _mean_sd(values)
                    row[name] = round(mean, 2)
                    row[f"{name}_sd"] = None if sd is None else round(sd, 2)
            rows[str(depth)] = row
            print(f"{table.id} d{depth}: pp {row.get('pp')} tg {row.get('tg')} acc {row.get('acceptance')}")
    wait_process_exit(server)
    artifact = session.artifact(table, mode=None, command=server.command, notes=[
        f"pp{prompt_tokens}/tg{output_tokens}; depth is a cached prefix of synthetic text; "
        f"tolerance max(32, {fraction:.3%}) on cache_n and prompt_n; actual counts per sample in rows",
        f"synthetic text: {tokenizer.ratio:.3f} tokens/word, template overhead {tokenizer.overhead} tokens",
    ])
    artifact["rows"] = rows
    session.store(artifact_path(cfg, table, session.target), artifact)


def _measure_depth(session: Session, base_url: str, tokenizer: Tokenizer, *, depth: int, prompt_tokens: int,
                   output_tokens: int, fraction: float, seed: int, repetition: int) -> RequestObservation:
    """Time pp/tg after a cached conversation prefix of about `depth` tokens.

    Both servers reuse a prior turn's state: the prefix is sent as its own turn
    (one generated token), and the measured request continues that conversation
    with a new user turn of about `prompt_tokens` tokens.
    """
    new_target = prompt_tokens - tokenizer.overhead
    ratio = tokenizer.ratio
    for attempt in range(4):
        messages: list[dict[str, str]] = []
        if depth > 0:
            prefix_target = depth - tokenizer.overhead - 1
            prefix = synthetic_text(seed + depth, max(1, round(prefix_target / ratio)))
            reply = session.chat_text(base_url, [{"role": "user", "content": prefix}], 1)
            messages = [{"role": "user", "content": prefix}, {"role": "assistant", "content": reply}]
        new_text = synthetic_text(100_000 + depth * 10 + repetition * 100 + attempt, max(1, round(new_target / ratio)))
        messages.append({"role": "user", "content": new_text})
        observation = session.request(base_url, new_text, output_tokens, index=attempt, messages=messages)
        cache_ok = abs(observation.cached_prompt_tokens - depth) <= _tolerance(depth, fraction)
        prefill_ok = abs(observation.prefill_tokens - prompt_tokens) <= _tolerance(prompt_tokens, fraction)
        if cache_ok and prefill_ok:
            return observation
        print(f"  d{depth} attempt {attempt}: cache_n {observation.cached_prompt_tokens} prompt_n "
              f"{observation.prefill_tokens}; recalibrating")
        measured = observation.prefill_tokens + observation.cached_prompt_tokens
        expected_words = sum(len(m["content"].split()) for m in messages if m["role"] == "user")
        ratio = (measured - tokenizer.overhead) / expected_words
    raise RuntimeError(f"depth {depth}: cached prefix outside tolerance after 4 attempts")


def run_multi(session: Session, table: TableSpec) -> None:
    cfg = session.config
    spec = table.spec
    cfg.require_files(table.variant)
    levels = [int(c) for c in spec["concurrency"]]
    keys = _selected_rows(session, table, {c: str(c) for c in levels})
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    suite = (cfg.model_dir / "artifacts" / spec["suite"]).resolve()
    cases = load_prompt_suite(suite, selected=set(spec["cases"]))
    modes = list(spec.get("modes", ["ar"])) if session.target == "gufo" else [None]
    for mode in modes:
        path = artifact_path(cfg, table, session.target, mode)
        reference = None
        ar_path = artifact_path(cfg, table, "gufo", "ar")
        if path != ar_path and ar_path.exists():
            reference = load_reference_report(ar_path)
        combined = load_artifact(path)
        for users in keys:
            context = int(spec["context"])
            server = session.server(table, mode=mode, context=context if session.target == "gufo" else context * users,
                                    sessions=users, tag=f"{mode or 'ref'}-c{users}")
            with server:
                report = run_corpus_benchmark(
                    base_url=server.base_url, model=MODEL_ALIAS, cases=cases,
                    workload_id=f"{cfg.model}-{table.id}-{mode or session.target}",
                    max_tokens=int(spec["output_tokens"]),
                    temperature=float(cfg.data["sampling"]["temperature"]),
                    concurrency_levels=[users], warmup_rounds=int(spec.get("warmup", 1)),
                    repetitions=int(spec.get("repetitions", 1)), timeout_seconds=REQUEST_TIMEOUT,
                    fingerprint=session.fingerprint or {}, source_revision=session.source["revision"],
                    source_dirty=session.source["dirty"], suite_bytes=suite.read_bytes(),
                    corpus_layout=spec.get("corpus_layout", "distinct"), endpoint_profile=session.profile,
                    cache_prompt=False if not spec.get("cache_prompt", False) else None,
                    reference=reference,
                    notes=[note for note in (session.reference_version(), " ".join(public_command(server.command)),
                                             "fresh server per concurrency level") if note],
                )
            wait_process_exit(server)
            if combined is None or combined.get("artifactType") != report.get("artifactType"):
                combined = report
            else:
                combined["results"].update(report["results"])
                combined["notes"] = report.get("notes", combined.get("notes"))
            combined["modelBench"] = {"table": table.id, "target": session.target, "mode": mode,
                                      "measuredOn": dt.date.today().isoformat()}
            save_artifact(path, combined)
            rate = report["results"][f"c{users}"]["aggregate"]["output_tokens_per_second"]["overall"]
            print(f"{table.id} {mode or session.target} C{users}: {rate:.2f} tok/s -> {path}")


class MemoryPoller:
    def __init__(self) -> None:
        self.peak: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = rocm_used_gib()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used
            time.sleep(0.25)

    def __enter__(self) -> "MemoryPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def run_memory(session: Session, table: TableSpec) -> None:
    cfg = session.config
    spec = table.spec
    cfg.require_files(table.variant)
    workloads = {w["id"]: w for w in spec["workloads"]}
    keys = _selected_rows(session, table, {k: k for k in workloads})
    if not keys:
        print(f"{table.id}: nothing to do")
        return
    if rocm_used_gib() is None:
        raise SystemExit("memory table needs rocm-smi with --showmemuse --json")
    mode = cfg.speculative["mode"] if session.target == "gufo" else "ar"
    server = session.server(table, mode=mode, context=int(spec["context"]), sessions=1, tag="memory")
    rows: dict[str, Any] = {}
    with server:
        tokenizer = Tokenizer(session, server.base_url)
        for key in keys:
            workload = workloads[key]
            depth = int(workload["depth"])
            with MemoryPoller() as poller:
                _measure_depth(session, server.base_url, tokenizer, depth=depth,
                               prompt_tokens=int(workload["prompt_tokens"]),
                               output_tokens=int(workload["output_tokens"]), fraction=0.02, seed=1, repetition=0)
            rows[key] = {"gib": None if poller.peak is None else round(poller.peak, 2)}
            print(f"{table.id} {key}: {rows[key]['gib']} GiB")
    wait_process_exit(server)
    artifact = session.artifact(table, mode=None, command=server.command,
                                notes=["peak device-wide VRAM use sampled with rocm-smi every 250 ms during the request"])
    artifact["rows"] = rows
    session.store(artifact_path(cfg, table, session.target), artifact)


def run_image_encoder(session: Session, table: TableSpec) -> None:
    raise SystemExit(f"{table.id}: image-encoder measurement is not implemented yet; leave the cells TODO")


RUNNERS = {
    "loading": run_loading,
    "single": run_single,
    "multi": run_multi,
    "memory": run_memory,
    "image-encoder": run_image_encoder,
}


def run_table(session: Session, table: TableSpec) -> None:
    RUNNERS[table.kind](session, table)
