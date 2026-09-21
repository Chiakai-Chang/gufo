"""Render BENCHMARKS.md tables from artifacts, keeping hand-entered cells."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .artifacts import artifact_path, load_artifact
from .config import BenchConfig, TableSpec

TODO = "TODO"
MARKER_RE = re.compile(
    r"<!-- bench:(?P<id>[\w-]+) -->\n(?P<body>.*?)<!-- /bench -->", re.DOTALL
)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


@dataclass
class Column:
    header: str
    owner: str  # gufo | reference | gain | exact | label
    align: str = "---:"


@dataclass
class Layout:
    columns: list[Column]
    rows: list[str]  # row labels in order


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.replace("**", "").strip()
    if text == TODO or text in ("", "-", "n/a"):
        return None
    match = NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return TODO
    return f"{value:.{digits}f}{suffix}"


def _fmt_stat(row: dict[str, Any] | None, key: str, digits: int = 2, suffix: str = "") -> str | None:
    if row is None or row.get(key) is None:
        return None
    text = _fmt(row[key], digits)
    sd = row.get(f"{key}_sd")
    if sd is not None:
        text += f" ± {sd:.{digits}f}"
    return text + suffix


def gain(gufo: float | None, reference: float | None, better: str) -> str:
    if gufo is None or reference is None or gufo <= 0 or reference <= 0:
        return TODO
    ratio = gufo / reference if better == "higher" else reference / gufo
    return f"{(ratio - 1) * 100:+.1f}%"


def _depth_label(depth: int) -> str:
    return f"{depth:,}"


def _merged_tokens(size: int) -> int:
    return (size // 32) ** 2


def layout_for(config: BenchConfig, table: TableSpec) -> Layout:
    ref = config.reference_name
    spec_label = config.speculative["label"]
    kind = table.kind
    if kind == "loading":
        return Layout(
            [Column("Target", "label", "---"), Column("Gufo ready", "gufo"),
             Column(f"{ref} ready", "reference"), Column("Gain", "gain")],
            [config.variant_label(v) for v in config.variants],
        )
    if kind == "single" and not table.speculative:
        return Layout(
            [Column("Depth", "label"), Column("Gufo pp", "gufo"), Column(f"{ref} pp", "reference"),
             Column("Gain", "gain"), Column("Gufo tg", "gufo"), Column(f"{ref} tg", "reference"),
             Column("Gain", "gain")],
            [_depth_label(d) for d in table.spec["depths"]],
        )
    if kind == "single":
        if config.reference_speculative:
            reference = [Column(f"{ref} {spec_label} tg", "reference"), Column("Gain", "gain")]
        else:
            reference = [Column(f"{ref} AR tg", "reference"), Column(f"Gain vs {ref} AR", "gain")]
        return Layout(
            [Column("Depth", "label"), Column("Gufo pp", "gufo"), Column("Gufo tg", "gufo"),
             Column("Acceptance", "gufo"), *reference],
            [_depth_label(d) for d in table.spec["depths"]],
        )
    if kind == "multi":
        if config.reference_speculative:
            speculative = [Column(f"Gufo {spec_label}", "gufo"), Column(f"{ref} {spec_label}", "reference"),
                           Column("Gain", "gain")]
        else:
            speculative = [Column(f"Gufo {spec_label}", "gufo"), Column(f"Gain vs {ref} AR", "gain")]
        return Layout(
            [Column("Users", "label"), Column("Gufo AR", "gufo"), Column(f"{ref} AR", "reference"),
             Column("Gain", "gain"), *speculative, Column("Exact", "exact")],
            [str(c) for c in table.spec["concurrency"]],
        )
    if kind == "memory":
        return Layout(
            [Column("Workload", "label", "---"), Column("Gufo GiB", "gufo"),
             Column(f"{ref} GiB", "reference"), Column("Gain", "gain")],
            [w["id"] for w in table.spec["workloads"]],
        )
    if kind == "image-encoder":
        return Layout(
            [Column("RGB image", "label", "---"), Column("Merged tokens", "label"),
             Column("Gufo ms", "gufo"), Column(f"{ref} ms", "reference"), Column("Gain", "gain")],
            [f"{s}×{s}" for s in table.spec["sizes"]],
        )
    raise SystemExit(f"no renderer for table {table.id}")


def parse_table(body: str) -> dict[str, list[str]]:
    """Existing table cells keyed by first-column label."""
    lines = [line for line in body.strip().splitlines() if line.startswith("|")]
    rows: dict[str, list[str]] = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0].replace("**", "")] = cells[1:]
    return rows


def existing_tables(document: str) -> dict[str, dict[str, list[str]]]:
    return {m.group("id"): parse_table(m.group("body")) for m in MARKER_RE.finditer(document)}


def _row_values(config: BenchConfig, table: TableSpec) -> dict[str, dict[str, str | None]]:
    """Fresh cell text per row label from artifacts; None means no artifact value."""
    values: dict[str, dict[str, str | None]] = {}
    kind = table.kind
    if kind == "multi":
        mode = config.speculative["mode"]
        gufo_ar = load_artifact(artifact_path(config, table, "gufo", "ar"))
        gufo_spec = load_artifact(artifact_path(config, table, "gufo", mode))
        reference = load_artifact(artifact_path(config, table, "reference"))
        ref_spec = load_artifact(artifact_path(config, table, "reference", mode))
        for users in table.spec["concurrency"]:
            values[str(users)] = {
                "gufo_ar": _fmt(_serving_rate(gufo_ar, users)) if gufo_ar else None,
                "reference": _fmt(_serving_rate(reference, users)) if reference else None,
                "gufo_spec": _fmt(_serving_rate(gufo_spec, users)) if gufo_spec else None,
                "ref_spec": _fmt(_serving_rate(ref_spec, users)) if ref_spec else None,
                "exact": _serving_exact(reference, users) if reference else None,
            }
        return values

    gufo = load_artifact(artifact_path(config, table, "gufo"))
    if kind == "single" and table.speculative and not config.reference_speculative:
        ar_table = config.table("single-ar" + (f"-{table.variant}" if table.variant else ""))
        reference = load_artifact(artifact_path(config, ar_table, "reference"))
    else:
        reference = load_artifact(artifact_path(config, table, "reference"))
    g_rows = (gufo or {}).get("rows", {})
    r_rows = (reference or {}).get("rows", {})

    if kind == "loading":
        for variant in config.variants:
            values[config.variant_label(variant)] = {
                "gufo": _fmt_stat(g_rows.get(variant), "ready_s", 2, " s"),
                "reference": _fmt_stat(r_rows.get(variant), "ready_s", 2, " s"),
            }
    elif kind == "single":
        for depth in table.spec["depths"]:
            g, r = g_rows.get(str(depth)), r_rows.get(str(depth))
            values[_depth_label(depth)] = {
                "gufo_pp": _fmt_stat(g, "pp"), "gufo_tg": _fmt_stat(g, "tg"),
                "acceptance": _fmt_stat(g, "acceptance", 1, "%"),
                "ref_pp": _fmt_stat(r, "pp"), "ref_tg": _fmt_stat(r, "tg"),
            }
    elif kind == "memory":
        for workload in table.spec["workloads"]:
            key = workload["id"]
            values[key] = {"gufo": _fmt_stat(g_rows.get(key), "gib"),
                           "reference": _fmt_stat(r_rows.get(key), "gib")}
    elif kind == "image-encoder":
        for size in table.spec["sizes"]:
            key = str(size)
            values[f"{size}×{size}"] = {"gufo": _fmt_stat(g_rows.get(key), "ms", 1),
                                        "reference": _fmt_stat(r_rows.get(key), "ms", 1)}
    return values


def _serving_rate(report: dict[str, Any] | None, users: int) -> float | None:
    if report is None:
        return None
    result = report.get("results", {}).get(f"c{users}")
    if result is None:
        return None
    return result["aggregate"]["output_tokens_per_second"]["overall"]


def _serving_exact(report: dict[str, Any] | None, users: int) -> str | None:
    if report is None:
        return None
    result = report.get("results", {}).get(f"c{users}")
    exactness = (result or {}).get("completionExactness")
    if not isinstance(exactness, dict):
        return None
    return f"{exactness['exactRequests']}/{exactness['comparedRequests']}"


def _pick(fresh: str | None, existing: str | None) -> str:
    """Gufo cells keep hand-entered text when no artifact covers them."""
    if fresh is not None:
        return fresh
    return existing if existing else TODO


def _ref(fresh: str | None) -> str:
    """Reference cells come from artifacts only, so layout changes never carry stale values."""
    return fresh if fresh is not None else TODO


def render_table(config: BenchConfig, table: TableSpec, existing: dict[str, list[str]] | None) -> str:
    layout = layout_for(config, table)
    values = _row_values(config, table)
    better = table.spec.get("better", "higher")
    kind = table.kind
    lines = ["| " + " | ".join(c.header for c in layout.columns) + " |",
             "| " + " | ".join(c.align for c in layout.columns) + " |"]
    for label in layout.rows:
        old = (existing or {}).get(label, [])
        cell = lambda index: old[index] if index < len(old) else None  # noqa: E731
        fresh = values.get(label, {})
        if kind == "loading" or kind == "memory":
            g = _pick(fresh.get("gufo"), cell(0))
            r = _ref(fresh.get("reference"))
            cells = [g, r, gain(_number(g), _number(r), better)]
        elif kind == "single" and not table.speculative:
            gp = _pick(fresh.get("gufo_pp"), cell(0)); rp = _ref(fresh.get("ref_pp"))
            gt = _pick(fresh.get("gufo_tg"), cell(3)); rt = _ref(fresh.get("ref_tg"))
            cells = [gp, rp, gain(_number(gp), _number(rp), better),
                     gt, rt, gain(_number(gt), _number(rt), better)]
        elif kind == "single":
            gp = _pick(fresh.get("gufo_pp"), cell(0)); gt = _pick(fresh.get("gufo_tg"), cell(1))
            acc = _pick(fresh.get("acceptance"), cell(2)); rt = _ref(fresh.get("ref_tg"))
            cells = [gp, gt, acc, rt, gain(_number(gt), _number(rt), better)]
        elif kind == "multi":
            ga = _pick(fresh.get("gufo_ar"), cell(0)); ra = _ref(fresh.get("reference"))
            gs = _pick(fresh.get("gufo_spec"), cell(3)); ex = _ref(fresh.get("exact"))
            cells = [ga, ra, gain(_number(ga), _number(ra), better), gs]
            if config.reference_speculative:
                rs = _ref(fresh.get("ref_spec"))
                cells += [rs, gain(_number(gs), _number(rs), better), ex]
            else:
                cells += [gain(_number(gs), _number(ra), better), ex]
        elif kind == "image-encoder":
            size = int(label.split("×")[0])
            g = _pick(fresh.get("gufo"), cell(1)); r = _ref(fresh.get("reference"))
            cells = [str(_merged_tokens(size)), g, r, gain(_number(g), _number(r), better)]
        else:
            raise SystemExit(f"no renderer for table {table.id}")
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines) + "\n"


def render_document(config: BenchConfig, document: str, only: set[str] | None = None) -> tuple[str, list[str]]:
    existing = existing_tables(document)
    known = {t.id: t for t in config.tables()}
    rendered: list[str] = []

    def replace(match: re.Match[str]) -> str:
        table_id = match.group("id")
        if table_id not in known or (only and table_id not in only):
            return match.group(0)
        body = render_table(config, known[table_id], existing.get(table_id))
        rendered.append(table_id)
        return f"<!-- bench:{table_id} -->\n{body}<!-- /bench -->"

    return MARKER_RE.sub(replace, document), rendered


def todo_rows(config: BenchConfig, document: str, table: TableSpec, target: str) -> set[str] | None:
    """Row labels whose target-owned cells are TODO; None when the table is absent."""
    tables = existing_tables(document)
    if table.id not in tables:
        return None
    layout = layout_for(config, table)
    owned = [i for i, c in enumerate(layout.columns[1:]) if c.owner == target or (target == "reference" and c.owner == "exact")]
    todo: set[str] = set()
    for label, cells in tables[table.id].items():
        if any(i >= len(cells) or cells[i] == TODO for i in owned):
            todo.add(label)
    return todo
