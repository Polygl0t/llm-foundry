#!/usr/bin/env python
"""Aggregate lm-evaluation-harness results into a report



Usage:

    python scripts/aggregate_evals.py
    python scripts/aggregate_evals.py --input models/.evals
    python scripts/aggregate_evals.py --input models/.evals --outdir reports

Config file:

    <input_dir> must contain a `.evals_config.yaml` defining the benchmarks
    to report on and their random-chance baselines (used for the NPM), e.g.:

        benchmarks:
          - key: arc_challenge_poly_pt_acc_norm   # results.<key> in the yaml results
            label: ARC-C                          # short label (table header / plot axis)
            title: ARC-Challenge (PT)              # optional, longer description
          - key: belebele_por_Latn_acc_norm
            label: Belebele

        baselines:
          arc_challenge_poly_pt_acc_norm: 0.25
          belebele_por_Latn_acc_norm: 0.25

    RULER (long context) is shaped differently from everything else: it emits
    one metric per sub-task per context length instead of a single metric, so it
    is declared as a family rather than listed as a benchmark:

        # optional; omit the whole block to disable RULER handling
        ruler:
          contexts: [16000, 32000]   # lengths to report, in column order
          baseline: 0.0              # chance level, used for the NPM tables
          metrics:                   # sub-tasks averaged into the RULER score
            niah_pt_single_1: NIAH single needle (passkey retrieval)
            ruler_pt_vt: Variation tracing

    <input_dir> must also contain one *.yaml file per evaluated model, as
    produced by our lm-evaluation-harness scripts (see llm-foundry/evals), e.g.:

        model_name: Tucano2.1-0.5B-Base
        model_path: /path/to/checkpoint
        results:
          arc_challenge_poly_pt_acc_norm: 0.31
          arc_challenge_poly_pt_acc_norm_stderr: 0.01
          belebele_por_Latn_acc_norm: 0.27
          belebele_por_Latn_acc_norm_stderr: 0.01

Requirements:

    PyYAML:  pip install pyyaml
    matplotlib (optional, for plotting):  pip install matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required:  pip install pyyaml")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover
    plt = None
    np = None

# Populated at runtime from <input_dir>/.evals_config.yaml (see load_config()).
BENCHMARKS: list = []
BASELINES: dict = {}

# RULER (long context). Its results are one metric per sub-task per context
# length, so the config declares the (contexts x metrics) grid and load_config()
# appends one pseudo-benchmark per context length (RULER_PREFIX + "<length>").
RULER_PREFIX = "ruler@"
RULER_CONTEXTS: list = []  # ordered context lengths, e.g. [16000, 32000]
RULER_METRICS: list = []  # [(metric_key, label), ...] averaged into the score
RULER_BASELINE: float = 0.0

CONFIG_FILENAME = ".evals_config.yaml"

# Curated palette of up to 20 distinguishable, print-friendly colors
# (Okabe-Ito + Paul Tol muted palettes combined). Colors repeat past 20 series.
BAR_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#8C564B",
    "#7F7F7F",
    "#332288",
    "#117733",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
    "#661100",
    "#6699CC",
    "#888888",
    "#DDCC77",
    "#CC6677",
]


def _use_this_style():
    """Make things look pretty."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "legend.fontsize": 8,
            "legend.title_fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def ruler_key(context) -> str:
    """Pseudo-benchmark key standing for the RULER score at `context`."""
    return f"{RULER_PREFIX}{context}"


def ruler_context(key) -> str:
    """Inverse of ruler_key(): the context length behind a pseudo-key."""
    return key[len(RULER_PREFIX) :]


def ruler_label(context) -> str:
    """Short column label for a context length: 16000 -> "Ruler-16k"."""
    if context >= 1000 and context % 1000 == 0:
        return f"Ruler-{context // 1000}k"
    return f"Ruler-{context}"


def load_config(input_dir: Path):
    """Load BENCHMARKS/BASELINES globals from <input_dir>/.evals_config.yaml.

    Expected structure:

        benchmarks:
          - key: arc_challenge_poly_pt_acc_norm
            label: ARC-C
            title: ARC-Challenge (PT)
          ...
        baselines:
          arc_challenge_poly_pt_acc_norm: 0.25
          ...
    """
    global BENCHMARKS, BASELINES, RULER_CONTEXTS, RULER_METRICS, RULER_BASELINE
    config_path = input_dir / CONFIG_FILENAME
    if not config_path.is_file():
        sys.exit(
            f"Missing config file: {config_path}\n"
            "Define 'benchmarks' (list) and 'baselines' (mapping)."
        )
    with config_path.open(encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            sys.exit(f"{config_path} is not valid YAML:\n{exc}")

    raw_benchmarks = data.get("benchmarks")
    baselines = data.get("baselines")
    if not raw_benchmarks or not isinstance(baselines, dict):
        sys.exit(f"{config_path} must define 'benchmarks' (list) and 'baselines' (mapping)")

    BENCHMARKS = [(b["key"], b["label"], b.get("title", b["label"])) for b in raw_benchmarks]
    BASELINES = dict(baselines)

    ruler = data.get("ruler") or {}
    if not isinstance(ruler, dict):
        sys.exit(f"{config_path}: 'ruler' must be a mapping")
    contexts = ruler.get("contexts") or []
    metrics = ruler.get("metrics") or {}
    if contexts and not metrics:
        sys.exit(f"{config_path}: 'ruler.metrics' is required when 'ruler.contexts' is set")

    RULER_CONTEXTS = [int(c) for c in contexts]
    RULER_METRICS = [(str(k), str(v)) for k, v in metrics.items()]
    RULER_BASELINE = float(ruler.get("baseline", 0.0))

    for ctx in RULER_CONTEXTS:
        BENCHMARKS.append((ruler_key(ctx), ruler_label(ctx), f"RULER score at {ctx} tokens"))
        BASELINES.setdefault(ruler_key(ctx), RULER_BASELINE)


def load_results(input_dir: Path):
    """Return a list of dicts: {name, path, results} for every YAML found."""

    files = sorted(f for f in input_dir.glob("*.yaml") if f.name != CONFIG_FILENAME)
    if not files:
        sys.exit(f"No *.yaml result files found in {input_dir}")

    models = []
    for f in files:
        try:
            with f.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            print(f"  ! skipping {f.name}: {exc}", file=sys.stderr)
            continue

        results = data.get("results") or {}
        if not isinstance(results, dict) or not results:
            print(f"  ! skipping {f.name}: no 'results' mapping", file=sys.stderr)
            continue

        models.append(
            {
                "name": data.get("model_name") or f.stem,
                "path": data.get("model_path", ""),
                "results": results,
            }
        )

    if not models:
        sys.exit(f"No usable result files found in {input_dir}")
    return models


def _as_float(value):
    """float(value), or None when it is missing or non-numeric.

    lm-eval writes the string "N/A" where a metric has no stderr, and -1 where
    a task was declared but skipped, so neither `float()` nor a bare `is None`
    check is enough here.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score(model, key):
    """Return (value, stderr) for a raw metric key, or (None, None) if absent."""
    val = _as_float(model["results"].get(key))
    if val is None:
        return None, None
    return val, _as_float(model["results"].get(key + "_stderr"))


def ruler_metric_scores(model, context):
    """[(metric_key, value, stderr), ...] for the RULER sub-tasks at `context`.

    Keeps the config order and drops sub-tasks that are absent for this length or
    that lm-eval flagged as skipped (negative sentinel, e.g. -1).
    """
    parts = []
    for metric_key, _ in RULER_METRICS:
        val, err = score(model, f"{metric_key}_{context}")
        if val is None or val < 0:
            continue
        parts.append((metric_key, val, err))
    return parts


def ruler_score(model, context):
    """Aggregate RULER score at `context`: unweighted mean of its sub-tasks.

    The stderr is propagated from the sub-task stderrs (independent errors) and
    is only reported when every averaged sub-task has one.
    """
    parts = ruler_metric_scores(model, context)
    if not parts:
        return None, None
    mean = sum(val for _, val, _ in parts) / len(parts)
    errs = [err for _, _, err in parts]
    if any(err is None for err in errs):
        return mean, None
    return mean, math.sqrt(sum(err * err for err in errs)) / len(errs)


def ruler_count(model, context):
    """How many of the configured RULER sub-tasks are present at `context`."""
    return len(ruler_metric_scores(model, context))


def resolve(model, key):
    """(value, stderr) for a report column: RULER pseudo-key or raw metric."""
    if key.startswith(RULER_PREFIX):
        return ruler_score(model, ruler_context(key))
    return score(model, key)


def mean_score(model):
    """Return (mean over available columns, number present)."""
    vals = [resolve(model, k)[0] for k, _, _ in BENCHMARKS]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def compute_npm(val, key):
    """Normalized Preferred Metric: 100 * (score - baseline) / (1 - baseline)."""
    baseline = BASELINES.get(key)
    if val is None or baseline is None:
        return None
    return 100 * (val - baseline) / (1.0 - baseline)


def npm_score(model, key):
    """Return (NPM value, NPM stderr) for a benchmark, or (None, None) if absent."""
    val, err = resolve(model, key)
    npm = compute_npm(val, key)
    if npm is None:
        return None, None
    baseline = BASELINES[key]
    err_npm = 100 * err / (1.0 - baseline) if err is not None else None
    return npm, err_npm


def mean_npm(model):
    """Return (mean NPM over available benchmarks, number present)."""
    vals = [npm_score(model, k)[0] for k, _, _ in BENCHMARKS]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def rank_key(model):
    """Sort key: fully-evaluated models first, then by mean descending."""
    mean, n = mean_score(model)
    complete = 1 if n == len(BENCHMARKS) else 0
    return (-complete, -n, -(mean or 0.0))


def fmt(val, err, decimals=4):
    if val is None:
        return " - "
    if err is None:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} \u00b1 {err:.{decimals}f}"


def build_markdown(models, input_dir: Path, decimals=4, npm_decimals=2) -> str:
    models = sorted(models, key=rank_key)
    n_bench = len(BENCHMARKS)
    n_ruler = len(RULER_CONTEXTS)

    headers = ["Model"] + [label for _, label, _ in BENCHMARKS] + ["Mean"]
    sep = ["---"] * len(headers)

    suite = f"{n_bench - n_ruler} benchmarks"
    legend = (
        "Cells show **score \u00b1 stderr**. Higher is better; best score per column is **bold**."
    )
    if n_ruler:
        suite += " + RULER @ " + ", ".join(str(c) for c in RULER_CONTEXTS)
        legend += (
            " A `Ruler-*` column is the unweighted mean of that context length's "
            "sub-tasks (see [RULER](#ruler-long-context) below), and **Mean** "
            "averages every column, RULER columns included."
        )

    lines = [
        "# Portuguese benchmark results",
        "",
        f"Source: `{input_dir}`  \u2022  {len(models)} model(s)  \u2022  {suite}",
        "",
        legend,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    # best (max) value per column for bolding
    best = {}
    for key, _, _ in BENCHMARKS:
        vals = [resolve(m, key)[0] for m in models]
        vals = [v for v in vals if v is not None]
        if vals:
            best[key] = max(vals)

    for m in models:
        row = [f"`{m['name']}`"]
        for key, _, _ in BENCHMARKS:
            val, err = resolve(m, key)
            cell = fmt(val, err, decimals)
            if val is not None and best.get(key) is not None and val == best[key]:
                cell = f"**{cell}**"
            row.append(cell)
        avg, n = mean_score(m)
        if avg is None:
            row.append(" - ")
        elif n == n_bench:
            row.append(f"**{avg:.{decimals}f}**")
        else:
            row.append(f"{avg:.{decimals}f} ({n}/{n_bench})")
        lines.append("| " + " | ".join(row) + " |")

    # second, stats-friendly table with full metric names
    lines += [
        "",
        "## Metric keys",
        "",
        "| Label | Accuracy key | stderr key |",
        "| --- | --- | --- |",
    ]
    for key, label, _ in BENCHMARKS:
        if key.startswith(RULER_PREFIX):
            ctx = ruler_context(key)
            lines.append(
                f"| {label} | mean of the {len(RULER_METRICS)} `ruler.metrics` "
                f"keys suffixed `_{ctx}` | propagated |"
            )
        else:
            lines.append(f"| {label} | `{key}` | `{key}_stderr` |")

    lines += [
        "",
        "## Ranking by mean score",
        "",
        "| # | Model | Mean | Coverage |",
        "| --- | --- | --- | --- |",
    ]
    for i, m in enumerate(models, 1):
        avg, n = mean_score(m)
        mean_txt = f"{avg:.{decimals}f}" if avg is not None else "-"
        lines.append(f"| {i} | `{m['name']}` | {mean_txt} | {n}/{n_bench} |")
    lines.append("")

    # Aggregate NPM (Normalized Preferred Metric): a single score per model,
    # averaging 100 * (score - baseline) / (1 - baseline) over all benchmarks.
    lines += [
        "## Aggregate NPM (Normalized Preferred Metric)",
        "",
        "A single normalized score per model, averaging each benchmark's "
        "NPM (rescaled against its random-chance baseline, so 0 = chance "
        "and 100 = perfect) over the whole eval suite. Higher is better; "
        "best value is **bold**.",
        "",
        "| # | Model | Aggregate NPM | Coverage |",
        "| --- | --- | --- | --- |",
    ]
    npm_ranked = sorted(models, key=lambda m: -(mean_npm(m)[0] or float("-inf")))
    best_agg_npm = max((mean_npm(m)[0] for m in models if mean_npm(m)[0] is not None), default=None)
    for i, m in enumerate(npm_ranked, 1):
        avg, n = mean_npm(m)
        if avg is None:
            npm_txt = "-"
        elif best_agg_npm is not None and avg == best_agg_npm:
            npm_txt = f"**{avg:.{npm_decimals}f}**"
        else:
            npm_txt = f"{avg:.{npm_decimals}f}"
        lines.append(f"| {i} | `{m['name']}` | {npm_txt} | {n}/{n_bench} |")
    lines.append("")

    lines += build_ruler_markdown(models, decimals)
    return "\n".join(lines)


def build_ruler_markdown(models, decimals=4) -> list:
    """Markdown lines for the RULER section (empty when RULER is disabled)."""
    if not RULER_CONTEXTS:
        return []

    order = sorted(models, key=rank_key)
    n_sub = len(RULER_METRICS)
    lines = [
        "## RULER (long context)",
        "",
        f"RULER is scored as one metric per sub-task per context length; a "
        f"model's **RULER score** at a length is the unweighted mean of that "
        f"length's sub-tasks. Only the `ruler.contexts` lengths "
        f"({', '.join(str(c) for c in RULER_CONTEXTS)}) are reported, and a "
        f"sub-task with no value - or with lm-eval's negative \u201cskipped\u201d "
        f"sentinel - is excluded from the mean.",
        "",
        "### RULER score",
        "",
        "Best value per column is **bold**; `Mean` averages the context lengths "
        "shown, and `Coverage` counts the sub-tasks actually present "
        f"(out of {n_sub} x {len(RULER_CONTEXTS)}).",
        "",
        "| # | Model | "
        + " | ".join(ruler_label(ctx) for ctx in RULER_CONTEXTS)
        + " | Mean | Coverage |",
        "| --- | --- | " + " | ".join("---" for _ in RULER_CONTEXTS) + " | --- | --- |",
    ]

    best = {}
    for ctx in RULER_CONTEXTS:
        vals = [ruler_score(m, ctx)[0] for m in order]
        vals = [v for v in vals if v is not None]
        if vals:
            best[ctx] = max(vals)

    for i, m in enumerate(order, 1):
        row = [str(i), f"`{m['name']}`"]
        present = []
        for ctx in RULER_CONTEXTS:
            val, err = ruler_score(m, ctx)
            cell = fmt(val, err, decimals)
            if val is not None:
                present.append(val)
                if best.get(ctx) is not None and val == best[ctx]:
                    cell = f"**{cell}**"
            row.append(cell)
        row.append(f"{sum(present) / len(present):.{decimals}f}" if present else " - ")
        row.append(
            f"{sum(ruler_count(m, c) for c in RULER_CONTEXTS)}/{n_sub * len(RULER_CONTEXTS)}"
        )
        lines.append("| " + " | ".join(row) + " |")

    # One breakdown table per context length: rows are the sub-tasks, columns the
    # models, so a regression at a given length is easy to localise.
    for ctx in RULER_CONTEXTS:
        lines += [
            "",
            f"### RULER breakdown @ {ctx} tokens",
            "",
            "| Sub-task | " + " | ".join(f"`{m['name']}`" for m in order) + " |",
            "| --- | " + " | ".join("---" for _ in order) + " |",
        ]
        row_best = {}
        for metric_key, _ in RULER_METRICS:
            vals = []
            for m in order:
                val, _ = score(m, f"{metric_key}_{ctx}")
                vals.append(val if val is not None and val >= 0 else None)
            present = [v for v in vals if v is not None]
            row_best[metric_key] = max(present) if present else None
        for metric_key, label in RULER_METRICS:
            cells = []
            for m in order:
                val, err = score(m, f"{metric_key}_{ctx}")
                if val is not None and val < 0:
                    val, err = None, None
                cell = fmt(val, err, decimals)
                if val is not None and val == row_best[metric_key]:
                    cell = f"**{cell}**"
                cells.append(cell)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        cells = []
        for m in order:
            val, err = ruler_score(m, ctx)
            cell = fmt(val, err, decimals)
            if val is not None and val == best.get(ctx):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| **RULER score** | " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def build_plot(models, out_path: Path):
    if plt is None:
        print("  ! matplotlib not available - skipping plot", file=sys.stderr)
        return False

    _use_this_style()
    models = sorted(models, key=rank_key)

    labels = [label for _, label, _ in BENCHMARKS]
    n_models = len(models)
    x = np.arange(len(labels))
    total_width = 0.8
    width = total_width / n_models

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 6.5))

    for i, m in enumerate(models):
        vals, errs = [], []
        for key, _, _ in BENCHMARKS:
            v, e = resolve(m, key)
            vals.append(v if v is not None else np.nan)
            errs.append(e if e is not None else 0.0)
        offset = -total_width / 2 + width * (i + 0.5)
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            capsize=2,
            error_kw={"elinewidth": 0.7, "ecolor": "#333333"},
            label=m["name"],
            color=BAR_COLORS[i % len(BAR_COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Score (accuracy / acc\u2009norm)")
    ax.set_title("Portuguese benchmark comparison", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(3, n_models),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def build_npm_plot(models, out_path: Path):
    """Single bar per model comparing the aggregate NPM over the whole suite."""
    if plt is None:
        print("  ! matplotlib not available - skipping plot", file=sys.stderr)
        return False

    _use_this_style()
    models = sorted(models, key=lambda m: -(mean_npm(m)[0] or float("-inf")))
    names = [m["name"] for m in models]
    vals = []
    for m in models:
        avg, _ = mean_npm(m)
        vals.append(avg if avg is not None else np.nan)

    x = np.arange(len(names))
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5.5))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    for rect, v in zip(bars, vals, strict=False):
        if v is None or np.isnan(v):
            continue
        va = "bottom" if v >= 0 else "top"
        offset = 1.0 if v >= 0 else -1.0
        ax.annotate(
            f"{v:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, v),
            xytext=(0, offset * 3),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.set_ylabel("Aggregate NPM  (0 = chance, 100 = perfect)")
    ax.set_title("Portuguese eval suite \u2014 Aggregate NPM per model", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def build_ruler_plot(models, out_path: Path):
    """Grouped bars: RULER score per context length, one bar per model."""
    if plt is None:
        print("  ! matplotlib not available - skipping plot", file=sys.stderr)
        return False
    if not RULER_CONTEXTS:
        return False

    _use_this_style()
    models = sorted(models, key=rank_key)
    n_models = len(models)
    n_ctx = len(RULER_CONTEXTS)
    x = np.arange(n_ctx)
    total_width = 0.8
    width = total_width / n_models

    fig, ax = plt.subplots(figsize=(max(7, n_ctx * 2.2), 5.0))
    for i, m in enumerate(models):
        vals, errs = [], []
        for ctx in RULER_CONTEXTS:
            v, e = ruler_score(m, ctx)
            vals.append(v if v is not None else np.nan)
            errs.append(e if e is not None else 0.0)
        ax.bar(
            x - total_width / 2 + width * (i + 0.5),
            vals,
            width,
            yerr=errs,
            capsize=2,
            error_kw={"elinewidth": 0.7, "ecolor": "#333333"},
            label=m["name"],
            color=BAR_COLORS[i % len(BAR_COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("RULER score (mean over sub-tasks)")
    ax.set_title("RULER \u2014 long-context scores by context length", pad=12)
    ax.set_xlabel("Context length (tokens)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ctx:,}" for ctx in RULER_CONTEXTS])
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, n_models),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_csv(path: Path, header, rows):
    """Write a small CSV with `\"\"` (empty cell) for missing values."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def write_ruler_csvs(models, outdir: Path) -> list:
    """Write ruler_summary.csv and one unified_ruler_results_<ctx>.csv per length.

    Same shape as the ad-hoc RULER post-processing scripts these replace, so
    existing spreadsheet/plotting workflows keep working.
    """
    if not RULER_CONTEXTS:
        return []

    order = sorted(models, key=rank_key)
    written = []

    # model x context, plus the mean over the reported lengths
    summary = outdir / "ruler_summary.csv"
    rows = []
    for m in order:
        vals = [ruler_score(m, ctx)[0] for ctx in RULER_CONTEXTS]
        present = [v for v in vals if v is not None]
        cells = ["" if v is None else f"{v:.6f}" for v in vals]
        cells.append(f"{sum(present) / len(present):.6f}" if present else "")
        rows.append([m["name"]] + cells)
    _write_csv(summary, ["Model"] + [str(c) for c in RULER_CONTEXTS] + ["Mean"], rows)
    written.append(summary)

    # metric x model, one file per context length
    for ctx in RULER_CONTEXTS:
        path = outdir / f"unified_ruler_results_{ctx}.csv"
        rows = []
        for metric_key, label in RULER_METRICS:
            cells = []
            for m in order:
                val, _ = score(m, f"{metric_key}_{ctx}")
                if val is not None and val < 0:
                    val = None
                cells.append("" if val is None else f"{val:.6f}")
            rows.append([label] + cells)
        cells = []
        for m in order:
            val, _ = ruler_score(m, ctx)
            cells.append("" if val is None else f"{val:.6f}")
        rows.append(["RULER Score"] + cells)
        _write_csv(path, ["Metric"] + [m["name"] for m in order], rows)
        written.append(path)

    return written


def main(args):
    input_dir = args.input
    if not input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {input_dir}")
    outdir = args.outdir or input_dir
    outdir.mkdir(parents=True, exist_ok=True)

    load_config(input_dir)

    models = load_results(input_dir)
    print(f"Loaded {len(models)} model(s) from {input_dir}")
    for m in models:
        print(f"  - {m['name']}")
    if RULER_CONTEXTS:
        print(
            f"RULER: {len(RULER_METRICS)} sub-tasks \u00d7 "
            f"{len(RULER_CONTEXTS)} context length(s) {RULER_CONTEXTS}"
        )

    md_path = outdir / "eval_report.md"
    md_path.write_text(build_markdown(models, input_dir, args.decimals), encoding="utf-8")
    print(f"Wrote {md_path}")

    if not args.no_plot:
        png_path = outdir / "eval_scores.png"
        if build_plot(models, png_path):
            print(f"Wrote {png_path}")

        npm_png_path = outdir / "eval_npm_scores.png"
        if build_npm_plot(models, npm_png_path):
            print(f"Wrote {npm_png_path}")

        ruler_png_path = outdir / "eval_ruler_scores.png"
        if build_ruler_plot(models, ruler_png_path):
            print(f"Wrote {ruler_png_path}")

    for path in write_ruler_csvs(models, outdir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        "-i",
        type=Path,
        default=".evals",
        help="directory with *.yaml result files (default: .evals)",
    )
    ap.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=None,
        help="output directory for the report/plot (default: same as --input)",
    )
    ap.add_argument("--decimals", type=int, default=4, help="decimal places (default 4)")
    ap.add_argument("--no-plot", action="store_true", help="skip generating the bar plot")
    args = ap.parse_args()
    main(args)
