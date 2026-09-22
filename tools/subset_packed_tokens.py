"""
subset_packed_tokens.py -- carve a token-budgeted subset out of a packed
(parquet) dataset folder by linking a selection of its shards into a new
sibling folder.

Usage:
    # report only (safe)
    python3 subset_packed_tokens.py \
        --source data/portuguese/packed_4096/gigaverbo_v2_3 --tokens 12.8b

    # create gigaverbo_v2_3_12.8b, fineweb_edu_5.9b and codeparrot_5.3b
    python3 subset_packed_tokens.py --apply \
        --source data/portuguese/packed_4096/gigaverbo_v2_3 \
                 data/portuguese/packed_4096/fineweb_edu \
                 data/portuguese/packed_4096/codeparrot \
        --tokens 12.8b 5.9b 5.3b

    # put the subsets in a dedicated folder instead of beside their source
    python3 subset_packed_tokens.py --apply \
        --source DIR --tokens 5.3b --output-dir data/portuguese/packed_4096/_mix

Requires: pyarrow.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pyarrow.parquet as pq

METADATA_NAME = ".metadata"
SUBSET_MARKER = "subset_of"
TMP_SUFFIX = ".subsetting.tmp"
MODE_CHOICES = ("symlink", "hardlink", "copy")
SELECTION_CHOICES = ("spread", "first", "shuffle")


def _norm_key(key: str) -> str:
    """`.metadata` keys are compared case/space-insensitively."""
    return re.sub(r"[^a-z0-9]", "", key.lower())


def parse_tokens(text: str) -> int:
    """Accept `5300000000`, `5_300_000_000`, `5.3b`, `12000k`, `0.5B`."""
    cleaned = str(text).strip().replace("_", "").lower()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kmbt]?)", cleaned)
    if not match:
        raise argparse.ArgumentTypeError(f"cannot parse token count: {text!r}")
    multiplier = {"": 1, "k": 10**3, "m": 10**6, "b": 10**9, "t": 10**12}[match.group(2)]
    value = float(match.group(1)) * multiplier
    if value < 1:
        raise argparse.ArgumentTypeError(f"token count must be >= 1: {text!r}")
    return int(round(value))


def pretty_tokens(tokens: int) -> str:
    """12800000000 -> `12.8b`; used to build the default output folder name."""
    if tokens >= 10**12:
        value, unit = tokens / 10**12, "t"
    elif tokens >= 10**9:
        value, unit = tokens / 10**9, "b"
    elif tokens >= 10**6:
        value, unit = tokens / 10**6, "m"
    elif tokens >= 10**3:
        value, unit = tokens / 10**3, "k"
    else:
        return str(tokens)
    return f"{value:.2f}".rstrip("0").rstrip(".") + unit


def read_metadata(folder: str) -> dict[str, int]:
    """Return the numeric keys of `<folder>/.metadata`, normalised. Non-numeric lines skipped."""
    values: dict[str, int] = {}
    path = os.path.join(folder, METADATA_NAME)
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            match = re.match(r"^([^:]+):\s*(.*?)\s*$", line)
            if not match:
                continue
            try:
                values[_norm_key(match.group(1))] = int(match.group(2))
            except ValueError:
                continue
    return values


def looks_like_subset(folder: str) -> bool:
    """True when `folder` is a directory this tool created (safe to overwrite)."""
    path = os.path.join(folder, METADATA_NAME)
    if not os.path.isdir(folder) or not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as fh:
        return any(line.startswith(f"{SUBSET_MARKER}:") for line in fh)


def discover_shards(folder: str) -> list[str]:
    """All `*.parquet` files one level inside `folder`, name-sorted."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"source folder does not exist: {folder}")
    shards = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found in: {folder}")
    return shards


def block_size_of(folder: str, override: int | None) -> int:
    """Tokens per row, from `--block-size` or `.metadata`."""
    if override:
        return override
    meta = read_metadata(folder)
    if "blocksize" in meta:
        return meta["blocksize"]
    samples, tokens = meta.get("samples"), meta.get("tokens")
    if samples and tokens:
        return tokens // samples
    raise SystemExit(
        f"error: cannot determine block_size for {folder} "
        "(no `block_size`/`samples`+`tokens` in .metadata); pass --block-size."
    )


def shard_rows(shard: str) -> int:
    """Row count from the parquet footer -- no data is read."""
    return pq.ParquetFile(shard).metadata.num_rows


def inventory(shards: list[str], block_size: int, num_proc: int) -> list[tuple[str, int, int]]:
    """Return (path, rows, tokens) per shard."""
    with ThreadPoolExecutor(max_workers=max(1, num_proc)) as pool:
        rows = list(pool.map(shard_rows, shards))
    return [(shard, n, n * block_size) for shard, n in zip(shards, rows, strict=False)]


def spread_indices(n: int, k: int) -> list[int]:
    """`k` indices evenly spaced over `[0, n)`, endpoints included."""
    if k <= 1:
        return [n // 2]
    if k >= n:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _prefix_sum_pick(tokens: list[int], order: list[int], target: int) -> tuple[list[int], int]:
    """First `k` entries of `order` landing closest to `target` (prefix sums are monotonic)."""
    prefix = 0
    best_k, best_total, best_gap = 1, 0, None
    for k, index in enumerate(order, start=1):
        prefix += tokens[index]
        gap = abs(prefix - target)
        if best_gap is None or gap < best_gap:
            best_k, best_total, best_gap = k, prefix, gap
    return order[:best_k], best_total


def select(
    inv: list[tuple[str, int, int]], target: int, selection: str, seed: int
) -> tuple[list[int], int]:
    """Choose shard indices totalling as close to `target` tokens as possible."""
    n = len(inv)
    tokens = [t for _, _, t in inv]
    total = sum(tokens)

    if selection == "spread":
        mean = total / n
        k0 = max(1, min(n, round(target / mean)))
        best: tuple[list[int], int] | None = None
        for k in range(max(1, k0 - 8), min(n, k0 + 8) + 1):
            indices = spread_indices(n, k)
            subtotal = sum(tokens[i] for i in indices)
            if best is None or (abs(subtotal - target), len(indices)) < (
                abs(best[1] - target),
                len(best[0]),
            ):
                best = (indices, subtotal)
        assert best is not None
        return sorted(best[0]), best[1]

    if selection == "first":
        order = list(range(n))
    else:  # shuffle
        import numpy as np

        order = list(np.random.default_rng(seed).permutation(n))
    chosen, subtotal = _prefix_sum_pick(tokens, order, target)
    return sorted(chosen), subtotal


def write_metadata(
    output_dir: str,
    *,
    source: str,
    shard_count: int,
    source_shard_count: int,
    tokens: int,
    rows: int,
    block_size: int,
    target: int,
    selection: str,
    mode: str,
) -> None:
    chunks = max(shard_count, 1)
    lines = [
        f"samples: {rows}",
        f"tokens: {tokens}",
        f"tokens_per_chunk: {tokens // chunks}",
        f"chunks: {chunks}",
        "strategy: concatenate",
        f"block_size: {block_size}",
        "packed_columns: input_ids",
        f"{SUBSET_MARKER}: {os.path.abspath(source)}",
        f"subset_target_tokens: {target}",
        f"subset_tokens: {tokens}",
        f"subset_shards: {shard_count}/{source_shard_count}",
        f"subset_selection: {selection}",
        f"subset_mode: {mode}",
        f"subset_created: {datetime.now(UTC).isoformat(timespec='seconds')}",
    ]
    with open(os.path.join(output_dir, METADATA_NAME), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def link_shard(source: str, destination: str, mode: str) -> None:
    absolute = os.path.abspath(source)
    if mode == "symlink":
        os.symlink(absolute, destination)
    elif mode == "hardlink":
        os.link(absolute, destination)
    else:  # copy
        shutil.copy2(absolute, destination)


def build_subset(
    inv: list[tuple[str, int, int]],
    indices: list[int],
    target: int,
    selection: str,
    mode: str,
    source: str,
    output_dir: str,
    overwrite: bool,
    apply: bool,
    quiet: bool,
) -> dict:
    """Create (or plan) one subset folder. Returns a summary dict."""
    tokens = sum(inv[i][2] for i in indices)
    rows = sum(inv[i][1] for i in indices)
    block_size = inv[0][2] // inv[0][1] if inv[0][1] else 0
    label = os.path.basename(output_dir.rstrip("/"))
    delta = tokens - target
    summary = {
        "output_dir": output_dir,
        "label": label,
        "available": len(inv),
        "selected": len(indices),
        "tokens": tokens,
        "rows": rows,
        "target": target,
        "delta": delta,
        "written": False,
    }

    if not quiet:
        print(f"  -> {label}")
        print(f"       source         : {source}")
        print(f"       shards         : {len(indices)} of {len(inv)}  ({selection})")
        print(f"       tokens         : {tokens:,d}  (target {target:,d}, {delta / target:+.2%})")
        print(f"       rows           : {rows:,d}  (block_size {block_size})")
        print(f"       mode           : {mode}")

    if not apply:
        return summary

    if os.path.exists(output_dir):
        if not overwrite:
            raise SystemExit(
                f"error: {output_dir} already exists (pass --overwrite to replace it)."
            )
        if not looks_like_subset(output_dir):
            raise SystemExit(
                f"error: refusing to delete {output_dir}: it exists but was not created by this "
                f"tool (no `{SUBSET_MARKER}:` line in its {METADATA_NAME}). Delete it yourself."
            )
        shutil.rmtree(output_dir)

    staging = output_dir + TMP_SUFFIX
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    try:
        for index in indices:
            shard = inv[index][0]
            link_shard(shard, os.path.join(staging, os.path.basename(shard)), mode)
        write_metadata(
            staging,
            source=source,
            shard_count=len(indices),
            source_shard_count=len(inv),
            tokens=tokens,
            rows=rows,
            block_size=block_size,
            target=target,
            selection=selection,
            mode=mode,
        )
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    summary["written"] = True
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Source packed dataset folder(s) (the ones with `*.parquet` + `.metadata`).",
    )
    ap.add_argument(
        "--tokens",
        nargs="+",
        required=True,
        type=parse_tokens,
        help="Target token count(s), e.g. `12.8b 5.9b 5.3b`. One value is broadcast to every "
        "source; one source with several values produces several subsets.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Where to put the subset(s). Default: beside the source folder.",
    )
    ap.add_argument(
        "--name",
        nargs="+",
        default=None,
        help="Output folder name(s) (basename). Default: `<source>_<tokens>b`.",
    )
    ap.add_argument(
        "--selection",
        choices=SELECTION_CHOICES,
        default="spread",
        help="How to choose shards: evenly spread, the first k, or a seeded random k.",
    )
    ap.add_argument("--seed", type=int, default=1337, help="Seed for `--selection shuffle`.")
    ap.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="symlink",
        help="How to reference the shards: symlink (no extra disk), hardlink, or copy.",
    )
    ap.add_argument(
        "--block-size", type=int, default=None, help="Tokens per row. Default: `.metadata`."
    )
    ap.add_argument(
        "--num-proc", type=int, default=16, help="Threads used to read parquet footers."
    )
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing subset folder.")
    ap.add_argument(
        "--apply", action="store_true", help="Write the subset(s). Without this it is a dry run."
    )
    ap.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    return ap.parse_args(argv)


def broadcast(values: list, count: int, what: str) -> list:
    """One value -> repeated; one-per-source -> as-is."""
    if len(values) == count:
        return list(values)
    if len(values) == 1:
        return list(values) * count
    raise SystemExit(
        f"error: got {len(values)} {what} for {count} source(s); give either one or {count}."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sources = [os.path.abspath(s.rstrip("/")) for s in args.source]
    if len(args.tokens) > 1 and len(sources) == 1:
        sources = sources * len(args.tokens)  # several subsets of the same source
    tokens = broadcast(list(args.tokens), len(sources), "--tokens")

    if args.name is not None:
        names = broadcast(list(args.name), len(sources), "--name")
    else:
        names = [
            f"{os.path.basename(s)}_{pretty_tokens(t)}"
            for s, t in zip(sources, tokens, strict=False)
        ]

    jobs = []
    for source, target, name in zip(sources, tokens, names, strict=False):
        if not os.path.isdir(source):
            print(f"error: source folder does not exist: {source}", file=sys.stderr)
            return 2
        out_dir = args.output_dir or os.path.dirname(source)
        jobs.append((source, target, os.path.join(os.path.abspath(out_dir), name)))

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] {len(jobs)} subset(s), selection={args.selection}, links={args.mode}")

    summaries = []
    for source, target, out_dir in jobs:
        if not args.quiet:
            print(f"\n  source folder: {source}")
        block_size = block_size_of(source, args.block_size)
        shards = discover_shards(source)
        inv = inventory(shards, block_size, args.num_proc)

        source_tokens = sum(t for _, _, t in inv)
        meta = read_metadata(source)
        recorded = meta.get("tokens")
        if recorded is not None and recorded != source_tokens:
            print(
                f"  note: .metadata records {recorded:,d} tokens but the parquet footers sum to "
                f"{source_tokens:,d} ({source_tokens - recorded:+,d}); using the footers.",
                file=sys.stderr,
            )
        if target > source_tokens:
            print(
                f"  warning: requested {target:,d} tokens but the source only holds "
                f"{source_tokens:,d}; the whole folder will be linked.",
                file=sys.stderr,
            )

        indices, _ = select(inv, min(target, source_tokens), args.selection, args.seed)
        summaries.append(
            build_subset(
                inv,
                indices,
                target,
                args.selection,
                args.mode,
                source,
                out_dir,
                args.overwrite,
                args.apply,
                args.quiet,
            )
        )

    print("\n" + "=" * 78)
    print(f"{'subset':32s} {'shards':>13s} {'tokens':>16s} {'vs target':>10s}")
    grand = 0
    for s in summaries:
        grand += s["tokens"]
        shard_note = "{}/{}".format(s["selected"], s["available"])
        print(
            f"{s['label']:32s} {shard_note:>13s} "
            f"{s['tokens']:>16,d} {s['delta'] / s['target']:>+9.2%}"
        )
    print(f"{'TOTAL':32s} {'':>13s} {grand:>16,d}")
    if args.apply:
        print(f"\nCreated {len(summaries)} subset folder(s).")
    else:
        print("\nThis was a DRY RUN. Re-run with --apply to create the folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
