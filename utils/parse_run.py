"""
Parse training log files and/or emissions data into structured output files.

Usage:
    python parse_run.py --log path/to/run.log [--emissions path/to/emissions.csv] [--nodes N] [--output-dir ./output] [--plot]

Outputs (in output-dir):
    metadata.json     — structured metadata from the log's [metadata] sections,
                        plus emissions info (null if no emissions file is provided).
    training.jsonl    — one JSON object per line for training stats.
    validation.jsonl  — one JSON object per line for validation stats (if any exist).
    ./plots/          — PNG plots of all numeric fields in training/validation stats (if --plot is used).
"""

import argparse
import json
import os
import sys


def _split_by_sep(lines, sep):
    """Split a list of lines on lines that match sep, returning a list of segments."""
    segments = []
    current = []
    for line in lines:
        if line.strip() == sep:
            segments.append(current)
            current = []
        else:
            current.append(line)
    segments.append(current)
    return segments


def parse_metadata_lines(lines):
    """
    Parse the raw lines of a [metadata] section into a structured dict.

    The expected format is:
        <preamble text>
        ==================================================
          key | value
          ...
        ==================================================
        Section Name:
          key | value
          ...
        ==================================================
        <trailing notes>
    """
    SEP = "=" * 50
    segments = _split_by_sep(lines, SEP)

    result = {}

    # segments[0] is the preamble (text before first ===)
    preamble = [l.strip() for l in segments[0] if l.strip()]
    if preamble:
        result["initialization_notes"] = preamble

    # Middle segments are key-value blocks, possibly with a section header
    for seg in segments[1:-1]:
        non_empty = [l for l in seg if l.strip()]
        if not non_empty:
            continue

        first = non_empty[0].strip()
        if first.endswith(":") and "|" not in first:
            section_name = first.rstrip(":")
            kv_lines = non_empty[1:]
        else:
            section_name = "run_info"
            kv_lines = non_empty

        kv_dict = {}
        for kl in kv_lines:
            if "|" in kl:
                key, _, value = kl.partition("|")
                key, value = key.strip(), value.strip()
                if key == "Model config" and os.path.isfile(value):
                    try:
                        with open(value, "r", encoding="utf-8") as cfg_fh:
                            value = json.load(cfg_fh)
                    except (json.JSONDecodeError, OSError):
                        pass  # fall back to keeping the path string
                kv_dict[key] = value

        result[section_name] = kv_dict

    # Last segment is trailing text
    if len(segments) > 1:
        trailing = [l.strip() for l in segments[-1] if l.strip()]
        if trailing:
            result["notes"] = trailing

    return result


def parse_log_file(log_path):
    """
    Parse a training log file.

    Returns:
        metadata  (dict)  — stage-organised metadata with resume counts.
                            Structure:
                            {
                              "resume_count": <int>,   # total resumes across all stages
                              "stages": {
                                "<stage_name>": {
                                  "resume_count": <int>,
                                  "initialization_notes": [...],
                                  "events": [...],   # lightweight mid-run annotations
                                  "run_info": {...},
                                  ...               # other config sections
                                },
                                ...
                              }
                            }
        stats_by_status (dict[str, list[dict]]) — stats grouped by their "status" field
    """
    raw_metadata_sections = []
    stats_by_status = {}

    current_section = None
    current_lines = []

    def _flush(section, lines):
        if section == "metadata" and lines:
            raw_metadata_sections.append(parse_metadata_lines(lines))
        elif section == "stats":
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    status = entry.get("status", "unknown")
                    stats_by_status.setdefault(status, []).append(entry)
                except json.JSONDecodeError:
                    pass

    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            stripped = line.strip()

            if stripped == "---":
                _flush(current_section, current_lines)
                current_section = None
                current_lines = []
            elif stripped == "[metadata]":
                _flush(current_section, current_lines)
                current_section = "metadata"
                current_lines = []
            elif stripped == "[stats]":
                _flush(current_section, current_lines)
                current_section = "stats"
                current_lines = []
            elif current_section is not None:
                current_lines.append(line)

    _flush(current_section, current_lines)

    # Organise metadata sections by stage, processing in document order so
    # resume/event sections can be attributed to the most-recently-seen stage.
    #
    # Three kinds of [metadata] sections:
    #   1. Full config  — has "run_info" (initial start or not: same structure)
    #   2. Resume       — no "run_info", but contains "Resumed model from checkpoint"
    #   3. Event        — no "run_info"; a lightweight annotation (e.g. LR stage change)
    stages = {}
    current_stage_name = None
    total_resume_count = 0

    for md in raw_metadata_sections:
        notes = md.get("initialization_notes", [])
        is_resume = any("Resumed model from checkpoint" in note for note in notes)
        is_config = "run_info" in md

        if is_config:
            stage_name = md["run_info"].get("Training stage", "unknown")
            current_stage_name = stage_name

            if stage_name not in stages:
                stages[stage_name] = _empty_stage()

            stage = stages[stage_name]
            stage["initialization_notes"].extend(notes)

            # Copy all config sections; last value wins on key collision
            # (contents are identical across resumes of the same stage).
            for key, value in md.items():
                if key not in ("initialization_notes", "notes"):
                    stage[key] = value

        elif is_resume:
            # Resume of the current stage — no config block is repeated.
            target = current_stage_name or "_global"
            stages.setdefault(target, _empty_stage())
            stages[target]["resume_count"] += 1
            stages[target]["initialization_notes"].extend(notes)
            total_resume_count += 1

        else:
            # Lightweight event annotation (e.g. LR stage change).
            target = current_stage_name or "_global"
            stages.setdefault(target, _empty_stage())["events"].extend(notes)

    return {"resume_count": total_resume_count, "stages": stages}, stats_by_status


def _empty_stage():
    return {"resume_count": 0, "initialization_notes": [], "events": []}


_SKIP_PLOT_FIELDS = {"status", "stage_name", "lr_stage"}


def plot_stats(entries, split_name, plots_dir):
    """
    Create one PNG per numeric stat field in *entries*.
    Plots are written to plots_dir/<split_name>/.
    The x-axis is always 'step'.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "ERROR: matplotlib is required for --plot. "
            "Install it with: pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)

    if not entries:
        return

    out_dir = os.path.join(plots_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    # Collect all numeric fields (excluding step and skip list)
    numeric_fields = [
        k for k in entries[0]
        if k != "step" and k not in _SKIP_PLOT_FIELDS
        and isinstance(entries[0][k], (int, float))
    ]

    steps = [e["step"] for e in entries]

    for field in numeric_fields:
        values = [e.get(field) for e in entries]
        # Drop entries where the field is missing or non-numeric
        pairs = [(s, v) for s, v in zip(steps, values) if isinstance(v, (int, float))]
        if not pairs:
            continue
        xs, ys = zip(*pairs)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys, linewidth=0.9)
        ax.set_xlabel("step")
        ax.set_ylabel(field)
        ax.set_title(f"{split_name} — {field}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{field}.png"), dpi=120)
        plt.close(fig)

    print(f"Written: {out_dir}/  ({len(numeric_fields)} plots)")


def compute_emissions(emissions_path, num_nodes):
    """
    Compute aggregated emissions metrics from a CodeCarbon emissions CSV.

    For each unique run_id, takes the last row (cumulative values), sums across
    run_ids, then scales energy and emissions by num_nodes.

    Returns a dict with duration_hours, energy_consumed_kwh, co2_emissions_kgco2eq.
    """
    try:
        import pandas as pd
    except ImportError:
        print(
            "ERROR: pandas is required to process an emissions file. "
            "Install it with: pip install pandas",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(emissions_path)

    total_duration = 0.0
    total_energy = 0.0
    total_emissions = 0.0

    for run_id in df["run_id"].unique():
        df_run = df[df["run_id"] == run_id]
        last = df_run.iloc[-1]
        total_duration += last["duration"]
        total_energy += last["energy_consumed"]
        total_emissions += last["emissions"]

    total_energy *= num_nodes
    total_emissions *= num_nodes
    total_duration_hrs = total_duration / 3600.0

    return {
        "duration_hours": round(total_duration_hrs, 4),
        "energy_consumed_kwh": round(total_energy, 6),
        "co2_emissions_kgco2eq": round(total_emissions, 6),
        "num_nodes": num_nodes,
    }


def main(args):

    if not args.log and not args.emissions:
        parser.error("At least one of --log or --emissions must be provided.")

    if args.emissions and args.nodes is None:
        print(
            "Warning: --nodes not specified; assuming 1 node for emissions calculation. "
            "Pass --nodes N for multi-node training.",
            file=sys.stderr,
        )
        num_nodes = 1
    else:
        num_nodes = args.nodes if args.nodes is not None else 1

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {}
    stats_by_status = {}

    if args.log:
        metadata, stats_by_status = parse_log_file(args.log)

    # Parse emissions
    if args.emissions:
        emissions_info = compute_emissions(args.emissions, num_nodes)
    else:
        emissions_info = None

    metadata["emissions"] = emissions_info

    # Write metadata.json
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Written: {metadata_path}")

    # If no log was provided, nothing else to write.
    if not args.log:
        return

    # Write training.jsonl
    training_entries = stats_by_status.get("training", [])
    if training_entries:
        training_path = os.path.join(args.output_dir, "training.jsonl")
        with open(training_path, "w", encoding="utf-8") as fh:
            for entry in training_entries:
                fh.write(json.dumps(entry) + "\n")
        print(f"Written: {training_path}  ({len(training_entries)} entries)")
    else:
        print("No training stats found in log file.")

    # Write validation.jsonl
    validation_entries = stats_by_status.get("validation", [])
    if validation_entries:
        validation_path = os.path.join(args.output_dir, "validation.jsonl")
        with open(validation_path, "w", encoding="utf-8") as fh:
            for entry in validation_entries:
                fh.write(json.dumps(entry) + "\n")
        print(f"Written: {validation_path}  ({len(validation_entries)} entries)")
    else:
        print("No validation stats found in log file.")

    # Write any other status types
    known = {"training", "validation"}
    for status, entries in stats_by_status.items():
        if status in known:
            continue
        other_path = os.path.join(args.output_dir, f"{status}.jsonl")
        with open(other_path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
        print(f"Written: {other_path}  ({len(entries)} entries)")

    # Plots
    if args.plot:
        plots_dir = os.path.join(args.output_dir, "plots")
        plot_stats(training_entries, "training", plots_dir)
        plot_stats(validation_entries, "validation", plots_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description=(
            "Parse a training log file and/or an emissions CSV into structured output files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--log", metavar="PATH", help="Path to training log file.")
    parser.add_argument("--emissions", metavar="PATH", help="Path to emissions CSV file.")
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of nodes used in multi-node training. "
            "Required for correct emissions scaling when --emissions is provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=".",
        help="Directory to write output files (default: current directory).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a PNG plot for every stat in training/validation logs.",
    )

    args = parser.parse_args()

    main(args)
