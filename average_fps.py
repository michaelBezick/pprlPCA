#!/usr/bin/env python3
"""
Parse RL training logs and compute average FPS per (experiment, method),
where both are encoded in "Group name is: <GROUP>".

Example group names:
  PC_PCA
  OCD_Baseline
  TF_baseline
  OCDoor_PCA

Outputs:
  - per_file_fps.csv: mean FPS per log file (+ counts)
  - agg_fps.csv: aggregated stats per (experiment, method)

Usage:
  python parse_fps.py /path/to/logs --recursive
  python parse_fps.py /path/to/logs --glob "*.out" "*.log"
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Optional, Tuple


GROUP_RE = re.compile(r"^\s*Group name is:\s*(?P<group>\S+)\s*$", re.MULTILINE)
# Matches the FPS value inside the ASCII tables, e.g. "|    fps              | 54.9     |"
FPS_RE = re.compile(r"^\s*\|\s*fps\s*\|\s*(?P<fps>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|", re.MULTILINE)

# Map various spellings to canonical method labels
METHOD_CANON = {
    "pca": "PCA",
    "baseline": "baseline",
    "base": "baseline",
    "b": "baseline",
}

# Canonical experiments (you can extend this if needed)
EXPERIMENTS = {"pc": "PC", "ocd": "OCD", "tf": "TF", "ocdoor": "OCDoor"}


@dataclass
class FileResult:
    path: str
    group: str
    experiment: str
    method: str
    fps_count: int
    fps_mean: float


def canonicalize_group(group: str) -> Tuple[str, str]:
    """
    Split group name into (experiment, method).
    Expected formats:
      EXP_METHOD
      EXP-METHOD
      EXP:METHOD
    Where EXP in {PC, OCD, TF, OCDoor} (case-insensitive)
    and METHOD in {PCA, baseline} (case-insensitive, allow "Baseline"/"baseline"/etc).
    """
    # Normalize separators to underscore and split
    norm = re.sub(r"[-:]+", "_", group.strip())
    parts = [p for p in norm.split("_") if p]
    if len(parts) < 2:
        raise ValueError(f"Group '{group}' does not look like EXP_METHOD (got parts={parts})")

    exp_raw = parts[0].lower()
    method_raw = parts[1].lower()

    if exp_raw not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{parts[0]}' in group '{group}'")
    experiment = EXPERIMENTS[exp_raw]

    if method_raw not in METHOD_CANON:
        # allow things like "Baseline" / "PCA" / "baselines"
        if "pca" in method_raw:
            method = "PCA"
        elif "base" in method_raw:
            method = "baseline"
        else:
            raise ValueError(f"Unknown method '{parts[1]}' in group '{group}'")
    else:
        method = METHOD_CANON[method_raw]

    return experiment, method


def extract_group(text: str) -> Optional[str]:
    m = GROUP_RE.search(text)
    return m.group("group") if m else None


def extract_fps_values(text: str) -> List[float]:
    vals: List[float] = []
    for m in FPS_RE.finditer(text):
        try:
            vals.append(float(m.group("fps")))
        except ValueError:
            pass
    return vals


def iter_files(root: Path, patterns: Iterable[str], recursive: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return

    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)


def parse_file(path: Path) -> Optional[FileResult]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None

    group = extract_group(text)
    if not group:
        return None

    fps_vals = extract_fps_values(text)
    if not fps_vals:
        return None

    try:
        experiment, method = canonicalize_group(group)
    except ValueError:
        return None

    return FileResult(
        path=str(path),
        group=group,
        experiment=experiment,
        method=method,
        fps_count=len(fps_vals),
        fps_mean=float(mean(fps_vals)),
    )


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=str, help="Directory containing log files (or a single log file path)")
    ap.add_argument("--glob", nargs="+", default=["*.out", "*.log", "*.txt"], help="Glob patterns to match")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--outdir", type=str, default=".", help="Where to write CSV outputs")
    args = ap.parse_args()

    root = Path(args.log_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    results: List[FileResult] = []
    for fp in iter_files(root, args.glob, args.recursive):
        r = parse_file(fp)
        if r is not None:
            results.append(r)

    # Per-file CSV
    per_file_rows = [
        {
            "path": r.path,
            "group": r.group,
            "experiment": r.experiment,
            "method": r.method,
            "fps_count": r.fps_count,
            "fps_mean": f"{r.fps_mean:.6g}",
        }
        for r in sorted(results, key=lambda x: (x.experiment, x.method, x.path))
    ]
    write_csv(outdir / "per_file_fps.csv", per_file_rows,
              ["path", "group", "experiment", "method", "fps_count", "fps_mean"])

    # Aggregate by (experiment, method)
    buckets: dict[Tuple[str, str], List[FileResult]] = {}
    for r in results:
        buckets.setdefault((r.experiment, r.method), []).append(r)

    agg_rows: List[dict] = []
    for (exp, method), rs in sorted(buckets.items()):
        file_means = [x.fps_mean for x in rs]
        agg_rows.append(
            {
                "experiment": exp,
                "method": method,
                "n_files": len(rs),
                "mean_fps_over_files": f"{mean(file_means):.6g}",
                "std_fps_over_files": f"{(pstdev(file_means) if len(file_means) > 1 else 0.0):.6g}",
                "min_file_mean_fps": f"{min(file_means):.6g}",
                "max_file_mean_fps": f"{max(file_means):.6g}",
                "total_fps_samples": sum(x.fps_count for x in rs),
            }
        )

    write_csv(
        outdir / "agg_fps.csv",
        agg_rows,
        [
            "experiment",
            "method",
            "n_files",
            "mean_fps_over_files",
            "std_fps_over_files",
            "min_file_mean_fps",
            "max_file_mean_fps",
            "total_fps_samples",
        ],
    )

    # Friendly terminal print
    print(f"Parsed {len(results)} log files with group+fps.")
    print(f"Wrote: {outdir / 'per_file_fps.csv'}")
    print(f"Wrote: {outdir / 'agg_fps.csv'}")
    if not results:
        print("No matches found. If your logs have different patterns, tweak GROUP_RE / FPS_RE or --glob/--recursive.")


if __name__ == "__main__":
    main()

