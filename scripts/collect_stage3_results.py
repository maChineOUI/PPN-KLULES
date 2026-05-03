#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {k: (v or "").strip() for k, v in row.items()}
    return {}


def detect_flag(text: str, needle: str) -> str:
    return "yes" if needle in text else "no"


def extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def collect_case(metrics_path: Path) -> dict[str, str]:
    row = read_single_row_csv(metrics_path)
    case_id = row.get("case_id", "")
    report_dir = metrics_path.parent

    log_path = report_dir / f"{case_id}.log"
    slurm_out_path = report_dir / f"{case_id}.slurm.out"
    slurm_err_path = report_dir / f"{case_id}.slurm.err"
    manifest_path = report_dir / "manifest.csv"

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    slurm_out_text = (
        slurm_out_path.read_text(encoding="utf-8", errors="replace")
        if slurm_out_path.exists()
        else ""
    )
    slurm_err_text = (
        slurm_err_path.read_text(encoding="utf-8", errors="replace")
        if slurm_err_path.exists()
        else ""
    )
    combined_text = "\n".join([log_text, slurm_out_text, slurm_err_text])

    row["report_dir"] = str(report_dir)
    row["metrics_path"] = str(metrics_path)
    row["log_path"] = str(log_path) if log_path.exists() else ""
    row["slurm_out_path"] = str(slurm_out_path) if slurm_out_path.exists() else ""
    row["slurm_err_path"] = str(slurm_err_path) if slurm_err_path.exists() else ""
    row["manifest_path"] = str(manifest_path) if manifest_path.exists() else ""

    row["warn_openfabrics"] = detect_flag(combined_text, "OpenFabrics device")
    row["warn_ucx"] = detect_flag(combined_text, "UCX  WARN")
    row["warn_pmix"] = detect_flag(combined_text, "libpmix.so.2")
    row["warn_orte"] = detect_flag(combined_text, "orte_init failed")
    row["run_completed"] = detect_flag(combined_text, "Run completed:")
    row["local_host"] = extract_first(r"Local host:\s+(\S+)", combined_text)
    row["local_device"] = extract_first(r"Local device:\s+(\S+)", combined_text)
    row["num_processors_seen"] = extract_first(r"Num processors:\s+(\S+)", combined_text)
    row["num_threads_seen"] = extract_first(r"Num threads:\s+(\S+)", combined_text)

    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect stage3 cluster result metrics into one CSV."
    )
    parser.add_argument(
        "--reports-dir",
        default="cluster/stage3/reports",
        help="Root directory containing stage3-cluster-* result folders.",
    )
    parser.add_argument(
        "--pattern",
        default="stage3-cluster-*",
        help="Glob for result directories under --reports-dir.",
    )
    parser.add_argument(
        "--output",
        default="cluster/stage3/reports/stage3_results_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--compact-output",
        default="cluster/stage3/reports/stage3_results_compact.csv",
        help="Compact CSV path with only analysis-friendly columns.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir).resolve()
    output_path = Path(args.output).resolve()
    compact_output_path = Path(args.compact_output).resolve()

    metrics_files: list[Path] = []
    for report_dir in sorted(reports_dir.glob(args.pattern)):
        if report_dir.is_dir():
            metrics_files.extend(sorted(report_dir.glob("*.metrics.csv")))

    rows = [collect_case(path) for path in metrics_files]

    fieldnames = [
        "report_dir",
        "case_id",
        "version",
        "section",
        "mpi_ranks",
        "omp_threads",
        "size",
        "iters",
        "elapsed_s",
        "fom",
        "energy",
        "max_abs_diff",
        "status",
        "run_completed",
        "warn_openfabrics",
        "warn_ucx",
        "warn_pmix",
        "warn_orte",
        "local_host",
        "local_device",
        "num_processors_seen",
        "num_threads_seen",
        "metrics_path",
        "log_path",
        "slurm_out_path",
        "slurm_err_path",
        "manifest_path",
    ]
    compact_fieldnames = [
        "case_id",
        "version",
        "section",
        "mpi_ranks",
        "omp_threads",
        "size",
        "iters",
        "elapsed_s",
        "fom",
        "energy",
        "max_abs_diff",
        "status",
        "run_completed",
        "warn_openfabrics",
        "warn_ucx",
        "warn_pmix",
        "warn_orte",
        "local_host",
        "local_device",
        "num_processors_seen",
        "num_threads_seen",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    compact_output_path.parent.mkdir(parents=True, exist_ok=True)
    with compact_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compact_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in compact_fieldnames})

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote {len(rows)} rows to {compact_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
