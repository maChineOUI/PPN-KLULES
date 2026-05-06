#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

C_BASE = "#E05A2B"
C_KC = "#2B7BE0"
C_STAGE3 = "#7A7A7A"
C_STAGE4 = "#2E8B57"

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def is_valid(row: dict[str, str]) -> bool:
    return row.get("status") == "PASS" and row.get("run_completed") == "yes"


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pick_unique(rows: list[dict[str, str]], **conditions: str | int) -> dict[str, str]:
    matched = []
    for row in rows:
        ok = True
        for key, value in conditions.items():
            if str(row.get(key, "")) != str(value):
                ok = False
                break
        if ok:
            matched.append(row)
    if len(matched) != 1:
        raise ValueError(f"Expected exactly 1 row for {conditions}, found {len(matched)}")
    return matched[0]


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_stage3_source(path: Path) -> Path:
    return path


def plot_stage4_weak_fom(stage4_rows: list[dict[str, str]], out_dir: Path) -> None:
    sizes = [80, 120]
    baseline = [
        as_float(
            pick_unique(
                stage4_rows,
                scaling_mode="weak",
                node_count=8,
                size=size,
                version="baseline",
            ),
            "fom",
        )
        for size in sizes
    ]
    kc = [
        as_float(
            pick_unique(
                stage4_rows,
                scaling_mode="weak",
                node_count=8,
                size=size,
                version="kc",
            ),
            "fom",
        )
        for size in sizes
    ]

    x = np.arange(len(sizes))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, baseline, width=w, color=C_BASE, label="baseline")
    ax.bar(x + w / 2, kc, width=w, color=C_KC, label="kc")

    for xpos, value in zip(x - w / 2, baseline):
        ax.text(xpos, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + w / 2, kc):
        ax.text(xpos, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Stage 4 Weak Scaling on 8 Nodes (Bind-to-Core)")
    ax.set_xlabel("Local problem size s")
    ax.set_ylabel("FOM (z/s)")
    ax.set_xticks(x, [str(size) for size in sizes])
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "stage4_weak_fom.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage4_strong_fom(stage4_rows: list[dict[str, str]], out_dir: Path) -> None:
    baseline = as_float(
        pick_unique(
            stage4_rows,
            scaling_mode="strong",
            node_count=8,
            base_size=120,
            size=60,
            version="baseline",
        ),
        "fom",
    )
    kc = as_float(
        pick_unique(
            stage4_rows,
            scaling_mode="strong",
            node_count=8,
            base_size=120,
            size=60,
            version="kc",
        ),
        "fom",
    )

    labels = ["baseline", "kc"]
    values = [baseline, kc]
    colors = [C_BASE, C_KC]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, values, color=colors, width=0.55)
    for xpos, value in zip(x, values):
        ax.text(xpos, value, f"{value:.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Stage 4 Strong Scaling on 8 Nodes (Bind-to-Core)")
    ax.set_ylabel("FOM (z/s)")
    ax.set_xticks(x, labels)

    fig.tight_layout()
    fig.savefig(out_dir / "stage4_strong_fom.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage3_stage4_fom(
    stage3_rows: list[dict[str, str]], stage4_rows: list[dict[str, str]], out_dir: Path
) -> None:
    sizes = [80, 120]
    labels = []
    stage3_values = []
    stage4_values = []

    for size in sizes:
        for version in ("baseline", "kc"):
            s3 = pick_unique(
                stage3_rows,
                section="hybrid",
                mpi_ranks=8,
                omp_threads=12,
                size=size,
                version=version,
            )
            s4 = pick_unique(
                stage4_rows,
                scaling_mode="weak",
                node_count=8,
                size=size,
                version=version,
            )
            labels.append(f"s={size}\n{version}")
            stage3_values.append(as_float(s3, "fom"))
            stage4_values.append(as_float(s4, "fom"))

    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, stage3_values, width=w, color=C_STAGE3, label="stage3: 1 node")
    ax.bar(x + w / 2, stage4_values, width=w, color=C_STAGE4, label="stage4: 8 nodes")

    for xpos, value in zip(x - w / 2, stage3_values):
        ax.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + w / 2, stage4_values):
        ax.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Stage 3 vs Stage 4 FOM (bind-to-core results)")
    ax.set_ylabel("FOM (z/s)")
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "stage3_vs_stage4_fom.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate stage4 comparison plots from CSV data.")
    parser.add_argument(
        "--stage4",
        default=str(ROOT / "cluster/stage4/my_stage4_compact.csv"),
        help="Path to stage4 compact CSV (bind-to-core results).",
    )
    parser.add_argument(
        "--stage3",
        default=str(ROOT / "cluster/stage3-slotpe-rank/stage3_slotpe_rank_results_compact.csv"),
        help="Path to the stage3 compact CSV used for the comparison plot.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory for output figures.",
    )
    args = parser.parse_args()

    stage4_path = Path(args.stage4).resolve()
    stage3_path = resolve_stage3_source(Path(args.stage3).resolve())
    out_dir = Path(args.out_dir).resolve()
    ensure_out_dir(out_dir)

    stage4_rows = [row for row in load_csv(stage4_path) if is_valid(row)]

    plot_stage4_weak_fom(stage4_rows, out_dir)
    print(f"Wrote {out_dir / 'stage4_weak_fom.png'}")

    plot_stage4_strong_fom(stage4_rows, out_dir)
    print(f"Wrote {out_dir / 'stage4_strong_fom.png'}")

    if stage3_path.exists():
        raw_stage3_rows = load_csv(stage3_path)
        stage3_rows = [row for row in raw_stage3_rows if row.get("status") == "PASS"]
        plot_stage3_stage4_fom(stage3_rows, stage4_rows, out_dir)
        print(f"Wrote {out_dir / 'stage3_vs_stage4_fom.png'}")
    else:
        print(f"Skip stage3 comparison: missing {stage3_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
