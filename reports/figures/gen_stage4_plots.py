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
C_UP = "#178F57"
C_FILL = "#F4F7FB"

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
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


def pick_best_by_fom(rows: list[dict[str, str]], **conditions: str | int) -> dict[str, str]:
    matched = []
    for row in rows:
        ok = True
        for key, value in conditions.items():
            if str(row.get(key, "")) != str(value):
                ok = False
                break
        if ok:
            matched.append(row)
    if not matched:
        raise ValueError(f"Expected at least 1 row for {conditions}, found 0")
    return max(matched, key=lambda row: as_float(row, "fom"))


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_stage3_source(path: Path) -> Path:
    return path


def common_stage3_stage4_sizes(
    stage3_rows: list[dict[str, str]], stage4_rows: list[dict[str, str]]
) -> list[int]:
    stage3_sizes = {
        int(row["size"])
        for row in stage3_rows
        if row.get("section") == "hybrid"
        and row.get("mpi_ranks") == "8"
        and row.get("omp_threads") == "12"
        and row.get("version") in {"baseline", "kc"}
    }
    stage4_sizes = {
        int(row["size"])
        for row in stage4_rows
        if row.get("scaling_mode") == "weak"
        and row.get("node_count") == "8"
        and row.get("version") in {"baseline", "kc"}
    }
    return sorted(stage3_sizes & stage4_sizes)


def stage4_weak_sizes(stage4_rows: list[dict[str, str]]) -> list[int]:
    sizes = {
        int(row["size"])
        for row in stage4_rows
        if row.get("scaling_mode") == "weak"
        and row.get("node_count") == "8"
        and row.get("version") in {"baseline", "kc"}
    }
    return sorted(sizes)


def stage4_strong_points(
    stage4_rows: list[dict[str, str]],
) -> tuple[list[int], list[float], list[float]]:
    base_sizes = sorted(
        {
            int(row["base_size"])
            for row in stage4_rows
            if row.get("scaling_mode") == "strong"
            and row.get("node_count") == "8"
            and row.get("version") in {"baseline", "kc"}
        }
    )
    actual_sizes = []
    baseline = []
    kc = []
    for base_size in base_sizes:
        b_row = pick_unique(
            stage4_rows,
            scaling_mode="strong",
            node_count=8,
            base_size=base_size,
            version="baseline",
        )
        k_row = pick_unique(
            stage4_rows,
            scaling_mode="strong",
            node_count=8,
            base_size=base_size,
            version="kc",
        )
        actual_sizes.append(int(b_row["size"]))
        baseline.append(as_float(b_row, "fom"))
        kc.append(as_float(k_row, "fom"))
    return actual_sizes, baseline, kc


def plot_stage4_weak_fom(stage4_rows: list[dict[str, str]], out_dir: Path) -> None:
    sizes = stage4_weak_sizes(stage4_rows)
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

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_facecolor(C_FILL)
    ax.fill_between(sizes, baseline, kc, color="#8FB8FF", alpha=0.16, zorder=1)
    ax.plot(
        sizes,
        baseline,
        color=C_BASE,
        marker="o",
        markersize=10,
        linewidth=3,
        label="baseline",
    )
    ax.plot(
        sizes,
        kc,
        color=C_KC,
        marker="o",
        markersize=10,
        linewidth=3,
        label="kc",
    )

    for xpos, value in zip(sizes, baseline):
        ax.annotate(
            f"{value:.0f}",
            (xpos, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=C_BASE,
            fontsize=9,
            fontweight="bold",
        )
    for xpos, value in zip(sizes, kc):
        ax.annotate(
            f"{value:.0f}",
            (xpos, value),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            color=C_KC,
            fontsize=9,
            fontweight="bold",
        )

    for xpos, base_val, kc_val in zip(sizes, baseline, kc):
        uplift = ((kc_val - base_val) / base_val) * 100.0
        ax.annotate(
            f"kc +{uplift:.1f}%",
            (xpos, (base_val + kc_val) / 2.0),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=C_UP,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_UP, lw=1.2),
        )

    ax.set_title("Stage 4 Weak Scaling FOM on 8 Nodes")
    ax.set_xlabel("Local problem size s")
    ax.set_ylabel("FOM (z/s)")
    ax.set_xticks(sizes, [str(size) for size in sizes])
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "stage4_weak_fom.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage4_strong_fom(stage4_rows: list[dict[str, str]], out_dir: Path) -> None:
    actual_sizes, baseline, kc = stage4_strong_points(stage4_rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_facecolor(C_FILL)
    ax.fill_between(actual_sizes, baseline, kc, color="#8FB8FF", alpha=0.16, zorder=1)
    ax.plot(
        actual_sizes,
        baseline,
        color=C_BASE,
        marker="o",
        markersize=10,
        linewidth=3,
        label="baseline",
    )
    ax.plot(
        actual_sizes,
        kc,
        color=C_KC,
        marker="o",
        markersize=10,
        linewidth=3,
        label="kc",
    )

    for xpos, value in zip(actual_sizes, baseline):
        ax.annotate(
            f"{value:.0f}",
            (xpos, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=C_BASE,
            fontsize=9,
            fontweight="bold",
        )
    for xpos, value in zip(actual_sizes, kc):
        ax.annotate(
            f"{value:.0f}",
            (xpos, value),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            color=C_KC,
            fontsize=9,
            fontweight="bold",
        )
    for xpos, base_val, kc_val in zip(actual_sizes, baseline, kc):
        uplift = ((kc_val - base_val) / base_val) * 100.0
        ax.annotate(
            f"kc +{uplift:.1f}%",
            (xpos, (base_val + kc_val) / 2.0),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=C_UP,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_UP, lw=1.2),
        )

    ax.set_title("Stage 4 Strong Scaling FOM on 8 Nodes")
    ax.set_xlabel("Local problem size s")
    ax.set_ylabel("FOM (z/s)")
    ax.set_xticks(actual_sizes, [str(size) for size in actual_sizes])
    ax.legend(frameon=False, loc="upper left",bbox_to_anchor=(0.0, 1.03), ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "stage4_strong_fom.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage3_stage4_fom(
    stage3_rows: list[dict[str, str]], stage4_rows: list[dict[str, str]], out_dir: Path
) -> None:
    sizes = common_stage3_stage4_sizes(stage3_rows, stage4_rows)
    if not sizes:
        raise ValueError("No common weak-scaling sizes found between stage3 and stage4 datasets")

    s3_baseline = []
    s3_kc = []
    s4_baseline = []
    s4_kc = []

    for size in sizes:
        s3_baseline.append(
            as_float(
                pick_best_by_fom(
                    stage3_rows,
                    section="hybrid",
                    mpi_ranks=8,
                    omp_threads=12,
                    size=size,
                    version="baseline",
                ),
                "fom",
            )
        )
        s3_kc.append(
            as_float(
                pick_best_by_fom(
                    stage3_rows,
                    section="hybrid",
                    mpi_ranks=8,
                    omp_threads=12,
                    size=size,
                    version="kc",
                ),
                "fom",
            )
        )
        s4_baseline.append(
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
        )
        s4_kc.append(
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
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), sharey=True)
    panels = [
        ("baseline", s3_baseline, s4_baseline, C_BASE),
        ("kc", s3_kc, s4_kc, C_KC),
    ]

    for idx, (ax, (label, stage3_vals, stage4_vals, color)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(C_FILL)
        ax.plot(
            sizes,
            stage3_vals,
            color=C_STAGE3,
            linestyle="--",
            marker="o",
            markersize=7,
            linewidth=2.2,
            label="stage3: 1 node",
        )
        ax.plot(
            sizes,
            stage4_vals,
            color=color,
            linestyle="-",
            marker="o",
            markersize=10,
            linewidth=3.2,
            label="stage4: 8 nodes",
        )
        ax.fill_between(sizes, stage3_vals, stage4_vals, color=color, alpha=0.15, zorder=1)

        for xpos, s3_val, s4_val in zip(sizes, stage3_vals, stage4_vals):
            uplift = ((s4_val - s3_val) / s3_val) * 100.0
            ax.annotate(
                f"{s3_val:.0f}",
                (xpos, s3_val),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=C_STAGE3,
                fontweight="bold",
            )
            ax.annotate(
                f"{s4_val:.0f}",
                (xpos, s4_val),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
            ax.annotate(
                f"{uplift:+.0f}%",
                (xpos, (s3_val + s4_val) / 2.0),
                xytext=(0, -4),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=color, lw=1.0),
            )

        ax.set_title(label)
        ax.set_xticks(sizes, [str(size) for size in sizes])
        ax.set_xlabel("Local problem size s")
        if idx == 0:
            ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.10))
        else:
            ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.10))

    axes[0].set_ylabel("FOM (z/s)")
    fig.suptitle("Stage 3 vs Stage 4 FOM (Bind-to-Core)", fontsize=15, y=1.02)
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
