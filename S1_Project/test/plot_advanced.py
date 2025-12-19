# ===== 统一视觉编码 =====
MARKERS = {
    30: "s",   # 正方形
    50: "^",   # 三角形
    100: "o"   # 圆形
}

COLORS = {
    "kokkos": "tab:blue",      # 蓝色
    "baseline": "tab:orange"   # 黄色
}

LINESTYLES = {
    "kokkos": "-",
    "baseline": "--"
}

import pandas as pd
import matplotlib.pyplot as plt

SIZES = [30, 50, 100]

def plot_speedup_combined():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, size in zip(axes, SIZES):
        df = pd.read_csv(f"summary_s{size}.csv")

        for impl in ["baseline", "kokkos"]:
            d = df[df["impl"] == impl].sort_values("threads")
            t0 = d[d["threads"] == 4]["time"].values[0]
            speedup = t0 / d["time"]

            ax.plot(
                d["threads"],
                speedup,
                marker=MARKERS[size],
                color=COLORS[impl],
                linestyle=LINESTYLES[impl],
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=impl.capitalize()
            )

        # Ideal line
        threads = sorted(df["threads"].unique())
        ax.plot(threads, [t / 4 for t in threads],
                "--", color="gray", label="Ideal")

        ax.set_title(f"size = {size}")
        ax.set_xlabel("Threads")
        ax.grid(True)

    axes[0].set_ylabel("Speedup")

    # ===== 全局图例 =====
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92)
    )

    fig.suptitle("Speedup Comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    plt.savefig("speedup.png", dpi=300)
    plt.close()

plot_speedup_combined()
print("Saved speedup.png")

def plot_efficiency_combined():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, size in zip(axes, SIZES):
        df = pd.read_csv(f"summary_s{size}.csv")

        for impl in ["baseline", "kokkos"]:
            d = df[df["impl"] == impl].sort_values("threads")
            t0 = d[d["threads"] == 4]["time"].values[0]
            speedup = t0 / d["time"]
            efficiency = speedup / (d["threads"] / 4)

            ax.plot(
                d["threads"],
                efficiency,
                marker=MARKERS[size],
                color=COLORS[impl],
                linestyle=LINESTYLES[impl],
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=impl.capitalize()
            )

        ax.set_title(f"size = {size}")
        ax.set_xlabel("Threads")
        ax.grid(True)

    axes[0].set_ylabel("Parallel Efficiency")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92)
    )

    fig.suptitle("Parallel Efficiency Comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    plt.savefig("efficiency.png", dpi=300)
    plt.close()

plot_efficiency_combined()
print("Saved efficiency.png")

def plot_overhead_combined():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, size in zip(axes, SIZES):
        df = pd.read_csv(f"summary_s{size}.csv")

        base = df[df["impl"] == "baseline"].sort_values("threads")
        kokk = df[df["impl"] == "kokkos"].sort_values("threads")

        ratio = kokk["time"].values / base["time"].values

        ax.plot(
            base["threads"],
            ratio,
            marker=MARKERS[size],
            color=COLORS["kokkos"],
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=f"size={size}"
        )

        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"size = {size}")
        ax.set_xlabel("Threads")
        ax.grid(True)

    axes[0].set_ylabel("Time Ratio (Kokkos / Baseline)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92)
    )

    fig.suptitle("Implementation Overhead of Kokkos", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    plt.savefig("overhead.png", dpi=300)
    plt.close()

plot_overhead_combined()
print("Saved overhead.png")

def plot_size_compare_combined():
    plt.figure(figsize=(7,5))

    for size in [30, 50, 100]:
        df = pd.read_csv(f"summary_s{size}.csv")

        for impl in ["baseline", "kokkos"]:
            d = df[df["impl"] == impl].sort_values("threads")

            plt.plot(
                d["threads"],
                d["fom"],
                marker=MARKERS[size],
                color=COLORS[impl],
                linestyle=LINESTYLES[impl],
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=f"{impl.capitalize()}, size={size}"
            )

    plt.xlabel("Threads")
    plt.ylabel("FOM (zones/s)")
    plt.title("Problem Size Sensitivity: Baseline vs Kokkos")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("size_compare_combined.png", dpi=300)
    plt.close()

plot_size_compare_combined()
print("Saved size_compare_combined.png")

