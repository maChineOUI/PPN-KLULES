# ===== 统一视觉编码 =====
MARKERS = {
    30: "s",
    50: "^",
    100: "o"
}

COLORS = {
    "kokkos": "tab:blue",
    "baseline": "tab:orange"
}

LINESTYLES = {
    "kokkos": "-",
    "baseline": "--"
}

import pandas as pd
import matplotlib.pyplot as plt

# ===== 统一视觉编码 =====
MARKERS = {30: "s", 50: "^", 100: "o"}
COLORS = {"kokkos": "tab:blue", "baseline": "tab:orange"}
LINESTYLES = {"kokkos": "-", "baseline": "--"}

size = 100  # 推荐用 100³
df = pd.read_csv(f"summary_s{size}.csv")

base = df[df["impl"] == "baseline"].sort_values("threads")
kokk = df[df["impl"] == "kokkos"].sort_values("threads")

threads = base["threads"].values
overhead = kokk["time"].values / base["time"].values

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

# ===== 左 y 轴：绝对性能（FOM）=====
ax1.plot(
    threads,
    base["fom"],
    marker=MARKERS[size],
    color=COLORS["baseline"],
    linestyle=LINESTYLES["baseline"],
    markeredgecolor="black",
    markeredgewidth=0.8,
    label="Baseline LULESH"
)

ax1.plot(
    threads,
    kokk["fom"],
    marker=MARKERS[size],
    color=COLORS["kokkos"],
    linestyle=LINESTYLES["kokkos"],
    markeredgecolor="black",
    markeredgewidth=0.8,
    label="Kokkos LULESH"
)

ax1.set_xlabel("Number of Threads")
ax1.set_ylabel("FOM (zones/s)")
ax1.tick_params(axis="y")

# ===== 右 y 轴：实现开销（Overhead）=====
ax2 = ax1.twinx()
ax2.plot(
    threads,
    overhead,
    marker=MARKERS[size],
    color="red",
    linestyle=":",
    markeredgecolor="black",
    markeredgewidth=0.8,
    label="Overhead"
)

ax2.axhline(1.0, linestyle="--", color="red", linewidth=1)
ax2.set_ylabel("Time Ratio (Kokkos / Baseline)")
ax2.tick_params(axis="y")

# ===== 合并图例，放到图外 =====
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

fig.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper center",
    ncol=3,
    frameon=False
)

plt.title(f"Performance vs Implementation Overhead (size={size})")
fig.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(f"dual_axis_fom_overhead_s{size}.png", dpi=300)
plt.close()

print(f"Saved dual_axis_fom_overhead_s{size}.png")
