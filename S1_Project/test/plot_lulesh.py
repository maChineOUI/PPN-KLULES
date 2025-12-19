import pandas as pd
import matplotlib.pyplot as plt

MARKERS = {30: "s", 50: "^", 100: "o"}
COLORS = {"kokkos": "tab:blue", "baseline": "tab:orange"}
LINESTYLES = {"kokkos": "-", "baseline": "--"}

sizes = [30, 50, 100]

# ===== 创建 1×3 子图 =====
fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 4),
    sharey=True
)

for ax, size in zip(axes, sizes):
    df = pd.read_csv(f"summary_s{size}.csv")

    for impl in ["baseline", "kokkos"]:
        d = df[df["impl"] == impl].sort_values("threads")

        ax.plot(
            d["threads"],
            d["fom"],
            marker=MARKERS[size],
            color=COLORS[impl],
            linestyle=LINESTYLES[impl],
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=impl.capitalize()
        )

    ax.set_title(f"Size = {size}³")
    ax.set_xlabel("Threads")
    ax.grid(True)

# 统一 y 轴标签
axes[0].set_ylabel("FOM (zones/s)")

# ===== 全局图例（只放一次）=====
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.92)  # 控制图例位置
)

fig.suptitle("LULESH Strong Scaling", fontsize=14, y=0.98)

# 给上方留空间，避免挤压
fig.tight_layout(rect=[0, 0, 1, 0.88])

plt.savefig("scaling.png", dpi=300)
plt.close()

print("Saved scaling.png")