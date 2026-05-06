"""Generate all benchmark figures for cluster-node-report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ── Color palette ──────────────────────────────────────────────────────────
C_BASE = "#E05A2B"   # warm red for baseline
C_KC   = "#2B7BE0"   # blue for kc
C_IDEAL = "#888888"  # grey for ideal

SIZES = [20, 40, 80]
SIZE_MARKS = ['s', '^', 'o']
SIZE_COLORS_B = ["#F4A582", "#D6604D", "#B2182B"]
SIZE_COLORS_K = ["#92C5DE", "#4393C3", "#1A6FAE"]

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

# --- Pure MPI ---
mpi_data = {
    "baseline": {20: {1: 838.7, 8: 6356.8, 27: 22369.5},
                 40: {1: 1809.5, 8: 13878.4, 27: 38255.8},
                 80: {1: 1894.4, 8: 15380.7, 27: 37833.5, 64: 50410.2}},
    "kc":       {20: {1: 852.3, 8: 7145.8, 27: 27639.7},
                 40: {1: 1986.3, 8: 15889.1, 27: 44817.2},
                 80: {1: 2096.3, 8: 16894.5, 27: 45228.5, 64: 60823.9}},
}

# --- Pure OMP (all thread counts, size=80 and size=40 and size=20) ---
omp_threads = [1, 2, 4, 8, 15, 20, 24, 28, 29, 30, 31, 32, 36, 40,
               58, 59, 60, 61, 62, 64]

omp_data = {
    "baseline": {
        20: [886.9, 637.3, 447.1, 263.4, 177.9, 162.5, 126.8, 110.4, 108.2, 104.8,
             101.9, 97.7, 88.4, 80.7, 60.5, 56.9, 55.7, 54.9, 52.9, 51.3],
        40: [1820.3, 1567.5, 1439.6, 1180.1, 904.5, 770.2, 703.9, 635.8, 617.5, 603.8,
             591.7, 579.7, 527.7, 491.2, 372.0, 369.4, 363.2, 354.8, 353.0, 345.9],
        80: [1888.0, 1615.3, 1546.4, 1541.0, 1445.5, 1497.8, 1394.4, 1422.7, 1348.6, 1362.9,
             1353.1, 1375.9, 1267.3, 1297.5, 1081.7, 1073.9, 1056.7, 1058.4, 1038.9, 1026.2],
    },
    "kc": {
        20: [906.7, 855.7, 797.9, 659.9, 522.3, 443.0, 494.3, 382.1, 376.0, 358.3,
             358.3, 340.7, 320.1, 302.7, 222.1, 222.9, 226.8, 226.5, 222.0, 216.1],
        40: [2100.4, 2099.0, 2059.4, 1915.1, 1725.9, 1639.6, 1592.3, 1510.7, 1490.1, 1503.3,
             1388.5, 1393.7, 1322.1, 1310.8, 1114.7, 1062.5, 1053.5, 1006.5, 1010.0, 1023.0],
        80: [2103.3, 2123.1, 2136.3, 2135.6, 2077.2, 2070.2, 2061.0, 2088.6, 2034.4, 2029.3,
             2021.2, 2038.3, 1982.8, 2008.3, 1827.7, 1843.8, 1819.1, 1820.8, 1831.9, 1817.4],
    },
}

# --- Hybrid (s80) ---
hybrid_configs = ["1×96", "8×4", "8×8", "8×10", "8×11", "8×12", "27×3"]
hybrid_total_t = [96, 32, 64, 80, 88, 96, 81]

hybrid_s80 = {
    "baseline": [21088.1, 15272.4, 34684.3, 38988.7, 50762.3, 53726.0, 38815.4],
    "kc":       [55014.4, 19412.3, 63574.5, 78974.3, 86674.4, 92474.0, 48948.0],
}
hybrid_s40 = {
    "baseline": [3431.7, 20400.3, 31821.0, 34595.0, 35661.9, 41503.0, 51450.2],
    "kc":       [12357.7, 33976.7, 60348.9, 68208.5, 70891.5, 75839.0, 79824.8],
}
hybrid_s20 = {
    "baseline": [484.1, 10084.4, 11169.3,  9618.3,  9977.1, 12558.0, 29090.4],
    "kc":       [1680.8, 15513.8, 21979.2, 21746.9, 24733.8, 21631.0, 44444.4],
}


def size_label(sz: int) -> str:
    return f"s{sz} ({sz}^3 = {sz**3:,} zones)"


def staggered_ticklabels(values: list[int]) -> list[str]:
    labels = []
    for idx, value in enumerate(values):
        labels.append(f"{value}" if idx % 2 == 0 else f"\n{value}")
    return labels

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – MPI Strong Scaling (FOM vs np)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, sz, cb, ck in zip(axes, SIZES, SIZE_COLORS_B, SIZE_COLORS_K):
    bdata = mpi_data["baseline"][sz]
    kdata = mpi_data["kc"][sz]
    nps_b = sorted(bdata.keys())
    nps_k = sorted(kdata.keys())
    nps_all = sorted(set(nps_b) | set(nps_k))

    # Ideal scaling line (from np=1 baseline)
    np1_b = bdata[1]
    nps_ideal = np.array(nps_all)
    ideal = np1_b * nps_ideal
    ax.plot(nps_ideal, ideal, "--", color=C_IDEAL, lw=1.5, label="Ideal linear scaling")

    ax.plot(nps_b, [bdata[n] for n in nps_b], 's-', color=cb,
            lw=2, ms=8, label="baseline")
    ax.plot(nps_k, [kdata[n] for n in nps_k], 'o-', color=ck,
            lw=2, ms=8, label="kc")

    # Annotate speedup at each point
    for n in nps_k:
        if n in bdata:
            sp = kdata[n] / bdata[n]
            ax.annotate(f"+{(sp-1)*100:.0f}%",
                        xy=(n, kdata[n]), xytext=(4, 6),
                        textcoords='offset points', fontsize=8,
                        color=ck)

    ax.set_title(size_label(sz))
    ax.set_xlabel("MPI ranks (np)")
    ax.set_ylabel("FOM (kzc/s)" if ax == axes[0] else "")
    ax.set_xticks(nps_all)
    ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_mpi_scaling.png", bbox_inches='tight')
plt.close()
print("fig1 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – MPI Scaling Efficiency
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, sz in zip(axes, [40, 80]):
    bdata = mpi_data["baseline"][sz]
    kdata = mpi_data["kc"][sz]
    nps   = sorted(set(bdata.keys()) | set(kdata.keys()))
    nps_no1 = [n for n in nps if n in {8, 27}]

    eff_b = [bdata[n] / bdata[1] / n * 100 for n in nps_no1 if n in bdata]
    eff_k = [kdata[n] / kdata[1] / n * 100 for n in nps_no1 if n in kdata]
    xb = [n for n in nps_no1 if n in bdata]
    xk = [n for n in nps_no1 if n in kdata]

    ax.axhline(100, color=C_IDEAL, ls='--', lw=1.5, label="Ideal efficiency 100%")
    ax.bar([x - 1.5 for x in xb], eff_b, width=3, color=C_BASE, alpha=0.85, label="baseline")
    ax.bar([x + 1.5 for x in xk], eff_k, width=3, color=C_KC, alpha=0.85, label="kc")

    for x, e in zip(xb, eff_b):
        ax.text(x - 1.5, e + 0.8, f"{e:.1f}%", ha='center', va='bottom', fontsize=9, color=C_BASE)
    for x, e in zip(xk, eff_k):
        ax.text(x + 1.5, e + 0.8, f"{e:.1f}%", ha='center', va='bottom', fontsize=9, color=C_KC)

    ax.set_title(f"{size_label(sz)}: MPI parallel efficiency")
    ax.set_xlabel("MPI ranks (np)")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_xticks(xb)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_mpi_efficiency.png", bbox_inches='tight')
plt.close()
print("fig2 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – OMP: FOM vs Thread Count (all 3 sizes, both versions)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), sharey=False)

for ax, sz, cb, ck in zip(axes, SIZES, SIZE_COLORS_B, SIZE_COLORS_K):
    ts    = omp_threads
    fom_b = omp_data["baseline"][sz]
    fom_k = omp_data["kc"][sz]
    ymax  = max(max(fom_b), max(fom_k)) * 1.10

    ax.plot(ts, fom_b, 's-', color=cb, lw=2, ms=5, label="baseline")
    ax.plot(ts, fom_k, 'o-', color=ck, lw=2, ms=5, label="kc")

    # NUMA boundary markers (drawn once, after ylim)
    ax.axvline(30, color='#CC4400', lw=1.4, ls=':', alpha=0.8)
    ax.axvline(60, color='#8B0055', lw=1.4, ls=':', alpha=0.8)
    ax.text(30.5, ymax * 0.98, "NUMA 0/1", fontsize=8, color='#CC4400', va='top',
            rotation=90, ha='left')
    ax.text(60.5, ymax * 0.98, "Socket", fontsize=8, color='#8B0055', va='top',
            rotation=90, ha='left')

    ax.set_title(size_label(sz))
    ax.set_xlabel("OMP threads (T)")
    ax.set_ylabel("FOM (kzc/s)" if ax == axes[0] else "")
    ticks = [1, 8, 15, 30, 31, 60, 61, 64]
    ax.set_xticks(ticks)
    ax.set_xticklabels(staggered_ticklabels(ticks))
    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylim(0, ymax)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_omp_fom.png", bbox_inches='tight')
plt.close()
print("fig3 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 – OMP: kc/baseline Speedup Ratio vs Thread Count
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

for sz, ck, mk in zip(SIZES, SIZE_COLORS_K, SIZE_MARKS):
    ratio = [k / b for k, b in zip(omp_data["kc"][sz], omp_data["baseline"][sz])]
    ax.plot(omp_threads, ratio, marker=mk, color=ck, lw=2, ms=6,
            label=size_label(sz))

ax.axhline(1.0, color=C_IDEAL, ls='--', lw=1.5, label="Baseline reference (1.0)")
ax.axvline(30, color='#CC4400', lw=1.2, ls=':', alpha=0.8, label="NUMA 0/1 boundary (t=30)")
ax.axvline(60, color='#8B0055', lw=1.2, ls=':', alpha=0.8, label="Socket boundary (t=60)")

ax.fill_between(omp_threads,
                [k / b for k, b in zip(omp_data["kc"][80], omp_data["baseline"][80])],
                1.0, alpha=0.08, color=C_KC)

ax.set_title("Pure OMP: kc / baseline FOM ratio")
ax.set_xlabel("OMP threads (T)")
ax.set_ylabel("kc / baseline FOM ratio")
ticks = [1, 8, 15, 30, 31, 60, 61, 64]
ax.set_xticks(ticks)
ax.set_xticklabels(staggered_ticklabels(ticks))
ax.tick_params(axis='x', labelsize=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
ax.set_ylim(bottom=0.9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_omp_speedup_ratio.png", bbox_inches='tight')
plt.close()
print("fig4 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 – OMP NUMA Boundary Zoom (s40 only, t=24 to 40)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

zoom_t = [24, 28, 29, 30, 31, 32, 36, 40]
zoom_idx = [omp_threads.index(t) for t in zoom_t]

for ax, sz, title in zip(axes, [40, 80], ["s40", "s80"]):
    fom_b = [omp_data["baseline"][sz][i] for i in zoom_idx]
    fom_k = [omp_data["kc"][sz][i] for i in zoom_idx]
    cb = SIZE_COLORS_B[SIZES.index(sz)]
    ck = SIZE_COLORS_K[SIZES.index(sz)]

    ax.plot(zoom_t, fom_b, 's-', color=cb, lw=2.5, ms=9, label="baseline")
    ax.plot(zoom_t, fom_k, 'o-', color=ck, lw=2.5, ms=9, label="kc")

    # Shade the NUMA crossing
    ax.axvspan(30, 31, alpha=0.12, color='#CC4400', label="Cross-NUMA region")
    ax.axvline(30, color='#CC4400', lw=2, ls='-', alpha=0.6)
    ax.axvline(31, color='#CC4400', lw=2, ls='--', alpha=0.6)

    # Annotate the drop at t=31 for kc
    v30_k = omp_data["kc"][sz][omp_threads.index(30)]
    v31_k = omp_data["kc"][sz][omp_threads.index(31)]
    drop_k = (v31_k - v30_k) / v30_k * 100
    ax.annotate(f"kc: {drop_k:+.1f}%",
                xy=(31, v31_k), xytext=(32, (v30_k + v31_k)/2),
                arrowprops=dict(arrowstyle='->', color=ck, lw=1.5),
                fontsize=10, color=ck, fontweight='bold')

    v30_b = omp_data["baseline"][sz][omp_threads.index(30)]
    v31_b = omp_data["baseline"][sz][omp_threads.index(31)]
    drop_b = (v31_b - v30_b) / v30_b * 100
    ax.annotate(f"baseline: {drop_b:+.1f}%",
                xy=(31, v31_b), xytext=(32, v31_b * 0.97),
                arrowprops=dict(arrowstyle='->', color=cb, lw=1.5),
                fontsize=10, color=cb, fontweight='bold')

    ax.set_title(f"{size_label(sz)}: NUMA boundary zoom")
    ax.set_xlabel("OMP threads (T)")
    ax.set_ylabel("FOM (kzc/s)")
    ax.set_xticks(zoom_t)
    ax.legend()
    ax.text(30.5, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(fom_b)*0.98,
            "NUMA 0->1", ha='center', fontsize=8.5, color='#CC4400')

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_numa_boundary_zoom.png", bbox_inches='tight')
plt.close()
print("fig5 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Hybrid FOM: Grouped Bar Chart (3 sizes × 7 configs)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for ax, (sz, hdata) in zip(axes, [(20, hybrid_s20), (40, hybrid_s40), (80, hybrid_s80)]):
    n = len(hybrid_configs)
    x = np.arange(n)
    w = 0.38

    bars_b = ax.bar(x - w/2, hdata["baseline"], w, color=C_BASE, alpha=0.85, label="baseline")
    bars_k = ax.bar(x + w/2, hdata["kc"],       w, color=C_KC,   alpha=0.85, label="kc")

    # Annotate speedup ratio
    for xi, (b, k) in enumerate(zip(hdata["baseline"], hdata["kc"])):
        r = k / b
        ax.text(xi + w/2, k + ax.get_ylim()[1]*0.01 if ax.get_ylim()[1]!=1.0 else k*1.02,
                f"×{r:.2f}", ha='center', va='bottom', fontsize=8, color=C_KC, fontweight='bold')

    ax.set_title(size_label(sz))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({t} cores)" for c, t in zip(hybrid_configs, hybrid_total_t)],
                       fontsize=8.5)
    ax.set_xlabel("Hybrid configuration (np x T)")
    ax.set_ylabel("FOM (kzc/s)" if ax == axes[0] else "")
    ax.legend()

# Fix annotations after ylim
for ax, (sz, hdata) in zip(axes, [(20, hybrid_s20), (40, hybrid_s40), (80, hybrid_s80)]):
    ymax = max(max(hdata["baseline"]), max(hdata["kc"])) * 1.18
    ax.set_ylim(0, ymax)
    for xi, (b, k) in enumerate(zip(hdata["baseline"], hdata["kc"])):
        r = k / b
        ax.text(xi + w/2, k + ymax * 0.012,
                f"×{r:.2f}", ha='center', va='bottom', fontsize=8, color=C_KC, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_hybrid_bar.png", bbox_inches='tight')
plt.close()
print("fig6 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Hybrid Thread Efficiency (s80): how well OMP threads scale
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))

# np=8 configs for s80
np8_t   = [1, 4, 8, 10, 11, 12]   # T values
np8_k   = [16894.5, 19412.3, 63574.5, 78974.3, 86674.4, 92474.0]
np8_b   = [15380.7, 15272.4, 34684.3, 38988.7, 50762.3, 53726.0]

eff_k = [np8_k[i] / np8_k[0] / np8_t[i] * 100 for i in range(len(np8_t))]
eff_b = [np8_b[i] / np8_b[0] / np8_t[i] * 100 for i in range(len(np8_t))]

ax.axhline(100, color=C_IDEAL, ls='--', lw=1.5, label="Ideal efficiency 100%")
ax.plot(np8_t, eff_b, 's-', color=C_BASE, lw=2.5, ms=9, label="baseline np=8")
ax.plot(np8_t, eff_k, 'o-', color=C_KC,  lw=2.5, ms=9, label="kc np=8")

for t, e in zip(np8_t[1:], eff_b[1:]):
    ax.annotate(f"{e:.0f}%", xy=(t, e), xytext=(0, -16),
                textcoords='offset points', ha='center', fontsize=9.5, color=C_BASE)
for t, e in zip(np8_t[1:], eff_k[1:]):
    ax.annotate(f"{e:.0f}%", xy=(t, e), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9.5, color=C_KC)

ax.fill_between(np8_t, eff_k, eff_b, alpha=0.10, color=C_KC)
ax.set_title("s80: OMP thread efficiency at fixed np=8")
ax.set_xlabel("OMP threads per rank T (fixed np=8)")
ax.set_ylabel("OMP thread efficiency (%)")
ax.set_xticks(np8_t)
ax.set_ylim(0, 120)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/fig7_hybrid_thread_eff.png", bbox_inches='tight')
plt.close()
print("fig7 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Comprehensive Summary: Peak FOM across all modes (s80)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

labels = [
    "MPI\nnp=1", "MPI\nnp=8", "MPI\nnp=27",
    "OMP\nt=1",  "OMP\nt=30", "OMP\nt=64",
    "Hybrid\n8x4",  "Hybrid\n8x8",  "Hybrid\n8x12",
    "Hybrid\n1x96", "Hybrid\n27x3",
]
fom_b_all = [1894.4, 15380.7, 37833.5,
             1888.0, 1362.9, 1026.2,
             15272.4, 34684.3, 53726.0,
             21088.1, 38815.4]
fom_k_all = [2096.3, 16894.5, 45228.5,
             2103.3, 2029.3, 1817.4,
             19412.3, 63574.5, 92474.0,
             55014.4, 48948.0]

n = len(labels)
x = np.arange(n)
w = 0.38

ax.bar(x - w/2, fom_b_all, w, color=C_BASE, alpha=0.85, label="baseline")
ax.bar(x + w/2, fom_k_all, w, color=C_KC,   alpha=0.85, label="kc")

ymax = max(fom_k_all) * 1.2
ax.set_ylim(0, ymax)

for xi, (b, k) in enumerate(zip(fom_b_all, fom_k_all)):
    sp = (k - b) / b * 100
    sign = "+" if sp >= 0 else ""
    ax.text(xi, max(b, k) + ymax * 0.01,
            f"{sign}{sp:.0f}%", ha='center', va='bottom',
            fontsize=8, color='#333333')

# Section dividers
ax.axvline(2.5, color='#999', lw=1.0, ls='-', alpha=0.5)
ax.axvline(5.5, color='#999', lw=1.0, ls='-', alpha=0.5)
ax.text(1,   ymax * 0.92, "Pure MPI", ha='center', fontsize=10, color='#555', style='italic')
ax.text(4.5, ymax * 0.92, "Pure OMP", ha='center', fontsize=10, color='#555', style='italic')
ax.text(8.5, ymax * 0.92, "Hybrid MPI+OMP", ha='center', fontsize=10, color='#555', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("FOM (kzc/s)")
ax.set_title("s80: peak FOM across parallel modes")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_summary_all_modes.png", bbox_inches='tight')
plt.close()
print("fig8 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9 – OMP Degradation Rate: % loss vs t=1
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

for ax, sz in zip(axes, [40, 80]):
    cb = SIZE_COLORS_B[SIZES.index(sz)]
    ck = SIZE_COLORS_K[SIZES.index(sz)]
    base_b = omp_data["baseline"][sz][0]
    base_k = omp_data["kc"][sz][0]

    pct_b = [(v / base_b) * 100 for v in omp_data["baseline"][sz]]
    pct_k = [(v / base_k) * 100 for v in omp_data["kc"][sz]]

    ax.plot(omp_threads, pct_b, 's-', color=cb, lw=2, ms=6, label="baseline")
    ax.plot(omp_threads, pct_k, 'o-', color=ck, lw=2, ms=6, label="kc")
    ax.axhline(100, color=C_IDEAL, ls='--', lw=1.3, label="t=1 reference")
    ax.fill_between(omp_threads, pct_k, pct_b, alpha=0.08, color=C_KC)

    ax.axvline(30, color='#CC4400', lw=1.2, ls=':', alpha=0.7)
    ax.axvline(60, color='#8B0055', lw=1.2, ls=':', alpha=0.7)
    ax.text(30.5, 15, "NUMA 0/1", fontsize=8, color='#CC4400', rotation=90, va='bottom')
    ax.text(60.5, 15, "Socket", fontsize=8, color='#8B0055', rotation=90, va='bottom')

    # Annotate final values
    ax.annotate(f"t=64: {pct_b[-1]:.0f}%", xy=(64, pct_b[-1]),
                xytext=(-48, -22), textcoords='offset points',
                fontsize=9, color=cb, arrowprops=dict(arrowstyle='->', color=cb))
    ax.annotate(f"t=64: {pct_k[-1]:.0f}%", xy=(64, pct_k[-1]),
                xytext=(-48, 14), textcoords='offset points',
                fontsize=9, color=ck, arrowprops=dict(arrowstyle='->', color=ck))

    ax.set_title(f"{size_label(sz)}: OMP retention")
    ax.set_xlabel("OMP threads (T)")
    ax.set_ylabel("FOM retention vs t=1 (%)")
    ticks = [1, 8, 15, 30, 31, 60, 61, 64]
    ax.set_xticks(ticks)
    ax.set_xticklabels(staggered_ticklabels(ticks))
    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylim(0, 120)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/fig9_omp_retention.png", bbox_inches='tight')
plt.close()
print("fig9 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Amdahl Analysis: baseline 1x96 hybrid parallel fraction
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))

# Measured speedup for 1×96 (vs np=1 pure MPI single-thread)
# baseline: 21088/1894 = 11.1x, kc: 55014/2096 = 26.2x
# Amdahl: S(p) = 1/( (1-f) + f/p )  => f = (1/S - 1/p)/(1 - 1/p)
p = 96
S_b = 21088.1 / 1894.4   # 11.13×
S_k = 55014.4 / 2096.3   # 26.24×

# Amdahl: S(p) = 1/((1-f)+f/p) where f = parallel fraction
# => f = (1 - 1/S) / (1 - 1/p)
f_par_b = (1 - 1/S_b) / (1 - 1/p)   # parallel fraction baseline ≈ 92%
f_par_k = (1 - 1/S_k) / (1 - 1/p)   # parallel fraction kc       ≈ 97%
s_b = 1 - f_par_b                     # serial fraction baseline   ≈ 8%
s_k = 1 - f_par_k                     # serial fraction kc         ≈ 3%

p_range = np.arange(1, 200)
amdahl_b = 1 / ((1 - f_par_b) + f_par_b / p_range)
amdahl_k = 1 / ((1 - f_par_k) + f_par_k / p_range)
ideal_line = p_range.astype(float)

# Theoretical maximum speedup
smax_b = 1 / s_b
smax_k = 1 / s_k

ax.plot(p_range, ideal_line,  '--', color=C_IDEAL, lw=1.5, label="Ideal linear scaling")
ax.plot(p_range, amdahl_b, '-', color=C_BASE, lw=2.5,
        label=f"baseline (serial fraction {s_b*100:.1f}%, limit {smax_b:.0f}x)")
ax.plot(p_range, amdahl_k, '-', color=C_KC,  lw=2.5,
        label=f"kc (serial fraction {s_k*100:.1f}%, limit {smax_k:.0f}x)")

# Horizontal asymptotes
ax.axhline(smax_b, color=C_BASE, lw=1.0, ls='--', alpha=0.45)
ax.axhline(smax_k, color=C_KC,   lw=1.0, ls='--', alpha=0.45)
ax.text(148, smax_b - 1.0, f"Limit {smax_b:.0f}x", fontsize=8.5, color=C_BASE,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
ax.text(148, smax_k + 0.6, f"Limit {smax_k:.0f}x", fontsize=8.5, color=C_KC,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# Measured points
ax.scatter([p], [S_b], color=C_BASE, zorder=5, s=90, marker='D')
ax.scatter([p], [S_k], color=C_KC,  zorder=5, s=90, marker='D')
ax.annotate(f"Measured {p} cores\n{S_b:.1f}x", xy=(p, S_b),
            xytext=(p+10, S_b - 3.0), fontsize=9.5, color=C_BASE,
            arrowprops=dict(arrowstyle='->', color=C_BASE, lw=1.2))
ax.annotate(f"Measured {p} cores\n{S_k:.1f}x", xy=(p, S_k),
            xytext=(p+10, S_k + 2.0), fontsize=9.5, color=C_KC,
            arrowprops=dict(arrowstyle='->', color=C_KC, lw=1.2))

ax.set_title("s80 hybrid 1x96: Amdahl fit")
ax.set_xlabel("Threads / ranks p")
ax.set_ylabel("Speedup S(p)")
ax.set_xlim(1, 200)
ax.set_ylim(0, max(smax_k, S_k) * 1.15)
ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig10_amdahl.png", bbox_inches='tight')
plt.close()
print("fig10 done")

print("\nAll figures saved to", OUT)
