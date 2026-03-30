#!/usr/bin/env bash
# collect-gpu-info-v2.sh
# LULESH GPU 性能数据采集（改进版，支持 FOM/Elapsed 正确抓取 & ncu 绝对路径）

set -euo pipefail

BINARY="${1:-./scripts/build-cuda/lulesh2.0}"
[[ ! -f "$BINARY" ]] && { echo "[错误] 找不到二进制文件: $BINARY"; exit 1; }

OUTDIR="reports/gpu-profile-$(date +%Y%m%d_%H%M)"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"

# 确认 ncu 绝对路径
NCU_BIN="/usr/local/cuda-12.8/bin/ncu"
if [[ ! -x "$NCU_BIN" ]]; then
    echo "[错误] 找不到 ncu，请确认 CUDA 安装路径：$NCU_BIN"
    exit 1
fi

echo "==================== LULESH GPU 数据采集 ====================" | tee "$SUMMARY"
echo "Binary: $BINARY" | tee -a "$SUMMARY"
echo "Date: $(date '+%Y年 %m月 %d日 %A %H:%M:%S %Z')" | tee -a "$SUMMARY"
echo "Output dir: $OUTDIR" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ------------------------
# Group 1: 基础 FOM
# ------------------------
echo "=== Group 1: 基础 FOM (5分钟) ===" | tee -a "$SUMMARY"
echo "s,FOM,Elapsed_s" > "$OUTDIR/fom_basic.csv"
for s in 30 45 60 90 120; do
    OUTPUT=$("$BINARY" -s $s -i 200 2>&1)
    FOM=$(echo "$OUTPUT" | grep -E "FOM" | grep -oP '[0-9]+')
    ELAPSED=$(echo "$OUTPUT" | grep -E "Elapsed" | grep -oP '[0-9.]+' | head -1)
    echo "$s,$FOM,$ELAPSED" >> "$OUTDIR/fom_basic.csv"
    echo "s=$s FOM=$FOM Elapsed=$ELAPSED" | tee -a "$SUMMARY"
done
echo "" | tee -a "$SUMMARY"

# ------------------------
# Group 2: P2-A 寄存器效果 (ncu)
# ------------------------
echo "=== Group 2: P2-A 寄存器效果 (ncu) ===" | tee -a "$SUMMARY"
P2A_OUT="$OUTDIR/ncu_p2a.ncu-rep"
sudo "$NCU_BIN" --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct \
    -o "$P2A_OUT" --force-overwrite "$BINARY" -s 45 -i 3
sudo "$NCU_BIN" --import "$P2A_OUT" --print-summary per-kernel | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ------------------------
# Group 3: P2-B 内存带宽效果 (ncu)
# ------------------------
echo "=== Group 3: P2-B 内存带宽效果 (ncu) ===" | tee -a "$SUMMARY"
P2B_OUT="$OUTDIR/ncu_p2b.ncu-rep"
sudo "$NCU_BIN" --metrics \
dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
-o "$P2B_OUT" --force-overwrite "$BINARY" -s 45 -i 3
sudo "$NCU_BIN" --import "$P2B_OUT" --print-summary per-kernel | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ------------------------
# Group 4: Roofline 位置
# ------------------------
echo "=== Group 4: Roofline 位置 ===" | tee -a "$SUMMARY"
ROOFLINE_OUT="$OUTDIR/ncu_roofline.ncu-rep"
sudo "$NCU_BIN" --set roofline -o "$ROOFLINE_OUT" --force-overwrite "$BINARY" -s 45 -i 2
sudo "$NCU_BIN" --import "$ROOFLINE_OUT" --print-summary per-kernel | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ------------------------
# Group 5: EOS 串行碎片 (nsys)
# ------------------------
echo "=== Group 5: EOS 串行碎片 (nsys) ===" | tee -a "$SUMMARY"
NSYS_OUT="$OUTDIR/nsys_eos"
nsys profile --stats=true --force-overwrite=true -o "$NSYS_OUT" "$BINARY" -s 45 -i 10 -q
nsys stats --report cuda_gpu_sum "$NSYS_OUT.nsys-rep" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ------------------------
# Group 6: Warp 延迟 (ncu)
# ------------------------
echo "=== Group 6: Warp 延迟 ===" | tee -a "$SUMMARY"
WARP_OUT="$OUTDIR/ncu_warp.ncu-rep"
sudo "$NCU_BIN" --metrics smsp__average_warp_latency_due_to_long_scoreboard.ratio,smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct,smsp__warp_cycles_per_issued_instruction.avg \
    -o "$WARP_OUT" --force-overwrite "$BINARY" -s 45 -i 3
sudo "$NCU_BIN" --import "$WARP_OUT" --print-summary per-kernel | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

echo "==================== 数据采集完成 ====================" | tee -a "$SUMMARY"
echo "汇总文件：$SUMMARY"