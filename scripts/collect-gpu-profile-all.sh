#!/usr/bin/env bash
# collect-gpu-profile-complete.sh
# 汇总 LULESH GPU / CPU 性能数据，补全缺失信息（gpukern, ncu, dmon, CPU OpenMP）
# 使用示例：./collect-gpu-profile-complete.sh ./scripts/build-cuda/lulesh2.0

set -euo pipefail

BINARY="${1:-./scripts/build-cuda/lulesh2.0}"
[[ ! -f "$BINARY" ]] && { echo "[错误] 找不到二进制文件: $BINARY"; exit 1; }

OUTDIR="reports/profile-complete-$(date +%Y%m%d_%H%M)"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"

echo "==================== LULESH 完整性能采集汇总 ====================" | tee "$SUMMARY"
echo "Binary: $BINARY" | tee -a "$SUMMARY"
echo "Date: $(date)" | tee -a "$SUMMARY"
echo "Output dir: $OUTDIR" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ------------------------
# Step 1: GPU 状态快照
# ------------------------
echo "=== Step 1: GPU 状态快照 ===" | tee -a "$SUMMARY"
nvidia-smi --query-gpu=name,temperature.gpu,clocks.current.sm,clocks.max.sm,power.draw,power.limit,clocks_throttle_reasons.active \
    --format=csv | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ------------------------
# Step 2: FOM 扩展规模测试
# ------------------------
echo "=== Step 2: FOM 扩展规模测试 ===" | tee -a "$SUMMARY"
echo "Size,Run,FOM,Time_s,Avg_Power_W,Avg_Clocks_MHz" > "$OUTDIR/fom_scale.csv"

for SIZE in 30 45 60 90 ; do
    echo ">>> Size $SIZE (GPU 预热中...)" | tee -a "$SUMMARY"
    "$BINARY" -s $SIZE -i 10 > /dev/null 2>&1  # GPU 预热

    total_fom=0
    valid_runs=0
    for RUN in $(seq 1 3); do
        echo -n "  [Run $RUN/3] " | tee -a "$SUMMARY"
        TEMP_OUT=$("$BINARY" -s $SIZE -i 200 2>&1) || { echo "运行失败" | tee -a "$SUMMARY"; FOM=0; TIME=0; continue; }

        # 安全抓取 FOM 和 Time
        FOM=$(echo "$TEMP_OUT" | grep -oP 'FOM\s*=\s*\K[0-9.]+|[0-9.]+(?=\s*z/s)' | head -1 || echo "0")
        TIME=$(echo "$TEMP_OUT" | grep -oP 'Elapsed\s*time\s*=\s*\K[0-9.]+' | head -1 || echo "0")

        # 获取功率和时钟
        POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
        CLOCK=$(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits)

        echo "$SIZE,$RUN,$FOM,$TIME,$POWER,$CLOCK" >> "$OUTDIR/fom_scale.csv"
        echo "$SIZE,$RUN,FOM=$FOM, Time_s=$TIME, Power=$POWER, Clock=$CLOCK" | tee -a "$SUMMARY"

        if [[ "$FOM" != "0" ]]; then
            total_fom=$(echo "$total_fom + $FOM" | bc)
            valid_runs=$((valid_runs + 1))
        fi
    done

    avg_fom=$(echo "scale=2; if($valid_runs>0) $total_fom/$valid_runs else 0" | bc)
    echo "  >> 平均 FOM: $avg_fom" | tee -a "$SUMMARY"
    echo "--------------------------------------------------" | tee -a "$SUMMARY"
    sleep 5
done

# ------------------------
# Step 3: Nsight Systems (兼容新版报表名)
# ------------------------
for SIZE in 30 45 60 90; do
    REP="$OUTDIR/nsys_s${SIZE}.nsys-rep"
    if command -v nsys &> /dev/null; then
        echo "=== Step 3: Nsight Systems Size=$SIZE ===" | tee -a "$SUMMARY"
        
        # 1. 采集数据
        nsys profile --trace=cuda --output="$OUTDIR/nsys_s${SIZE}" --force-overwrite=true "$BINARY" -s $SIZE -i 10 -q > /dev/null 2>&1

        # 2. 使用新版兼容的报表名统计
        echo "--- GPU Kernel 耗时分布 ---" | tee -a "$SUMMARY"
        # 尝试两种可能的报表名，确保万无一失
        nsys stats --report cuda_gpu_sum "$REP" 2>/dev/null | tee -a "$SUMMARY" || \
        nsys stats --report gputimesum "$REP" 2>/dev/null | tee -a "$SUMMARY"
        
        echo "" | tee -a "$SUMMARY"
    fi
done

# ------------------------
# Step 4: Nsight Compute (基于序号的稳健版)
# ------------------------
NCU_BIN="/usr/local/cuda-12.8/bin/ncu"
for SIZE in 30 90; do
    if [[ -f "$NCU_BIN" ]]; then
        echo "=== Step 4: Nsight Compute Size=$SIZE ===" | tee -a "$SUMMARY"
        
        # 核心逻辑：
        # --launch-skip 5: 避开前面的架构查询和内存分配
        # --launch-count 10: 连续抓取接下来的 10 个内核调用（确保涵盖主循环）
        sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH "$NCU_BIN" \
            --set full \
            --launch-skip 5 \
            --launch-count 10 \
            -o "$OUTDIR/ncu_s${SIZE}_batch" \
            --force-overwrite \
            "$BINARY" -s $SIZE -i 20 -q

        echo "--- 捕获的内核序列汇总 ---" | tee -a "$SUMMARY"
        sudo "$NCU_BIN" --import "$OUTDIR/ncu_s${SIZE}_batch.ncu-rep" --print-summary per-kernel | tee -a "$SUMMARY"
        echo "" | tee -a "$SUMMARY"
    fi
done


# ------------------------
# Step 5: GPU 实时监控（频率/功率/温度）
# ------------------------
echo "=== Step 5: GPU dmon 实时监控 s=90 ===" | tee -a "$SUMMARY"
sudo nvidia-smi dmon -s pucvt -d 1 > "$OUTDIR/gpu_dmon_s90.txt" &
DMON_PID=$!
"$BINARY" -s 90 -i 200 -q
kill $DMON_PID
echo "GPU dmon 输出：$OUTDIR/gpu_dmon_s90.txt" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

echo "==================== 数据采集完成 ====================" | tee -a "$SUMMARY"
echo "汇总文件：$SUMMARY"