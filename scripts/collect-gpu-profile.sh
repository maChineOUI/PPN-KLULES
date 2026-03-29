#!/usr/bin/env bash
set -euo pipefail

# 1. 环境准备
BINARY="${1:-./build-cuda/lulesh2.0}"
[[ ! -f "$BINARY" ]] && { echo "[错误] 找不到二进制文件: $BINARY"; exit 1; }

# 对齐手动成功的环境
export OMP_PROC_BIND=false
export CUDA_LAUNCH_BLOCKING=0

OUT="reports/profile-$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"

echo "=================================================="
echo "  LULESH GPU 高级性能采集 (sm_89)"
echo "  输出目录: $OUT"
echo "=================================================="

# 2. 采集静态硬件详情
nvidia-smi -q -d MEMORY,POWER,CLOCK | grep -E "Product Name|Total|Max" > "$OUT/hardware_specs.txt"

echo "Size,Run,FOM,Time_s,Avg_Power_W,Avg_Clocks_MHz" | tee "$OUT/fom_scale.csv"

# 3. 测试循环
for SIZE in 30 45 60 90 120 150; do
    echo ">>> 规模 -s $SIZE (预热中...)"
    # 热机：让 GPU 升频
    "$BINARY" -s "$SIZE" -i 10 > /dev/null 2>&1
    
    total_fom=0
    
    for RUN in $(seq 1 3); do
        echo -n "  [Run $RUN/3] "
        
        # --- 异步监控 GPU 状态 (后台运行) ---
        # 记录 1 秒一次的功耗和频率
        monitor_pid=""
        nvidia-smi --query-gpu=power.draw,clocks.current.sm --format=csv,noheader,nounits -l 1 > "$OUT/gpu_trace_${SIZE}_${RUN}.tmp" &
        monitor_pid=$!

        # --- 执行 LULESH ---
        TEMP_OUT=$( "$BINARY" -s "$SIZE" -i 100 2>&1 ) || { kill $monitor_pid; echo "失败"; continue; }
        
        # 停止监控
        kill $monitor_pid 2>/dev/null || true
        
        # 解析数据
        FOM=$(echo "$TEMP_OUT" | grep "FOM" | awk '{print $3}')
        TIME=$(echo "$TEMP_OUT" | grep "Elapsed time" | awk '{print $4}')
        
        # 计算该次运行的平均功耗和频率 (使用 awk)
        GPU_STATS=$(awk '{p+=$1; c+=$2; n++} END {if(n>0) printf "%.1f,%.0f", p/n, c/n; else print "0,0"}' "$OUT/gpu_trace_${SIZE}_${RUN}.tmp")
        rm "$OUT/gpu_trace_${SIZE}_${RUN}.tmp"

        if [[ -n "$FOM" ]]; then
            echo "$SIZE,$RUN,$FOM,$TIME,$GPU_STATS" | tee -a "$OUT/fom_scale.csv"
            total_fom=$(echo "$total_fom + $FOM" | bc)
        else
            echo "错误: 无输出"
        fi
    done
    
    avg_fom=$(echo "scale=2; $total_fom / 3" | bc)
    echo "  >> 规模 $SIZE 平均 FOM: $avg_fom"
    echo "--------------------------------------------------"
    sleep 10 # 笔记本散热冷却
done

# 4. 自动采集 Nsight Stats (仅针对中大规模 -s 90)
if command -v nsys &> /dev/null; then
    echo ">>> 正在采集 Nsight Kernel 统计 (-s 90)..."
    nsys profile --trace=cuda --output="$OUT/nsys_s90" --force-overwrite=true "$BINARY" -s 90 -i 10 -q > /dev/null
    nsys stats --report cuda_kern_sum "$OUT/nsys_s90.nsys-rep" > "$OUT/kernel_bottlenecks.txt" 2>/dev/null
    echo "  [完成] 瓶颈分析见: kernel_bottlenecks.txt"
fi

echo "=================================================="
echo " 所有采集任务已结束。"