import subprocess
import re
import statistics
import pandas as pd
import os
import platform
import matplotlib.pyplot as plt

# -----------------------------
# 输出目录
OUT_DIR = "experiment_results"
RAW_DIR = os.path.join(OUT_DIR, "raw")
SUMMARY_DIR = os.path.join(OUT_DIR, "summary")
PERF_DIR = os.path.join(OUT_DIR, "perf")
for d in [RAW_DIR, SUMMARY_DIR, PERF_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------
# 全局 FLAMEGRAPH_PATH
FLAMEGRAPH_PATH = "/home/fzj/桌面/ppn/new/lulesh_3.3/PPN-KLULES/FlameGraph"

# -----------------------------
# TEST_MODE 开关
TEST_MODE = False  # True=小规模测试, False=全量测试

if TEST_MODE:
    SIZES = [30, 45, 60]  # 小规模测试只选取前三个问题规模
    THREADS_LIST = [1, 4,8, 16]  # 小规模测试只选取部分线程数
    REPEATS = 2
    PERF_SIZES = [30]     # 小规模测试只采样一个问题规模
    PERF_ITERS = 10
else:
    SIZES = [30, 45, 60, 90]
    THREADS_LIST = [1, 2, 4, 8, 16, 32]
    REPEATS = 10
    PERF_SIZES = SIZES    # 全量测试，对所有 Sizes 采样
    PERF_ITERS = 50

print("\n==============================")
print(f"TEST_MODE = {TEST_MODE} ({'小规模测试' if TEST_MODE else '全量测试'})")
print(f"Sizes to run = {SIZES}")
print(f"Threads List = {THREADS_LIST}")
print(f"Repeats per config = {REPEATS}")
print(f"Perf Sizes = {PERF_SIZES}")
print(f"Perf Iterations = {PERF_ITERS}")
print("==============================\n")

# -----------------------------
# CPU 性能模式 + OpenMP 核心绑定
print("锁定 CPU 性能模式为 performance...")
subprocess.run("sudo cpupower frequency-set -g performance", shell=True, check=True)

os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"
print("OpenMP 线程绑定已设置: OMP_PROC_BIND=true, OMP_PLACES=cores")

# -----------------------------
# 硬件信息采集
HW_FILE = os.path.join(OUT_DIR, "hardware_software_detail.txt")

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except:
        return "Error executing command"

def run_sudo_cmd(cmd):
    try:
        proc = subprocess.run(f"sudo {cmd}", shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True)
        return proc.stdout.decode().strip()
    except Exception as e:
        return f"Error executing sudo command: {e}"

print("正在采集详细系统信息...")
with open(HW_FILE, "w") as f:
    f.write("="*30 + "\nFULL SYSTEM SPECIFICATIONS\n" + "="*30 + "\n\n")
    f.write("--- [CPU INFO (lscpu)] ---\n")
    f.write(run_cmd("lscpu") + "\n\n")
    f.write("--- [MEMORY INFO (dmidecode)] ---\n")
    f.write(run_sudo_cmd("dmidecode -t memory | grep -A 20 'Memory Device' | grep -E 'Size|Type|Speed|Rank|Configured'") + "\n\n")
    f.write("--- [OS & KERNEL] ---\n")
    f.write(f"OS: {platform.platform()}\n")
    f.write(f"Kernel: {run_cmd('uname -r')}\n")
    f.write(f"Hostname: {platform.node()}\n\n")
    f.write("--- [COMPILER & FLAGS] ---\n")
    f.write(f"Compiler Path: {run_cmd('which g++-13')}\n")
    f.write(run_cmd("g++-13 --version | head -n1") + "\n")
    f.write("Build Mode: Release\n\n")
    f.write("--- [HPC TUNING] ---\n")
    f.write(f"CPU Governor: {run_cmd('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')}\n")
    f.write(f"Scaling Driver: {run_cmd('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver')}\n")
print(f"详细硬件信息已保存到 {HW_FILE}")

# -----------------------------
# 实验配置
VERSIONS = {
    "OpenMP": "/home/fzj/桌面/ppn/lulesh/LULESH/build/lulesh2.0",
    "Kokkos-OpenMP": "../build/release/lulesh2.0"
}
ITERATIONS = 100

def run_lulesh(exe_path, threads, size, warmup=False):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "true"
    env["OMP_PLACES"] = "cores"
    cmd = [exe_path, "-s", str(size), "-i", str(ITERATIONS)]
    if warmup:
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None, None
    try:
        result = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT).decode()
        fom = re.search(r"FOM\s+=\s+([\d.]+)", result)
        elap = re.search(r"Elapsed time\s+=\s+([\d.]+)", result)
        return (float(fom.group(1)), float(elap.group(1))) if fom and elap else (None, None)
    except Exception as e:
        print(f"Error running {exe_path} with size={size}, threads={threads}: {e}")
        return None, None

# -----------------------------
# 热机函数
def warmup_lulesh(exe_path, threads, size, repeats=3):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "true"
    env["OMP_PLACES"] = "cores"
    cmd = [exe_path, "-s", str(size), "-i", str(ITERATIONS)]
    for i in range(repeats):
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# -----------------------------
# 核心性能测试
raw_records = []
summary_results = []

for ver, path in VERSIONS.items():
    print(f"\n>>> 开始测试版本: {ver}")
    for s in SIZES:
        for t in THREADS_LIST:
            print(f"\n  Size {s}, Threads {t}: Warmup...", end="", flush=True)
            warmup_lulesh(path, t, s, repeats=3)

            foms, times = [], []
            for r in range(REPEATS):
                print(".", end="", flush=True)
                f_val, t_val = run_lulesh(path, t, s)
                print(f"DEBUG: Version={ver}, Size={s}, Threads={t}, Run={r+1}, FOM={f_val}, Time={t_val}")
                if f_val is not None:
                    foms.append(f_val)
                    times.append(t_val)
                    raw_records.append({
                        "Version": ver,
                        "Size": s,
                        "Threads": t,
                        "Run": r+1,
                        "FOM": f_val,
                        "Time": t_val
                    })

            if not foms:
                foms = [0]
                times = [0]

            mean_fom = statistics.mean(foms)
            mean_time = statistics.mean(times)
            stddev_fom = statistics.stdev(foms) if len(foms) > 1 else 0
            stddev_time = statistics.stdev(times) if len(times) > 1 else 0
            summary_results.append({
                "Version": ver,
                "Size": s,
                "Threads": t,
                "Mean_FOM": round(mean_fom, 2),
                "StdDev_FOM_%": f"{(stddev_fom/mean_fom*100):.2f}%",
                "Mean_Time": round(mean_time, 4),
                "Min_Time": min(times),
                "Max_Time": max(times),
                "Median_Time": statistics.median(times),
                "StdDev_Time": round(stddev_time, 4)
            })

            # -----------------------------
            # 当前配置完成后立即保存 CSV 并刷新磁盘
            raw_csv_file = os.path.join(RAW_DIR, "raw_data.csv")
            summary_csv_file = os.path.join(SUMMARY_DIR, "summary.csv")

            pd.DataFrame(raw_records).to_csv(raw_csv_file, index=False)
            pd.DataFrame(summary_results).to_csv(summary_csv_file, index=False)
            os.sync()  # 强制刷新磁盘
            print(f"\n>>> [Autosave] Version={ver}, Size={s}, Threads={t} 数据已保存到:")
            print(f"    {raw_csv_file}")
            print(f"    {summary_csv_file}")

# -----------------------------
# 计算并行效率
df_summary = pd.DataFrame(summary_results)
parallel_eff_list = []

for ver in df_summary['Version'].unique():
    for s in df_summary['Size'].unique():
        single_thread_time = df_summary[(df_summary['Version']==ver) & (df_summary['Size']==s) & (df_summary['Threads']==1)]['Mean_Time'].values[0]
        for t in df_summary['Threads'].unique():
            t_time = df_summary[(df_summary['Version']==ver) & (df_summary['Size']==s) & (df_summary['Threads']==t)]['Mean_Time'].values[0]
            speedup = single_thread_time / t_time
            parallel_eff = speedup / t
            parallel_eff_list.append({
                "Version": ver,
                "Size": s,
                "Threads": t,
                "Mean_Time": t_time,
                "Speedup": round(speedup, 4),
                "Parallel_Efficiency": round(parallel_eff, 4)
            })

df_parallel = pd.DataFrame(parallel_eff_list)
parallel_eff_file = os.path.join(OUT_DIR, "parallel_efficiency.csv")
df_parallel.to_csv(parallel_eff_file, index=False)
print(f"\nParallel efficiency saved to: {parallel_eff_file}")

# -----------------------------
# perf 采样 + 火焰图
for ver, path in VERSIONS.items():
    for s in PERF_SIZES:
        perf_file = os.path.join(PERF_DIR, f"profile_{ver}_size{s}.data")
        perf_script_out = os.path.join(PERF_DIR, f"profile_{ver}_size{s}.perf")
        folded_out = os.path.join(PERF_DIR, f"profile_{ver}_size{s}.folded")
        flamegraph_svg = os.path.join(PERF_DIR, f"profile_{ver}_size{s}_flamegraph.svg")

        print(f"\nProfiling {ver}, problem size {s} and generating flamegraph...")
        try:
            subprocess.run(
                f"perf record -g -o {perf_file} -- {path} -s {s} -i {PERF_ITERS}",
                shell=True, check=True
            )
            subprocess.run(f"perf script -i {perf_file} > {perf_script_out}", shell=True, check=True)
            subprocess.run(f"{FLAMEGRAPH_PATH}/stackcollapse-perf.pl {perf_script_out} > {folded_out}", shell=True, check=True)
            subprocess.run(f"{FLAMEGRAPH_PATH}/flamegraph.pl {folded_out} > {flamegraph_svg}", shell=True, check=True)
            print(f"Flamegraph completed: {flamegraph_svg}")
        except subprocess.CalledProcessError as e:
            print(f"Error during profiling {ver} size {s}: {e}")

print(f"\n[全流程完成] 结果已保存到: {OUT_DIR}")