import subprocess
import re
import os

# 配置路径
VERSIONS = {
    "OpenMP": "/home/fzj/桌面/ppn/lulesh/LULESH/build/lulesh2.0",
    "Kokkos-OpenMP": "../build/release/lulesh2.0"
}

SIZES = [30, 45, 60, 90]
ITERATIONS = 100
THREADS = 16  # 修改为 16 线程

def run_lulesh_energy(exe_path, size, version_name):
    # 继承环境并设置 16 线程
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    # 禁用显式的线程绑定，让系统自动分配，提高兼容性
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)

    cmd = [exe_path, "-s", str(size), "-i", str(ITERATIONS)]
    
    try:
        # 运行并捕获输出
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        output = proc.stdout
        
        # 打印原始结果，方便查看 16 线程下的运行速度 (FOM/Grind time)
        print(f"\n--- Raw Output for {version_name} (Size {size}, Threads {THREADS}) ---")
        if output.strip():
            # 仅打印最后几行关键信息，防止刷屏，或者根据需要打印全部内容
            # 这里选择打印全部内容以便你确认 "Num threads: 16"
            print(output.strip())
        else:
            print("[No Output captured]")
            if proc.stderr:
                print(f"Error Output:\n{proc.stderr}")
        print("-" * 60)

        # 匹配能量值
        energy_match = re.search(r"Final Origin Energy\s+=\s+([0-9.eE+-]+)", output)
        if energy_match:
            return float(energy_match.group(1))
        return None
            
    except Exception as e:
        print(f"Execution Error: {e}")
        return None

# --- 执行循环 ---
for s in SIZES:
    results = {}
    print(f"\n" + "*"*60)
    print(f"  RUNNING PROBLEM SIZE: {s} x {s} x {s} (Threads: {THREADS})")
    print("*"*60)
    
    for ver, path in VERSIONS.items():
        energy = run_lulesh_energy(path, s, ver)
        results[ver] = energy
    
    # --- 最终总结部分 ---
    print(f"\n>>> Results Summary (Size {s}):")
    for ver, energy in results.items():
        print(f"  {ver}: {energy}")
    
    val_list = list(results.values())
    if None in val_list:
        print("Status: ⚠️ Could not get energy for one or more versions.")
    elif val_list[0] == val_list[1]:
        print("Status: ✅ Consistent (Exact Match at 16 threads)")
    else:
        # 注意：由于多线程并行加法顺序不同，可能会有 10^-10 级别的微小差异
        diff = abs(val_list[0] - val_list[1])
        print(f"Status: ⚠️ Mismatch detected! Absolute Diff: {diff}")