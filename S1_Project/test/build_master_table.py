import pandas as pd

all_rows = []

for size in [30, 50, 100]:
    df = pd.read_csv(f"summary_s{size}.csv")
    df["size"] = size

    # 以 4 线程为基准计算 speedup / efficiency
    for impl in ["baseline", "kokkos"]:
        mask = df["impl"] == impl
        t0 = df[mask & (df["threads"] == 4)]["time"].values[0]

        df.loc[mask, "speedup"] = t0 / df.loc[mask, "time"]
        df.loc[mask, "efficiency"] = df.loc[mask, "speedup"] / (df.loc[mask, "threads"] / 4)

    # implementation overhead（只对 kokkos 有意义）
    base = df[df["impl"] == "baseline"].sort_values("threads")
    kokk = df[df["impl"] == "kokkos"].sort_values("threads")

    overhead = kokk["time"].values / base["time"].values
    df.loc[df["impl"] == "baseline", "overhead"] = 1.0
    df.loc[df["impl"] == "kokkos", "overhead"] = overhead

    all_rows.append(df)

master = pd.concat(all_rows, ignore_index=True)

# 列顺序整理
master = master[[
    "impl",
    "size",
    "threads",
    "time",
    "fom",
    "speedup",
    "efficiency",
    "overhead"
]]

master.to_csv("lulesh_master_table.csv", index=False)

print("Generated lulesh_master_table.csv")