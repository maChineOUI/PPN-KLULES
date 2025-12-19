import re
import csv
import glob
import os

def parse_file(path):
    with open(path) as f:
        text = f.read()

    threads = int(re.search(r"Num threads:\s+(\d+)", text).group(1))
    time = float(re.search(
        r"Elapsed time\s+=\s+([\d\.eE+-]+)", text
    ).group(1))
    fom = float(re.search(
        r"FOM\s+=\s+([\d\.eE+-]+)", text
    ).group(1))

    return threads, time, fom


def collect(impl, size):
    rows = []
    files = glob.glob(f"{impl}/*_s{size}_t*.txt")

    for f in files:
        threads, time, fom = parse_file(f)
        rows.append([threads, impl, time, fom])

    rows.sort(key=lambda x: x[0])
    return rows


for size in [30, 50, 100]:
    rows = []
    rows += collect("baseline", size)
    rows += collect("kokkos", size)

    out = f"summary_s{size}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "impl", "time", "fom"])
        writer.writerows(rows)

    print(f"Generated {out}")
