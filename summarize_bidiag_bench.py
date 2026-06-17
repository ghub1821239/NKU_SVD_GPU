import csv
import re
import statistics
import sys
from pathlib import Path


RESULT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./NEW_perf_results/bidiag_bench")
OUT_CSV = RESULT_DIR / "summary_bidiag_bench.csv"
OUT_MD = RESULT_DIR / "summary_bidiag_bench.md"

PATTERN_BIDIAG = re.compile(r"总上二对角化耗时\(ms\):\s*([0-9.eE+-]+)")
PATTERN_GKH = re.compile(r"总GKH迭代耗时\(ms\):\s*([0-9.eE+-]+)")
PATTERN_PASS = re.compile(r"通过:\s*(\d+)\s*/\s*(\d+)")


def mean_or_zero(values):
    return statistics.mean(values) if values else 0.0


rows = []

for file in sorted(RESULT_DIR.glob("*_program.out")):
    text = file.read_text(encoding="utf-8", errors="ignore")
    label = file.name.replace("_program.out", "")

    bidiag_times = [float(x) for x in PATTERN_BIDIAG.findall(text)]
    gkh_times = [float(x) for x in PATTERN_GKH.findall(text)]
    pass_pairs = PATTERN_PASS.findall(text)

    if not bidiag_times or not gkh_times:
        print(f"Skip {file.name}: missing timing data")
        continue

    runs = min(len(bidiag_times), len(gkh_times))
    bidiag_times = bidiag_times[:runs]
    gkh_times = gkh_times[:runs]
    total_times = [b + g for b, g in zip(bidiag_times, gkh_times)]

    pass_ok = 0
    for passed, total in pass_pairs[:runs]:
        if passed == total:
            pass_ok += 1

    rows.append(
        {
            "version": label,
            "runs": runs,
            "pass_runs": pass_ok,
            "avg_bidiag_ms": mean_or_zero(bidiag_times),
            "min_bidiag_ms": min(bidiag_times),
            "max_bidiag_ms": max(bidiag_times),
            "avg_gkh_ms": mean_or_zero(gkh_times),
            "avg_total_ms": mean_or_zero(total_times),
        }
    )

rows.sort(key=lambda row: row["version"])

cpu_row = next((row for row in rows if row["version"] == "cpu"), None)
cpu_avg = cpu_row["avg_bidiag_ms"] if cpu_row else None

for row in rows:
    if cpu_avg and row["avg_bidiag_ms"] > 0:
        row["bidiag_speedup_vs_cpu"] = cpu_avg / row["avg_bidiag_ms"]
    else:
        row["bidiag_speedup_vs_cpu"] = 0.0


fieldnames = [
    "version",
    "runs",
    "pass_runs",
    "avg_bidiag_ms",
    "min_bidiag_ms",
    "max_bidiag_ms",
    "avg_gkh_ms",
    "avg_total_ms",
    "bidiag_speedup_vs_cpu",
]

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def fmt(x):
    if isinstance(x, int):
        return str(x)
    return f"{x:.6f}"


md_lines = [
    "| version | runs | pass_runs | avg_bidiag_ms | min_bidiag_ms | max_bidiag_ms | avg_gkh_ms | avg_total_ms | bidiag_speedup_vs_cpu |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    md_lines.append(
        "| {version} | {runs} | {pass_runs} | {avg_bidiag_ms} | {min_bidiag_ms} | {max_bidiag_ms} | {avg_gkh_ms} | {avg_total_ms} | {bidiag_speedup_vs_cpu} |".format(
            version=row["version"],
            runs=row["runs"],
            pass_runs=row["pass_runs"],
            avg_bidiag_ms=fmt(row["avg_bidiag_ms"]),
            min_bidiag_ms=fmt(row["min_bidiag_ms"]),
            max_bidiag_ms=fmt(row["max_bidiag_ms"]),
            avg_gkh_ms=fmt(row["avg_gkh_ms"]),
            avg_total_ms=fmt(row["avg_total_ms"]),
            bidiag_speedup_vs_cpu=fmt(row["bidiag_speedup_vs_cpu"]),
        )
    )

OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_MD}")
