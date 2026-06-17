#!/bin/bash

set -euo pipefail

RUNS="${1:-10}"
SEED="${2:-20260408}"
OUT_DIR="${3:-./NEW_perf_results/bidiag_bench}"

mkdir -p "${OUT_DIR}"

declare -a PROGRAMS=(
  "cpu:./main_cpu"
  "hipblas:./main_hipblas"
  "hipkernel:./main_hipkernel"
)

echo "runs=${RUNS}" > "${OUT_DIR}/meta.txt"
echo "seed=${SEED}" >> "${OUT_DIR}/meta.txt"
echo "out_dir=${OUT_DIR}" >> "${OUT_DIR}/meta.txt"

for entry in "${PROGRAMS[@]}"; do
    label="${entry%%:*}"
    exe="${entry#*:}"
    out_file="${OUT_DIR}/${label}_program.out"

    if [ ! -x "${exe}" ]; then
        echo "Skip ${label}: ${exe} not found or not executable"
        continue
    fi

    : > "${out_file}"

    echo "========================================" | tee -a "${out_file}"
    echo "Benchmark ${label}" | tee -a "${out_file}"
    echo "Executable: ${exe}" | tee -a "${out_file}"
    echo "Runs: ${RUNS}" | tee -a "${out_file}"
    echo "Seed: ${SEED}" | tee -a "${out_file}"
    echo "========================================" | tee -a "${out_file}"

    for ((i = 1; i <= RUNS; ++i)); do
        echo "" | tee -a "${out_file}"
        echo "========== RUN ${i}/${RUNS} BEGIN ==========" | tee -a "${out_file}"
        "${exe}" "${SEED}" 2>&1 | tee -a "${out_file}"
        echo "========== RUN ${i}/${RUNS} END ==========" | tee -a "${out_file}"
    done
done

python3 summarize_bidiag_bench.py "${OUT_DIR}"

echo "Benchmark done. Raw logs are in ${OUT_DIR}"
echo "Summary CSV: ${OUT_DIR}/summary_bidiag_bench.csv"
echo "Summary MD : ${OUT_DIR}/summary_bidiag_bench.md"
