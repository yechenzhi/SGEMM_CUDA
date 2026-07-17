#!/usr/bin/env bash

set -uo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${1:-"$repo_dir/benchmark_results/kernel_13_43"}
sgemm_bin=${SGEMM_BIN:-"$repo_dir/build/sgemm"}
device=${DEVICE:-0}
kernels=({13..28} {30..37} {39..43})

if [[ ! -x "$sgemm_bin" ]]; then
  printf 'Missing executable: %s\n' "$sgemm_bin" >&2
  exit 2
fi

mkdir -p "$output_dir"
printf 'kernel\tstatus\n' > "$output_dir/status.tsv"

failed=0
for kernel in "${kernels[@]}"; do
  printf '\n===== kernel %s =====\n' "$kernel"
  DEVICE="$device" "$sgemm_bin" "$kernel" 2>&1 |
    tee "$output_dir/kernel_${kernel}.log"
  status=${PIPESTATUS[0]}
  printf '%s\t%s\n' "$kernel" "$status" >> "$output_dir/status.tsv"
  if ((status != 0)); then
    failed=1
  fi
done

printf '\nResults: %s\n' "$output_dir"
printf 'Status file: %s\n' "$output_dir/status.tsv"
exit "$failed"
