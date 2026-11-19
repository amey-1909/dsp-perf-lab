#!/usr/bin/env bash
# Create the next GEMM step as a copy of the previous one.
#   scripts/new_gemm_step.sh <short_name> "<what you are about to change>"
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:?usage: $0 <short_name> \"<description>\"}"
desc="${2:?usage: $0 <short_name> \"<description>\"}"
[[ "$name" =~ ^[a-z0-9_]+$ ]] || { echo "short_name: lowercase letters, digits, underscores only" >&2; exit 1; }
if [[ "$desc" == *'"'* || "$desc" == *'\'* || "$desc" == *'&'* || "$desc" == *'|'* ]]; then
  echo "description must not contain double quotes, backslashes, & or |" >&2; exit 1
fi
dir=kernels/avx2/gemm
last=$(ls "$dir"/gemm_s[0-9]*_*.c | sed -E 's|.*/gemm_s([0-9]+)_.*|\1|' | sort -n | tail -1)
next=$((last + 1))
prev_file=$(ls "$dir"/gemm_s"${last}"_*.c)
new_file="$dir/gemm_s${next}_${name}.c"
[[ -e "$new_file" ]] && { echo "$new_file already exists" >&2; exit 1; }
sed -e "s/gemm_s8_avx2_s${last}\b/gemm_s8_avx2_s${next}/g" "$prev_file" > "$new_file"
sed -i "1s|.*|/* Step ${next}: ${desc} (copied from step ${last}; change ONE thing) */|" "$new_file"
sed -i "/NEW PROTOTYPES GO ABOVE/i void gemm_s8_avx2_s${next}(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);" "$dir/gemm_steps.h"
sed -i "/NEW STEPS GO ABOVE/i \\    {\"s${next}_${name}\", \"${desc}\", gemm_s8_avx2_s${next}}," "$dir/gemm_steps.c"
echo "Created step ${next}: $new_file (copied from $prev_file)"
echo "Next: edit it (ONE change), then scripts/gemm_check.sh"
