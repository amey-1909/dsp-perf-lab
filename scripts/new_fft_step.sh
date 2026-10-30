#!/usr/bin/env bash
# Create the next FFT optimization step as a copy of the previous one.
#
#   scripts/new_fft_step.sh <short_name> "<what you are about to change>"
#
# Example:
#   scripts/new_fft_step.sh vector_store "replace scalar store loop with vector stores"
#
# This copies the latest kernels/avx2/fft/fft_s<N>_*.c to fft_s<N+1>_<name>.c,
# renames the function, and registers it (prototype + registry line). Edit the
# new file, change ONE thing, then run scripts/fft_check.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

name="${1:?usage: $0 <short_name> \"<description>\"}"
desc="${2:?usage: $0 <short_name> \"<description>\"}"

if ! [[ "$name" =~ ^[a-z0-9_]+$ ]]; then
  echo "short_name must contain only lowercase letters, digits and underscores" >&2
  exit 1
fi
if [[ "$desc" == *'"'* || "$desc" == *'\'* || "$desc" == *'&'* || "$desc" == *'|'* ]]; then
  echo "description must not contain double quotes, backslashes, & or |" >&2
  exit 1
fi

dir=kernels/avx2/fft
last=$(ls "$dir"/fft_s[0-9]*_*.c | sed -E 's|.*/fft_s([0-9]+)_.*|\1|' | sort -n | tail -1)
next=$((last + 1))
prev_file=$(ls "$dir"/fft_s"${last}"_*.c)
new_file="$dir/fft_s${next}_${name}.c"

if [[ -e "$new_file" ]]; then
  echo "$new_file already exists" >&2
  exit 1
fi

sed -e "s/fft_f32_avx2_s${last}\b/fft_f32_avx2_s${next}/g" "$prev_file" > "$new_file"
sed -i "1s|.*|/* Step ${next}: ${desc} (copied from step ${last}; change ONE thing) */|" "$new_file"

sed -i "/NEW PROTOTYPES GO ABOVE/i void fft_f32_avx2_s${next}(const fft_plan *p, float *d);" "$dir/fft_steps.h"
sed -i "/NEW STEPS GO ABOVE/i \\    {\"s${next}_${name}\", \"${desc}\", fft_f32_avx2_s${next}}," "$dir/fft_steps.c"

echo "Created step ${next}: $new_file (copied from $prev_file)"
echo "Next: edit it (change ONE thing), then run scripts/fft_check.sh"
