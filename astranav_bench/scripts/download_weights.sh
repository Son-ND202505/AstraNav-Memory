#!/usr/bin/env bash
# Downloads (run from this repo's root):
#   astranav_bench/weights/ds16_context100   the original AstraNav-Memory weights (ModelScope, ~7.4 GB)
#   astranav_bench/weights/astranav_g1_ft.pt the fine-tune for the MuJoCo houses (LoRA + compressor + merger)
set -euo pipefail
W=astranav_bench/weights
mkdir -p "$W"
if [ ! -f "$W/ds16_context100/config.json" ]; then
  modelscope download --model chongchongjj/AstraNav_Memory_ds16_context100 --local_dir "$W/ds16_context100"
fi
URL="${URL:-https://github.com/Son-ND202505/AstraNav-Memory/releases/download/v1.0/astranav_g1_ft.pt}"
SHA=716f32ef412310601c0815a6dc6f89febbe9d9ecb489d28a5cc327a49fc98a92
[ -f "$W/astranav_g1_ft.pt" ] || curl -L --fail -o "$W/astranav_g1_ft.pt" "$URL"
echo "$SHA  $W/astranav_g1_ft.pt" | sha256sum -c -
