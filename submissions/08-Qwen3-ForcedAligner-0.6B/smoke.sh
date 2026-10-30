#!/usr/bin/env bash
# 强制对齐 smoke：跑 data/ 下的中英两条样例，结果写入 evidence/
set -o pipefail
cd "$(dirname "$0")"
source ./env.sh
H=/root/hygon
source /opt/dtk/env.sh
export ROCM_PATH="$H/rootfs/opt/dtk"
export DTKROOT="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH/hip"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$ROCM_PATH/aillvm/bin:$ROCM_PATH/hip/bin:/opt/hyhal/bin:$PATH"
export PYTHONPATH="$H/rootfs/usr/local/lib/python3.10/dist-packages:$H/rootfs/usr/lib/python3/dist-packages"

mkdir -p evidence
echo "== 英文样例 (data/en.mp3)"
OUT_JSON=evidence/en.json python3 fa_align.py data/en.mp3 \
  "The tribal chief then called for the boy and presented him with fifty pieces of gold." English

echo "== 中文样例 (data/zh.mp3，逐字切分)"
OUT_JSON=evidence/zh.json python3 fa_align.py data/zh.mp3 \
  "開 放 時 間 早 上 九 點 至 下 午 五 點" Chinese

echo "== 完成，结果在 evidence/*.json"
