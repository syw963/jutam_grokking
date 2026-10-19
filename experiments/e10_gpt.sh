#!/usr/bin/env bash
# E10 GPT로 E6·E7 일부 반복: 모델 구조가 달라도 "coverage가 정한다"가 유지되는가.
#   규칙 8개(a+b 4개, a+3b 4개) × train 896~1,536 (5개) × seed 5 = 200 run, 모델 gpt
# 여러 GPU: GPUS=all JOBS=3 bash experiments/e10_gpt.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-40000}; OUT=runs_e10
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
DATA=""
for c in 896 1024 1152 1280 1536; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c
  DATA="$DATA data/c$c"
done
RULES="w5_next w6_p1 w2_m12 w1_m0 w5_next_n3 w6_p1_n3 w2_m12_n3 w1_m0_n3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sweep.py --data $DATA --rules $RULES --models gpt --seeds 0 1 2 3 4 --runs-dir $OUT \
  --multi --group-size 20 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 100
python analyze_boundary.py $OUT --out e10_report || true
tar czf e10_results.tgz --exclude='*.pt' $OUT e10_report
echo "끝: e10_results.tgz"
