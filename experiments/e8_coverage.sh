#!/usr/bin/env bash
# E8 coverage 직접 조작: 같은 규칙, 같은 train 개수에서 train 값 집합만 골라 연산 표 coverage를 바꾼다.
#   규칙 2개(w5_next: 쌍 5, w1_m0: 쌍 4) × split 6개(train 768·1,024 × 무작위/높게/낮게) × seed 10 = 120 run
#   핵심 비교: (1) N이 같고 coverage만 다른 split, (2) coverage가 같고 N이 다른 split
#   (예: w5_next 1,024개 coverage 0.923 vs 768개 무작위 0.923)
# 여러 GPU: GPUS=all JOBS=3 bash experiments/e8_coverage.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-40000}; OUT=runs_e8
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
mk() { [ -d data/$2 ] || python -m src.curate --rule $1 --train-count $3 ${4:+--target $4} --out data/$2; }
# w5_next: 무작위 coverage 768개 0.923, 1,024개 0.951
mk w5_next e8_next_n768_rand 768;        mk w5_next e8_next_n768_hi 768 0.944;   mk w5_next e8_next_n768_lo 768 0.88
mk w5_next e8_next_n1024_rand 1024;      mk w5_next e8_next_n1024_lo 1024 0.923; mk w5_next e8_next_n1024_lo2 1024 0.90
# w1_m0: 무작위 coverage 768개 0.884, 1,024개 0.929
mk w1_m0 e8_m0_n768_rand 768;            mk w1_m0 e8_m0_n768_hi 768 0.90;        mk w1_m0 e8_m0_n768_lo 768 0.85
mk w1_m0 e8_m0_n1024_rand 1024;          mk w1_m0 e8_m0_n1024_lo 1024 0.884;     mk w1_m0 e8_m0_n1024_lo2 1024 0.86
COMMON="--models m0 --seeds 0 1 2 3 4 5 6 7 8 9 --runs-dir $OUT --multi --group-size 20 --jobs $JOBS $EXTRA"
TRAIN="--lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 100"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sweep.py --data data/e8_next_* --rules w5_next $COMMON -- $TRAIN
python sweep.py --data data/e8_m0_* --rules w1_m0 $COMMON -- $TRAIN
python analyze.py "$OUT/*" --out $OUT/results.csv --group-out $OUT/results_grouped.csv
tar czf e8_results.tgz --exclude='*.pt' $OUT data/e8_*/meta.json
echo "끝: e8_results.tgz"
