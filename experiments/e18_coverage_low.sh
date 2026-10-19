#!/usr/bin/env bash
# E18 coverage를 만들 수 있는 한 낮추기 (E8·E12B 연장). train 768개, seed 10, 최대 60,000 step. 5090 한 장 기준 약 1~1.5시간.
#   w5_next: 0.82 / 0.80 / 0.78  (E8·E12B: 0.944 → 6,900 … 0.842 → 2,950. 0.77 아래는 만들 수 없음)
#   w1_m0:   0.79 / 0.77 / 0.74  (E8·E12B: 0.898 → 21,000 … 0.811 → 5,350)
#   = 60 run. split 만들기에 1~4분씩 걸린다(동시에 만든다).
#   사전 예측: w5_next는 이미 평평해지는 중이라 0.78에서도 약 2,500~3,000 근처(더 빨라지지 않거나 오히려 느려짐 = 최적점).
#     w1_m0은 0.74까지 계속 빨라진다(약 4,000). 안 나온 칸이 많아질수록(0.74면 test 칸의 약 25%) 언젠가 반대로 느려질 것.
# 실행: GPUS=all JOBS=3 bash experiments/e18_coverage_low.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-3}; OUT=runs_e18
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
mk() { [ -d data/$2 ] || python -m src.curate --rule $1 --train-count 768 --target $3 --out data/$2; }
mk w5_next e18_next_c82 0.82 & mk w5_next e18_next_c80 0.80 & mk w5_next e18_next_c78 0.78 &
mk w1_m0 e18_m0_c79 0.79 & mk w1_m0 e18_m0_c77 0.77 & mk w1_m0 e18_m0_c74 0.74 &
wait
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 100"
S10="0 1 2 3 4 5 6 7 8 9"
python sweep.py --data data/e18_next_* --rules w5_next --models m0 --seeds $S10 --runs-dir $OUT \
  --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN > runs_e18_next.sweep.log 2>&1 &
python sweep.py --data data/e18_m0_* --rules w1_m0 --models m0 --seeds $S10 --runs-dir $OUT \
  --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN > runs_e18_m0.sweep.log 2>&1 &
wait
grep -h "실패" runs_e18_*.sweep.log || true
python analyze.py "$OUT/*" --out e18_summary.csv --group-out e18_grouped.csv || true
tar czf e18_results.tgz --exclude='*.pt' $OUT e18_summary.csv e18_grouped.csv data/e18_*/meta.json
echo "끝: e18_results.tgz"
