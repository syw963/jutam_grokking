#!/usr/bin/env bash
# E20 (E19 보강) 덧셈 두 번(comp_next_next), n=50을 오래 학습: E19는 6만 step 안에 학습이 끝나지 않은 run이 많았다.
#   작은 모델(d128·2층) × N 16,384 / 32,768 / 65,536 / 131,072 (칸당 관찰 수 1.3 / 2.6 / 5.1 / 10.3) × seed 3 = 12 run
#   batch 2,048 복원추출, 최대 200,000 step, 일반화 후 2,000 step 더 보고 멈춤
#   사전 예측: 1.3회는 20만 step에도 실패(step 부족이 아니라 관찰 수 부족), 2.6회 이상은 대부분 일반화.
#   4090 한 장 기준 약 2시간 (일반화한 묶음은 일찍 끝남)
# 실행: GPUS=all JOBS=4 bash experiments/e20_comp_long.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-200000}; OUT=runs_e20
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
D=""; for c in 16384 32768 65536 131072; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
  D="$D data/c$c"; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sweep.py --data $D --rules comp_next_next --models m0 --seeds 0 1 2 --runs-dir $OUT \
  --multi --group-size 3 --jobs $JOBS $EXTRA \
  -- --lr 3e-4 --steps $STEPS --stop-after-gen 2000 --eval-every 500 --batch-size 2048
python analyze.py "$OUT/*" --out e20_summary.csv --group-out e20_grouped.csv || true
tar czf e20_results.tgz --exclude='*.pt' $OUT e20_summary.csv e20_grouped.csv
echo "끝: e20_results.tgz"
