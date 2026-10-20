#!/usr/bin/env bash
# E23 weight decay 계열: grokking 연구에서 핵심 손잡이인 weight decay를 0~3으로 바꿔도 "칸당 관찰 수" 순서가 유지되는가.
#   w5_next(a+b), w5_next_n3(a+3b) × 관찰 수 2.6 / 3.4 × weight decay 0 / 0.1 / 3.0 × seed 5 = 60 run, 최대 100,000 step
#     a+b N 663 / 867,  a+3b N 1,300 / 1,700   (기본 wd 1.0 결과는 E12, wd 0.3은 E14에 있음)
#   사전 예측: 지연은 대략 1 / weight decay에 비례(E14에서 wd 0.3이 약 3배 느렸음) → wd 3.0은 약 3배 빠르고 0.1은 약 10배 느림.
#     wd 0은 10만 step 안에 일반화하지 않는다(외우기에 머묾). 어떤 wd에서도 관찰 수 3.4가 2.6보다 빠르다.
#   5090 한 장 약 1.5~2시간 (wd 0·0.1 묶음이 끝까지 돈다)
# 실행: GPUS=all JOBS=4 bash experiments/e23_weight_decay.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in 663 867 1300 1700; do [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null; done
TRAIN="--lr 3e-4 --steps 100000 --stop-after-gen 1000 --eval-every 200"
python sweep.py --data data/c663 data/c867 --rules w5_next --models m0 --seeds 0 1 2 3 4 --wds 0 0.1 3.0 \
  --runs-dir runs_e23_ab --multi --group-size 15 --jobs $JOBS $EXTRA -- $TRAIN > runs_e23_ab.sweep.log 2>&1 &
python sweep.py --data data/c1300 data/c1700 --rules w5_next_n3 --models m0 --seeds 0 1 2 3 4 --wds 0 0.1 3.0 \
  --runs-dir runs_e23_n3 --multi --group-size 15 --jobs $JOBS $EXTRA -- $TRAIN > runs_e23_n3.sweep.log 2>&1 &
wait
grep -h "실패" runs_e23_*.sweep.log || true
python analyze.py "runs_e23_*/*" --out e23_summary.csv --group-out e23_grouped.csv || true
tar czf e23_results.tgz --exclude='*.pt' runs_e23_* e23_summary.csv e23_grouped.csv
echo "끝: e23_results.tgz"
