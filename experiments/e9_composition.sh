#!/usr/bin/env bash
# E9 합성 파일럿: 단일 부품 3개와 그 합성 4개가 어느 train 크기부터 학습되는가, 모델을 키우면 달라지는가.
#   규칙 7개 × train 1,024~16,384 (5개) × seed 3 × 모델 2개(기본 d128·2층, 큰 d256·4층) = 210 run, 최대 50,000 step
#   train이 커도 step당 계산량이 같도록 batch 2,048 복원추출 (다른 실험의 full batch와 학습 방식이 다름에 주의)
#   sweep 두 개(기본·큰 모델)가 동시에 돈다 → GPU 하나당 묶음 2×JOBS개
# 여러 GPU: GPUS=all JOBS=3 bash experiments/e9_composition.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-50000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
DATA=""
for c in 1024 2048 4096 8192 16384; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c
  DATA="$DATA data/c$c"
done
RULES="p_sort p_next p_first comp_sort_next comp_next_next comp_first_sort comp_next_sort"
TRAIN="--lr 3e-4 --steps $STEPS --stop-after-gen 2000 --eval-every 200 --batch-size 2048"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sweep.py --data $DATA --rules $RULES --models m0 --seeds 0 1 2 --runs-dir runs_e9_small \
  --multi --group-size 21 --jobs $JOBS $EXTRA -- $TRAIN &
python sweep.py --data $DATA --rules $RULES --models m0 --seeds 0 1 2 --runs-dir runs_e9_big \
  --multi --group-size 21 --jobs $JOBS $EXTRA -- $TRAIN --d 256 --layers 4 --heads 8 &
wait
for o in runs_e9_small runs_e9_big; do python analyze.py "$o/*" --out $o/results.csv --group-out $o/results_grouped.csv; done
tar czf e9_results.tgz --exclude='*.pt' runs_e9_small runs_e9_big
echo "끝: e9_results.tgz"
