#!/usr/bin/env bash
# 경계 구간 정밀 측정: 배선 대칭 비교
#   규칙 13개 (등변군 크기 5·6·2·1인 배선 각 3개 + 첫 칸) × train 768~1280 (128 간격, 5개) × seed 5개 = 325 run
#   train 크기마다 한 묶음(65 run) → 묶음 5개를 동시에 실행
#   배선들의 연산 표 coverage는 거의 같아서(1,024개에서 85~87%) 대칭만 다른 비교가 된다. 첫 칸은 coverage가 낮은 대조군.
# 사용: bash experiments/boundary.sh     (JOBS=동시 묶음 수, STEPS=최대 step, DRY=1이면 명령만 출력)
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-5}
STEPS=${STEPS:-40000}
OUT=runs_boundary
DRYFLAG=""
[ "${DRY:-0}" = "1" ] && DRYFLAG="--dry-run"
SIZES="768 896 1024 1152 1280"
RULES="w5_next w5_p0 w5_p2 w6_p1 w6_p4 w6_p6 w2_m4 w2_m12 w2_m15 w1_m0 w1_m2 w1_m8 l1_first"

DATA=""
for c in $SIZES; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c
  DATA="$DATA data/c$c"
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sweep.py \
  --data $DATA --rules $RULES --models m0 --seeds 0 1 2 3 4 \
  --runs-dir $OUT --multi --group-size 65 --jobs $JOBS $DRYFLAG \
  -- --lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 100

[ -n "$DRYFLAG" ] && exit 0
python analyze.py "$OUT/*" --out $OUT/results.csv --group-out $OUT/results_grouped.csv
tar czf boundary_results.tgz --exclude='*.pt' $OUT
echo "끝: boundary_results.tgz"
