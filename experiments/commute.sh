#!/usr/bin/env bash
# E7 교환법칙 실험: 같은 배선에서 연산만 a+b → a+3b(교환법칙 없음)로 바꾼다.
#   E6에서 경계·지연은 대칭 크기가 아니라 "한 예제가 주는 서로 다른 무순서 쌍 수"(참조 그래프 간선 수)를 따랐다.
#   a+3b에서는 모든 배선이 순서쌍 5개를 주므로, 정보량 가설이 맞다면 배선 간 차이가 사라져야 한다.
#   대칭(등변군 크기)은 연산과 무관하게 같다.
#   규칙 8개(_n3) × train 768~1280 (5개) × seed 5 = 200 run. E6(a+b)와 같은 데이터·seed라 그대로 비교한다.
# GPU 여러 개: GPUS=all JOBS=2 GROUP=20 bash experiments/X  (GPU마다 묶음 JOBS개, 묶음당 run GROUP개)
#        서버 여러 대: SHARD=0/3 (첫 서버), SHARD=1/3, SHARD=2/3 처럼 각 서버에서 실행
# 사용: bash experiments/commute.sh     (JOBS=동시 묶음 수, STEPS=최대 step, DRY=1이면 명령만 출력)
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-5}
STEPS=${STEPS:-40000}
OUT=runs_commute
DRYFLAG=""
EXTRA=""
[ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
[ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
[ "${DRY:-0}" = "1" ] && DRYFLAG="--dry-run"
SIZES="768 896 1024 1152 1280"
RULES="w5_next_n3 w5_p2_n3 w6_p1_n3 w6_p4_n3 w2_m4_n3 w2_m12_n3 w2_m15_n3 w1_m0_n3"

DATA=""
for c in $SIZES; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c
  DATA="$DATA data/c$c"
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sweep.py \
  --data $DATA --rules $RULES --models m0 --seeds 0 1 2 3 4 \
  --runs-dir $OUT --multi --group-size ${GROUP:-40} --jobs $JOBS $DRYFLAG $EXTRA \
  -- --lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 100

[ -n "$DRYFLAG" ] && exit 0
python analyze_boundary.py $OUT --out commute_report || true
tar czf commute_results.tgz --exclude='*.pt' $OUT commute_report
echo "끝: commute_results.tgz"
