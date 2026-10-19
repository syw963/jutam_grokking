#!/usr/bin/env bash
# 파일럿 실험: 연구 질문·방법 확정용
#   P1  train 크기 3개 × 규칙 8개 (m0)       → grokking이 보이는 train 크기, 규칙별 지연
#   P2  사슬 규칙 3개 (1%, m0)               → RQ2 실현 가능성
#   P3  같은 조건 seed 하나 더 (1%, m0)       → 반복성
#   P4  모델 대칭 일치 (minv / meq, 1%)       → RQ1-3
#   P5  GPT (1%)                              → 이전 RJ 구조와 비교
# 사용: bash experiments/pilot.sh        (JOBS=동시 실행 수, STEPS=최대 step, DRY=1이면 명령만 출력)
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-4}
STEPS=${STEPS:-100000}
OUT=runs_pilot
DRYFLAG=""
[ "${DRY:-0}" = "1" ] && DRYFLAG="--dry-run"
TRAIN_ARGS="--steps $STEPS --stop-after-gen 1000 --eval-every 200"

for f in 0.001 0.003 0.01; do
  [ -d data/pilot_f$f ] || python -m src.data --n 50 --m 5 --train-frac $f --test-count 20000 --seed 0 --out data/pilot_f$f
done
D1=data/pilot_f0.001; D3=data/pilot_f0.003; D10=data/pilot_f0.01
RULES="sort l1_sortnext l1_sum l1_first l1_next l1_map c_sum_sort_nb ctrl_add_mul_sort"

run() { python sweep.py --runs-dir $OUT --jobs $JOBS $DRYFLAG "$@"; }

echo "== P1: train 크기 × 규칙 (m0, seed 0)"
run --data $D1 $D3 $D10 --rules $RULES --models m0 --seeds 0 -- $TRAIN_ARGS
echo "== P2: RQ2 사슬 (1%, m0, seed 0)"
run --data $D10 --rules c_sort_nb_sort c_nb_sort_pos c_pos_sort_sum --models m0 --seeds 0 -- $TRAIN_ARGS
echo "== P3: 반복성 (1%, m0, seed 1)"
run --data $D10 --rules $RULES --models m0 --seeds 1 -- $TRAIN_ARGS
echo "== P4: 모델 대칭 일치 (1%)"
run --data $D10 --rules sort l1_sortnext --models minv --seeds 0 1 -- $TRAIN_ARGS
run --data $D10 --rules l1_sum add_total --models meq --seeds 0 1 -- $TRAIN_ARGS
run --data $D10 --rules add_total --models m0 --seeds 0 1 -- $TRAIN_ARGS
echo "== P5: GPT (1%, seed 0)"
run --data $D10 --rules $RULES --models gpt --seeds 0 -- $TRAIN_ARGS

[ -n "$DRYFLAG" ] && exit 0

echo "== 결과 정리"
python analyze.py "$OUT/*" --out $OUT/results.csv --group-out $OUT/results_grouped.csv
python analyze.py "$OUT/*" --gen-thr 0.90 --out $OUT/results_gen90.csv --group-out $OUT/results_grouped_gen90.csv
mkdir -p $OUT/plots
for r in $RULES add_total; do
  python plot.py "$OUT/${r}_*" --out $OUT/plots/$r.png || true
done
python plot.py "$OUT/c_*" --out $OUT/plots/chains.png || true
tar czf pilot_results.tgz --exclude='*.pt' $OUT
echo "끝: pilot_results.tgz 를 채팅에 첨부해 주세요"
