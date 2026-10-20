#!/usr/bin/env bash
# E26·E27 (RQ2 합성 규칙) — RQ1에서 얻은 "필요한 예제 수 법칙"을 합성 규칙에 적용해 사전 예측한다.
#   RQ1 법칙(E6~E25, 9개 조건에 ±10% 안): 일반화 50% 경계 N0 ≈ 2.81 × C^0.764,  C = 배워야 할 연산 표의 칸 수
#   (a+b: n(n+1)/2, a+3b: n²). 대안: 칸당 관찰 수 2.3회 하한 N ≈ 2.3 × C / 5.
#
# E26 덧셈 두 번(comp_next_next = x_i + 2x_{i+1} + x_{i+2}, 세 값 함수, C = n × n(n+1)/2). 작은 n이라 full batch로 최적화 문제 없음.
#   n=19 (C 3,610):  N 900 / 1,300 / 1,800 / 2,400     예측 N0: 법칙 1,468 / 관찰 수 하한 1,661
#   n=23 (C 6,348):  N 1,600 / 2,200 / 3,000 / 4,000   예측 N0: 법칙 2,259 / 하한 2,920
#   n=31 (C 15,376): N 3,000 / 4,500 / 6,500 / 9,000   예측 N0: 법칙 4,441 / 하한 7,073
#   seed 5 = 60 run, 최대 100,000 step. "합성 비용" 가설이면 두 예측보다 훨씬(2배 이상) 뒤.
# E27 정렬이 들어간 합성의 비용 (n=50): 부품 p_sort·p_first와 합성 comp_sort_next·comp_first_sort를 같은 N에서
#   N 768 / 1,024 / 1,280 / 1,536 / 2,048 × seed 5 = 4 × 5 × 5 = 100 run, full batch, 최대 60,000 step
#   사전 예측: 부품 경계 약 800~1,000(E5·E9), 합성 경계는 약 1,500(부품의 1.5~2배, E9에서 1,024 실패·2,048 성공).
#   대안 "합성 = 더 어려운 부품": 합성 경계도 약 1,000.
# 5090 한 장에 ONLY=e26 / ONLY=e27로 나눠 돌리면 각 1.5~2시간. 한 대면 ONLY 없이.
# 실행: GPUS=all JOBS=2 ONLY=e26 bash experiments/e26_e27_rq2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}; ONLY=${ONLY:-all}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mk() { [ -d data/$1 ] || python -m src.data --n $2 --train-count $3 --test-count $4 --seed 0 --out data/$1 > /dev/null; }
run() { local out=$1 rules=$2 seeds=$3 steps=$4; shift 4; local d=""; for c in "$@"; do d="$d data/$c"; done
  python sweep.py --data $d --rules $rules --models m0 --seeds $seeds --runs-dir $out \
    --multi --group-size 10 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $steps --stop-after-gen 1000 --eval-every 200 > $out.sweep.log 2>&1; }
S5="0 1 2 3 4"
if [ "$ONLY" != e27 ]; then
  for c in 900 1300 1800 2400; do mk n19_c$c 19 $c 8000; done
  for c in 1600 2200 3000 4000; do mk n23_c$c 23 $c 20000; done
  for c in 3000 4500 6500 9000; do mk n31_c$c 31 $c 20000; done
  run runs_e26_n19 comp_next_next "$S5" 100000 n19_c900 n19_c1300 n19_c1800 n19_c2400 &
  run runs_e26_n23 comp_next_next "$S5" 100000 n23_c1600 n23_c2200 n23_c3000 n23_c4000 &
  run runs_e26_n31 comp_next_next "$S5" 100000 n31_c3000 n31_c4500 n31_c6500 n31_c9000 &
fi
if [ "$ONLY" != e26 ]; then
  for c in 768 1024 1280 1536 2048; do mk c$c 50 $c 20000; done
  run runs_e27_parts "p_sort p_first" "$S5" 60000 c768 c1024 c1280 c1536 c2048 &
  run runs_e27_comp  "comp_sort_next comp_first_sort" "$S5" 60000 c768 c1024 c1280 c1536 c2048 &
fi
wait
grep -h "실패" runs_e2[67]_*.sweep.log || true
python analyze.py "runs_e2[67]_*/*" --out e26_summary.csv --group-out e26_grouped.csv || true
tar czf e26_e27_${ONLY}_results.tgz --exclude='*.pt' $(ls -d runs_e26_* runs_e27_* 2>/dev/null) e26_summary.csv e26_grouped.csv
echo "끝: e26_e27_${ONLY}_results.tgz"
