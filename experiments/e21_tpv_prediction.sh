#!/usr/bin/env bash
# E21 "정답 값당 등장 횟수" 사전 예측 검증 (E15 이후 결과를 보고 세운 설명을 새 조건에서 시험)
#   E6·E7·E12·E13·E15로 적합한 로지스틱: logit = −40.91 + 4.23 × 칸당 관찰 수 + 16.29 × log10(정답 값당 등장 횟수)
#   처음 보는 표 크기 n=41, 61에서, 같은 관찰 수인데 정답 값당 횟수가 다른 세 조건을 비교한다.
#     n41 a+b  (861칸, 값당 = 관찰 수 × 21)  N = 379 / 448 / 517
#     n61 a+b  (1,891칸, 값당 = 관찰 수 × 31) N = 832 / 983 / 1,135
#     n41 a+3b (1,681칸, 값당 = 관찰 수 × 41) N = 740 / 874 / 1,009
#   관찰 수 2.2 / 2.6 / 3.0 × seed 10 = 90 run, 최대 60,000 step
#   사전 예측(일반화 비율): n41 a+b 0.01 / 0.17 / 0.75,  n61 a+b 0.15 / 0.76 / 0.98,  n41 a+3b 0.56 / 0.96 / 1.00
#   핵심: 관찰 수 2.6에서 세 조건이 0.17 / 0.76 / 0.96으로 갈린다. 관찰 수만 중요하다면 셋이 같아야 한다.
#   4090 한 장 기준 약 1~1.5시간
# 실행: GPUS=all JOBS=2 bash experiments/e21_tpv_prediction.sh  (sweep 3개 × 2 = 묶음 6개 동시, 24GB 안)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}; STEPS=${STEPS:-60000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
TRAIN="--lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 100"
mkd() { [ -d data/n$1_c$2 ] || python -m src.data --n $1 --train-count $2 --test-count 20000 --seed 0 --out data/n$1_c$2 > /dev/null; }
for c in 379 448 517 740 874 1009; do mkd 41 $c; done
for c in 832 983 1135; do mkd 61 $c; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S10="0 1 2 3 4 5 6 7 8 9"
run() { local out=$1 rule=$2; shift 2; local d=""; for c in "$@"; do d="$d data/$c"; done
  python sweep.py --data $d --rules $rule --models m0 --seeds $S10 --runs-dir $out \
    --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN > $out.sweep.log 2>&1; }
run runs_e21_n41_ab w5_next    n41_c379 n41_c448 n41_c517 &
run runs_e21_n61_ab w5_next    n61_c832 n61_c983 n61_c1135 &
run runs_e21_n41_n3 w5_next_n3 n41_c740 n41_c874 n41_c1009 &
wait
grep -h "실패" runs_e21_*.sweep.log || true
python analyze_obs.py "runs_e21_*" --out e21_report || true
tar czf e21_results.tgz --exclude='*.pt' runs_e21_* e21_report
echo "끝: e21_results.tgz"
