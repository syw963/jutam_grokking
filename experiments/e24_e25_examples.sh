#!/usr/bin/env bash
# E24·E25: "일반화에 필요한 예제 수" 가설을 새 조건에서 사전 예측으로 시험. 5090 한 장 기준 약 1.5시간.
#
# 지금까지(E13·E16·E21·E22) 보면 n=50에서 필요한 예제 수가 수열 길이 m=2~6과 상관없이 약 650~750개로 모이고,
# 표 크기 n에 따라 대략 N0 ≈ 2·n^1.5 (n31 → 345, n41 → 525, n50 → 707, n61 → 953, n71 → 1,197)로 커진다.
# 경쟁 가설은 "칸당 관찰 수 약 2.3회"(m=5): N ≈ 0.23·n(n+1) (n31 → 228, n50 → 586, n71 → 1,176).
# 두 가설은 n=23에서 크게 갈린다(127 vs 221).
#
# E24 표 크기 (m=5, w5_next, a+b) × seed 10
#   n=23:  N 130 / 160 / 190 / 220 / 260   (관찰 수 2.4 / 2.9 / 3.4 / 4.0 / 4.7)
#   n=101: N 1,900 / 2,150 / 2,400 / 2,700  (관찰 수 1.8 / 2.1 / 2.3 / 2.6)
#   사전 예측(예제 수 가설 + 관찰 수 하한, N0 = max(2·n^1.5, 0.23·n(n+1))):
#     n=23은 220개 근처에서 50% (130·160은 거의 실패). 관찰 수만의 가설이면 130개부터 대부분 성공.
#     n=101은 관찰 수 항(0.23·n(n+1) = 2,370)이 더 커서 약 2,370개 근처에서 50% (예제 수 항은 2,030).
#     (10월 23일 수정: 예전 주석 "두 가설 모두 약 2,370"은 틀림)
# E25 수열 길이별 필요한 예제 수 (n=50, p_next) × seed 10
#   m=4: N 450 / 550 / 650 (관찰 수 1.4 / 1.7 / 2.0),  m=6: N 800 / 900 / 1,050 (관찰 수 3.8 / 4.2 / 4.9)
#   사전 예측(예제 수 가설): m=4는 450·550 실패, 650 근처 50% / m=6은 800 이상 대부분 성공(E22 723개 7/10).
#   = 90 + 60 = 150 run, 최대 60,000 step
# 실행: GPUS=all JOBS=2 bash experiments/e24_e25_examples.sh   (한 서버에서 전부, 묶음 8개 동시)
#   서버 두 대로 나누기: ONLY=e24 ... / ONLY=e25 ...
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}; ONLY=${ONLY:-all}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mk() { [ -d data/$1 ] || python -m src.data --n $2 --m $3 --train-count $4 --test-count 20000 --seed 0 --out data/$1 > /dev/null; }
for c in 130 160 190 220 260; do mk n23_c$c 23 5 $c; done
for c in 1900 2150 2400 2700; do mk n101_c$c 101 5 $c; done
for c in 450 550 650; do mk m4_c$c 50 4 $c; done
for c in 800 900 1050; do mk m6_c$c 50 6 $c; done
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 100"
S10="0 1 2 3 4 5 6 7 8 9"
run() { local out=$1 rule=$2; shift 2; local d=""; for c in "$@"; do d="$d data/$c"; done
  python sweep.py --data $d --rules $rule --models m0 --seeds $S10 --runs-dir $out \
    --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN > $out.sweep.log 2>&1; }
if [ "$ONLY" != e25 ]; then
  run runs_e24_n23  w5_next n23_c130 n23_c160 n23_c190 n23_c220 n23_c260 &
  run runs_e24_n101 w5_next n101_c1900 n101_c2150 n101_c2400 n101_c2700 &
fi
if [ "$ONLY" != e24 ]; then
  run runs_e25_m4   p_next  m4_c450 m4_c550 m4_c650 &
  run runs_e25_m6   p_next  m6_c800 m6_c900 m6_c1050 &
fi
wait
grep -h "실패" runs_e24_*.sweep.log runs_e25_*.sweep.log || true
python analyze.py "runs_e2[45]_*/*" --out e24_summary.csv --group-out e24_grouped.csv || true
tar czf e24_e25_${ONLY}_results.tgz --exclude='*.pt' $(ls -d runs_e24_* runs_e25_* 2>/dev/null) e24_summary.csv e24_grouped.csv
echo "끝: e24_e25_${ONLY}_results.tgz"
