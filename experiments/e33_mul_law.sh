#!/usr/bin/env bash
# E33 곱셈 규칙도 같은 법칙인가: 필요한 예제 수는 연산 종류와 무관하게 표 크기 C로만 정해지는가
#   규칙 w5_next_mul: y_i = (x_i × x_{i+1}) mod n + 1 (교환 연산이라 C = n(n+1)/2, 덧셈과 같은 칸 수)
#   모델 m0 (기본), n = 31, 41, 61, 71 (소수: 곱셈 표가 순환군 구조) + 50 (합성수: 0으로 나누어지는 값이 있어 구조가 다름)
#   N = 덧셈 법칙 경계(2.81 × C^0.764)의 0.55 / 0.75 / 1.0 / 1.35 / 1.8배
#     n31: 175 240 320 435 580        n41: 270 370 490 665 885        n50: 365 495 665 895 1195
#     n61: 495 670 895 1210 1610      n71: 620 845 1125 1520 2030
#   × seed 6 = 150 run, 최대 60,000 step. 비교 기준(덧셈, 같은 n)은 E32의 기본 모델 run.
# 사전 예측 (10월 22일 2시 기록)
#   P1 소수 n(31·41·61·71)에서 곱셈의 경계는 덧셈 법칙의 ±25% 안 (연산이 달라도 표 크기로 정해진다).
#   P2 소수 n 네 점의 적합 지수 구간이 0.764를 포함.
#   P3 (탐색적) n=50 합성수는 소수 n보다 어렵다(법칙의 1.3배 이상): 곱셈 표 값이 고르게 퍼지지 않음.
#   반대 결과(곱셈이 일관되게 더 어렵거나 쉬움)면 "상수는 연산마다 다르고 지수만 같다" 또는 "법칙은 덧셈에 한정"으로 고친다.
# 실행: GPUS=all JOBS=10 bash experiments/e33_mul_law.sh     (서버 여러 대면 SHARD=0/2 ...)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A GRID=( [31]="175 240 320 435 580" [41]="270 370 490 665 885" [50]="365 495 665 895 1195"
                  [61]="495 670 895 1210 1610" [71]="620 845 1125 1520 2030" )
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 200 --sym-every 0"
for n in 31 41 50 61 71; do
  D=""
  for c in ${GRID[$n]}; do
    [ -d data/n${n}_c$c ] || python -m src.data --n $n --train-count $c --test-count 20000 --seed 0 --out data/n${n}_c$c > /dev/null
    D="$D data/n${n}_c$c"
  done
  python sweep.py --data $D --rules w5_next_mul --models m0 --seeds 0 1 2 3 4 5 --runs-dir runs_e33_n$n \
    --multi --group-size 6 --jobs $JOBS $EXTRA -- $TRAIN > runs_e33_n$n.sweep.log 2>&1 &
done
wait
grep -h "실패" runs_e33_*.sweep.log || true
python analyze_law.py "runs_e33_*" --out e33_report || true
T=e33_results${SHARD:+_$(echo $SHARD | tr / -)}.tgz
tar czf $T --exclude='*.pt' runs_e33_* e33_report
echo "끝: $T"
