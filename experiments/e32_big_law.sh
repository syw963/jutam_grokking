#!/usr/bin/env bash
# E32 큰 모델에서도 필요한 예제 수 법칙이 같은가 (모델 크기: d128·2층 → d256·4층·8헤드)
#   규칙 w5_next (a+b), n = 31, 41, 50, 61, 71. N = m0 법칙 경계(2.81 × C^0.764)의 0.4 / 0.55 / 0.75 / 1.0 / 1.35배
#     n31: 130 175 240 320 435        n41: 195 270 370 490 665        n50: 265 365 495 665 895
#     n61: 360 495 670 895 1210       n71: 450 620 845 1125 1520
#   모델 2개(기본 m0, 큰 m0) × 25 데이터 × seed 6 = 300 run, 최대 60,000 step
#   기본 모델은 같은 격자·같은 seed에서 비교 기준 (n41·n61은 법칙의 새 표본 밖 점이기도 하다)
# 사전 예측 (10월 22일 1시 기록)
#   P1 큰 모델의 지수는 m0와 같다(부트스트랩·격자 구간이 0.764를 포함). GPT(E31)처럼 상수만 바뀐다.
#   P2 큰 모델의 상수는 m0의 0.5~1.0배(경계가 더 앞). 근거: E14에서 큰 모델이 같은 데이터에서 약 2배 빨리 일반화.
#   P3 기본 모델의 경계는 n41·n61에서도 법칙의 ±15% 안.
# 실행: GPUS=all JOBS=4 bash experiments/e32_big_law.sh     (서버 여러 대면 SHARD=0/2 ...)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A GRID=( [31]="130 175 240 320 435" [41]="195 270 370 490 665" [50]="265 365 495 665 895"
                  [61]="360 495 670 895 1210" [71]="450 620 845 1125 1520" )
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 200 --sym-every 0"
for n in 31 41 50 61 71; do
  D=""
  for c in ${GRID[$n]}; do
    [ -d data/n${n}_c$c ] || python -m src.data --n $n --train-count $c --test-count 20000 --seed 0 --out data/n${n}_c$c > /dev/null
    D="$D data/n${n}_c$c"
  done
  python sweep.py --data $D --rules w5_next --models m0 --seeds 0 1 2 3 4 5 --runs-dir runs_e32_big_n$n \
    --multi --group-size 6 --jobs $JOBS $EXTRA -- $TRAIN --d 256 --layers 4 --heads 8 > runs_e32_big_n$n.sweep.log 2>&1 &
  python sweep.py --data $D --rules w5_next --models m0 --seeds 0 1 2 3 4 5 --runs-dir runs_e32_base_n$n \
    --multi --group-size 6 --jobs $JOBS $EXTRA -- $TRAIN > runs_e32_base_n$n.sweep.log 2>&1 &
done
wait
grep -h "실패" runs_e32_*.sweep.log || true
python analyze_law.py "runs_e32_*" --out e32_report || true
T=e32_results${SHARD:+_$(echo $SHARD | tr / -)}.tgz
tar czf $T --exclude='*.pt' runs_e32_* e32_report
echo "끝: $T"
