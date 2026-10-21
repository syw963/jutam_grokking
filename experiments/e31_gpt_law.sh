#!/usr/bin/env bash
# E31 GPT의 필요한 예제 수 법칙을 촘촘히: 지수가 정말 m0(0.764)보다 작은가 (E28은 seed 3, n 두 개라 ±0.1 불확실)
#   규칙 w5_next (a+b), 모델 gpt, n = 31, 41, 50, 61, 71 (C = 496, 861, 1,275, 1,891, 2,556)
#   E28 적합(N0 ≈ 19.3 × C^0.651)의 예측 경계 × 0.65 / 0.8 / 1.0 / 1.25 / 1.55 의 N × seed 6 = 150 run, 최대 60,000 step
#     n31: 715 880 1100 1375 1705      (예측 1,100)
#     n41: 1020 1260 1570 1960 2440    (예측 1,570)
#     n50: 1320 1620 2030 2540 3150    (예측 2,030, E10b 약 1,800)
#     n61: 1700 2100 2620 3280 4060    (예측 2,620)
#     n71: 2080 2560 3200 4000 4960    (예측 3,200)
# 사전 예측 (10월 21일 기록): 다섯 경계의 적합 지수 0.60~0.70 (95% 구간이 0.764를 포함하지 않음).
#   반대: 구간이 0.764를 포함하면 "GPT도 같은 지수, 상수만 다름"으로 고친다.
# 실행: GPUS=all JOBS=4 bash experiments/e31_gpt_law.sh      (서버 여러 대면 SHARD=0/2 ...)
#   5090 한 장 약 6~8시간, 네 장이면 약 1.5~2시간 (E28 속도 기준 예상)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A GRID=( [31]="715 880 1100 1375 1705" [41]="1020 1260 1570 1960 2440" [50]="1320 1620 2030 2540 3150"
                  [61]="1700 2100 2620 3280 4060" [71]="2080 2560 3200 4000 4960" )
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 200 --sym-every 0"
for n in 31 41 50 61 71; do
  D=""
  for c in ${GRID[$n]}; do
    [ -d data/n${n}_c$c ] || python -m src.data --n $n --train-count $c --test-count 20000 --seed 0 --out data/n${n}_c$c > /dev/null
    D="$D data/n${n}_c$c"
  done
  python sweep.py --data $D --rules w5_next --models gpt --seeds 0 1 2 3 4 5 --runs-dir runs_e31_n$n \
    --multi --group-size 6 --jobs $JOBS $EXTRA -- $TRAIN > runs_e31_n$n.sweep.log 2>&1 &
done
wait
grep -h "실패" runs_e31_*.sweep.log || true
python analyze_law.py "runs_e31_*" --out e31_report || true
T=e31_results${SHARD:+_$(echo $SHARD | tr / -)}.tgz
tar czf $T --exclude='*.pt' runs_e31_* e31_report
echo "끝: $T"
