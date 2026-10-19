#!/usr/bin/env bash
# E16 고전 grokking 과제와 비교 + E17 무작위 연산표. 5090 한 장 기준 약 1.5~2시간.
#
# E16: 입력 길이 m을 줄인 같은 규칙(p_next: y_i = x_i + x_{i+1}, 순환). m=2는 기존 grokking 연구의 "a+b" 과제와 같다
#   (예제 하나가 표 칸 하나, 같은 칸을 두 번 볼 수 없다 → 칸당 관찰 수 ≤ 1).
#   m=2: N 400 / 600 / 800 / 1,000 (전체 1,225쌍 중, 나머지가 test), 최대 100,000 step
#   m=3: N 425 / 638 / 935 / 1,275 (관찰 수 1.0 / 1.5 / 2.2 / 3.0), 최대 60,000 step
#   seed 5 = 40 run
#   사전 예측(우리 법칙): m=2는 관찰 수 ≤ 0.8이라 모두 실패. m=3은 3.0(정답 값당 76회)에서만 일부 일반화.
#   반대 가설(고전 grokking): m=2도 train 비율이 높으면(65~80%) 오래 학습하면 일반화한다. 그렇다면 우리 법칙은 "수열 과제" 한정.
#
# E17: 무작위 연산표(op=table, 규칙 없는 50×50 표) vs a+3b(같은 2,500칸, 같은 정답 값당 횟수). 둘 다 5-순환 배선.
#   N 1,100 / 1,500 / 2,000 (관찰 수 2.2 / 3.0 / 4.0) × seed 3 = 18 run, 최대 30,000 step, 500 step마다 체크포인트
#   사전 예측: 무작위 표는 "train에 나온 칸" 정확도만 오르고(③) "안 나온 칸"은 우연(0.02)에 머문다(④ 없음).
#     ③의 시점(나온 칸 50%)은 a+3b와 비슷하다(③은 연산 구조가 필요 없으므로).
# 실행: GPUS=all JOBS=3 bash experiments/e16_e17.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-3}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in 400 600 800 1000; do
  [ -d data/m2_c$c ] || python -m src.data --m 2 --train-count $c --test-count $((1225 - c)) --seed 0 --out data/m2_c$c > /dev/null; done
for c in 425 638 935 1275; do
  [ -d data/m3_c$c ] || python -m src.data --m 3 --train-count $c --test-count 10000 --seed 0 --out data/m3_c$c > /dev/null; done
for c in 1100 1500 2000; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null; done
S5="0 1 2 3 4"
python sweep.py --data data/m2_c400 data/m2_c600 data/m2_c800 data/m2_c1000 --rules p_next --models m0 --seeds $S5 \
  --runs-dir runs_e16_m2 --multi --group-size 5 --jobs $JOBS $EXTRA \
  -- --lr 3e-4 --steps 100000 --stop-after-gen 1000 --eval-every 100 > runs_e16_m2.sweep.log 2>&1 &
python sweep.py --data data/m3_c425 data/m3_c638 data/m3_c935 data/m3_c1275 --rules p_next --models m0 --seeds $S5 \
  --runs-dir runs_e16_m3 --multi --group-size 5 --jobs $JOBS $EXTRA \
  -- --lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 100 > runs_e16_m3.sweep.log 2>&1 &
python sweep.py --data data/c1100 data/c1500 data/c2000 --rules "REF:next(op=table,seed=0)" w5_next_n3 --models m0 --seeds 0 1 2 \
  --runs-dir runs_e17 --multi --group-size 6 --jobs $JOBS $EXTRA \
  -- --lr 3e-4 --steps 30000 --stop-after-gen 3000 --eval-every 100 --ckpt-log --ckpt-dense 500 > runs_e17.sweep.log 2>&1 &
wait
grep -h "실패" runs_e16_*.sweep.log runs_e17.sweep.log || true
python analyze.py "runs_e16_*/*" --out e16_summary.csv --group-out e16_grouped.csv || true
python analyze_internal.py "runs_e17/*" --out e17_internal || true
tar czf e16_e17_results.tgz --exclude='*.pt' runs_e16_* runs_e17 e16_summary.csv e16_grouped.csv e17_internal
echo "끝: e16_e17_results.tgz"
