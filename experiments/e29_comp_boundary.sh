#!/usr/bin/env bash
# E29 (RQ2 보완) 덧셈 두 번(comp_next_next)의 실제 경계 재기 + 학습 방식(full batch vs 미니배치) 비교
#   E26(full batch): n=19 2,400개·n=23 4,000개까지 10만 step 동안 0/5 → 경계는 표 크기 법칙(1,468 / 2,259)보다 훨씬 뒤.
#   E19(미니배치 2,048): n=31에서 8,192~16,384개(법칙 4,441의 2~4배)에서 일반화.
#   n=19 (C 3,610):  N 3,600 / 5,000 / 7,200  (법칙의 2.5 / 3.4 / 4.9배)
#   n=23 (C 6,348):  N 6,000 / 9,000 / 12,000 (법칙의 2.7 / 4.0 / 5.3배)
#   × 학습 방식 {full batch, 미니배치 2,048 복원추출} × seed 3 = 36 run (묶음 12개, 묶음마다 seed 3개), 최대 100,000 step
#   사전 예측: 경계는 법칙의 약 3~4배(n=19 약 5,000, n=23 약 8,000). 미니배치가 full batch보다 경계가 앞이다(E19·E26 차이).
#   반대: 학습 방식 차이가 없다면 두 방식의 경계가 같다.
# 속도 우선: 묶음 12개가 GPU 12장에 하나씩 가도록 GPU가 많을수록 빠르다. GPU 8장 서버면 2~2.5시간, 12장 이상이면 약 1.5~2시간.
# 서버 여러 대: SHARD=0/2, SHARD=1/2 처럼 나누면 묶음을 나눠 맡는다.
# 실행: GPUS=all JOBS=2 bash experiments/e29_comp_boundary.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mk() { [ -d data/$1 ] || python -m src.data --n $2 --train-count $3 --test-count $4 --seed 0 --out data/$1 > /dev/null; }
for c in 3600 5000 7200; do mk n19_c$c 19 $c 4000; done
for c in 6000 9000 12000; do mk n23_c$c 23 $c 20000; done
D="data/n19_c3600 data/n19_c5000 data/n19_c7200 data/n23_c6000 data/n23_c9000 data/n23_c12000"
TRAIN="--lr 3e-4 --steps 100000 --stop-after-gen 1000 --eval-every 500"
python sweep.py --data $D --rules comp_next_next --models m0 --seeds 0 1 2 --runs-dir runs_e29_full \
  --multi --group-size 3 --jobs $JOBS $EXTRA -- $TRAIN > runs_e29_full.sweep.log 2>&1 &
python sweep.py --data $D --rules comp_next_next --models m0 --seeds 0 1 2 --runs-dir runs_e29_mb \
  --multi --group-size 3 --jobs $JOBS $EXTRA -- $TRAIN --batch-size 2048 > runs_e29_mb.sweep.log 2>&1 &
wait
grep -h "실패" runs_e29_*.sweep.log || true
python analyze.py "runs_e29_*/*" --out e29_summary.csv --group-out e29_grouped.csv || true
tar czf e29_results${SHARD:+_$(echo $SHARD | tr / -)}.tgz --exclude='*.pt' runs_e29_* e29_summary.csv e29_grouped.csv
echo "끝: e29_results${SHARD:+_$(echo $SHARD | tr / -)}.tgz"
