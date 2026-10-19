#!/usr/bin/env bash
# E11 내부 분석용 체크포인트: 일반화 전후의 모델을 1·2·5·10·20·… step마다 저장한다.
#   규칙 4개(w5_next, w6_p1, w5_next_n3, w6_p1_n3) × train 1,024·1,280 × seed 2 = 16 run, 일반화 후 5,000 step 더
#   체크포인트(.pt)도 결과에 포함한다 (용량 수백 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}; OUT=runs_e11
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
for c in 1024 1280; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c
done
python sweep.py --data data/c1024 data/c1280 --rules w5_next w6_p1 w5_next_n3 w6_p1_n3 --models m0 --seeds 0 1 \
  --runs-dir $OUT --multi --group-size 4 --jobs $JOBS $EXTRA \
  -- --lr 3e-4 --steps 40000 --stop-after-gen 5000 --eval-every 100 --ckpt-log --sym-every 100
tar czf e11_results.tgz $OUT
echo "끝: e11_results.tgz"
