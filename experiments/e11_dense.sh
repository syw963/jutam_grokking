#!/usr/bin/env bash
# E11 촘촘한 체크포인트: E11과 같은 16 run을 200 step마다 체크포인트를 저장하며 다시 학습하고,
# 바로 내부 분석(analyze_internal.py)까지 한다. 결과: e11d_internal.tgz (표·그림만, 작음)
#   "train에 나온 칸 → 안 나온 칸" 순서(3·4단계)를 모든 run에서 구분하려는 목적
# 실행: GPUS=all JOBS=2 bash experiments/e11_dense.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}; OUT=runs_e11d
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
for c in 1024 1280; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c
done
python sweep.py --data data/c1024 data/c1280 --rules w5_next w6_p1 w5_next_n3 w6_p1_n3 --models m0 --seeds 0 1 \
  --runs-dir $OUT --multi --group-size 4 --jobs $JOBS $EXTRA \
  -- --lr 3e-4 --steps 40000 --stop-after-gen 3000 --eval-every 100 --ckpt-log --ckpt-dense 200 --sym-every 200
python analyze_internal.py "$OUT/*" --out e11d_internal
tar czf e11d_internal.tgz e11d_internal
echo "끝: e11d_internal.tgz"
