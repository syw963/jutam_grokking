#!/usr/bin/env bash
# E10b GPT 경계 찾기: E10에서 GPT는 1,536개까지 전혀 일반화하지 않았다(외우기만, test 토큰 0.03~0.11).
#   train을 늘려 GPT의 경계를 찾고, 경계가 칸당 관찰 수로 정렬되는지 본다.
#   w5_next(a+b 5쌍), w2_m12(a+b 3쌍), w5_next_n3(a+3b) × train 2,048~8,192 × seed 3 = 45 run, 최대 60,000 step
#   사전 예측: GPT도 경계가 생기면 순서는 m0와 같다(관찰 수가 많은 쪽이 먼저). 단 경계 위치(관찰 수)는 m0의 약 2.3보다 크다.
# 실행: GPUS=all JOBS=2 bash experiments/e10b_gpt_large.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}; STEPS=${STEPS:-60000}; OUT=runs_e10b
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
DATA=""
for c in 2048 3072 4096 6144 8192; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
  DATA="$DATA data/c$c"
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sweep.py --data $DATA --rules w5_next w2_m12 w5_next_n3 --models gpt --seeds 0 1 2 --runs-dir $OUT \
  --multi --group-size 9 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 200
python analyze_obs.py "$OUT" --out e10b_report || true
tar czf e10b_results.tgz --exclude='*.pt' $OUT e10b_report
echo "끝: e10b_results.tgz"
