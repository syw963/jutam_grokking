#!/usr/bin/env bash
# E28 GPT에서도 "필요한 예제 수 ∝ C^0.76"인가 (법칙의 지수가 모델 구조와 무관한지)
#   m0 법칙: N0 = 2.81 × C^0.764. GPT는 n=50 a+b(w5_next)에서 경계 약 1,800(E10·E10b: 1,536 0/5, 2,048 3/3) → 상수만 약 2.7배라고 가정하면
#   GPT 사전 예측 N0 = 7.6 × C^0.764: n=31(C 496) 약 880, n=71(C 2,556) 약 3,050.
#   대안: 지수가 다르면 n=31·71의 경계 비율이 3.5배(=(2556/496)^0.764)에서 벗어난다.
#   w5_next × n=31: N 600 / 800 / 1,000 / 1,300,  n=71: N 2,000 / 2,600 / 3,400 / 4,400 × seed 3 = 24 run, 모델 gpt, 최대 60,000 step
#   5090 한 장 약 1.5시간
# 실행: GPUS=all JOBS=4 bash experiments/e28_gpt_law.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mk() { [ -d data/$1 ] || python -m src.data --n $2 --train-count $3 --test-count 20000 --seed 0 --out data/$1 > /dev/null; }
for c in 600 800 1000 1300; do mk n31_c$c 31 $c; done
for c in 2000 2600 3400 4400; do mk n71_c$c 71 $c; done
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 200"
python sweep.py --data data/n31_c600 data/n31_c800 data/n31_c1000 data/n31_c1300 --rules w5_next --models gpt --seeds 0 1 2 \
  --runs-dir runs_e28_n31 --multi --group-size 3 --jobs $JOBS $EXTRA -- $TRAIN > runs_e28_n31.sweep.log 2>&1 &
python sweep.py --data data/n71_c2000 data/n71_c2600 data/n71_c3400 data/n71_c4400 --rules w5_next --models gpt --seeds 0 1 2 \
  --runs-dir runs_e28_n71 --multi --group-size 3 --jobs $JOBS $EXTRA -- $TRAIN > runs_e28_n71.sweep.log 2>&1 &
wait
grep -h "실패" runs_e28_*.sweep.log || true
python analyze.py "runs_e28_*/*" --out e28_summary.csv --group-out e28_grouped.csv || true
tar czf e28_results.tgz --exclude='*.pt' runs_e28_* e28_summary.csv e28_grouped.csv
echo "끝: e28_results.tgz"
