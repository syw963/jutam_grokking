#!/usr/bin/env bash
# E19 합성 규칙 RQ2: "덧셈 두 번"(comp_next_next = x_i + 2x_{i+1} + x_{i+2})이 E9에서 16,384개까지 전혀 학습되지 않은 이유를
#   칸당 관찰 수로 예측한다. 이 규칙의 출력은 세 값의 함수라 연산 표가 n × n(n+1)/2칸(n=50: 63,750칸, n=31: 15,376칸)이다.
#   예제당 서로 다른 세 값 묶음 5개 → 칸당 관찰 수 = 5N / 칸 수. E6~E12의 경계 약 2.3회를 적용하면
#     n=31: N ≈ 7,100에서 경계,  n=50: N ≈ 29,000에서 경계 (E9 최대 16,384는 1.3회라 실패가 예측과 맞음)
#   반대 가설(모델이 덧셈 두 번으로 나눠 배움): 덧셈 칸은 이미 충분히 관찰되므로 N에 덜 민감하고 n=31·50의 경계 비율이 4배가 아니다.
#   n31: N 4,096 / 8,192 / 16,384 / 32,768,  n50: N 32,768 / 65,536 / 131,072 × seed 3 × 모델 2개(d128·2층, d256·4층) = 42 run
#   E9와 같은 학습 설정: batch 2,048 복원추출, 최대 60,000 step
# 실행: GPUS=all JOBS=3 bash experiments/e19_comp_table.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-3}; STEPS=${STEPS:-60000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
TRAIN="--lr 3e-4 --steps $STEPS --stop-after-gen 2000 --eval-every 200 --batch-size 2048"
D31=""; for c in 4096 8192 16384 32768; do
  [ -d data/n31_c$c ] || python -m src.data --n 31 --train-count $c --test-count 20000 --seed 0 --out data/n31_c$c > /dev/null
  D31="$D31 data/n31_c$c"; done
D50=""; for c in 32768 65536 131072; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
  D50="$D50 data/c$c"; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BIG="--d 256 --layers 4 --heads 8"
for size in small big; do
  X=""; [ $size = big ] && X="$BIG"
  python sweep.py --data $D31 $D50 --rules comp_next_next --models m0 --seeds 0 1 2 --runs-dir runs_e19_$size \
    --multi --group-size 3 --jobs $JOBS $EXTRA -- $TRAIN $X > runs_e19_$size.sweep.log 2>&1 &
done
wait
grep -h "실패" runs_e19_*.sweep.log || true
python analyze.py "runs_e19_*/*" --out e19_summary.csv --group-out e19_grouped.csv || true
tar czf e19_results.tgz --exclude='*.pt' runs_e19_* e19_summary.csv e19_grouped.csv
echo "끝: e19_results.tgz"
