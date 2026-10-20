#!/usr/bin/env bash
# E22 수열 길이(m) 바꾸기: 두 번째 요인이 "예제 수"인지 "학습 정답 총개수"인지 가른다.
#   같은 규칙 p_next(y_i = x_i + x_{i+1}, 순환, a+b, 예제당 서로 다른 쌍 m개)를 m=4·6으로. m=5는 E12의 w5_next.
#   칸당 관찰 수 2.2 / 2.6 / 3.0 / 3.4가 되게 N = 관찰 수 × 1,275 / m
#     m=4: 701 / 829 / 956 / 1,084      m=6: 468 / 553 / 638 / 723
#   같은 관찰 수면 학습 정답 총개수(m × N)와 정답 값당 횟수는 m과 상관없이 같고, 예제 수 N만 다르다.
#   사전 예측(학습 정답 총개수 가설): m=4·5·6이 같은 관찰 수에서 같다 → E12 w5_next와 같게
#     0/10, 8/10 · 약 11,000, 10/10 · 약 4,900, 10/10 · 약 2,900
#   반대 가설(예제 수 가설): m=4(예제 많음)가 m=6보다 확실히 쉽다.
#   2 × 4 × seed 10 = 80 run, 최대 60,000 step. 5090 한 장 약 1시간.
# 실행: GPUS=all JOBS=4 bash experiments/e22_seq_length.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in 701 829 956 1084; do [ -d data/m4_c$c ] || python -m src.data --m 4 --train-count $c --test-count 20000 --seed 0 --out data/m4_c$c > /dev/null; done
for c in 468 553 638 723; do [ -d data/m6_c$c ] || python -m src.data --m 6 --train-count $c --test-count 20000 --seed 0 --out data/m6_c$c > /dev/null; done
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 100"
S10="0 1 2 3 4 5 6 7 8 9"
python sweep.py --data data/m4_c701 data/m4_c829 data/m4_c956 data/m4_c1084 --rules p_next --models m0 --seeds $S10 \
  --runs-dir runs_e22_m4 --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN > runs_e22_m4.sweep.log 2>&1 &
python sweep.py --data data/m6_c468 data/m6_c553 data/m6_c638 data/m6_c723 --rules p_next --models m0 --seeds $S10 \
  --runs-dir runs_e22_m6 --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN > runs_e22_m6.sweep.log 2>&1 &
wait
grep -h "실패" runs_e22_*.sweep.log || true
python analyze.py "runs_e22_*/*" --out e22_summary.csv --group-out e22_grouped.csv || true
tar czf e22_results.tgz --exclude='*.pt' runs_e22_* e22_summary.csv e22_grouped.csv
echo "끝: e22_results.tgz"
