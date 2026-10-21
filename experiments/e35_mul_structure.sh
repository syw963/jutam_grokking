#!/usr/bin/env bash
# E35 같은 표 크기, 다른 구조: n=50 곱셈이 유독 쉬운 이유가 표의 구조(합성수)인가 (E33 후속)
#   곱셈 mod n의 값 분포: n=47·53(소수)은 거의 고르다(엔트로피 5.54/5.55, 5.72/5.73 비트),
#   n=50(=2×25)은 덜 고르다(5.35/5.64, 0이 되는 칸 7.8% vs 소수 약 4%).
#   A. w5_next_mul, n=47 (C 1,128, 덧셈 법칙 경계 603): N 330 455 605 815 1085
#                   n=53 (C 1,431, 법칙 724):              N 400 545 725 975 1305
#   B. n=50 곱셈의 경계를 E33 격자(365에서 이미 6/6) 아래에서: N 120 165 220 290
#   × seed 6 = 84 run, 모델 m0, 최대 60,000 step
# 사전 예측 (10월 22일 8시 30분 기록)
#   P1 소수 n=47·53의 경계는 E33 소수 n(31·41·61·71)의 적합선(1.26 × C^0.85, 덧셈 법칙의 약 0.8배) ±20% 안: 약 495, 605.
#   P2 n=50의 경계는 그 선의 0.6배 이하(약 330 이하) — 같은 크기의 소수 표보다 확실히 쉽다(구조 효과).
#   반대: n=47·53도 n=50만큼 쉬우면 "합성수 구조"가 아니라 n≈50 부근 또는 다른 원인.
# 실행: JUTAM_MAX_PER_GPU=6 GPUS=all JOBS=4 bash experiments/e35_mul_structure.sh   (5090 한 장 약 1시간)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A GRID=( [47]="330 455 605 815 1085" [53]="400 545 725 975 1305" [50]="120 165 220 290" )
TRAIN="--lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 200 --sym-every 0"
for n in 47 53 50; do
  D=""
  for c in ${GRID[$n]}; do
    [ -d data/n${n}_c$c ] || python -m src.data --n $n --train-count $c --test-count 20000 --seed 0 --out data/n${n}_c$c > /dev/null
    D="$D data/n${n}_c$c"
  done
  python sweep.py --data $D --rules w5_next_mul --models m0 --seeds 0 1 2 3 4 5 --runs-dir runs_e35_n$n \
    --multi --group-size 6 --jobs $JOBS $EXTRA -- $TRAIN > runs_e35_n$n.sweep.log 2>&1 &
done
wait
grep -h "실패" runs_e35_*.sweep.log || true
python analyze_law.py "runs_e35_*" --out e35_report || true
tar czf e35_results.tgz --exclude='*.pt' runs_e35_* e35_report
echo "끝: e35_results.tgz"
