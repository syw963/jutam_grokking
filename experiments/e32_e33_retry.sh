#!/usr/bin/env bash
# E32·E33 빠진 묶음만 다시 돌리기 (10월 22일 아침).
#   1차 실행은 GPU 메모리 부족으로 묶음 21개(E32 11개, E33 10개)가 빠졌다.
#   results/의 1차 결과를 풀어 두고 같은 스크립트를 다시 실행하면 끝난 run은 건너뛰고 빠진 묶음만 돈다.
#   서버 전체 동시 학습을 GPU당 MAX_PER_GPU개로 제한한다(sweep.py의 JUTAM_MAX_PER_GPU, 기본 5).
# 실행: MAX_PER_GPU=5 bash experiments/e32_e33_retry.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d runs_e32_big_n31 ] || tar xzf results/E32_big_law_partial.tgz
[ -d runs_e33_n31 ] || tar xzf results/E33_mul_law_partial.tgz
export JUTAM_MAX_PER_GPU=${MAX_PER_GPU:-5}
GPUS=all JOBS=4 bash experiments/e32_big_law.sh > e32_retry.out 2>&1 &
GPUS=all JOBS=4 bash experiments/e33_mul_law.sh > e33_retry.out 2>&1 &
wait
ls -la e32_results.tgz e33_results.tgz
echo "끝: 재실행 완료"
