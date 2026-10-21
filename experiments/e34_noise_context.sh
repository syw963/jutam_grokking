#!/usr/bin/env bash
# E34 외운 오답이 지워지지 않는 이유가 "문맥" 때문인가 (E30이 Doshi 외 2024와 다른 이유 확인)
#   가설: m=5 수열에서는 오답 칸 옆의 다른 숫자(문맥)로 "이 예제만 예외"를 외울 수 있어 규칙과 부딪히지 않는다 → 지울 이유가 없다.
#         두 수 과제(m=2)는 입력 자체가 표 한 칸이라 오답이 규칙과 정면으로 부딪힌다 → weight decay가 오답을 지운다(Doshi의 역전).
#   A. m=2 (n=50, 1,225쌍 중 train 1,100, test 125), 규칙 p_next (y = a+b, b+a: 두 출력이 같은 칸)
#      오답: cell 방식 = 예제 하나의 정답 전체(두 칸)를 같은 오답으로 (Doshi와 같은 방식)
#      오답 토큰 비율 0.2 / 0.4 = 오답 예제 10% / 20%, 대조군 오답 없음 × wd 1·2·3 × seed 5 = 45 run
#   B. m=5 비교: E30과 같은 w5_next N 2,048 에 wd 3 추가 (scatter 0.1, cell 0.1) × seed 5 = 10 run
#   최대 200,000 step, 멈추지 않음
# 사전 예측 (10월 22일 8시 기록)
#   P1 (문맥 가설) m=2에서는 20만 step 안에 외운 오답이 크게 준다: wd 2·3에서 mem < 0.5, 그동안 test 정확도는 높게 유지(역전).
#   P2 m=5는 wd 3에서도 mem ≥ 0.9 (E30과 같음).
#   반대: m=2에서도 mem ≈ 1이면 차이는 문맥이 아니라 손실 함수(CE vs MSE)·학습률 등 학습 설정 때문.
# 실행: JUTAM_MAX_PER_GPU=6 GPUS=all JOBS=4 bash experiments/e34_noise_context.sh   (5090 한 장 약 1~2시간)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-200000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d data/m2_c1100 ] || python -m src.data --m 2 --train-count 1100 --test-count 125 --seed 0 --out data/m2_c1100 > /dev/null
[ -d data/c2048 ] || python -m src.data --train-count 2048 --test-count 20000 --seed 0 --out data/c2048 > /dev/null
TRAIN="--lr 3e-4 --steps $STEPS --eval-every 1000 --sym-every 0"
m2() {  # m2 <이름> <옵션...>
  local out=runs_e34_m2_$1; shift
  python sweep.py --data data/m2_c1100 --rules p_next --models m0 --seeds 0 1 2 3 4 --wds 1.0 2.0 3.0 \
    --runs-dir $out --multi --group-size 5 --jobs $JOBS $EXTRA -- $TRAIN "$@" > $out.sweep.log 2>&1
}
m5() {
  local out=runs_e34_m5_$1; shift
  python sweep.py --data data/c2048 --rules w5_next --models m0 --seeds 0 1 2 3 4 --wds 3.0 \
    --runs-dir $out --multi --group-size 5 --jobs $JOBS $EXTRA -- $TRAIN "$@" > $out.sweep.log 2>&1
}
m2 none &
m2 cell0.2 --noise-frac 0.2 --noise-mode cell &
m2 cell0.4 --noise-frac 0.4 --noise-mode cell &
m5 scatter0.1 --noise-frac 0.1 --noise-mode scatter &
m5 cell0.1 --noise-frac 0.1 --noise-mode cell &
wait
grep -h "실패" runs_e34_*.sweep.log || true
python analyze_noise.py "runs_e34_*" --out e34_report || true
tar czf e34_results.tgz --exclude='*.pt' runs_e34_* e34_report
echo "끝: e34_results.tgz"
