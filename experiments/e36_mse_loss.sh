#!/usr/bin/env bash
# E36 외운 오답이 지워지지 않는 이유가 손실 함수(CE)인가: Doshi 외(2024)는 MSE 손실에서 weight decay가 오답을 지웠다(역전).
#   E30·E34: CE 손실에서는 m=5·m=2, wd 1~3, 20만 step 모두 외운 오답 100% 유지.
#   과제 m=2 (E34와 같은 data/m2_c1100), p_next, 오답 = 예제 정답 전체(오답 예제 10%, --noise-frac 0.2 --noise-mode cell)
#   A. MSE 손실 × lr 3e-4·1e-3 × wd 1·3·10 × {오답 없음, 오답 10%} × seed 4 = 48 run
#   B. CE 손실, lr 3e-4, wd 10 × {없음, 10%} × seed 4 = 8 run  (wd만 크게 하면 CE에서도 지워지는가)
#   최대 200,000 step, 멈추지 않음
# 사전 예측 (10월 22일 10시 30분 기록)
#   P1 (손실 가설) MSE + 오답에서 wd 3 또는 10일 때 20만 step 안에 외운 오답이 크게 준다(mem < 0.5)면서 처음 보는 문제 정확도 ≥ 0.9 유지(역전).
#   P2 CE는 wd 10에서도 mem ≥ 0.9 (지우지 않음; 일반화 실패나 규칙 붕괴는 있을 수 있음).
#   반대: MSE에서도 mem ≈ 1이면 차이는 손실이 아니라 모델(Transformer vs MLP)·최적화 설정 → 여기서 멈추고 한계로 적는다.
#   참고: CPU 시험에서 MSE는 CE보다 외우기가 느리다(300 step에 train 29% vs CE 100%). 오답 없는 MSE가 일반화하지 못하면 그 조건은 판정 불가.
# 실행: JUTAM_MAX_PER_GPU=6 GPUS=all JOBS=4 bash experiments/e36_mse_loss.sh   (5090 한 장 약 1~1.5시간)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-200000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d data/m2_c1100 ] || python -m src.data --m 2 --train-count 1100 --test-count 125 --seed 0 --out data/m2_c1100 > /dev/null
run() {  # run <이름> "<wd 목록>" <학습 옵션...>
  local out=runs_e36_$1 wds=$2; shift 2
  python sweep.py --data data/m2_c1100 --rules p_next --models m0 --seeds 0 1 2 3 --wds $wds \
    --runs-dir $out --multi --group-size 4 --jobs $JOBS $EXTRA -- --steps $STEPS --eval-every 1000 --sym-every 0 "$@" > $out.sweep.log 2>&1
}
for lr in 3e-4 1e-3; do
  run mse_lr${lr}_none  "1.0 3.0 10.0" --loss mse --lr $lr &
  run mse_lr${lr}_noise "1.0 3.0 10.0" --loss mse --lr $lr --noise-frac 0.2 --noise-mode cell &
done
run ce_wd10_none  "10.0" --lr 3e-4 &
run ce_wd10_noise "10.0" --lr 3e-4 --noise-frac 0.2 --noise-mode cell &
wait
grep -h "실패" runs_e36_*.sweep.log || true
for d in runs_e36_*; do python analyze_noise.py "$d" --out e36_report/$d || true; done
tar czf e36_results.tgz --exclude='*.pt' runs_e36_* e36_report
echo "끝: e36_results.tgz"
