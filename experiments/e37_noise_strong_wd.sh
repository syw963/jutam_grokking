#!/usr/bin/env bash
# E37 숫자 5개 수열(m=5)에서도 강한 규제면 외운 오답을 고치는가 (E36 후속)
#   E36: 두 수 과제(m=2)는 wd 10이면 일반화 순간 외운 오답을 고침(CE). E30·E34의 m=5는 wd ≤ 3에서 100% 유지.
#   w5_next(n=50), m0, CE, lr 3e-4, N 768·2,048 × wd 5·10 × {오답 없음, scatter 10%, cell 10%} × seed 4 = 48 run, 최대 100,000 step
# 사전 예측 (10월 22일 13시 40분 기록)
#   P1 m=5도 wd 10이면 scatter 오답을 고친다(mem < 0.5), 일반화 무렵에.
#   P2 wd 5는 그 사이(일부만 고침 또는 늦게 고침).
#   P3 cell(같은 칸에 일관된 오답)은 scatter보다 덜 고친다(표 자체가 바뀐 것처럼 여러 데이터가 같은 오답을 뒷받침하므로).
# 실행: JUTAM_MAX_PER_GPU=8 GPUS=all JOBS=4 bash experiments/e37_noise_strong_wd.sh   (5090 한 장 약 40분~1시간)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-100000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in 768 2048; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
done
run() {
  local out=runs_e37_$1; shift
  python sweep.py --data data/c768 data/c2048 --rules w5_next --models m0 --seeds 0 1 2 3 --wds 5.0 10.0 \
    --runs-dir $out --multi --group-size 8 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $STEPS --eval-every 1000 --sym-every 0 "$@" > $out.sweep.log 2>&1
}
run none &
run scatter0.1 --noise-frac 0.1 --noise-mode scatter &
run cell0.1 --noise-frac 0.1 --noise-mode cell &
wait
grep -h "실패" runs_e37_*.sweep.log || true
python analyze_noise.py "runs_e37_*" --out e37_report || true
tar czf e37_results.tgz --exclude='*.pt' runs_e37_* e37_report
echo "끝: e37_results.tgz"
