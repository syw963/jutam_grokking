#!/usr/bin/env bash
# E39 외운 오답은 모델 안 어디에 있는가 (오답 실험 내부 분석, E30·E37 후속)
#   w5_next(n=50), m0, CE, lr 3e-4, 오답 10%, seed 0~3, 50,000 step
#   N 2,048: {scatter, cell} × wd {1, 3, 5, 10} + 오답 없음 × wd {1, 10}   (40 run)
#   N   768: {scatter, cell} × wd {1, 5}  (오답을 먼저 외우는 조건, 16 run)     → 총 56 run
#   체크포인트: 처음 3,000 step은 50 step마다(오답 암기·교정이 step 200~2,000에 일어남), 이후 2,500 step마다 + 로그 간격.
#   분석: analyze_noise_internal.py (층별 읽기, 부품 끄기, A/B로 나눈 오답 담당 뉴런 끄기 vs 무작위·규칙 담당, 뉴런 크기, cell 번짐)
# 사전 예측 (10월 22일 23시 20분 기록)
#   P1 (층별 읽기) wd 1 오답 자리에서 1층 MLP 직전(L1_attn)까지는 원래 정답이 1위인 비율 ≥ 0.5, 1층 MLP를 지나며 외운 오답으로 바뀐다.
#   P2 (뉴런 끄기, N 2,048·wd 1·scatter) 오답 담당 상위 32개(1,024개 중, A 자리로 고름)를 끄면 고르지 않은 B 자리의 외운 오답 < 0.3,
#      test > 0.9. 무작위 32개는 외운 오답 > 0.9 그대로. 규칙 담당 상위 32개를 끄면 test가 외운 오답보다 더 크게 떨어진다.
#   P3 (wd 10) 오답이 고쳐지는 시점에 오답 담당 뉴런의 크기(노름 비율)가 줄고, 고쳐지기 전에도 L1_attn에서는 원래 정답이 우세하다.
#   P4 (cell) 같은 칸 오답은 상위 뉴런을 꺼도 scatter보다 덜 사라지고, 끄면 test 번짐도 함께 줄어든다.
#   P5 (N 768, 오답 먼저 외움) 오답 담당 뉴런과 규칙 담당 뉴런의 상위 32개 겹침이 N 2,048보다 크다(같은 데서 같이 외움).
# 실행: JUTAM_MAX_PER_GPU=8 GPUS=all JOBS=4 bash experiments/e39_noise_internal.sh
#   (5090 2장 기준 학습 약 1시간 + 분석 약 30~60분. 체크포인트 약 8GB → 디스크 30GB 이상)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-50000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in 768 2048; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
done
CK="--ckpt-log --ckpt-early-every 50 --ckpt-early-until 3000 --ckpt-dense 2500"
run() {  # run <N> <방식> <wd 목록> [오답 인자...]
  local N=$1 mode=$2 wds=$3; shift 3
  local out=runs_e39_c${N}_$mode
  python sweep.py --data data/c$N --rules w5_next --models m0 --seeds 0 1 2 3 --wds $wds \
    --runs-dir $out --multi --group-size 4 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $STEPS --eval-every 500 --sym-every 0 \
    $CK "$@" > $out.sweep.log 2>&1
}
run 2048 scatter "1.0 3.0 5.0 10.0" --noise-frac 0.1 --noise-mode scatter &
run 2048 cell "1.0 3.0 5.0 10.0" --noise-frac 0.1 --noise-mode cell &
run 2048 none "1.0 10.0" &
run 768 scatter "1.0 5.0" --noise-frac 0.1 --noise-mode scatter &
run 768 cell "1.0 5.0" --noise-frac 0.1 --noise-mode cell &
wait
grep -h "실패" runs_e39_*.sweep.log || true
mkdir -p e39_report
python analyze_noise.py "runs_e39_*" --out e39_report/outputs > e39_report/outputs.log 2>&1 || true
# 내부 분석: run 폴더마다 하나씩, GPU에 나눠서
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
i=0
for d in runs_e39_c*_*; do
  [ -d "$d" ] || continue
  CUDA_VISIBLE_DEVICES=$((i % NG)) python analyze_noise_internal.py "$d/w5_*" --out e39_report/internal_${d#runs_e39_} \
    > e39_report/internal_${d#runs_e39_}.log 2>&1 &
  i=$((i + 1))
done
wait
python analyze_noise_internal.py --merge "e39_report/internal_*" --out e39_report/internal > e39_report/internal_merge.log 2>&1 || true
# seed 0의 마지막 체크포인트만 결과에 넣는다(나중에 GPU 없이 추가 분석용, GitHub 파일 크기 제한 때문에 seed 0만)
KEEP=$(for d in runs_e39_*/w5_*_s0; do ls $d/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1; done)
tar czf e39_results.tgz --exclude='*.pt' runs_e39_* e39_report
tar czf e39_ckpt_subset.tgz $KEEP || true
echo "끝: e39_results.tgz e39_ckpt_subset.tgz"
