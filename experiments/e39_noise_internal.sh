#!/usr/bin/env bash
# E39 외운 오답은 모델 안 어디에 있는가 (오답 실험 내부 분석, E30·E37 후속)
#   w5_next(n=50), m0, CE, lr 3e-4, N 2,048, 오답 10% × {scatter, cell} × wd {1 (공존), 10 (고침)} × seed 0·1 = 8 run
#   체크포인트: 로그 간격 + wd 1은 500 step, wd 10은 250 step마다. wd 1은 60,000 step, wd 10은 20,000 step.
#   분석: analyze_noise_internal.py (층별 읽기, 부품 끄기, 오답 담당 뉴런 끄기)
# 사전 예측 (10월 22일 23시 기록)
#   P1 (층별 읽기) wd 1 오답 자리에서 1층 MLP 직전(L1_attn)까지는 원래 정답이 1위인 비율이 0.5 이상이고, 1층 MLP를 지나며 외운 오답으로 바뀐다.
#   P2 (뉴런 끄기) wd 1 scatter에서 오답 담당 상위 32개 뉴런(1층 MLP 512개 중)을 끄면 외운 오답 < 0.3, test 정확도 > 0.9.
#                  같은 수의 무작위 뉴런을 끄면 외운 오답 > 0.9 그대로.
#   P3 (wd 10) 오답이 고쳐지는 시점에 출력의 (오답 − 정답) logit이 음수로 바뀌고, 그 전에도 L1_attn에서는 원래 정답이 우세하다.
#   P4 (cell) 같은 칸에 일관된 오답은 표의 일부처럼 저장되어, 상위 뉴런을 꺼도 scatter보다 덜 사라진다.
# 실행: JUTAM_MAX_PER_GPU=8 GPUS=all JOBS=4 bash experiments/e39_noise_internal.sh   (GPU 1장 약 30~50분, 분석 포함)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d data/c2048 ] || python -m src.data --train-count 2048 --test-count 20000 --seed 0 --out data/c2048 > /dev/null
run() {  # run <이름> <wd> <steps> <dense> <오답 인자...>
  local name=$1 wd=$2 steps=$3 dense=$4; shift 4
  local out=runs_e39_${name}_wd${wd}
  python sweep.py --data data/c2048 --rules w5_next --models m0 --seeds 0 1 --wds $wd \
    --runs-dir $out --multi --group-size 2 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $steps --eval-every 500 --sym-every 0 \
    --ckpt-log --ckpt-dense $dense "$@" > $out.sweep.log 2>&1
}
run scatter 1.0 60000 500 --noise-frac 0.1 --noise-mode scatter &
run cell 1.0 60000 500 --noise-frac 0.1 --noise-mode cell &
run scatter 10.0 20000 250 --noise-frac 0.1 --noise-mode scatter &
run cell 10.0 20000 250 --noise-frac 0.1 --noise-mode cell &
wait
grep -h "실패" runs_e39_*.sweep.log || true
python analyze_noise.py "runs_e39_*" --out e39_report || true
python analyze_noise_internal.py "runs_e39_*/*" --out e39_report/internal || true
# 마지막 체크포인트만 결과에 넣는다(나중에 GPU 없이 추가 분석용, run당 약 2MB)
LAST=$(for d in runs_e39_*/w5_*; do ls $d/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1; done)
tar czf e39_results.tgz --exclude='*.pt' runs_e39_* e39_report
tar czf e39_last_ckpt.tgz $LAST || true
echo "끝: e39_results.tgz e39_last_ckpt.tgz"
