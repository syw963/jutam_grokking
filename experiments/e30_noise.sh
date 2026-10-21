#!/usr/bin/env bash
# E30 오답 실험을 우리 과제에서: 규칙을 깨친 뒤 외운 오답은 남는가, 언제 지워지는가, 처음 보는 문제로 번지는가
#   규칙 w5_next (n=50, a+b, 이웃 덧셈), 모델 m0, full batch, lr 3e-4, 최대 200,000 step (멈추지 않음)
#   train N: 768 (경계 근처, 오답 없을 때 지연 약 5,000) / 2,048 (경계보다 훨씬 위, 지연 거의 0)
#   오답: 없음 / scatter(예제마다 한 칸) / cell(표 칸 단위로 일관) × 비율 0.05·0.1·0.2 (오답 토큰 수 = 비율 × N, 두 방식 같게)
#   wd 1.0, 2.0 × seed 5 → 7 조건 × N 2 × wd 2 × seed 5 = 140 run (묶음 14개, 묶음마다 wd 2 × seed 5)
#
# 사전 예측 (10월 21일 기록)
#   P1 (scatter, 일반화 시점): 외운 오답 유지 비율 mem ≥ 0.8 — 다른 대화의 GPT 실험(91~99%)과 같다.
#   P2 (scatter, 장기): Doshi 외(ICLR 2024)처럼 weight decay가 외운 오답을 지운다면, 20만 step 안에 mem이 크게 준다
#       (N=2,048, wd 2.0에서 0.5 아래). 줄더라도 wd 2.0이 wd 1.0보다 빨리 준다(지연 ∝ 1/wd와 같은 방향).
#       반대 결과(mem 그대로)면 "이 조건에서는 20만 step 안에 지우지 않는다".
#   P3 (번짐): cell 방식은 오답이 처음 보는 문제로 번진다(test_spread ≥ 0.5, 칸 단위로 표를 배우므로).
#       scatter 방식은 거의 번지지 않는다(test_spread ≤ 0.1, 같은 칸이 다른 예제에서는 정답으로 나오므로).
#   P4 (경계): 오답 비율이 클수록 N=768의 일반화가 늦어지거나 실패한다(오답 칸만큼 정답 관찰이 줄어든다).
#
# 실행: GPUS=all JOBS=4 bash experiments/e30_noise.sh      (서버 여러 대면 SHARD=0/2, SHARD=1/2)
#   4090·5090 한 장 약 3~5시간, 네 장이면 약 1~1.5시간 (예상)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-200000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in 768 2048; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
done
TRAIN="--lr 3e-4 --steps $STEPS --eval-every 1000 --sym-every 0"
run() {  # run <이름> <학습 옵션...>
  local out=runs_e30_$1; shift
  python sweep.py --data data/c768 data/c2048 --rules w5_next --models m0 --seeds 0 1 2 3 4 --wds 1.0 2.0 \
    --runs-dir $out --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN "$@" > $out.sweep.log 2>&1
}
run none &
for f in 0.05 0.1 0.2; do
  run scatter$f --noise-frac $f --noise-mode scatter &
  run cell$f    --noise-frac $f --noise-mode cell &
done
wait
grep -h "실패" runs_e30_*.sweep.log || true
python analyze_noise.py "runs_e30_*" --out e30_report || true
python analyze.py "runs_e30_*/*" --out e30_report/summary.csv --group-out e30_report/grouped_metrics.csv || true
T=e30_results${SHARD:+_$(echo $SHARD | tr / -)}.tgz
tar czf $T --exclude='*.pt' runs_e30_* e30_report
echo "끝: $T"
