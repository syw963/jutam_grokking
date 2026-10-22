#!/usr/bin/env bash
# E38 학습 설정(weight decay)을 바꿔도 필요한 데이터 수 법칙의 꼴이 유지되는가
#   E37: 데이터 768개(n=50)에서 wd 10은 아무것도 못 배움 → 경계가 wd에 따라 움직인다. 지수까지 바뀌는지 잰다.
#   기준(wd 1.0)은 E32 기본 모델(같은 n, seed 6).
#   모델 m0, 규칙 w5_next(a+b), n = 31·50·71 (C 496·1,275·2,556)
#   wd 0.3 (최대 150,000 step: 지연이 약 3배 길다) · wd 3.0 (60,000 step): N = 법칙의 0.55·0.75·1.0·1.35·1.8배
#     n31: 175 240 320 435 580   n50: 365 495 665 895 1195   n71: 620 845 1125 1520 2030
#   wd 10.0 (60,000 step): N = 법칙의 1.0·1.35·1.8·2.4·3.2배
#     n31: 320 435 580 775 1030  n50: 665 895 1195 1590 2120  n71: 1125 1520 2030 2705 3605
#   × seed 5 = 225 run
# 사전 예측 (10월 22일 17시 30분 기록)
#   P1 세 wd 모두 지수가 0.70~0.85(기본 0.764와 같은 꼴).
#   P2 wd 3의 경계는 wd 1(E32 기본)의 ±20% 안.
#   P3 wd 10의 경계는 wd 1의 1.3배 이상(n=50: 768개에서 실패했으므로).
#   P4 wd 0.3의 경계는 wd 1의 ±25% 안(150,000 step이면 늦어도 일반화한다는 가정).
# 실행: JUTAM_MAX_PER_GPU=8 GPUS=all JOBS=4 bash experiments/e38_law_wd.sh   (5090 2장 약 2~3시간)
# 재실행(wd 0.3·3만): WDS="0.3 3.0" JUTAM_MAX_PER_GPU=8 GPUS=all JOBS=4 bash experiments/e38_law_wd.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"; [ -n "${SHARD:-}" ] && EXTRA="$EXTRA --shard $SHARD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A GA=( [31]="175 240 320 435 580" [50]="365 495 665 895 1195" [71]="620 845 1125 1520 2030" )
declare -A GB=( [31]="320 435 580 775 1030" [50]="665 895 1195 1590 2120" [71]="1125 1520 2030 2705 3605" )
mk() { [ -d data/n${1}_c$2 ] || python -m src.data --n $1 --train-count $2 --test-count 20000 --seed 0 --out data/n${1}_c$2 > /dev/null; }
# 데이터는 먼저 한 번에 만든다(여러 묶음이 같은 폴더를 동시에 만들면 충돌해 묶음이 시작되지 않는다: 10월 22일 1차 실행에서 wd 0.3·3 전부 누락)
for n in 31 50 71; do for c in ${GA[$n]} ${GB[$n]}; do mk $n $c; done; done
run() {  # run <n> <wd> <steps> <grid>
  local n=$1 wd=$2 steps=$3 D=""; shift 3
  for c in $@; do D="$D data/n${n}_c$c"; done
  python sweep.py --data $D --rules w5_next --models m0 --seeds 0 1 2 3 4 --wds $wd --runs-dir runs_e38_wd${wd}_n$n \
    --multi --group-size 5 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $steps --stop-after-gen 1000 --eval-every 200 --sym-every 0 \
    > runs_e38_wd${wd}_n$n.sweep.log 2>&1
}
WDS=${WDS:-"0.3 3.0 10.0"}   # 일부만: WDS="0.3 3.0"
for n in 31 50 71; do
  for wd in $WDS; do
    case $wd in
      0.3) run $n 0.3 150000 ${GA[$n]} & ;;
      3.0) run $n 3.0 60000 ${GA[$n]} & ;;
      10.0) run $n 10.0 60000 ${GB[$n]} & ;;
    esac
  done
done
wait
grep -h "실패" runs_e38_*.sweep.log || true
python analyze_law.py "runs_e38_*" --out e38_report || true
T=e38_results${SHARD:+_$(echo $SHARD | tr / -)}.tgz
tar czf $T --exclude='*.pt' runs_e38_* e38_report
echo "끝: $T"
