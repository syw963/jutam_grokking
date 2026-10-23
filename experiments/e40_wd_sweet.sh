#!/usr/bin/env bash
# E40 weight decay의 권장 구간: 필요한 데이터(경계) 손해와 외운 오답 교정을 같은 wd 축에서 정량화 (E38·E37·E39 후속)
#   A. 경계: w5_next(a+b), m0, n 31·50·71, wd 2·5·7·15 × 경계 주변 N 5~6점 × seed 5 (330 run)
#      설정은 E32 기본·E38과 같다(lr 3e-4, 6만 step, 일반화 후 1,000 step에서 멈춤) → wd 0.3·1·3·10(E32·E38)과 합쳐 wd 8개의 곡선.
#   B. 오답: n=50, N 1,024·2,048·4,096 × wd 1·3·5·7·10·15 × {오답 없음, 흩어진 10%, 같은 칸 10%} × seed 4 (216 run), 6만 step
#   분석: analyze_wd_sweet.py (wd별 경계 배수·지수, wd·N별 학습 성공·외운 오답·교정 시점) → 그림 하나에 두 곡선
# 사전 예측 (10월 23일 12시 50분 기록)
#   P1 경계 배수(wd 1 대비, n=50)는 wd에 따라 매끄럽게 커진다: wd 2 ≈ 1.0(±15%), wd 5 1.1~1.35, wd 7 1.25~1.5, wd 15 ≥ 1.8.
#   P2 경계 지수는 wd 3 이하에서 0.70~0.85, wd 7 이상에서 0.85 이상(큰 표일수록 손해가 큼).
#   P3 흩어진 오답: N 2,048·4,096에서 wd ≥ 5면 고친다(마지막 오답 < 0.15). 같은 칸 오답: wd 7에서 고친다(< 0.15), wd 5에서는 못 고친다(> 0.8).
#   P4 N 1,024(wd 1 경계의 약 1.6배)에서는 wd 10·15가 학습에 실패하고(오답 없어도 ≤ 2/4), wd 7은 성공한다.
#   P5 권장 구간: 두 오답을 모두 고치는 가장 작은 wd는 7이고, 그때 경계 손해는 1.5배 이하.
# 실행: JUTAM_MAX_PER_GPU=8 GPUS=all JOBS=4 bash experiments/e40_wd_sweet.sh   (L40S 6장 약 2~3시간)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 경계 격자 (E38의 격자를 재사용해 데이터가 같다)
declare -A G2=(  [31]="175 240 320 435 580"       [50]="365 495 665 895 1195"       [71]="620 845 1125 1520 2030" )
declare -A G57=( [31]="240 320 435 580 775 1030"  [50]="495 665 895 1195 1590 2120" [71]="845 1125 1520 2030 2705 3605" )
declare -A G15=( [31]="435 580 775 1030 1375"     [50]="895 1195 1590 2120 2830"    [71]="1520 2030 2705 3605 4810" )
mk() { [ -d data/n${1}_c$2 ] || python -m src.data --n $1 --train-count $2 --test-count 20000 --seed 0 --out data/n${1}_c$2 > /dev/null; }
# 데이터는 먼저 한 번에 만든다(동시에 만들면 충돌: E38 1차 실행에서 겪음)
for n in 31 50 71; do for c in ${G2[$n]} ${G57[$n]} ${G15[$n]}; do mk $n $c; done; done
for c in 1024 2048 4096; do [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null; done
law() {  # law <n> <wd> <N...>
  local n=$1 wd=$2 D=""; shift 2
  for c in $@; do D="$D data/n${n}_c$c"; done
  python sweep.py --data $D --rules w5_next --models m0 --seeds 0 1 2 3 4 --wds $wd --runs-dir runs_e40_law_wd${wd}_n$n \
    --multi --group-size 5 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps 60000 --stop-after-gen 1000 --eval-every 200 --sym-every 0 \
    > runs_e40_law_wd${wd}_n$n.sweep.log 2>&1
}
noise() {  # noise <이름> [오답 인자...]
  local out=runs_e40_noise_$1; shift
  python sweep.py --data data/c1024 data/c2048 data/c4096 --rules w5_next --models m0 --seeds 0 1 2 3 --wds 1.0 3.0 5.0 7.0 10.0 15.0 \
    --runs-dir $out --multi --group-size 24 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps 60000 --eval-every 250 --sym-every 0 "$@" \
    > $out.sweep.log 2>&1
}
noise scatter --noise-frac 0.1 --noise-mode scatter &
noise cell --noise-frac 0.1 --noise-mode cell &
noise none &
for n in 31 50 71; do
  law $n 2.0 ${G2[$n]} &
  law $n 5.0 ${G57[$n]} &
  law $n 7.0 ${G57[$n]} &
  law $n 15.0 ${G15[$n]} &
done
wait
grep -h "실패" runs_e40_*.sweep.log || true
python analyze_wd_sweet.py --law "runs_e40_law_*" --noise "runs_e40_noise_*" --out e40_report > e40_report.log 2>&1 || true
python analyze_noise.py "runs_e40_noise_*" --out e40_report/noise > /dev/null 2>&1 || true
tar czf e40_results.tgz --exclude='*.pt' runs_e40_* e40_report e40_report.log
echo "끝: e40_results.tgz"
