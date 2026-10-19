#!/usr/bin/env bash
# E13·E14: "칸당 관찰 수" 법칙이 다른 조건에서도 성립하는가 (E12와 같은 사전 예측선으로 비교)
#
# E13 연산 표 크기 바꾸기: n = 31, 71 (3과 서로소라 a+3b도 일대일). 칸 수 a+b n(n+1)/2, a+3b n².
#   w5_next(a+b, 5쌍), w5_next_n3(a+3b, 5쌍) × 관찰 수 2.2 / 2.6 / 3.0 / 3.4 × seed 10 = 160 run
#     n31 a+b  N = 218, 258, 298, 337     n31 a+3b N = 423, 500, 577, 654
#     n71 a+b  N = 1125, 1329, 1534, 1738 n71 a+3b N = 2218, 2621, 3025, 3428
#   예측: n이 달라도 관찰 수가 같으면 지연이 같다 (E6·E7 적합선 10^(5.78 − 0.754 × 관찰 수) 근처).
#
# E14 학습 설정 바꾸기 (n=50): w5_next, w2_m12, w5_next_n3 × 관찰 수 2.6 / 3.0 × seed 5
#   (가) weight decay 0.3 (기본 1.0)  (나) 큰 모델 d256·4층·8헤드 → 60 run
#   예측: 절대 지연은 달라질 수 있지만, 같은 설정 안에서는 관찰 수가 같으면 규칙이 달라도 지연이 비슷하다.
#
# sweep 10개가 동시에 돈다(묶음 28개). GPU 4장 이상 권장.
# 실행: GPUS=all JOBS=1 bash experiments/e13_e14_general.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-1}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
TRAIN="--lr 3e-4 --stop-after-gen 1000 --eval-every 100"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkd() {  # mkd <n> <N>
  [ -d data/n$1_c$2 ] || python -m src.data --n $1 --train-count $2 --test-count 20000 --seed 0 --out data/n$1_c$2 > /dev/null
}
for c in 218 258 298 337 423 500 577 654; do mkd 31 $c; done
for c in 1125 1329 1534 1738 2218 2621 3025 3428; do mkd 71 $c; done
for c in 663 765 1105 1275 1300 1500; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
done

run() {  # run <runs 폴더> <규칙> "<seeds>" "<데이터 폴더들>" <학습 옵션...>
  local out=$1 rule=$2 seeds=$3 data=$4; shift 4
  local d=""; for c in $data; do d="$d data/$c"; done
  python sweep.py --data $d --rules $rule --models m0 --seeds $seeds --runs-dir $out \
    --multi --group-size 10 --jobs $JOBS $EXTRA "$@" > $out.sweep.log 2>&1
}
S10="0 1 2 3 4 5 6 7 8 9"; S5="0 1 2 3 4"
# E13
run runs_e13_n31_ab w5_next    "$S10" "n31_c218 n31_c258 n31_c298 n31_c337"     -- $TRAIN --steps 60000 &
run runs_e13_n31_n3 w5_next_n3 "$S10" "n31_c423 n31_c500 n31_c577 n31_c654"     -- $TRAIN --steps 60000 &
run runs_e13_n71_ab w5_next    "$S10" "n71_c1125 n71_c1329 n71_c1534 n71_c1738" -- $TRAIN --steps 60000 &
run runs_e13_n71_n3 w5_next_n3 "$S10" "n71_c2218 n71_c2621 n71_c3025 n71_c3428" -- $TRAIN --steps 60000 &
# E14 (가) weight decay 0.3
run runs_e14_wd_w5   w5_next    "$S5" "c663 c765"   --wds 0.3 -- $TRAIN --steps 100000 &
run runs_e14_wd_w2   w2_m12     "$S5" "c1105 c1275" --wds 0.3 -- $TRAIN --steps 100000 &
run runs_e14_wd_n3   w5_next_n3 "$S5" "c1300 c1500" --wds 0.3 -- $TRAIN --steps 100000 &
# E14 (나) 큰 모델
BIG="--d 256 --layers 4 --heads 8"
run runs_e14_big_w5  w5_next    "$S5" "c663 c765"   -- $TRAIN --steps 60000 $BIG &
run runs_e14_big_w2  w2_m12     "$S5" "c1105 c1275" -- $TRAIN --steps 60000 $BIG &
run runs_e14_big_n3  w5_next_n3 "$S5" "c1300 c1500" -- $TRAIN --steps 60000 $BIG &
wait
grep -h "실패" runs_e13*.sweep.log runs_e14*.sweep.log || true

python analyze_obs.py "runs_e13_*" --out e13_report || true
python analyze_obs.py "runs_e14_wd_*" --out e14_wd_report || true
python analyze_obs.py "runs_e14_big_*" --out e14_big_report || true
tar czf e13_e14_results.tgz --exclude='*.pt' runs_e13_* runs_e14_* e13_report e14_wd_report e14_big_report
echo "끝: e13_e14_results.tgz"
