#!/usr/bin/env bash
# E15 두 번째 요인 직접 검증: 칸당 관찰 수가 같을 때 "서로 다른 예제 수"가 많을수록 빠른가.
#
# A. 출력 칸 가리기(--train-mask k): train 예제마다 출력 5칸 중 k칸만 학습에 쓴다(예제별 고정 무작위).
#    규칙·모델·입력 분포는 그대로 두고, 같은 관찰 수를 k=5(가리지 않음)·3·2로 채움 → 예제 수가 1배·1.67배·2.5배.
#    w5_next(a+b, 1,275칸), w5_next_n3(a+3b, 2,500칸) × 관찰 수 2.2·2.6·3.0 × k 5·3·2 × seed 10 = 180 run
#      a+b   k5: 561 663 765    k3: 935 1105 1275    k2: 1403 1658 1913
#      a+3b  k5: 1100 1300 1500 k3: 1833 2167 2500   k2: 2750 3250 3750
#    사전 예측(E12 회귀 log10 지연 = 9.05 − 0.72·관찰 수 − 1.09·log10 N): 같은 관찰 수에서 k2가 k5보다 약 2.7배 빠르다.
#      관찰 수 2.2: a+b k5는 거의 실패(E12에서 0/10), k2는 대부분 일반화.
#    반대 가설(관찰 수만 중요): k에 상관없이 같다.
# B. split seed 바꾸기: 지금까지 모든 실험이 split seed 0. seed 1·2로 E12 일부 반복.
#    w5_next(663, 765), w2_m12(1105, 1275), w5_next_n3(1300, 1500) × split seed 1·2 × 모델 seed 5 = 60 run
#    예측: E12 값과 seed 흩어짐(약 1.5배) 안에서 같다.
#
# sweep 9개가 동시에 돈다(묶음 30개). GPU 4장 이상 권장.
# 실행: GPUS=all JOBS=1 bash experiments/e15_examples.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-1}; STEPS=${STEPS:-60000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
TRAIN="--lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 100"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for c in 561 663 765 935 1105 1275 1403 1658 1913 1100 1300 1500 1833 2167 2500 2750 3250 3750; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
done
for sd in 1 2; do for c in 663 765 1105 1275 1300 1500; do
  [ -d data/s${sd}_c$c ] || python -m src.data --train-count $c --test-count 20000 --seed $sd --out data/s${sd}_c$c > /dev/null
done; done

run() {  # run <runs 폴더> <규칙> "<seeds>" "<데이터 폴더들>" <학습 옵션...>
  local out=$1 rule=$2 seeds=$3 data=$4; shift 4
  local d=""; for c in $data; do d="$d data/$c"; done
  python sweep.py --data $d --rules $rule --models m0 --seeds $seeds --runs-dir $out \
    --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN "$@" > $out.sweep.log 2>&1
}
S10="0 1 2 3 4 5 6 7 8 9"; S5="0 1 2 3 4"
# A
run runs_e15_ab_k5 w5_next    "$S10" "c561 c663 c765" &
run runs_e15_ab_k3 w5_next    "$S10" "c935 c1105 c1275"  --train-mask 3 &
run runs_e15_ab_k2 w5_next    "$S10" "c1403 c1658 c1913" --train-mask 2 &
run runs_e15_n3_k5 w5_next_n3 "$S10" "c1100 c1300 c1500" &
run runs_e15_n3_k3 w5_next_n3 "$S10" "c1833 c2167 c2500" --train-mask 3 &
run runs_e15_n3_k2 w5_next_n3 "$S10" "c2750 c3250 c3750" --train-mask 2 &
# B
run runs_e15s_w5 w5_next    "$S5" "s1_c663 s1_c765 s2_c663 s2_c765" &
run runs_e15s_w2 w2_m12     "$S5" "s1_c1105 s1_c1275 s2_c1105 s2_c1275" &
run runs_e15s_n3 w5_next_n3 "$S5" "s1_c1300 s1_c1500 s2_c1300 s2_c1500" &
wait
grep -h "실패" runs_e15*.sweep.log || true

python analyze_obs.py "runs_e15_*" --out e15_report || true
python analyze_obs.py "runs_e15s_*" --out e15s_report || true
tar czf e15_results.tgz --exclude='*.pt' runs_e15* e15_report e15s_report
echo "끝: e15_results.tgz"
