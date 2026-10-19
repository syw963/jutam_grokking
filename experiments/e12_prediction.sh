#!/usr/bin/env bash
# E12 예측 검증: "칸당 평균 관찰 수"가 같으면 규칙·연산이 달라도 지연이 같은가.
#   칸당 관찰 수 = train 개수 × 예제당 서로 다른 쌍 수 / 연산 표 칸 수 (a+b 1,275칸, a+3b 2,500칸)
#
# A. 관찰 수 2.2 / 2.6 / 3.0 / 3.4가 되도록 규칙마다 train 개수를 골라 학습 (6규칙 × 4수준 × seed 10 = 240 run)
#      w5_next   (a+b, 5쌍)        N = 관찰 수 × 255     → 561, 663, 765, 867
#      w1_m0     (a+b, 4쌍, 연결)  N = 관찰 수 × 318.75  → 701, 829, 956, 1084
#      w6_p1     (a+b, 4쌍, 끊김)  N = 관찰 수 × 318.75  → 701, 829, 956, 1084
#      w2_m12    (a+b, 3쌍)        N = 관찰 수 × 425     → 935, 1105, 1275, 1445
#      w5_next_n3, w6_p1_n3 (a+3b, 5쌍) N = 관찰 수 × 500 → 1100, 1300, 1500, 1700
#    사전 예측 (E6·E7 적합, 일반화율 0.9 이상 칸): 지연 ≈ 10^(5.78 − 0.754 × 관찰 수)
#      → 2.2: 약 13,000 / 2.6: 약 6,700 / 3.0: 약 3,300 / 3.4: 약 1,700. 2.2에서는 일부 seed가 실패할 것.
# B. E8 확장: 768개에서 coverage를 더 낮춘 split (같은 칸 반복을 더 늘림). 예측: 더 빨라진다.
#      w5_next 0.86, 0.84 / w1_m0 0.83, 0.81 × seed 10 = 40 run
#
# 규칙마다 sweep을 따로 띄워 동시에 돈다(묶음 28개). GPU 4장 이상 권장.
# 실행: GPUS=all JOBS=1 bash experiments/e12_prediction.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-1}; STEPS=${STEPS:-60000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
TRAIN="--lr 3e-4 --steps $STEPS --stop-after-gen 1000 --eval-every 100"
SEEDS="0 1 2 3 4 5 6 7 8 9"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 데이터 (무작위 split, seed 0)
for c in 561 663 765 867 701 829 956 1084 935 1105 1275 1445 1100 1300 1500 1700; do
  [ -d data/c$c ] || python -m src.data --train-count $c --test-count 20000 --seed 0 --out data/c$c > /dev/null
done
# B용 coverage 조절 split
mk() { [ -d data/$2 ] || python -m src.curate --rule $1 --train-count $3 --target $4 --out data/$2; }
mk w5_next e12_next_n768_c86 768 0.86 & mk w5_next e12_next_n768_c84 768 0.84 &
mk w1_m0 e12_m0_n768_c83 768 0.83 & mk w1_m0 e12_m0_n768_c81 768 0.81 &
wait

run() {  # run <runs 폴더> <규칙> <데이터...>
  local out=$1 rule=$2; shift 2
  local d=""; for c in "$@"; do d="$d data/$c"; done
  python sweep.py --data $d --rules $rule --models m0 --seeds $SEEDS --runs-dir $out \
    --multi --group-size 10 --jobs $JOBS $EXTRA -- $TRAIN > $out.sweep.log 2>&1
}
run runs_e12_w5      w5_next    c561 c663 c765 c867 &
run runs_e12_w1      w1_m0      c701 c829 c956 c1084 &
run runs_e12_w6      w6_p1      c701 c829 c956 c1084 &
run runs_e12_w2      w2_m12     c935 c1105 c1275 c1445 &
run runs_e12_w5n3    w5_next_n3 c1100 c1300 c1500 c1700 &
run runs_e12_w6n3    w6_p1_n3   c1100 c1300 c1500 c1700 &
run runs_e12b_next   w5_next    e12_next_n768_c86 e12_next_n768_c84 &
run runs_e12b_m0     w1_m0      e12_m0_n768_c83 e12_m0_n768_c81 &
wait
grep -h "실패" runs_e12*.sweep.log || true

python analyze_boundary.py runs_e12_w5 runs_e12_w1 runs_e12_w6 runs_e12_w2 runs_e12_w5n3 runs_e12_w6n3 --out e12_report || true
python analyze_obs.py e12_report/by_graph.csv --out e12_report/obs_curve.png || true
python analyze.py "runs_e12b_*/*" --out e12_report/partB.csv --group-out e12_report/partB_grouped.csv || true
tar czf e12_results.tgz --exclude='*.pt' runs_e12* e12_report data/e12_*/meta.json
echo "끝: e12_results.tgz"
