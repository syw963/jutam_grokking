#!/usr/bin/env bash
# E41 두 수 과제(m=2)의 wd 빈칸 채우기: 보고서 그림(오답 vs weight decay)에서 두 수 과제는 wd 1·2·3(E34)·10(E36)만 있어 3→10을 점선으로 이었다.
#   E34·E36과 같은 설정: data/m2_c1100(n=50, 1,225쌍 중 train 1,100, test 125), 규칙 p_next, m0, CE, lr 3e-4, 20만 step
#   오답: cell 0.2 = 예제 10%의 정답 전체(두 칸)를 같은 오답으로 (Doshi와 같은 방식). 대조군 오답 없음.
#   wd 4·5·7·15 × {오답, 없음} × seed 0~4 = 40 run
# 사전 예측 (10월 23일 15시 50분 기록)
#   P1 wd 4: 외운 오답 대부분 유지(마지막 mem > 0.8).
#   P2 wd 5: 부분 교정 또는 늦은 교정(20만 step에 mem 0.1~0.8).
#   P3 wd 7: 교정(mem < 0.1), 일반화 무렵에.
#   P4 wd 15: 학습은 되고(오답 없는 대조군 ≥ 4/5 일반화) 오답도 교정.
# 실행: JUTAM_MAX_PER_GPU=8 GPUS=all JOBS=4 bash experiments/e41_m2_wd.sh   (GPU 1장 약 30~60분)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}; STEPS=${STEPS:-200000}
EXTRA=""; [ -n "${GPUS:-}" ] && EXTRA="$EXTRA --gpus $GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d data/m2_c1100 ] || python -m src.data --m 2 --train-count 1100 --test-count 125 --seed 0 --out data/m2_c1100 > /dev/null
run() {  # run <이름> <학습 옵션...>
  local out=runs_e41_$1; shift
  python sweep.py --data data/m2_c1100 --rules p_next --models m0 --seeds 0 1 2 3 4 --wds 4.0 5.0 7.0 15.0 \
    --runs-dir $out --multi --group-size 5 --jobs $JOBS $EXTRA -- --lr 3e-4 --steps $STEPS --eval-every 1000 --sym-every 0 "$@" > $out.sweep.log 2>&1
}
run noise --noise-frac 0.2 --noise-mode cell &
run none &
wait
grep -h "실패" runs_e41_*.sweep.log || true
python analyze_noise.py "runs_e41_*" --out e41_report > e41_report.log 2>&1 || true
tar czf e41_results.tgz --exclude='*.pt' runs_e41_* e41_report e41_report.log
echo "끝: e41_results.tgz"
