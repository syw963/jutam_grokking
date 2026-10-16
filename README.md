# jutam_grokking

주제탐구: 수열 변환 규칙의 대칭성과 grokking

1~n의 서로 다른 정수 m개로 된 수열을 같은 길이의 수열로 바꾸는 규칙을 신경망에 학습시키고,
암기 이후 일반화가 언제 일어나는지(grokking)를 규칙의 성질(대칭, 의존 범위, 유효 깊이)과 함께 기록한다.

## 설치

```bash
pip install -r requirements.txt
python -m pytest -q tests        # 1분 안에 끝나는 검사
```

CUDA GPU, Mac(MPS), CPU를 자동으로 고른다 (`--device`로 직접 지정 가능). CUDA에서는 기본으로 bf16을 쓴다.

## 전체 흐름

```bash
# 1. split 만들기 (값 집합 단위, 순서 고정). 모든 규칙이 같은 split을 공유한다.
python -m src.data --n 50 --m 5 --train-frac 0.01 --test-count 20000 --seed 0
#   → data/n50_m5_tr21188_te20000_o1_s0

# 2. 규칙 성질 확인
python -m src.rules --list                      # 이름 붙인 규칙 목록
python -m src.rules l1_first                    # 대칭군, 의존 범위, 유효 깊이
python -m src.rules "S:sum > A:asc > AF:sum"

# 3. 학습 하나
python train.py --data data/n50_m5_tr21188_te20000_o1_s0 --rule sort --model m0

# 4. 여러 조건 한꺼번에
python sweep.py --data data/n50_m5_tr21188_te20000_o1_s0 \
  --rules l1_sortnext l1_sum l1_first l1_next l1_perm l1_map \
  --models m0 minv meq gpt --seeds 0 1 2 3 4 --jobs 4 \
  -- --steps 100000 --stop-after-gen 2000

# 5. 결과 표와 그래프
python analyze.py "runs/*" --out results.csv          # run별 + 조건별 요약
python analyze.py "runs/*" --gen-thr 0.90             # 판정 기준 바꿔 다시 계산
python plot.py "runs/l1_first_*" --out l1_first.png
```

## 규칙 적는 법

단계를 `>`로 잇는다. 모든 단계 뒤에 `(x mod n) + 1`이 한 번 적용된다 (Rule Lab과 같음).

```
A:asc                         정렬 (오름차순)
I:add(c=3) > J:mul(c=3)       괄호 안에 매개변수
AO:add(period=2|5|3)          목록은 | 로 구분
Z:even(fn=pow,fc=2)           조건 규칙의 변환
REF:first(op=add)             1층 배선 규칙
```

**Rule Lab 23개** (`A B C D E I J K M P Q R S AF AG X Y Z AB AJ AO BA BC`)는 Rule Lab 웹 화면과 같은 결과를 낸다.
모드와 매개변수 이름도 Rule Lab과 같다 (`src/rules.py`의 `CATALOG`).

**1층 배선 규칙 `REF`**: y_i = T(x_i, 참조값_i)

| 모드 | 참조값 | 대칭 (n=50, m=5에서 계산값) |
|---|---|---|
| `sortnext` | 정렬 후 오른쪽 이웃 (출력도 정렬 순서) | 불변 S₅ (120) |
| `sum` | 전체 합 | 등변 S₅ (120) |
| `first` | 첫 칸 | 등변 S₄ (24) |
| `next` | 오른쪽 이웃 (순환) | 등변 Z₅ (5) |
| `perm` | 고정 순열 π(i), 자기 자신 제외 | π에 따라 Z₅(5) 또는 6 |
| `map` | 고정 함수 g(i), 자기 자신 제외, 겹침 허용 | seed에 따라 1~6 (seed 0, 2는 1) |

`op=add`는 T(a,b)=a+b, `op=table`은 무작위 연산 표(`tseed`). `perm`·`map`의 배선은 `seed`로 바꾼다.

**이름 붙인 규칙** (`python -m src.rules --list`): 대칭 예시(`sort`, `add_c`, `add_total`, `add_first`, `nbsum`, `add_pos`),
1층(`l1_*`, 연산 표는 `l1t_*`), RQ2 사슬(`c_*`), 대조군(`ctrl_*`).

**2층 무작위 합성**:

```bash
python sample_rules.py --count 40 --min-steps 1 --max-steps 4 --out rules/layer2.json
python sweep.py --data ... --rules-file rules/layer2.json --models m0 --seeds 0 1 2
```

상수 출력, 유효 깊이 0(이름만 바꾼 항등·고정 순서 바꾸기), 원소별 표 찾기 규칙, 출력이 같은 중복 규칙은 빠진다
(`--keep-trivial`, `--keep-pointwise`로 남길 수 있음).

## 규칙 성질 (자동 계산)

`python -m src.rules <규칙>`과 run마다 저장되는 `rule_info.json`.

- **대칭군**: 입력 순서를 바꾸는 순열 σ (m!개)를 모두 넣어 본다. (σx)_i = x_{σ(i)}.
  - `inv_size`: f(σx) = f(x)인 σ의 수
  - `eq_size`: f(σx) = σf(x)인 σ의 수
  - `permuting_size`: f(σx) = τf(x)인 τ가 있는 σ의 수 (불변·등변을 포함하는 넓은 정의)
- **의존 범위**: 입력 j를 다른 값으로 바꿨을 때 출력 i가 바뀌는지로 영향 행렬을 만들고, 출력 한 칸이 의존하는 입력 칸 수를 센다.
- **유효 깊이**: 각 단계를 `identity`, `fixed_perm`(값과 무관한 고정 순서 바꾸기), `relabel`(위치와 무관한 원소별 일대일 변환),
  `elementwise_lossy`, `core`로 분류하고, `core`와 `elementwise_lossy`만 센다.
  연속한 core 두 개가 하나로 합쳐지는지는 보지 않으므로 상한값이다.

모두 무작위 입력 200개(`--probe`)로 확인한 값이다.

## 모델 (`--model`)

| 이름 | 구조 | 모델이 보장하는 대칭 |
|---|---|---|
| `m0` | 병렬 예측. 입력 위치 임베딩 + 출력 질의 칸 m개 | 없음 |
| `minv` | 병렬 예측. 입력 위치 임베딩 없음 + 출력 질의 칸 | 입력 순서에 불변 |
| `meq` | 병렬 예측. 입력 위치 임베딩 없음, 입력 i번째 자리에서 y_i 예측 | 등변 |
| `m0eq` | `meq`에 입력 위치 임베딩을 넣은 대조군 | 없음 |
| `gpt` | decoder-only GPT, `BOS x SEP y`, 출력 쪽에만 loss (RJ와 같은 꼴) | 없음 |

기본: 2층, d=128, head 4, pre-LN, GELU MLP, dropout 없음. `gpt`만 임베딩과 출력 head를 공유한다 (`--tie`로 변경).

## 학습 설정 (`train.py`)

기본값: AdamW lr 1e-3, betas (0.9, 0.98), weight decay 1.0(모든 파라미터), warmup 100 후 고정 학습률,
full batch(`--batch-size -1`), 100 step마다 평가 + 1·2·5·10·20·… step에서 추가 평가.

- `--fit-thr 0.99`: t_fit = train 수열 전체 일치 정확도가 처음 이 값에 도달한 step
- `--gen-thr 0.95`: t_gen = test 수열 전체 일치 정확도가 처음 이 값에 도달한 step
- `--sym-thr 0.05`: t_sym = 대칭 위반(어긋난 출력 칸 비율)이 처음 이 값 이하가 된 step
- `--stop-after-gen K`: t_gen 이후 K step 더 하고 멈춤
- `--budget-mult B`: t_fit의 B배까지 t_gen이 없으면 멈춤 (미도달로 기록)
- `--ckpt-log`: 1·2·5·10·20·… step 체크포인트 저장 (내부 분석용)
- `--resume runs/<run>/ckpt_last.pt --steps 300000`: 이어서 학습

정확도는 모두 자유 예측 기준이다(`gpt`는 greedy 생성). teacher forcing 정확도는 `*_exact_tf` 열에 따로 있다.

## run 출력

```
runs/<규칙>_<모델>_<데이터>_wd<wd>_s<seed>/
  config.json  rule_info.json  metrics.csv  summary.json  ckpt_last.pt
```

`metrics.csv` 열: step, lr, elapsed, train/test loss·칸 정확도·수열 정확도, teacher-forced 수열 정확도,
param_norm, emb_norm, sym_viol_tok(어긋난 칸 비율), sym_viol_seq(수열 하나라도 어긋난 비율), sym_viol_tv(출력 분포 차이).
대칭 위반은 규칙의 대칭군(불변 또는 등변)에서 항등원을 뺀 σ 8개와 test 입력 512개로 잰다. 대칭이 없는 규칙은 비어 있다.

`analyze.py`는 metrics.csv에서 t_fit, t_gen, t_sym을 다시 계산하므로 판정 기준을 바꿔 민감도를 볼 수 있다.
미도달 run은 `censored=1`, `t_gen` 자리에 마지막 step이 들어간다.

## 파일

| 파일 | 역할 |
|---|---|
| `src/rules.py` | 규칙 정의·합성, 성질 계산, 2층 표본 추출 |
| `src/data.py` | 값 집합 단위 split |
| `src/model.py` | 모델 5종 |
| `src/metrics.py` | 정확도, 대칭 위반, norm |
| `train.py` | 학습 |
| `sweep.py` | 여러 조건 실행 (`--jobs`로 병렬, 끝난 run 건너뜀, 중단된 run 이어서) |
| `sample_rules.py` | 2층 규칙 목록 만들기 |
| `analyze.py` | 결과 표 |
| `plot.py` | 학습 곡선 |

## 참고한 이전 작업

| 저장소 | 내용 |
|---|---|
| [survibo/jujetamgoo](https://github.com/survibo/jujetamgoo) | 제안서, 포스터, 발표 자료, 예비 실험(a+b mod 87) |
| [survibo/RUN-jutam](https://github.com/survibo/RUN-jutam) | encoder-decoder 정렬 모델, relation-complete split, test strata, sweep |
| [survibo/RJ](https://github.com/survibo/RJ) | decoder-only GPT 정렬 모델, 6개 과제(shift 계열 포함) |
