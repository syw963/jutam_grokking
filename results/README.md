# 실험 결과 원본 (tgz)

GPU 서버에서 받은 결과 묶음. 각 파일 안에 run별 `config.json`, `metrics.csv`(step별 정확도), `summary.json`(t_fit, t_gen 등)이 있다.
체크포인트(.pt)는 크기 때문에 빠져 있다. 푸는 법: `tar xzf E12_prediction_results.tgz`

- E05~E07: 경계 지도, 배선 12개(a+b), 교환법칙 제거(a+3b)
- E08, E18: coverage 조절 split
- E09, E19, E20, E26, E27, E29: 합성 규칙(RQ2)
- E10, E10b, E28: GPT
- E11*: 체크포인트 내부 분석 (E11_checkpoints_metrics는 학습 곡선만, 체크포인트 원본은 4090 볼륨 서버 /workspace/jutam_e11)
- E12~E16, E21~E25: 칸당 관찰 수와 필요한 예제 수 법칙
- E17: 무작위 연산표 (E16_E17 안)
- 해석과 표: docs/결과_정리.md, docs/실험_기록.md
