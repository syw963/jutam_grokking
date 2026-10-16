"""여러 조건을 한꺼번에 실행한다.

예:
    # 1층 규칙 가족 × 모델 3개 × seed 5개
    python sweep.py --data data/n50_m5_tr21188_te20000_o1_s0 \
        --rules l1_sortnext l1_sum l1_first l1_next l1_perm l1_map \
        --models m0 minv meq --seeds 0 1 2 3 4 --jobs 4 -- --steps 100000 --stop-after-gen 2000

    # sample_rules.py가 만든 규칙 목록 사용
    python sweep.py --data ... --rules-file rules/layer2.json --models m0 --seeds 0 1 2

'--' 뒤 인자는 train.py에 그대로 넘긴다. 이미 끝난 run(summary.json에 stop_reason 있음)은 건너뛴다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import rule_name  # noqa: E402


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="split 디렉터리 (여러 개 가능)")
    ap.add_argument("--rules", nargs="*", default=[], help="규칙 스펙 또는 프리셋 이름")
    ap.add_argument("--rules-file", help="sample_rules.py 출력 JSON")
    ap.add_argument("--models", nargs="+", default=["m0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--wds", nargs="+", type=float, default=[1.0])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--jobs", type=int, default=1, help="동시에 돌릴 학습 수")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    rules = list(a.rules)
    if a.rules_file:
        rules += [r["spec"] for r in json.loads(Path(a.rules_file).read_text())]
    if not rules:
        sys.exit("--rules 또는 --rules-file이 필요합니다")

    jobs = []
    for data, rule, model, wd, seed in itertools.product(a.data, rules, a.models, a.wds, a.seeds):
        name = f"{rule_name(rule)}_{model}_{Path(data).name}_wd{wd:g}_s{seed}"
        run_dir = Path(a.runs_dir) / name
        summ = run_dir / "summary.json"
        if summ.exists() and json.loads(summ.read_text()).get("stop_reason"):
            continue
        cmd = [sys.executable, "train.py", "--data", data, "--rule", rule, "--model", model,
               "--wd", str(wd), "--seed", str(seed), "--runs-dir", a.runs_dir, "--name", name, "--quiet"]
        if (run_dir / "ckpt_last.pt").exists():
            cmd = [sys.executable, "train.py", "--resume", str(run_dir / "ckpt_last.pt"), "--quiet"]
            cmd += [x for x in extra if x != "--quiet"]
        else:
            cmd += extra
        jobs.append((name, cmd, run_dir))

    print(f"실행할 run {len(jobs)}개")
    if a.dry_run:
        for _, cmd, _ in jobs:
            print(" ".join(cmd))
        return

    Path(a.runs_dir).mkdir(parents=True, exist_ok=True)
    manifest = Path(a.runs_dir) / "sweep_log.jsonl"

    def run(job):
        name, cmd, run_dir = job
        t0 = time.time()
        log = Path(a.runs_dir) / f"{name}.log"
        with log.open("a") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=Path(__file__).parent)
        rec = dict(name=name, returncode=rc, seconds=round(time.time() - t0, 1), cmd=cmd)
        with manifest.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = "완료" if rc == 0 else f"실패({rc}) → {log}"
        print(f"[{status}] {name} {rec['seconds']}s", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rcs = list(ex.map(run, jobs))
    fails = sum(1 for r in rcs if r != 0)
    print(f"끝: {len(rcs) - fails}개 성공, {fails}개 실패")


if __name__ == "__main__":
    main()
