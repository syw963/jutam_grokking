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


def is_done(run_dir: Path) -> bool:
    summ = run_dir / "summary.json"
    return summ.exists() and bool(json.loads(summ.read_text()).get("stop_reason"))


def multi_jobs(a, rules, extra):
    """--multi: (데이터, 모델)별로 run을 묶는다.

    이미 만들어진 묶음(runs/_groups/<id>/members.json)에 끝나지 않은 run이 있으면 그 묶음을 이어서 돌리고,
    어느 묶음에도 속하지 않은 run만 새 묶음으로 만든다.
    """
    from train_multi import group_id, member_name
    runs = Path(a.runs_dir)
    groups_dir = runs / "_groups"
    in_group, jobs = set(), []
    for mfile in sorted(groups_dir.glob("*/members.json")) if groups_dir.exists() else []:
        spec = json.loads(mfile.read_text())
        names = [mb["name"] for mb in spec["members"]]
        in_group.update(names)
        if all(is_done(runs / nm) for nm in names):
            continue
        ckpt = mfile.parent / "ckpt_last.pt"
        if ckpt.exists():
            cmd = [sys.executable, "train_multi.py", "--resume", str(ckpt), "--quiet"]
            cmd += [x for x in extra if x != "--quiet"]
        else:
            cmd = spec["cmd"]
        jobs.append((mfile.parent.name, cmd, mfile.parent))

    pending = {}
    for data, model in itertools.product(a.data, a.models):
        for rule, wd, seed in itertools.product(rules, a.wds, a.seeds):
            nm = member_name(rule, model, data, wd, seed)
            if nm in in_group or is_done(runs / nm):
                continue
            pending.setdefault((data, model), []).append(dict(rule=rule, seed=seed, wd=wd, name=nm))
    for (data, model), mbs in pending.items():
        for s in range(0, len(mbs), a.group_size):
            chunk = mbs[s:s + a.group_size]
            gid = group_id([mb["name"] for mb in chunk])
            gdir = groups_dir / gid
            cmd = [sys.executable, "train_multi.py", "--data", data, "--model", model,
                   "--members", str(gdir / "members.json"), "--runs-dir", a.runs_dir, "--quiet"] + extra
            if not a.dry_run:
                gdir.mkdir(parents=True, exist_ok=True)
                (gdir / "members.json").write_text(json.dumps(dict(members=chunk, cmd=cmd), ensure_ascii=False,
                                                              indent=1))
            jobs.append((gid, cmd, gdir))
    return jobs


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
    ap.add_argument("--multi", action="store_true",
                    help="같은 데이터·모델의 run을 묶어 train_multi.py 한 프로세스로 학습")
    ap.add_argument("--group-size", type=int, default=32, help="--multi에서 한 묶음의 최대 run 수")
    ap.add_argument("--gpus", help="쓸 GPU 번호 목록 (예: 0,1,2,3) 또는 all. 주면 --jobs는 GPU 하나당 동시 실행 수")
    ap.add_argument("--shard", help="여러 서버에 나눠 돌릴 때 i/N (예: 0/3, 1/3, 2/3). 작업 목록의 i번째, i+N번째...만 실행")
    a = ap.parse_args(argv)

    rules = list(a.rules)
    if a.rules_file:
        rules += [r["spec"] for r in json.loads(Path(a.rules_file).read_text())]
    if not rules:
        sys.exit("--rules 또는 --rules-file이 필요합니다")

    jobs = multi_jobs(a, rules, extra) if a.multi else []
    for data, rule, model, wd, seed in ([] if a.multi else
                                        itertools.product(a.data, rules, a.models, a.wds, a.seeds)):
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

    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    gpus = None
    if a.gpus:
        if a.gpus == "all":
            import torch
            gpus = [str(g) for g in range(torch.cuda.device_count())]
        else:
            gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
        if not gpus:
            sys.exit("GPU를 찾지 못했습니다")
    print(f"실행할 {'묶음' if a.multi else 'run'} {len(jobs)}개" + (f", GPU {','.join(gpus)} × {a.jobs}" if gpus else ""))
    if a.dry_run:
        for _, cmd, _ in jobs:
            print(" ".join(cmd))
        return

    Path(a.runs_dir).mkdir(parents=True, exist_ok=True)
    manifest = Path(a.runs_dir) / "sweep_log.jsonl"

    import os
    import queue
    slots = queue.Queue()
    for _ in range(a.jobs):
        for g in (gpus or [None]):
            slots.put(g)

    def run(job):
        name, cmd, run_dir = job
        gpu = slots.get()
        t0 = time.time()
        log = Path(a.runs_dir) / f"{name}.log"
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        try:
            with log.open("a") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=Path(__file__).parent, env=env)
        finally:
            slots.put(gpu)
        rec = dict(name=name, returncode=rc, seconds=round(time.time() - t0, 1), gpu=gpu, cmd=cmd)
        with manifest.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = "완료" if rc == 0 else f"실패({rc}) → {log}"
        print(f"[{status}] {name} {rec['seconds']}s" + (f" (GPU {gpu})" if gpu is not None else ""), flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=a.jobs * (len(gpus) if gpus else 1)) as ex:
        rcs = list(ex.map(run, jobs))
    fails = sum(1 for r in rcs if r != 0)
    print(f"끝: {len(rcs) - fails}개 성공, {fails}개 실패")


if __name__ == "__main__":
    main()
