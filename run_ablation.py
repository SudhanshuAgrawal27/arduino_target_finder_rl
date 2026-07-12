#!/usr/bin/env python3
"""Parallel launcher for the multi-seed architecture ablation.

Trains every (architecture, seed) combination that isn't already complete,
up to --parallel jobs at a time. Each job is one `train.py` invocation pinned
to a single seed with a deterministic (timestamp-free) run directory, so a
completed job is detected by its epoch_<final> checkpoint and skipped on
re-run -- the launcher is safe to restart.

New fixed-code runs land in <run_name>_seed<seed> directories; the old
pre-fix runs (kept intact) use the _s<seed> suffix instead, so they never
collide and are never mistaken for completed new jobs.

    python3 run_ablation.py --dry-run           # list what would run
    python3 run_ablation.py --parallel 4        # run 4 at a time
    python3 run_ablation.py --seeds 43 44       # only some seeds

Runs on CPU (this project is CPU-pinned); each job's math libs are capped at
--threads so N parallel jobs don't oversubscribe the cores.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

CONF_DIR = Path("conf")
OUTPUT_ROOT = Path("trained_models")
LOG_DIR = OUTPUT_ROOT / "_ablation_logs"
FINAL_EPOCH = 150

# The nine ablation architectures (one config file each).
CONFIGS = [
    "config_train_h128_l2_hist4",   # baseline
    "config_train_h128_l2_hist3",
    "config_train_h64_l2_hist4",
    "config_train_h32_l2_hist4",
    "config_train_h16_l2_hist4",
    "config_train_h64_l3_hist4",
    "config_train_h64_l4_hist4",
    "config_train_h80_l2_hist4",
    "config_train_h96_l2_hist4",
]
DEFAULT_SEEDS = [42, 43, 44]


def run_name_of(config_name):
    return OmegaConf.load(CONF_DIR / f"{config_name}.yaml").output.run_name


def job_dir(run_name, seed):
    return OUTPUT_ROOT / f"{run_name}_seed{seed}"


def is_done(run_name, seed):
    return (job_dir(run_name, seed) / f"epoch_{FINAL_EPOCH}").exists()


def run_job(config_name, run_name, seed, threads):
    """Launch one training run to completion; return (run_name, seed, rc)."""
    log_path = LOG_DIR / f"{run_name}_seed{seed}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""          # CPU-pinned project
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    cmd = [
        sys.executable, "train.py",
        "--config-name", config_name,
        f"seeds=[{seed}]",
        "output.timestamp=false",             # deterministic, skippable dir name
    ]
    with open(log_path, "w") as lf:
        lf.write(f"$ {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return run_name, seed, proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=4, help="max concurrent jobs (default 4)")
    ap.add_argument("--threads", type=int, default=2, help="math-lib threads per job (default 2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="seeds to run (default 42 43 44)")
    ap.add_argument("--configs", nargs="+", default=CONFIGS, help="config names to run (default: all 9)")
    ap.add_argument("--dry-run", action="store_true", help="list jobs and exit")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], []
    for config_name in args.configs:
        run_name = run_name_of(config_name)
        for seed in args.seeds:
            (skipped if is_done(run_name, seed) else jobs).append((config_name, run_name, seed))

    for _, run_name, seed in skipped:
        print(f"skip (done): {run_name}_seed{seed}")
    print(f"\n{len(jobs)} job(s) to run, {args.parallel} at a time, {args.threads} threads each:")
    for _, run_name, seed in jobs:
        print(f"  {run_name}_seed{seed}")
    if args.dry_run or not jobs:
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_job, c, rn, s, args.threads): (rn, s) for c, rn, s in jobs}
        for fut in as_completed(futures):
            run_name, seed, rc = fut.result()
            print(f"[done] {run_name}_seed{seed}: {'OK' if rc == 0 else f'FAIL rc={rc}'}", flush=True)
            results.append((run_name, seed, rc))

    print("\n=== summary ===")
    n_ok = sum(1 for *_, rc in results if rc == 0)
    for run_name, seed, rc in sorted(results):
        print(f"  {run_name}_seed{seed}: {'OK' if rc == 0 else 'FAIL'}")
    print(f"{n_ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
