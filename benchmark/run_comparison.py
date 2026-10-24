"""Main Execution Script for OmniVoice Benchmark Comparison.

Executes each benchmark ablation mode in an isolated Python subprocess with
explicitly injected environment variables (e.g. MIOPEN_FIND_MODE=FAST)
to guarantee zero state leakage across modes.
"""

import os
import sys
import json
import argparse
import subprocess

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results")
DEFAULT_MODES = ["gpu_default_fast", "gpu_math_fast", "cpu_default_fast", "cpu_math"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run OmniVoice comparison benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Run only Style 1 with 1 run to verify execution.")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per style (default 5).")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="Ablation modes to test.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of styles (0 = all 8).")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    num_runs = 1 if args.dry_run else args.runs

    print(f"=======================================================")
    print(f"=== OMNIVOICE AMD BENCHMARK ABLATION SUITE ===")
    print(f"=======================================================")
    print(f"Modes to test: {args.modes}")
    print(f"Runs per style: {num_runs} (Dry-run: {args.dry_run})")
    print(f"Output directory: {args.out_dir}")

    all_results = {}
    for mode in args.modes:
        print(f"\n=======================================================")
        print(f">>> Launching isolated subprocess for mode: {mode} <<<")
        print(f"=======================================================")
        mode_json = os.path.join(args.out_dir, f"raw_{mode}.json")

        # Explicitly configure environment before subprocess launches
        child_env = os.environ.copy()
        if "fast" in mode.lower():
            child_env["MIOPEN_FIND_MODE"] = "FAST"
            print(f"[Env Injection] Injected MIOPEN_FIND_MODE=FAST for {mode}")
        else:
            child_env.pop("MIOPEN_FIND_MODE", None)

        child_env["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

        cmd = [
            sys.executable,
            "-m",
            "benchmark.benchmark_runner",
            "--mode",
            mode,
            "--out-dir",
            args.out_dir,
            "--runs",
            str(num_runs),
            "--save-json",
            mode_json,
        ]
        if args.dry_run or args.limit > 0:
            limit_val = 1 if args.dry_run else args.limit
            cmd.extend(["--limit", str(limit_val)])

        result = subprocess.run(cmd, env=child_env)
        if result.returncode != 0:
            print(f"[ERROR] Mode '{mode}' failed with return code {result.returncode}. Skipping...")
            continue

        if os.path.exists(mode_json):
            with open(mode_json, "r", encoding="utf-8") as f:
                all_results[mode] = json.load(f)

    # Save aggregated raw results
    raw_path = os.path.join(args.out_dir, "raw_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Full benchmark complete. Aggregated results saved to: {raw_path}")


if __name__ == "__main__":
    main()
