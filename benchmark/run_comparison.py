"""Main Execution Script for OmniVoice Benchmark Comparison.

Executes each benchmark mode in an isolated Python subprocess to ensure clean
environment variables (e.g. MIOPEN_FIND_MODE) and unpolluted PyTorch / ROCm state.
"""

import os
import sys
import json
import argparse
import subprocess

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Run OmniVoice comparison benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Run only Style 1 to verify execution.")
    parser.add_argument("--runs", type=int, default=1, help="Number of timed runs per style (default 1).")
    parser.add_argument("--modes", nargs="+", default=["hybrid", "all_gpu"], help="Modes to test.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== OMNIVOICE AMD BENCHMARK SUITE ===")
    print(f"Modes: {args.modes}")
    print(f"Runs per style: {args.runs} (Dry-run: {args.dry_run})")
    print(f"Output directory: {args.out_dir}")

    all_results = {}
    for mode in args.modes:
        print(f"\n=======================================================")
        print(f">>> Launching isolated subprocess for mode: {mode} <<<")
        print(f"=======================================================")
        mode_json = os.path.join(args.out_dir, f"raw_{mode}.json")

        cmd = [
            sys.executable,
            "-m",
            "benchmark.benchmark_runner",
            "--mode",
            mode,
            "--out-dir",
            args.out_dir,
            "--runs",
            str(args.runs),
            "--save-json",
            mode_json,
        ]
        if args.dry_run:
            cmd.extend(["--limit", "1"])

        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess for mode '{mode}' failed with returncode {result.returncode}")

        with open(mode_json, "r", encoding="utf-8") as f:
            all_results[mode] = json.load(f)

    # Save aggregated raw results
    raw_path = os.path.join(args.out_dir, "raw_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Full benchmark complete. Aggregated results saved to: {raw_path}")


if __name__ == "__main__":
    main()
