"""Rigorous Benchmark Runner for OmniVoice on AMD ROCm.

Implements 4-Way Ablation Matrix:
- Config A (gpu_default_fast): GPU Tokenizer + Default SDP (Memory-Efficient/Flash) + MIOPEN_FIND_MODE=FAST (Baseline)
- Config B (gpu_math_fast):    GPU Tokenizer + Math SDP + MIOPEN_FIND_MODE=FAST
- Config C (cpu_default_fast): CPU Tokenizer + Default SDP + MIOPEN_FIND_MODE=FAST
- Config D (cpu_math):         CPU Tokenizer + Math SDP (Our Decoupled Hybrid Fix)

Features:
- N runs per style with 1 warmup
- Reports Median, Min, Max, Mean, Std, and individual run times
- Pre- and post-GPU synchronization (torch.cuda.synchronize) with time.perf_counter()
- Cold-start model load time tracking
- Platform and hardware metadata recording
"""

import os
import sys
import time
import json
import platform
import argparse
import soundfile as sf
import numpy as np
import torch

REF_AUDIO_DEFAULT = r"K:\Cloned_Voices\master_ref_clip.wav"
REF_TEXT_DEFAULT = "สวัสดีครับ วันนี้เรามาเริ่มต้นอัปเดตโปรเจกต์ใหม่ในสตูดิโอส่วนตัวกันครับ ไม่ว่าจะเป็นในเรื่องของระบบเสียงออดิโอ ที่ต้องการความคมชัดระดับโปร"
SAMPLE_RATE = 24000

MODE_CONFIGS = {
    "gpu_default_fast": {
        "label": "Config A: GPU Tokenizer + Default SDP (+ FAST)",
        "tokenizer_device": "cuda",
        "math_sdp": False,
        "description": "Baseline All-GPU with MIOpen FAST and default attention",
    },
    "gpu_math_fast": {
        "label": "Config B: GPU Tokenizer + Math SDP (+ FAST)",
        "tokenizer_device": "cuda",
        "math_sdp": True,
        "description": "All-GPU with Math SDP to evaluate attention impact on GPU",
    },
    "cpu_default_fast": {
        "label": "Config C: CPU Tokenizer + Default SDP (+ FAST)",
        "tokenizer_device": "cpu",
        "math_sdp": False,
        "description": "CPU Tokenizer offload with default attention",
    },
    "cpu_math": {
        "label": "Config D: CPU Tokenizer + Math SDP (Hybrid Fix)",
        "tokenizer_device": "cpu",
        "math_sdp": True,
        "description": "Full Decoupled Hybrid: CPU Tokenizer + Math SDP",
    },
}

# Aliases for convenience
MODE_ALIASES = {
    "all_gpu": "gpu_default_fast",
    "hybrid": "cpu_math",
}


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode.lower(), mode.lower())


def compute_metrics(wall_time: float, audio_samples: int, sample_rate: int = SAMPLE_RATE, peak_vram_bytes: int = 0) -> dict:
    """Single-run metrics calculation."""
    audio_dur = audio_samples / sample_rate if sample_rate > 0 else 0.0
    rtf = wall_time / audio_dur if audio_dur > 0 else 0.0
    speedup_x = audio_dur / wall_time if wall_time > 0 else 0.0
    peak_vram_mb = peak_vram_bytes / (1024 * 1024) if peak_vram_bytes > 0 else 0.0
    return {
        "wall_time_s": round(wall_time, 3),
        "audio_duration_s": round(audio_dur, 3),
        "rtf": round(rtf, 3),
        "speedup_x": round(speedup_x, 2),
        "realtime_x": round(speedup_x, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }


def compute_run_stats(times: list[float], audio_samples: int, sample_rate: int = SAMPLE_RATE, peak_vram_bytes: int = 0) -> dict:
    """Multi-run statistical metrics (Median, Min, Max, Mean, Std)."""
    valid_times = [t for t in times if t > 0]
    audio_dur = audio_samples / sample_rate if sample_rate > 0 else 0.0

    if not valid_times:
        return {
            "runs_count": 0,
            "individual_runs_s": [],
            "median_time_s": 0.0,
            "min_time_s": 0.0,
            "max_time_s": 0.0,
            "mean_time_s": 0.0,
            "std_time_s": 0.0,
            "wall_time_s": 0.0,
            "audio_duration_s": round(audio_dur, 3),
            "rtf": 0.0,
            "speedup_x": 0.0,
            "realtime_x": 0.0,
            "peak_vram_mb": 0.0,
        }

    median_t = float(np.median(valid_times))
    min_t = float(np.min(valid_times))
    max_t = float(np.max(valid_times))
    mean_t = float(np.mean(valid_times))
    std_t = float(np.std(valid_times))

    rtf = median_t / audio_dur if audio_dur > 0 else 0.0
    realtime_x = audio_dur / median_t if median_t > 0 else 0.0
    peak_vram_mb = peak_vram_bytes / (1024 * 1024) if peak_vram_bytes > 0 else 0.0

    return {
        "runs_count": len(valid_times),
        "individual_runs_s": [round(t, 3) for t in valid_times],
        "median_time_s": round(median_t, 3),
        "min_time_s": round(min_t, 3),
        "max_time_s": round(max_t, 3),
        "mean_time_s": round(mean_t, 3),
        "std_time_s": round(std_t, 3),
        "wall_time_s": round(median_t, 3),  # default to median for robust comparisons
        "audio_duration_s": round(audio_dur, 3),
        "rtf": round(rtf, 3),
        "speedup_x": round(realtime_x, 2),
        "realtime_x": round(realtime_x, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }


def load_model_for_mode(mode: str):
    """Load and configure OmniVoice model according to ablation matrix."""
    norm_mode = normalize_mode(mode)
    if norm_mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODE_CONFIGS.keys())}")

    cfg = MODE_CONFIGS[norm_mode]
    from omnivoice import OmniVoice

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Configure attention SDP according to ablation matrix
    if cfg["math_sdp"]:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print(f"[Attention SDP] Forced Math SDP (Flash/MemEfficient disabled)")
    else:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print(f"[Attention SDP] Default Attention (Flash/MemEfficient enabled)")

    print(f"[Load] Loading model for {cfg['label']}...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)

    # Route tokenizer device
    target_tok_device = cfg["tokenizer_device"]
    if hasattr(model, "audio_tokenizer") and model.audio_tokenizer is not None:
        if target_tok_device == "cpu":
            model.audio_tokenizer = model.audio_tokenizer.to("cpu")
            print(f"[Tokenizer] Assigned audio_tokenizer -> CPU")
        elif target_tok_device == "cuda" and device == "cuda":
            model.audio_tokenizer = model.audio_tokenizer.to("cuda")
            print(f"[Tokenizer] Assigned audio_tokenizer -> CUDA (GPU)")

    return model, cfg


def run_benchmark_for_mode(
    mode: str,
    styles: list[dict],
    out_dir: str,
    ref_audio: str = REF_AUDIO_DEFAULT,
    ref_text: str = REF_TEXT_DEFAULT,
    num_runs: int = 5,
) -> dict:
    """Execute benchmark across all styles for a single mode with multi-run statistics."""
    norm_mode = normalize_mode(mode)
    wav_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"BENCHMARK RUN: {norm_mode.upper()} | Styles={len(styles)} | Runs/Style={num_runs}")
    print(f"MIOPEN_FIND_MODE in env: {os.environ.get('MIOPEN_FIND_MODE', '<unset>')}")
    print(f"=======================================================")

    t_load_start = time.perf_counter()
    model, cfg = load_model_for_mode(norm_mode)
    load_time_s = time.perf_counter() - t_load_start
    print(f"Model load (cold-start) completed in {load_time_s:.2f}s")

    print("\nCreating voice clone prompt from reference...")
    t_prompt_start = time.perf_counter()
    prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    prompt_time_s = time.perf_counter() - t_prompt_start
    print(f"Voice clone prompt created in {prompt_time_s:.2f}s")

    # Warmup run
    print("\n[Warmup] Running warmup synthesis...")
    warmup_text = "ทดสอบการทำงานเพื่อวอร์มไดรเวอร์กราฟิกและเคอร์เนล"
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_w0 = time.perf_counter()
        _ = model.generate(text=warmup_text, voice_clone_prompt=prompt, language="th")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[Warmup] Finished in {time.perf_counter() - t_w0:.2f}s")
    except Exception as e:
        print(f"[Warmup] Warning: {e}")

    style_results = []

    for idx, style in enumerate(styles, 1):
        print(f"\n--- [{idx}/{len(styles)}] {style['name']} ({style['id']}) ---")
        text = style["voice_text"]
        print(f"Text: \"{text}\"")

        run_times = []
        last_audio_data = None
        peak_vram = 0

        for r in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            try:
                audio = model.generate(text=text, voice_clone_prompt=prompt, language="th")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                run_times.append(elapsed)
                last_audio_data = audio[0]

                if torch.cuda.is_available():
                    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())

                print(f"  Run {r+1}/{num_runs}: {elapsed:.3f}s")
            except Exception as err:
                print(f"  Run {r+1} FAILED with error: {err}")
                run_times.append(-1.0)

        stats = compute_run_stats(
            times=run_times,
            audio_samples=len(last_audio_data) if last_audio_data is not None else 0,
            sample_rate=SAMPLE_RATE,
            peak_vram_bytes=peak_vram,
        )

        wav_filename = f"{norm_mode}_{style['id']}.wav"
        if last_audio_data is not None:
            out_wav_path = os.path.join(wav_dir, wav_filename)
            sf.write(out_wav_path, last_audio_data, SAMPLE_RATE)

        entry = {
            "id": style["id"],
            "name": style["name"],
            "style_type": style["style_type"],
            "mood": style.get("mood", ""),
            "mode": norm_mode,
            "voice_text": text,
            "display_text": style["display_text"],
            "wav_file": wav_filename if last_audio_data is not None else None,
            **stats,
            "status": "success" if stats["runs_count"] > 0 else "failed",
        }
        print(f"  -> Median: {stats['median_time_s']}s (Min: {stats['min_time_s']}s, Max: {stats['max_time_s']}s, Std: {stats['std_time_s']}s) | RTF: {stats['rtf']} ({stats['realtime_x']}x realtime)")
        style_results.append(entry)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "mode": norm_mode,
        "label": cfg["label"],
        "description": cfg["description"],
        "cold_start_load_s": round(load_time_s, 2),
        "prompt_creation_s": round(prompt_time_s, 2),
        "runs_per_style": num_runs,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE", "<unset>"),
    }

    return {
        "metadata": metadata,
        "styles": style_results,
    }


def main():
    from benchmark.benchmark_corpus import BENCHMARK_STYLES

    parser = argparse.ArgumentParser(description="Standalone runner for OmniVoice benchmark ablation mode.")
    parser.add_argument("--mode", type=str, required=True, choices=list(MODE_CONFIGS.keys()) + list(MODE_ALIASES.keys()), help="Mode to execute.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for wavs and json.")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per style (default 5).")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of styles (0 = all).")
    parser.add_argument("--save-json", type=str, default="", help="Path to save mode JSON output.")
    args = parser.parse_args()

    styles = BENCHMARK_STYLES[:args.limit] if args.limit > 0 else BENCHMARK_STYLES
    results = run_benchmark_for_mode(
        mode=args.mode,
        styles=styles,
        out_dir=args.out_dir,
        num_runs=args.runs
    )

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[Done] Mode '{args.mode}' results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
