"""Benchmark Runner for OmniVoice.

Compares:
1. "all_gpu": Peterc3 approach (100% GPU, MIOPEN_FIND_MODE=FAST)
2. "hybrid": Our decoupled approach (GPU Diffusion + CPU Higgs Audio Tokenizer + Math SDP bypass)
"""

import os
import sys
import time
import json
import argparse
import soundfile as sf
import numpy as np
import torch

REF_AUDIO_DEFAULT = r"K:\Cloned_Voices\master_ref_clip.wav"
REF_TEXT_DEFAULT = "สวัสดีครับ วันนี้เรามาเริ่มต้นอัปเดตโปรเจกต์ใหม่ในสตูดิโอส่วนตัวกันครับ ไม่ว่าจะเป็นในเรื่องของระบบเสียงออดิโอ ที่ต้องการความคมชัดระดับโปร"
SAMPLE_RATE = 24000


def compute_metrics(wall_time: float, audio_samples: int, sample_rate: int = SAMPLE_RATE, peak_vram_bytes: int = 0) -> dict:
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


def load_model_for_mode(mode: str):
    """Load OmniVoice model configured for specified mode."""
    from omnivoice import OmniVoice

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if mode == "hybrid":
        # Our approach: GPU Diffusion + CPU Higgs Tokenizer + Math SDP
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[Load] Mode: HYBRID (GPU Diffusion + CPU Audio Tokenizer + Math SDP)")
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
        # Ensure tokenizer is on CPU
        if hasattr(model, "audio_tokenizer") and model.audio_tokenizer is not None:
            model.audio_tokenizer = model.audio_tokenizer.to("cpu")
            print("[Load] Confirmed model.audio_tokenizer is on CPU")

    elif mode == "all_gpu":
        # Peterc3 approach: 100% on GPU with MIOPEN_FIND_MODE=FAST
        os.environ["MIOPEN_FIND_MODE"] = "FAST"
        print("[Load] Mode: ALL_GPU (Peterc3 approach: 100% GPU with MIOPEN_FIND_MODE=FAST)")
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
        # In all_gpu mode, explicitly move the tokenizer to GPU
        if hasattr(model, "audio_tokenizer") and model.audio_tokenizer is not None and device == "cuda":
            try:
                model.audio_tokenizer = model.audio_tokenizer.to("cuda")
                print("[Load] Explicitly moved model.audio_tokenizer to CUDA (All-GPU)")
            except Exception as e:
                print(f"[Load] Warning: moving audio_tokenizer to CUDA raised: {e}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model


def run_benchmark_for_mode(
    mode: str,
    styles: list[dict],
    out_dir: str,
    ref_audio: str = REF_AUDIO_DEFAULT,
    ref_text: str = REF_TEXT_DEFAULT,
    num_runs: int = 1,
) -> list[dict]:
    """Execute benchmark across all styles for a single mode."""
    wav_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"BENCHMARK RUN: Mode={mode.upper()} | Styles={len(styles)}")
    print(f"=======================================================")

    t_load_start = time.perf_counter()
    model = load_model_for_mode(mode)
    load_time_s = time.perf_counter() - t_load_start
    print(f"Model load completed in {load_time_s:.2f}s")

    print("\nCreating voice clone prompt from master reference...")
    t_prompt_start = time.perf_counter()
    prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    prompt_time_s = time.perf_counter() - t_prompt_start
    print(f"Voice clone prompt created in {prompt_time_s:.2f}s")

    # Warmup run to warm MIOpen / HIP dispatch
    print("\n[Warmup] Running warmup synthesis...")
    warmup_text = "ทดสอบการทำงานเพื่อวอร์มไดรเวอร์กราฟิกและเคอร์เนล"
    try:
        t_w0 = time.perf_counter()
        _ = model.generate(text=warmup_text, voice_clone_prompt=prompt, language="th")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[Warmup] Finished in {time.perf_counter() - t_w0:.2f}s")
    except Exception as e:
        print(f"[Warmup] Warning: Warmup exception: {e}")

    results = []

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

        valid_times = [t for t in run_times if t > 0]
        if valid_times and last_audio_data is not None:
            mean_time = float(np.mean(valid_times))
            metrics = compute_metrics(
                wall_time=mean_time,
                audio_samples=len(last_audio_data),
                sample_rate=SAMPLE_RATE,
                peak_vram_bytes=peak_vram,
            )

            # Save sample audio wav
            out_wav_path = os.path.join(wav_dir, f"{mode}_{style['id']}.wav")
            sf.write(out_wav_path, last_audio_data, SAMPLE_RATE)

            result_entry = {
                "id": style["id"],
                "name": style["name"],
                "style_type": style["style_type"],
                "mood": style.get("mood", ""),
                "mode": mode,
                "voice_text": text,
                "display_text": style["display_text"],
                "wav_file": os.path.basename(out_wav_path),
                **metrics,
                "status": "success",
            }
            print(f"  -> Audio: {metrics['audio_duration_s']}s | Wall: {metrics['wall_time_s']}s | RTF: {metrics['rtf']} ({metrics['speedup_x']}x realtime)")
        else:
            result_entry = {
                "id": style["id"],
                "name": style["name"],
                "style_type": style["style_type"],
                "mood": style.get("mood", ""),
                "mode": mode,
                "voice_text": text,
                "display_text": style["display_text"],
                "wav_file": None,
                "wall_time_s": 0.0,
                "audio_duration_s": 0.0,
                "rtf": 0.0,
                "speedup_x": 0.0,
                "realtime_x": 0.0,
                "peak_vram_mb": 0.0,
                "status": "failed",
            }
        results.append(result_entry)

    # Clean up model to free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    from benchmark.benchmark_corpus import BENCHMARK_STYLES

    parser = argparse.ArgumentParser(description="Standalone runner for a single OmniVoice mode.")
    parser.add_argument("--mode", type=str, required=True, choices=["hybrid", "all_gpu"], help="Mode to execute.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for wavs and json.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per style.")
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
