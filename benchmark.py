#!/usr/bin/env python3
"""Benchmark script to measure OmniVoice inference speed and Real-Time Factor (RTF)
on AMD GPUs under Windows ROCm.
"""

import os
import sys
import time
import torch
import soundfile as sf
from omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device

def main():
    print("=" * 65)
    print("      OmniVoice AMD GPU Benchmark (Windows ROCm Hybrid Engine)")
    print("=" * 65)

    device = get_best_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"\n[Environment] Device: {device} [{gpu_name}] | Dtype: {dtype}")

    print("\nLoading OmniVoice model...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    test_sentences = [
        "OmniVoice is an omnilingual zero-shot text-to-speech model with diffusion language models.",
        "Running native AMD ROCm on Windows with RDNA 4 architecture delivers blazing fast speed.",
        "Testing speech synthesis performance and calculating the real-time factor."
    ]

    print("\nSynthesizing benchmark sentences...")
    total_gen_time = 0.0
    total_audio_duration = 0.0

    for i, s in enumerate(test_sentences, 1):
        tg0 = time.time()
        audio = model.generate(text=s, language="en")
        gen_time = time.time() - tg0
        audio_dur = len(audio[0]) / model.sampling_rate

        total_gen_time += gen_time
        total_audio_duration += audio_dur
        rtf = gen_time / audio_dur if audio_dur > 0 else 0

        print(f"[{i}/{len(test_sentences)}] Gen: {gen_time:.2f}s | Audio: {audio_dur:.2f}s | RTF: {rtf:.3f}x")

    avg_rtf = total_gen_time / total_audio_duration if total_audio_duration > 0 else 0
    speed_factor = total_audio_duration / total_gen_time if total_gen_time > 0 else 0

    print("\n" + "=" * 65)
    print(f"BENCHMARK RESULTS on {gpu_name}:")
    print(f" - Total Generation Time: {total_gen_time:.2f}s")
    print(f" - Total Audio Generated: {total_audio_duration:.2f}s")
    print(f" - Real-Time Factor (RTF): {avg_rtf:.3f}x")
    print(f" - Speed: {speed_factor:.2f}x FASTER than real-time speech!")
    print("=" * 65)

if __name__ == "__main__":
    main()
