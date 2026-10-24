# OmniVoice AMD Benchmark & Architectural Report: Hybrid Pipeline vs All-GPU (Peterc3 Method)

**Target Hardware:** AMD Radeon RX 9070 XT (16GB, RDNA 4 / gfx1201)  
**Environment:** Windows 11 Pro, PyTorch 2.9.1+rocm7.2.1, HIP 7.2  
**Comparison Reference:** [Peterc3-dev/omnivoice-gfx1150](https://github.com/Peterc3-dev/omnivoice-gfx1150) (GPD Pocket 4, Radeon 890M / gfx1150, Linux CachyOS)

---

## 1. Executive Summary

In Peterc3's landmark report ([omnivoice-gfx1150](https://github.com/Peterc3-dev/omnivoice-gfx1150)), running OmniVoice on AMD APUs required `MIOPEN_FIND_MODE=FAST` to work around MIOpen `GemmFwdRest` workspace=0 fallback warnings, reaching **49s for 20s output (RTF ~2.45)** on warm runs.

On **Windows ROCm with discrete AMD RDNA 4 (RX 9070 XT)**, running 100% on GPU encounters two critical bottlenecks:
1. **MIOpen Conv1d Workspace Buffer Overflow**: `higgs-audio-v2-tokenizer` decodes large audio channels, allocating zero/unsupported workspace buffers that cause fallback penalties and high VRAM overhead.
2. **AOTriton JIT Stalls**: Windows ROCm lacks precompiled AOTriton kernels for RDNA 4, triggering JIT compilation lag during Flash/MemEfficient attention.

### Our Decoupled Hybrid Architecture Solution:
Instead of forcing 100% GPU execution:
- **Diffusion Backbone on GPU (RX 9070 XT)** with `enable_math_sdp(True)` to bypass AOTriton JIT compilation and execute directly on AMD compute units.
- **Higgs Audio Tokenizer on CPU**: Completely sidesteps MIOpen Conv1d buffer bugs and frees up ~765 MB of VRAM.
- **Thai Phonetic & Tonal Scripting**: Eliminates tone marks and silent character artifacts, producing smooth natural speech.

### Key Benchmark Results:
- **Total Synthesized Audio:** 49.37s across 8 styles
- **Hybrid Total Wall Time:** 19.28s (**Overall RTF: 0.391 | 2.56x Realtime**)
- **All-GPU Total Wall Time:** 21.29s (Overall RTF: 0.431)
- **Direct Speedup:** Hybrid is consistently **1.10x faster** than All-GPU even with `MIOPEN_FIND_MODE=FAST`.
- **VRAM Savings:** **+765.4 MB VRAM Saved (31.6% reduction)**, enabling voice cloning alongside other local LLMs.

---

## 2. Full 8-Style Benchmark Matrix

| Style / Prompt Name | Audio (s) | Hybrid Time (s) | All-GPU Time (s) | Speedup | Hybrid RTF | Realtime X | VRAM Saved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ultra-Short Hook (3s) | 3.52s | 1.78s | 1.96s | **1.10x** | 0.506 | 1.97x | +776.3 MB |
| Technical Explainer (7s) | 4.74s | 2.10s | 2.41s | **1.15x** | 0.443 | 2.26x | +773.1 MB |
| Emotion: Excited / Punchy (10s) | 5.63s | 2.30s | 2.59s | **1.13x** | 0.409 | 2.45x | +770.8 MB |
| Emotion: Serious / Problem (12s) | 6.36s | 2.35s | 2.63s | **1.12x** | 0.370 | 2.70x | +771.1 MB |
| Phonetic Stress Test (15s) | 6.50s | 2.37s | 2.63s | **1.11x** | 0.364 | 2.75x | +774.5 MB |
| Emotion: Confident / Analytical (18s) | 6.75s | 2.46s | 2.73s | **1.11x** | 0.364 | 2.75x | +773.9 MB |
| Long-Form Tech Narrative (25s) | 12.03s | 4.24s | 4.34s | **1.03x** | 0.352 | 2.84x | +765.4 MB |
| Call-to-Action / Summary (6s) | 3.84s | 1.68s | 1.98s | **1.18x** | 0.438 | 2.28x | +777.0 MB |

---

## 3. Style Breakdown & Acoustic Findings

| Style ID | Description | Hybrid RTF | Speedup vs All-GPU | Acoustic Notes |
| --- | --- | --- | --- | --- |
| `style_1_hook` | 3s Fast Hook | 0.506 (1.97x) | 1.10x | Immediate voice onset; no initial clipping |
| `style_2_tech` | Tech Explainer | 0.443 (2.26x) | 1.15x | English loanwords ('Tensor Core', 'RDNA') clear |
| `style_3_excited` | Punchy / Excited | 0.409 (2.45x) | 1.13x | High pitch dynamics maintained across generation |
| `style_4_serious` | Serious / Problem | 0.370 (2.70x) | 1.12x | Stable lower-register prosody without buzz |
| `style_5_phonetic` | Phonetic Stress | 0.364 (2.75x) | 1.11x | Fully articulated Thai numbers and fitness terms |
| `style_6_confident` | Confident / Deep | 0.364 (2.75x) | 1.11x | Authoritative cadence with balanced breathing pause |
| `style_7_long_narrative` | 25s Long Narrative | 0.352 (2.84x) | 1.03x | Single-pass speech generated in 4.2s |
| `style_8_cta` | Call to Action | 0.438 (2.28x) | 1.18x | Punchy ending without tail cut-off |

---

## 4. VRAM Utilization Comparison

| Configuration | Diffusion Backbone | Audio Tokenizer | Peak VRAM Allocated | Memory Profile |
|---|---|---|---|---|
| **All-GPU (Peterc3 approach)** | CUDA (RX 9070 XT) | CUDA (RX 9070 XT) | **2425.4 MB** | High VRAM contention, Conv1d workspace allocations |
| **Hybrid (Our approach)** | CUDA (RX 9070 XT) | CPU (System RAM) | **1660.0 MB** | **+765.4 MB Saved (-31.6%)**, zero MIOpen Conv1d warnings |

---

## 5. Architectural Code Implementation

The optimization is implemented directly in `omnivoice/models/omnivoice.py`:

```python
# Windows ROCm Optimization: Decouple Audio Tokenizer to CPU
# and force Math SDP to bypass AOTriton JIT compilation on RDNA 4.
is_rocm_windows = (
    os.name == "nt" and getattr(torch.version, "hip", None) is not None
)
tokenizer_device = (
    "cpu"
    if (str(model.device).startswith("mps") or is_rocm_windows)
    else model.device
)

if is_rocm_windows:
    # AOTriton lacks precompiled kernels for RDNA4/gfx1201 on Windows ROCm,
    # which triggers heavy JIT compilation lag. Disabling mem_efficient and flash
    # forces native math SDP which executes directly on AMD compute units.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
    audio_tokenizer_path, device_map=tokenizer_device
)
```

---

## 6. How to Reproduce

```bash
# 1. Activate Windows ROCm venv
.venv_rocm\Scripts\activate

# 2. Run the automated 8-style benchmark suite in isolated subprocesses
python benchmark/run_comparison.py --runs 2

# 3. Generate this comparison report
python benchmark/generate_report.py
```

Generated audio samples are available under `benchmark_results/wavs/`.
