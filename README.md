# 🚀 OmniVoice AMD Windows Fix: Unlock High-Speed Inference on AMD GPUs

[![AMD ROCm](https://img.shields.io/badge/AMD%20ROCm-7.2.1-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1%2Brocm7.2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Architecture-RDNA%204%20%7C%20RDNA%203-blue)](https://www.amd.com/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20x64-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Performance](https://img.shields.io/badge/Real--Time%20Factor-0.42x%20(2.4x%20Realtime)-brightgreen)](https://github.com/sherlek-dotcom/OmniVoice-AMD-Windows-Fix)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Zero-overhead Hybrid Pipeline patch for [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) running under native Windows with AMD ROCm.**  
> Fixes the severe **11x slowdown** and MIOpen buffer crash on modern AMD GPUs (tested on **Radeon RX 9070 XT**, RDNA 4 / `gfx1201`).

---

## ⚡ The Benchmark (Real Numbers)

Tested on **AMD Radeon RX 9070 XT** (16GB VRAM, RDNA 4) + Ryzen 16-Core CPU on Windows 11:

| Setup / Mode | Status / Warnings | Generation Time / Sentence | Full 6-Sentence Audio (33s) | Real-Time Factor (RTF) |
| :--- | :--- | :---: | :---: | :---: |
| **Vanilla GPU (ROCm Windows)** | ⚠️ `MIOpen Workspace 0` + `AOTriton JIT` | **25s – 60s+** | **11+ Minutes** | ~20.0x (Extremely Slow) |
| **CPU Only (16 Threads)** | ✅ oneDNN AVX | ~20s – 30s | ~1.5 – 2 Minutes | ~3.5x |
| **🚀 Hybrid Fix (This Repo)** | **✅ Clean (Zero Warnings, Zero Fallbacks)** | **~2.1s** | **~14.3 Seconds** | **0.42x (2.4x faster than speech!)** |

---

## 🔍 The Problem: Why Does OmniVoice Crawl on AMD Windows?

When users attempt to run OmniVoice on Windows with AMD ROCm (`device_map="cuda"`), generation stalls dramatically. We analyzed the execution trace down to kernel dispatches and identified **two critical bottlenecks**:

### 1. The MIOpen Workspace Buffer Bug (`size: 0`)
OmniVoice uses **[HiggsAudioV2TokenizerModel](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer)** (a deep 1D Convolutional Neural Audio Codec) to decode acoustic tokens into 24,000Hz waveform.  
On Windows ROCm, MIOpen's memory allocator fails to assign workspace memory, returning:
```
MIOpen(HIP): Warning [IsEnoughWorkspace] [GetSolutionsFallback WTI] Solver <GemmFwdRest>, workspace required: 362987520, provided ptr: 0000000000000000 size: 0
```
Because workspace is `0`, MIOpen **cannot execute Winograd or Direct Convolution**. It falls back to `GemmFwdRest` (a generic scalar GEMM solver), computing audio sample-by-sample and stalling generation for up to **11 minutes**.

### 2. AOTriton JIT Compilation Overhead on RDNA 4
On PyTorch ROCm, `scaled_dot_product_attention` defaults to memory-efficient attention powered by AOTriton. However, AOTriton lacks precompiled kernel binaries for newer architectures like RDNA 4 (`gfx1201`).  
This causes:
```
UserWarning: Using AOTriton backend for Efficient Attention forward...
```
Triggering repeated JIT compilation and synchronization stalls on all 32 diffusion steps across 28 transformer layers.

---

## 💡 The Solution: Hybrid Engine Architecture

Instead of fighting broken convolution drivers on GPU or accepting slow CPU generation, we split execution based on **algorithmic strengths**:

```
[ Input Text ]
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 🧠 LLM Diffusion Backbone (1B+ Params, 32 Steps)            │
│ ➔ Dispatched to GPU: AMD Radeon RX 9070 XT (FP16)           │
│ ➔ Forces Native PyTorch Math SDP (Bypasses AOTriton JIT)     │
│ ➔ Computes directly on AMD RDNA 4 Tensor Cores in ~1.9s     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              │ Lightweight Audio Tokens (few KB)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 🎵 HiggsAudioV2 Tokenizer (Neural Audio Codec Conv1D)       │
│ ➔ Dispatched to CPU (16 Cores, AVX/oneDNN)                  │
│ ➔ Decodes tokens into 24kHz waveform in just 0.05 seconds!  │
│ ➔ 100% immune to MIOpen Workspace 0 bug                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                     [ 24kHz Master Audio ]
```

---

## 🛠️ Quick Start (One-Click Setup on Windows)

### Prerequisites
1. **Windows 10 / 11 (64-bit)**
2. **AMD Radeon GPU**: RX 6000, RX 7000, or RX 9000 Series (RX 9070 / 9070 XT, RX 7900 XTX/XT, etc.)
3. **AMD Adrenalin Driver**: 25.x or 26.x+
4. **Python 3.12** (ROCm Windows wheels require Python 3.12)

### Automated Installation
Clone this repository and run `install_amd.bat`:
```cmd
git clone https://github.com/sherlek-dotcom/OmniVoice-AMD-Windows-Fix.git
cd OmniVoice-AMD-Windows-Fix
install_amd.bat
```
The script will:
1. Create a Python 3.12 virtual environment (`.venv_rocm`).
2. Download official AMD ROCm 7.2.1 wheels (`torch-2.9.1+rocm7.2.1`, `torchaudio`, SDK libraries).
3. Install OmniVoice and required dependencies.
4. Automatically apply the Hybrid Patch.

---

## 🚀 Running OmniVoice on AMD GPU

### Start the WebUI:
Double-click `run_webui.bat` or run:
```cmd
run_webui.bat
```
Access the Gradio WebUI at `http://127.0.0.1:8001`.

### Run Benchmark / Test Script:
```cmd
.venv_rocm\Scripts\python.exe benchmark.py
```

---

## 🧩 How to Apply to an Existing OmniVoice Repository

If you already have [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) cloned, you only need to modify **`omnivoice/models/omnivoice.py`**:

### Automated Patch:
```cmd
python patch_omnivoice.py --repo-path "C:\path\to\OmniVoice"
```

### Manual Code Diff:
In `omnivoice/models/omnivoice.py` around line 360, replace the tokenizer initialization with:

```python
# -------------------------------------------------------------------------
# AMD ROCm Windows Fix: Hybrid Tokenizer Dispatch & Math SDP
# -------------------------------------------------------------------------
is_rocm_windows = (
    os.name == "nt" and getattr(torch.version, "hip", None) is not None
)

# Route audio tokenizer to CPU on Windows ROCm to avoid MIOpen buffer=0 bug
tokenizer_device = (
    "cpu"
    if (str(model.device).startswith("mps") or is_rocm_windows)
    else model.device
)

if is_rocm_windows:
    # Disable AOTriton JIT overhead; force native HIP Math SDP on Tensor Cores
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
    audio_tokenizer_path, device_map=tokenizer_device
)
```

Also, set the stability flag before launching:
```cmd
set TORCH_BLAS_PREFER_HIPBLASLT=0
```

---

## 🇹🇭 สรุปสำหรับผู้ใช้งานชาวไทย (Thai Summary)

สำหรับผู้ใช้งานการ์ดจอค่ายแดง AMD บน Windows (โดยเฉพาะซีรีส์ **RX 7000 / RX 9000** สถาปัตยกรรม RDNA 3 / RDNA 4):
* **ปัญหาเดิม**: หากรันบน GPU ปกติ จะเจอบั๊ก `MIOpen Workspace 0` ในส่วนถอดรหัสคลื่นเสียง HiggsAudio ทำให้ระบบ fallback ไปรันแบบช้ามาก (ประโยคละเกือบ 1 นาที หรือช้าเป็น 10 นาที)
* **วิธีแก้ของโปรเจกต์นี้**: เราทำ **Hybrid Engine** โดยยกงานคำนวณสมองใหญ่ (Diffusion 1B+) ให้การ์ดจอ AMD ประมวลผลด้วยความเร็วสูงสุด (FP16 Math SDP) แล้วส่งโทเคนเสียงขนาดจิ๋วมาให้ CPU ถอดรหัสคลื่นเสียง Conv1D
* **ผลลัพธ์**: ได้ความเร็วในการสร้างเสียงระดับ **~2.1 วินาทีต่อประโยค (RTF 0.42x — เร็วกว่าเสียงพูดจริง 2.4 เท่า)** ลื่นไหล เสถียร ไร้ Error 100%

---

## 📜 Credits & Acknowledgments
* **OmniVoice**: Developed by [k2-fsa](https://github.com/k2-fsa/OmniVoice) (Han Zhu and contributors).
* **AMD ROCm Team**: For native Windows ROCm SDK and PyTorch 2.9 wheels.
* **HiggsAudio**: Developed by [eustlb](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer).

---

## 📄 License
This fix and documentation are licensed under the **Apache-2.0 License**, consistent with the upstream OmniVoice project.
