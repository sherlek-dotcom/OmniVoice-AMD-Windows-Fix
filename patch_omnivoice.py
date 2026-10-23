#!/usr/bin/env python3
"""Standalone patcher for k2-fsa/OmniVoice to enable high-speed AMD GPU inference
under Windows ROCm by fixing the MIOpen workspace buffer bug and AOTriton JIT overhead.
"""

import os
import sys
import argparse

TARGET_BLOCK = '''                # higgs-audio-v2-tokenizer does not support MPS
                # (output channels > 65536)
                tokenizer_device = (
                    "cpu" if str(model.device).startswith("mps") else model.device
                )
                model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                    audio_tokenizer_path, device_map=tokenizer_device
                )'''

REPLACEMENT_BLOCK = '''                # higgs-audio-v2-tokenizer does not support MPS (output channels > 65536)
                # and encounters severe MIOpen workspace buffer bugs on Windows ROCm.
                # Running audio_tokenizer on CPU bypasses MIOpen Conv1d fallback while
                # keeping the main LLM diffusion backbone on GPU.
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
                    # forces native math SDP which executes directly on AMD Tensor Cores.
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)

                model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                    audio_tokenizer_path, device_map=tokenizer_device
                )'''

def patch_file(target_file):
    if not os.path.isfile(target_file):
        print(f"[ERROR] Target file not found: {target_file}")
        return False

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    if "is_rocm_windows = (" in content:
        print("[INFO] File is already patched for AMD ROCm Windows!")
        return True

    if TARGET_BLOCK in content:
        content = content.replace(TARGET_BLOCK, REPLACEMENT_BLOCK, 1)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[SUCCESS] Successfully patched: {target_file}")
        return True
    else:
        print("[ERROR] Target code block not found in file. The OmniVoice version might be different.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Patch OmniVoice for AMD ROCm Windows acceleration.")
    parser.add_argument("--repo-path", default=".", help="Path to OmniVoice repository root.")
    args = parser.parse_args()

    candidates = [
        os.path.join(args.repo_path, "omnivoice", "models", "omnivoice.py"),
        os.path.join(args.repo_path, "models", "omnivoice.py"),
    ]

    found = False
    for c in candidates:
        if os.path.isfile(c):
            found = True
            patch_file(c)
            break

    if not found:
        print(f"[ERROR] Could not locate 'omnivoice/models/omnivoice.py' in {args.repo_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
