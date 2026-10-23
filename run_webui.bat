@echo off
title OmniVoice WebUI - AMD Radeon ROCm
cd /d "%~dp0"

set TORCH_BLAS_PREFER_HIPBLASLT=0
set HSA_OVERRIDE_GFX_VERSION=

echo =====================================================================
echo           Starting OmniVoice WebUI with AMD GPU Acceleration
echo           Target: AMD Radeon (RDNA 4 / RDNA 3 / ROCm)
echo           WebUI URL: http://127.0.0.1:8001
echo =====================================================================
echo.

if exist ".venv_rocm\Scripts\python.exe" (
    ".venv_rocm\Scripts\python.exe" -m omnivoice.cli.demo --ip 127.0.0.1 --port 8001
) else (
    echo [ERROR] Virtual environment '.venv_rocm' not found!
    echo Please run 'install_amd.bat' first.
)

pause
