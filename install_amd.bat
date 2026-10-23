@echo off
setlocal enabledelayedexpansion
title OmniVoice AMD Windows ROCm Installer
cd /d "%~dp0"

echo =====================================================================
echo        OmniVoice AMD Windows ROCm Installer (RDNA 4 / RDNA 3)
echo =====================================================================
echo.

:: 1. Check Python 3.12
echo [1/5] Checking Python 3.12...
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    python --version 2>&1 | findstr "3.12" >nul
    if %errorlevel% neq 0 (
        echo [ERROR] Python 3.12 is required for AMD ROCm on Windows.
        echo Please install Python 3.12 from python.org or run 'uv python install 3.12'.
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python
    )
) else (
    set PYTHON_CMD=py -3.12
)

echo Found Python 3.12: !PYTHON_CMD!

:: 2. Create Virtual Environment
echo.
echo [2/5] Creating virtual environment (.venv_rocm)...
if not exist ".venv_rocm" (
    !PYTHON_CMD! -m venv .venv_rocm
)
set VENV_PY=.venv_rocm\Scripts\python.exe
set VENV_PIP=.venv_rocm\Scripts\pip.exe

:: 3. Download & Install AMD ROCm 7.2.1 Wheels
echo.
echo [3/5] Installing AMD ROCm 7.2.1 PyTorch Packages...
!VENV_PIP! install --upgrade pip setuptools

echo Downloading and installing AMD ROCm core wheels...
!VENV_PIP! install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl

:: 4. Install Dependencies
echo.
echo [4/5] Installing OmniVoice dependencies...
!VENV_PIP! install transformers accelerate pydub gradio tensorboardX webdataset soundfile librosa "numpy<2.0.0"

:: 5. Verification
echo.
echo [5/5] Verifying AMD GPU Detection...
set TORCH_BLAS_PREFER_HIPBLASLT=0
!VENV_PY! -c "import torch; print('PyTorch ROCm:', torch.__version__); print('GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo =====================================================================
echo  Installation Complete! Run 'run_webui.bat' to start OmniVoice WebUI.
echo =====================================================================
pause
