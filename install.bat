@echo off
REM SD 3.5 Medium UI one-click installer for Windows.
REM
REM What it does:
REM   1. Checks for Python 3.12 via the py launcher, py -3.12.
REM   2. Creates a Python 3.12 virtual environment, .venv, in this folder.
REM   3. Installs PyTorch with CUDA support from the cu130 index, then
REM      the rest of the requirements, Diffusers 0.42, Gradio, ...
REM   4. Creates images_out\ for finished pictures.
REM
REM What it does NOT do:
REM   - Touch your Python 3.10 install, or any other Python you have.
REM     This app lives entirely in its own .venv, built from 3.12.
REM     Your other apps keep using whatever Python they already use.
REM   - Download the SD 3.5 Medium weights. Diffusers downloads the
REM     model from Hugging Face on the first Generate, about 6.5 GB
REM     on the default CLIP-only load, about 16 GB with T5 full,
REM     into the Hugging Face cache on your user profile -- not
REM     into this folder.
REM   - Accept the model's license for you, or make a token. SD 3.5
REM     Medium is GATED under the Stability AI Community License:
REM     YOU must accept that license on the model page AND paste a
REM     Hugging Face read token into the app's token box before the
REM     first Generate can succeed, see README.md, Before you start.
REM
REM Requirements: Windows, Python 3.12, see below, an NVIDIA GPU,
REM 12 GB is plenty -- the app defaults to the CLIP-only load plus
REM CPU offload, which is what makes it fit, about 20 GB free disk
REM for the first download plus room for pictures.

setlocal
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if not errorlevel 1 goto HavePy312
echo.
echo Python 3.12 was not found.
echo.
echo SD 3.5 UI needs Python 3.12, side by side with any other
echo Python you already have. Install it from python.org --
echo tick "Add python.exe to PATH" during setup -- then run
echo this file again. Nothing about your existing Python
echo installs is changed by this.
echo.
pause
exit /b 1

:HavePy312
echo Python 3.12 found:
py -3.12 --version

if exist .venv goto HaveVenv
echo Creating virtual environment from Python 3.12...
py -3.12 -m venv .venv
if not errorlevel 1 goto HaveVenv
echo Failed to create the virtual environment.
pause
exit /b 1

:HaveVenv
call .venv\Scripts\activate.bat
python -m pip install --no-warn-script-location --upgrade pip

echo Installing PyTorch with CUDA support - this is the big download...
pip install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cu130
if not errorlevel 1 goto TorchOk
echo.
echo PyTorch install failed. Check your internet connection and
echo that your GPU driver is up to date, then run this again.
echo.
pause
exit /b 1

:TorchOk
echo Installing the remaining packages...
pip install --no-warn-script-location -r requirements.txt
if not errorlevel 1 goto ReqOk
echo.
echo Package install failed. Run this again -- pip resumes
echo downloads it already finished.
echo.
pause
exit /b 1

:ReqOk
if not exist images_out mkdir images_out

echo.
echo Done. Double-click run.bat to start SD 3.5 Medium UI.
echo.
echo Before the first Generate: accept the Stability AI Community
echo License on the model's Hugging Face page and paste a Hugging
echo Face READ token into the token box in the app. The model is
echo gated -- without both, the download is refused.
echo.
pause
