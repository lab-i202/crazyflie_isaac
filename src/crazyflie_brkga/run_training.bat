@echo off
setlocal
cd /d "%~dp0"

set "ISAACLAB_BAT=C:\isaac-lab\IsaacLab\isaaclab.bat"
set "PYTHON_EXE=C:\Users\Admin\anaconda3\envs\env_isaaclab\python.exe"

if not exist "%ISAACLAB_BAT%" (
    echo ERROR: Isaac Lab launcher was not found at:
    echo %ISAACLAB_BAT%
    echo Edit run_training.bat if Isaac Lab is installed elsewhere.
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: Isaac Lab Python was not found at:
    echo %PYTHON_EXE%
    exit /b 1
)

"%PYTHON_EXE%" "%CD%\verify_isaac_environment.py"
if errorlevel 1 (
    echo.
    echo Run repair_isaac_environment.bat, then try again.
    exit /b 1
)

"%ISAACLAB_BAT%" -p "%CD%\train_isaac.py"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
