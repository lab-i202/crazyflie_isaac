@echo off
setlocal
cd /d "%~dp0"
set "ISAACLAB_BAT=C:\isaac-lab\IsaacLab\isaaclab.bat"
set "PYTHON_EXE=C:\Users\Admin\anaconda3\envs\env_isaaclab\python.exe"
if not exist "%ISAACLAB_BAT%" (
    echo ERROR: Isaac Lab launcher not found: %ISAACLAB_BAT%
    exit /b 1
)
"%PYTHON_EXE%" "%CD%\verify_isaac_environment.py"
if errorlevel 1 exit /b 1
"%ISAACLAB_BAT%" -p "%CD%\detector_capture_isaac.py"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
