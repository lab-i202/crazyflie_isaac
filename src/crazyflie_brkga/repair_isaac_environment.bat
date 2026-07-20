@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\Admin\anaconda3\envs\env_isaaclab\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Isaac Lab Python was not found at:
    echo %PYTHON_EXE%
    echo Edit PYTHON_EXE in repair_isaac_environment.bat if your environment is elsewhere.
    exit /b 1
)

echo [INFO] Repairing the Isaac Lab Python camera dependencies...
echo [INFO] Python: %PYTHON_EXE%

"%PYTHON_EXE%" -m pip install --no-cache-dir --upgrade --force-reinstall "numpy==1.26.4" "opencv-python==4.11.0.86"
if errorlevel 1 (
    echo ERROR: Dependency repair failed.
    exit /b 1
)

echo.
echo [INFO] Installed versions:
"%PYTHON_EXE%" -c "import numpy, cv2; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__)"
if errorlevel 1 exit /b 1

echo.
"%PYTHON_EXE%" "%CD%\verify_isaac_environment.py"
if errorlevel 1 exit /b 1

echo.
echo [OK] Isaac environment repaired. Run run_isaac_integrity.bat next.
endlocal
