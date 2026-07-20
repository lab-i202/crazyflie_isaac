@echo off
setlocal
cd /d "%~dp0"
set "CRAZYFLIE_BRKGA_CONFIG=%CD%\config\smoke.json"
python train_mock.py
endlocal
