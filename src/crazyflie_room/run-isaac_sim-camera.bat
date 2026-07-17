@echo off

if /i "%~1" neq "hidden" (
    powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%~f0' -ArgumentList 'hidden' -WindowStyle Hidden"
    exit /b
)

call "C:\Users\Admin\anaconda3\condabin\conda.bat" activate env_isaaclab

python "C:\Users\Admin\Documents\Github\crazyflie_isaac\src\crazyflie_room\last_frame_previewer.py"