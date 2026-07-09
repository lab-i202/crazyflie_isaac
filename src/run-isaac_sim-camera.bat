@echo off

call C:\Users\Admin\anaconda3\condabin\conda.bat activate env_isaaclab
cls

call C:\isaac-lab\IsaacLab\isaaclab.bat -p C:\Users\Admin\Documents\Github\crazyflie_isaac\src\crazyflie_room.py --config C:\Users\Admin\Documents\Github\crazyflie_isaac\src\crazyflie_room.yaml

pause