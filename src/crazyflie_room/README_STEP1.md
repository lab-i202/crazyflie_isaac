# Step 1 Crazyflie refactor files

Copy these files into `src/kite-isaac/` on the `development` branch.

New entry point:

```powershell
C:\isaac-lab\IsaacLab\isaaclab.bat -p C:\Users\Admin\Documents\Github\awes-isaac\src\kite-isaac\scenario_launcher.py
```

This step adds:

- Scenario launcher GUI
- Scenario registry
- Crazyflie room profile
- Crazyflie profile validator
- Crazyflie room scene runner
- Telemetry writer
- Local USD fallback folder for Crazyflie

RGB capture is deliberately disabled in the Step 1 runner. Camera prims are created, but capture folders receive `CAMERA_CAPTURE_DISABLED.txt`.
