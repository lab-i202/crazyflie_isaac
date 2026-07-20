# GUI, scenario preview, and landmark detector

## Main GUI

Start the application with:

```powershell
cd C:\Users\Admin\Documents\Github\crazyflie_isaac\src\crazyflie_brkga
.\run_gui.bat
```

The GUI is the main entry point for the project. It edits one JSON configuration that is shared by scenario preview, detector configuration, integrity tests, and BRKGA training.

### Main actions

- **Open** loads an existing JSON file.
- **Save** and **Save current JSON** overwrite the currently loaded JSON file.
- **Save As** creates a new JSON configuration.
- **Validate** checks the full scenario and training configuration.
- **Update scenario** saves the current JSON, starts a visible one-room Isaac preview when needed, and rebuilds the scene in the already-running preview process after later changes.
- **RUN TRAINING** saves the current JSON and starts the Isaac BRKGA training runner.
- **Landmark detector** opens the separate image-processing configurator with the same JSON file.
- **Open dashboard** starts Streamlit and selects the newest stable or timestamped database.

The **Tools and logs** tab also exposes mock integrity, Isaac integrity, scenario preview start/stop, detector configuration, dashboard, output-folder, and latest-database actions. Output from launched processes is shown inside the GUI.

## Scenario builder

The scenario builder contains:

- room dimensions, roof, wall thickness, environment spacing, and ambient light;
- floor and wall colors, opacity, roughness, metallic value, and reflectance;
- a top-down map containing the room, Crazyflie, landmark, obstacles, and custom assets;
- movable Crazyflie, landmark, obstacles, and assets;
- box, wall, sphere, cylinder, pillar, and floor-patch obstacles;
- obstacle position, size, color, collision, opacity, roughness, metallic value, and reflectance;
- recursive USD asset discovery under `assets/`;
- custom USD position, rotation, scale, and optional analytic collision size;
- landmark position, size, color, emissive intensity, light intensity, and exposure;
- Crazyflie initial pose, collision radius, asset path, and spawn randomization;
- onboard camera position, rotation, resolution, and optical settings;
- external isometric preview-camera framing;
- ranging and collision-sensor settings.

### Updating a visible preview

1. Open or create a JSON configuration.
2. Change the scenario fields.
3. Click **Update scenario**.
4. The first click starts Isaac Sim and creates one preview room.
5. Later clicks send a reload command to the same preview process. The stage is rebuilt from the saved JSON.
6. Use **Stop scenario preview** in the Tools tab when finished.

The training runner does not read unsaved widgets. Clicking **Update scenario** or **RUN TRAINING** always saves and validates the current JSON first.

## Landmark detector configurator

Open it from the main GUI or run:

```powershell
.\run_detector_configurator.bat
```

The tool is separate from the training form because it is an image-processing workspace. It can process an offline image or start a one-room Isaac camera worker.

The six bounded preview panels show:

1. RGB frame;
2. HSV visualization;
3. raw threshold mask;
4. processed mask;
5. grayscale image;
6. detection overlay.

Sliders and entries update the views immediately. Available processing controls include:

- HSV thresholding, brightness thresholding, HSV-or-brightness, grayscale thresholding, and adaptive thresholding;
- hue wrap for red landmarks;
- Gaussian blur;
- morphology open and close;
- erosion and dilation;
- mask inversion;
- minimum area, maximum area fraction, circularity, and brightest-candidate preference.

**Save** writes these settings into the currently loaded training JSON. Normal training then uses the same `vision` section on every camera frame. No separate detector configuration needs to be synchronized.

## Validation commands

```powershell
python -m compileall -q .
pytest -q
.\run_mock_test.bat
```

The included automated suite validates configuration compatibility and round-trip saving, BRKGA/model behavior, database behavior, the mock pipeline, and the configurable detector views.

Isaac-specific scene preview, camera capture, and training still require local testing because Isaac Sim is not available in the code-generation environment.
