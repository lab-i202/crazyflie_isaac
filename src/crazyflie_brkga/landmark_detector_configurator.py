from __future__ import annotations

import os

from cfbrkga.detector_gui import run_detector_gui


if __name__ == "__main__":
    run_detector_gui(os.environ.get("CRAZYFLIE_BRKGA_CONFIG"))
