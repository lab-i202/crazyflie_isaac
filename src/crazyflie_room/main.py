# main.py
#
# Compatibility wrapper.
#
# The old repository used main.py for the Bixler/tethered-glider prototype.
# For the modular refactor, scenario_launcher.py is the real entry point.

from scenario_launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
