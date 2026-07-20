from __future__ import annotations

import traceback

from cfbrkga.runtime_compat import format_runtime_versions, validate_isaac_runtime


def main() -> int:
    try:
        versions = validate_isaac_runtime(require_cuda=True)
    except Exception:
        print("\nISAAC ENVIRONMENT CHECK FAILED\n", flush=True)
        traceback.print_exc()
        return 1

    print("Isaac environment check passed.", flush=True)
    print(format_runtime_versions(versions), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
