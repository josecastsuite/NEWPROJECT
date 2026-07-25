"""Entry point for JoseCast Analyzer v8.0 Titan."""

import os
import sys


class _Tee:
    """Write stdout/stderr to both the original stream and a log file."""

    def __init__(self, stream, log_path):
        self._stream = stream
        self._log = open(log_path, "a", encoding="utf-8")
        self._log.write(f"\n--- JoseCast run started ---\n")

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()


# Keep a copy of console output in a file next to main.py for debugging.
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_LOG_DIR, "josecast.log")
sys.stdout = _Tee(sys.stdout, _LOG_FILE)
sys.stderr = _Tee(sys.stderr, _LOG_FILE)

from ui.main_window import main

if __name__ == "__main__":
    main()
