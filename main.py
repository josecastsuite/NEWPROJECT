"""Entry point for JoseCast Analyzer v8.0 Titan."""

import faulthandler
import os
import sys
import traceback


# Console encoding on Windows defaults to cp1254/cp1252; we want UTF-8 with a
# safe fallback so print statements containing arrows/Turkish chars do not crash.
for _std in (sys.stdout, sys.stderr):
    if _std is not None and hasattr(_std, "reconfigure"):
        try:
            _std.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class _Tee:
    """Write stdout/stderr to both the original stream and a log file."""

    def __init__(self, stream, log_path):
        self._stream = stream
        self._log = open(log_path, "a", encoding="utf-8")
        self._log.write("\n--- JoseCast run started ---\n")
        self._log.flush()

    def _safe_stream_write(self, data):
        """Write to the console stream, replacing unencodable characters."""
        try:
            self._stream.write(data)
            self._stream.flush()
            return
        except UnicodeEncodeError:
            pass
        # Fallback: encode with the stream's encoding and replace errors.
        try:
            enc = getattr(self._stream, "encoding", "utf-8") or "utf-8"
            safe = data.encode(enc, errors="replace").decode(enc, errors="replace")
            self._stream.write(safe)
            self._stream.flush()
        except Exception:
            pass

    def write(self, data):
        self._safe_stream_write(data)
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass


# Keep a copy of console output in a file next to main.py for debugging.
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_LOG_DIR, "josecast.log")
_FAULT_FILE = os.path.join(_LOG_DIR, "josecast_fault.log")

_fault_fd = open(_FAULT_FILE, "a", encoding="utf-8")
_fault_fd.write("\n--- JoseCast fault handler enabled ---\n")
_fault_fd.flush()
faulthandler.enable(_fault_fd)

sys.stdout = _Tee(sys.stdout, _LOG_FILE)
sys.stderr = _Tee(sys.stderr, _LOG_FILE)


def _excepthook(exc_type, exc_value, exc_tb):
    """Log uncaught exceptions before the process exits."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        sys.stderr.write(f"\n[UNCAUGHT EXCEPTION]\n{msg}\n")
        sys.stderr.flush()
    except Exception:
        pass
    # Re-raise the original hook so the process exits with the same behaviour.
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook

from ui.main_window import main

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            _fault_fd.flush()
        except Exception:
            pass
