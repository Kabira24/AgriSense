"""
main.py
───────
AgriSense-AI — Unified startup entry point.

Launches all six backend services as independent child processes so the
entire platform can be started with a single command:

    python main.py

Services & ports
    8000 – predict.py   Crop recommendation (ML)
    8001 – profit.py    Profit estimation
    8002 – weather.py   Weather / Open-Meteo
    8003 – risk.py      Risk assessment engine
    8004 – advisor.py   Gemini AI farm advisor
    8005 – planner.py   Crop lifecycle planner

Press Ctrl+C to stop all services.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).parent          # AgriSense-AI:/
_BACKEND   = _WORKSPACE / "backend:"       # AgriSense-AI:/backend:

# Load .env so GEMINI_API_KEY is available to child processes
try:
    from dotenv import load_dotenv
    _ENV_FILE = _WORKSPACE / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
        print(f"[main] Loaded environment from {_ENV_FILE}")
    else:
        print(f"[main] WARNING: .env not found at {_ENV_FILE}. "
              "GEMINI_API_KEY must be set in your shell environment.")
except ImportError:
    print("[main] WARNING: python-dotenv not installed. "
          "Run: pip install python-dotenv")

# ── Service definitions ────────────────────────────────────────────────────────
# (module_filename_stem, uvicorn_app_var, port)
SERVICES: list[tuple[str, str, int]] = [
    ("predict",  "app", 8000),
    ("profit",   "app", 8001),
    ("weather",  "app", 8002),
    ("risk",     "app", 8003),
    ("advisor",  "app", 8004),
    ("planner",  "app", 8005),
]

# ── Launcher ───────────────────────────────────────────────────────────────────

def _start_service(module: str, var: str, port: int) -> subprocess.Popen:
    """Start a single uvicorn service as a subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{module}:{var}",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "info",
    ]
    # Run from inside backend:/ so relative imports resolve correctly
    proc = subprocess.Popen(
        cmd,
        cwd=str(_BACKEND),
        env={**os.environ},   # propagate the .env variables we just loaded
    )
    return proc


def main() -> None:
    procs: list[tuple[str, int, subprocess.Popen]] = []

    print("\n" + "=" * 56)
    print("  AgriSense-AI -- Starting all services")
    print("=" * 56)

    for module, var, port in SERVICES:
        print(f"  >>  {module:<10}  ->  http://localhost:{port}")
        proc = _start_service(module, var, port)
        procs.append((module, port, proc))
        time.sleep(0.4)   # brief stagger so logs stay readable

    print("=" * 56)
    print("  All services started. Press Ctrl+C to stop.\n")

    def _shutdown(signum, frame):
        print("\n[main] Shutting down all services...")
        for name, port, proc in procs:
            proc.terminate()
            print(f"  ok  {name} (port {port}) stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait for all child processes
    for _, _, proc in procs:
        proc.wait()


if __name__ == "__main__":
    main()
