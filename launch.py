#!/usr/bin/env python3
"""Start the FringeLab app and open it in the default browser.

Usage: python launch.py        (the run_app launchers call this)
Stop the app by closing this window or pressing Ctrl+C.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def free_port(preferred: int = 8501) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def main():
    port = free_port()
    url = f"http://localhost:{port}"
    # Headless + opening the browser here avoids Streamlit's first-run e-mail prompt.
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless", "true", "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT,
    )
    try:
        for _ in range(120):  # wait up to 60 s for the server
            if proc.poll() is not None:
                sys.exit(proc.returncode)
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.5)
        print(f"\nFringeLab is running at {url}\nClose this window (or press Ctrl+C) to stop it.\n")
        webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
