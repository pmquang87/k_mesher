"""Start the k_mesher GUI.

Works from anywhere: double-click, `python start_gui.py`, or an IDE run
configuration. If the project virtualenv (.venv) exists and we are not
already running inside it, relaunch with its interpreter (pythonw.exe,
so no console window is left open).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, ".venv")


def _venv_interpreter() -> str | None:
    for name in ("pythonw.exe", "python.exe"):
        path = os.path.join(VENV_DIR, "Scripts", name)
        if os.path.isfile(path):
            return path
    return None


def main() -> None:
    venv_py = _venv_interpreter()
    in_venv = os.path.normcase(sys.prefix) == os.path.normcase(VENV_DIR)

    if venv_py and not in_venv:
        subprocess.Popen([venv_py, os.path.join(ROOT, "main.py")], cwd=ROOT)
        return

    sys.path.insert(0, ROOT)
    from gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
