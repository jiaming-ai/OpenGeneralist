from __future__ import annotations

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [str(ROOT)] + [entry for entry in sys.path if entry != str(SCRIPT_DIR)]
sys.modules.pop("train", None)


if __name__ == "__main__":
    runpy.run_module("open_gen.train.train", run_name="__main__")
