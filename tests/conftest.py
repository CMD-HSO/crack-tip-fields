"""pytest configuration: make the module (src/) and the scripts (scripts/, tests/) importable without installation."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src", "scripts", "tests"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib

matplotlib.use("Agg")
