from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"

for p in (ROOT, LEGACY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

for d in (DATA_DIR, OUTPUTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
