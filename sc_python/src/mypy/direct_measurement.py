# ~/path/to/Superconductivity/sc_python/src/mypy/direct_measurement.py
from __future__ import annotations

import config as cfg

from core import Measurement
from config import user_stripped
from pathlib import Path

# print(user_stripped(Path(__file__).resolve()))
cfg.configure()

m15 = Measurement.from_npz(cfg.DATA_CLEAN/"LHJG__Supraleitung_15.npz")