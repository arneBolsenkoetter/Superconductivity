# # ~/path/to/Superconductivity/sc_python/src/mypy/its90_table.py
# from __future__ import annotations
# from pathlib import Path
# import numpy as np

# THIS_FILE_PATH = Path(__file__).resolve()
# PARENT_PATH = Path(__file__).resolve().parent

# DTYPE = np.dtype([('T_K', 'f8'), ('p_kPa', 'f8')])

# def load_its90_table(path:str|Path) -> np.ndarray:
#     """Load 'temp(K),pressure(kPa)' into a structured array."""
#     return np.genfromtxt(path, delimiter=',', dtype=DTYPE)

# def T_from_p_kpa(p, table:np.ndarray):
#     """Interpolate temperature (K) from pressure (kPa)."""
#     p = np.atleast_1d(p).astype(float)
#     idx = np.argsort(table['p_kpa'])
#     out = np.interp(p, table['p_kpa'][idx], table['temp'][idx])
#     return out if out.ndim else out.item()

# def p_kpa_from_T(T, table:np.ndarray):
#     """Interpolate pressure (kPa) from temperature (K)."""
#     T = np.atleast_1d(T).astype(float)
#     idx = np.argsort(table['temp'])
#     out = np.interp(T, table['temp'][idx], table['p_kpa'][idx])
#     return out if out.ndim else out.item()


# ITS90_STRUCT = load_its90_table(PARENT_PATH/'ITS90.py')
import sys, numpy as np
from core import ITS90_STRUCT, print_struct

print_struct(ITS90_STRUCT)

p = ITS90_STRUCT['p_kPa']
is_strict_inc = np.all(np.diff(p) > 0)
is_nondecr = np.all(np.diff(p) >= 0)

print("strictly increasing:", is_strict_inc)
print("non-decreasing:", is_nondecr)
