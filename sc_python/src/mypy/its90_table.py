# # ~/path/to/Superconductivity/sc_python/src/mypy/its90_table.py
import sys, numpy as np
from core import ITS90_STRUCT, print_struct

print_struct(ITS90_STRUCT)

p = ITS90_STRUCT['p_kPa']
is_strict_inc = np.all(np.diff(p) > 0)
is_nondecr = np.all(np.diff(p) >= 0)

print("strictly increasing:", is_strict_inc)
print("non-decreasing:", is_nondecr)
