# ~/path/to/Superconductivity/sc_pthon/src/mySC/Dataviewer.py

# --------------------------------- imports -----------------------------------
from __future__ import annotations

import config

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from config import PROJECT_ROOT, PY_DIR, DATA_DIR, LATEX_DIR, figrect

from sc_data.Druck_Spaunung_Korrelation import npDV_si as npDV


# --------------------------------- tester ------------------------------------
tester = False
if tester:
    print(npDV['volts'])

    exit()


# ------------------------------- mpl config ----------------------------------
config.use_mpl


# ------------------------------ main workflow --------------------------------
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figrect(1,1,1,1))
ax.scatter(npDV['volts'],npDV['druck'],s=3)
ax.set_xlabel(r"Voltage [V]")
ax.set_ylabel(r"Pressure [bar]")

ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

fig.tight_layout()