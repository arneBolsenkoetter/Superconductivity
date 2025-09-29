import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from config import PROJECT_ROOT, PY_DIR, DATA_DIR, LATEX_DIR, figrect

from sc_data.Druck_Spaunung_Korrelation import npDV_si as npDV


fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figrect)
ax.