# ~/path/to/Superconductivity/sc_python/src/mypy/dataloader_test.py
import config as cfg
import matplotlib.pyplot as plt

from config import figrect
from pathlib import Path
from mypy.core import load_lhjg_dat, RENAMES, DEFAULT_COLS

m = load_lhjg_dat(Path(cfg.DATA_DIR)/"LHJG__Supraleitung_1.dat").with_voltage_errors(range_v=0.1)  # <- beware the double underscore before 'Supraleitung'

print(m.meta["Startzeit"])

t = m.data["t_s"]
Uprobe = m.data["U_probe"]

if __name__=="__main__":
    cfg.configure()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figrect())
    for name,rename in ((c,RENAMES[c]) for c in DEFAULT_COLS[1:]):
        ax.errorbar(x=t, y=m.data[str(rename)], yerr=m.data[f'{str(rename)}_err'], label=str(name))
    ax.legend()
    plt.show()