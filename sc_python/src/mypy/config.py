# ~/path/to/Superconductivity/sc_python/src/mySC/config.py

# --------------------------------- imports -----------------------------------
from __future__ import annotations
# always first

import numpy as np

from pathlib import Path


# ---------------------------- Project Locations ------------------------------
# directory MUST be sctructured like: 
# ~/path/to/Superconductivity/sc_python/src/mySC/config.py
config_path     = Path(__file__).resolve()  
PROJECT_ROOT    = config_path.parents[3]
FIG_DIR         = PROJECT_ROOT / 'figures'
PY_DIR          = PROJECT_ROOT / 'sc_python'
PY_SRC          = PROJECT_ROOT / 'sc_python' / 'src'
PY_SRC_MYPY     = PROJECT_ROOT / 'sc_python' / 'src' / 'mypy'
DATA_DIR        = PROJECT_ROOT / 'sc_data'
LATEX_DIR       = PROJECT_ROOT / 'sc_latex'


# ------------------------- physical constants (SI) ---------------------------
c0      = 299792458.0
h       = 6.62607015e-34
hbar    = h/(2*np.pi)
eps0    = 8.8541878128e-12
me      = 9.1093837015e-31
qe      = 1.602176634e-19


# ------------------------------- mpl config ----------------------------------
figwidth = 3.375    # review letter syle column width - in inches!
figheight = 0.75*figwidth
figrect_norm=(figwidth,figheight)
SCATTER_KW = dict(s=1, alpha=0.65, linewidths=0, edgecolors='none', rasterized=True)

mpl_rc: dict[str,object] = {
    'axes.axisbelow':       True,
    'axes.grid':            True,
    'axes.grid.axis':       'both',
    "axes.labelsize":       8,
    "axes.titlesize":       8,
    "axes.linewidth":       0.6,

    'grid.alpha':           0.6,
    'grid.linewidth':       0.4,

    "font.size":            8,
    "legend.fontsize":      7,

    "xtick.direction":      "in",
    "xtick.labelsize":      7,
    "xtick.major.size":     3,
    "xtick.major.width":    0.6,
    "xtick.minor.size":     1.5,
    "xtick.minor.width":    0.5,
    'xtick.top':            True,

    "ytick.direction":      "in",
    "ytick.labelsize":      7,
    "ytick.major.size":     3,
    "ytick.major.width":    0.6,
    "ytick.minor.size":     1.5,
    "ytick.minor.width":    0.5,
    "ytick.right":          True,

    'text.usetex':          True,
    "mathtext.fontset":     "stix",     # serif-like math without LaTeX dependency
    "font.family":          "serif",

    'figure.dpi':           200,
    "savefig.dpi":          300,
}

def figrect(m:int=1,n:int=1,sw:float=1.0,sh:float=1.0) -> tuple[float,float]:
    """
        Returns a tuple with the size (width,height) for a matplotlib.pyplot (plt) subplots-figure.

        Parameters:
        - m:    int,    multiplies the figure-widtch with 'ncols' = number of columns;
        - n:    int,    multiplies the figure-height with 'nrows' = number of rows;
        - sw:   float,  abbreviation for 'Sqeeze Width'. Stretches of thwats the figure-width by the set value;
        - sh:   float,  abbreviation for 'Squeeze Height', Stretches or thwats the figure_height by set value;

        Return:
        - (fig-width, fig-height):  tuple[float,float], size for a scientific-review-paper-styled image, in INCHES!
    """
    return (figwidth*m*sw,figheight*n*sh)

def config_plot(overrides:dict[str,object]|None=None, backend:str|None=None) -> None:
    """
        Set files matplotlib settings to the physical review letters "scientific" standard

        Usage:
            - default:
                import config
                config.config_plot()    or      config.use_mpl
            - optional with tweaks:
                config.config_plot({'axes.grid': False})
    """
    import matplotlib as mpl
    if backend is not None:
        mpl.use(backend,force=True)
    rc = mpl_rc if overrides is None else {**mpl_rc, **overrides}
    mpl.rcParams.update(rc)

# Optional ultra-short alias    -   without optional overrides
use_mpl = config_plot