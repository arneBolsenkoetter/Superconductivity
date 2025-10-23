# ~/path/to/Superconductivity/sc_python/src/mySC/config.py

# This file tries to condense all relevant parameters to make matplotlib (mpl) figures stylised as demanded in scientific publishings, like the size of a figure (see figrect()), the individual mpl.rcParams (see mpl_rc) and provides a way how to easily initialise the im another script (see configure()).
# There are even more advanced collections out there. I found one at: https://github.com/hosilva/physrev_mplstyle/blob/main/physrev.mplstyle

# Furthermore, it introduces a solution which hands-off the comilation of text in mpl.plot to LaTeX (see export_final). In order to do so in-line with the rest of the document, it uses a minimal preamble_mpl.tex to draw its settings from, which can be shared with the main preamble.tex (at the very start of preamble.tex: \input(preamble_mpl.tex)).

# Last but not lest this document defines a 'breaker' which allows you to set breakpoints manually in python-scripts (see breaker()).


# ------------------------------------- imports ----------------------------------------
from __future__ import annotations  # always first

import atexit
import os, sys, re
import numpy as np

from pathlib import Path


# --------------------------------- auxiliary helper -----------------------------------
def confirm(msg="Continue?", default=False):
    """
        Returns True to continue, False to abort.
        - default applies if stdin isn't a TTY (e.g., piped) or user just hits Enter.

        Usage:
        if not confirm("Proceed with the next step?"):
            sys.exit(0)  # graceful exit
    """
    if not sys.stdin.isatty():
        return default
    try:
        ans = input(f"{msg} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # clean newline
        return False
    return ans in ("y", "yes")

# some functions to print in colour with terminal
def prRed(s): print("\033[91m {}\033[00m".format(s))
def prGreen(s): print("\033[92m {}\033[00m".format(s))
def prYellow(s): print("\033[93m {}\033[00m".format(s))
def prLightPurple(s): print("\033[94m {}\033[00m".format(s))
def prPurple(s): print("\033[95m {}\033[00m".format(s))
def prCyan(s): print("\033[96m {}\033[00m".format(s))
def prLightGray(s): print("\033[97m {}\033[00m".format(s))
def prBlack(s): print("\033[90m {}\033[00m".format(s))

# mask user-directory on print ->   NEVER SHOW <$HOME> to A.I.
def user_stripped(path:Path) -> str:
    """ 
        This is just a tiny substitution. It allows the user to hide its $HOME directory on print-out. 
        Suplementing this function to any path about to be printed, like so

            'print(user_stripped(this_or_that_directory))',

        lets you safly copy-paste your error-code to AIs without unwillingly sharing sensible personal data.


        # P.S.:
        # You might want to add this following substitution-function to your ~/.zshrc, ~/.bashrc or similar (written for zsh):
        # # Run any command, show ~ instead of $HOME in its output
        # with-tilde() {
        # # -u = unbuffered so lines appear immediately
        # command "$@" 2> >(sed -u "s|$HOME|~|g; s|<$HOME>|~|g" >&2) | sed -u "s|$HOME|$
        # return $pipestatus[1]
        # }

        # # optional convenience aliases for noisy tools
        # alias git='with-tilde git'
        # alias pip='with-tilde pip'
        # alias python='with-tilde python'
        # alias pytest='with-tilde pytest'
    """
    from pathlib import Path
    home = Path.home()
    return re.sub(f'{home}','~',str(path))

# Dictate distinguished Dictionaries - class-definition:
class AttrDict(dict):
    __getattr__ = dict.get
    def __setattr__(self,k,v): self[k]=v
    def __delattr__(self,k): del self[k]
# function to convert any dict into AttrDict
def attrmap(x:dict[object,object]) -> AttrDict:
    """
        Usage:
        Say you have already defined a dictionary   Dict: dict[...,...] = {...,...},    then:

            Dict = config.attrmap(Dict)

        Afterwards you can call your values by simply concatenating their keys with dots (works for dicts of dicts of dicts ... too):

            Dict.key1.key2.key3...

        BACKWARD-COMPATIBLE: Still allows you to acces dicts the standard way:      dict[key1][key2]...
    """
    if isinstance(x,dict):
        return AttrDict({k: attrmap(v) for k,v in x.items()})
    if isinstance(x,list):
        return [attrmap(v) for v in x]
    if isinstance(x,tuple):
        return tuple(attrmap(v) for v in x)
    return x


# -------------------------------- Project Locations -----------------------------------
config_path     = Path(__file__).resolve()          # !!! BEWARE: The path-setup relies on relative paths to this file;     
PROJECT_ROOT    = config_path.parents[3]            # Either make sure you use the same layout and have placed this file in your equivalent to '~/path/to/PROJECT_ROOT', 
                                                    # or adjust the below paths accordingly. 
                                                    # To disperse any confusion: 'PROJECT_ROOT' is the generalisation of 'Superconductivity'
FIG_DIR         = PROJECT_ROOT / 'figures'
FIG_FINAL       = PROJECT_ROOT / 'figures' / 'final'
PY_DIR          = PROJECT_ROOT / 'sc_python'
SRC             = PROJECT_ROOT / 'sc_python' / 'src'
MYPY            = PROJECT_ROOT / 'sc_python' / 'src' / 'mypy'
DATA_DIR        = PROJECT_ROOT / 'sc_data'
DATA_CLEAN      = PROJECT_ROOT / 'sc_data' / 'clean'
LATEX_DIR       = PROJECT_ROOT / 'sc_latex'


# ----------------------------- physical constants (SI) --------------------------------
c0      = 299792458.0
h       = 6.62607015e-34
hbar    = h/(2*np.pi)
eps0    = 8.8541878128e-12
me      = 9.1093837015e-31
qe      = 1.602176634e-19


# ----------------------------------- mpl config ---------------------------------------
figwidth = 3.375    # review letter syle column width - in inches!
figheight = 0.75*figwidth
figrect_norm=(figwidth,figheight)

SCATTER_KW = dict(s=1, alpha=0.65, linewidths=0, edgecolors='face', rasterized=True)

def figrect(ncols:int|None=1, nrows:int|None=1, sw:float|None=1.0, sh:float|None=1.0) -> tuple[float,float]:
    """
        Returns a tuple with the size (width,height) for a matplotlib.pyplot (plt) subplots-figure.

        Parameters:
        - m:    int,    multiplies the figure-widtch with 'ncols' = number of columns;
        - n:    int,    multiplies the figure-height with 'nrows' = number of rows;
        - sw:   float,  abbreviation for 'Sqeeze Width'. Stretches of thwats the figure-width by the set value;
        - sh:   float,  abbreviation for 'Squeeze Height', Stretches or thwats the figure_height by set value;

        Returns:
        - (fig-width, fig-height):  tuple[float,float],     size for a scientific-review-paper-styled image,    in INCHES!
                                    default:                (3.375", 0.75*3.375")
    """
    figwidth = 3.375
    figheight = 0.75*figwidth
    return (figwidth*ncols*sw, figheight*nrows*sh)

def configure(overrides:dict[str,object]|None=None, final:bool|None=None, backend:str|None=None) -> None:
    """
        Apply standardised rcParams (config.mpl_rc); allows to override the standard rcParams locally for executing script via overrides; 
        if config.export_final == True, or config.configure(final=True)     -> merge-in LaTeX/PGF extras.

        Usage:
            - import:
                import config (as cfg)

            - default:
                at the very top (just after imports) of each script that plots graphs using matplotlib(.pyplot), you want to aply the standardised mpl rcParams to
                call:   config.configure()
            - overrides:
                If you want to make local (bound to the current script) changes,
                call:   config.configure(overrides={'lines.linewidth':0.5})     <- exemplary
            - final:
                for finalised plots, hand compilation off to LaTeX via:     'text.usetex':True
                first set:  config.export_final=True,
                AFTER that: config.configure( (overrides=...) )

                alternatively (no config.export_final=True in script):
                call:       config.configure(final=True)
    """
    import matplotlib as mpl
    if backend is not None:
        mpl.use(backend, force=True)
    use_final = export_final if final is None else final
    rc = dict(mpl_rc_base)
    if use_final:
        rc.update(mpl_rc_final)
    if overrides:
        rc.update(overrides)
    mpl.rcParams.update(rc)
    global errorbar_linewidth
    errorbar_linewidth = max(rc['lines.linewidth']-0.2,0.1)

def savefig(fig, name:str, ext:str='pdf', final:bool|None=None, **kwargs) -> Path:
    """Save with (or without) final LaTeX settings, independent of global state."""
    import matplotlib as mpl
    name_path = Path(name)
    print(name_path)
    ext_norm = str(ext).lower().lstrip('.')
    print(ext_norm)
    if name_path.suffix:
        stem = name_path.stem
        print(stem)
        use_ext = ext_norm if ext_norm else name_path.suffix.lstrip(".")
        print(use_ext)
    else:
        stem = name_path.as_posix()
        print(stem)
        use_ext = ext_norm or "pdf"
        print(use_ext)
    use_final = export_final if final is None else final
    directory = FIG_FINAL if use_final else FIG_DIR
    suffix = '_final' if use_final else ''
    destination = (directory / f"{stem}{suffix}.{use_ext}").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_userstripped=re.sub('/Users/arnebolsenkoetter','~',str(destination))
    if use_final:
        with mpl.rc_context(mpl_rc_final):
            fig.savefig(destination, **kwargs)
    else:
        fig.savefig(destination, **kwargs)
    print(f'Saved as {destination_userstripped}')
    return destination

mpl_rc: dict[str,object] = {
    'axes.axisbelow':       True,
    'axes.grid':            True,
    'axes.grid.axis':       'both',
    "axes.labelsize":       8,
    "axes.titlesize":       8,
    "axes.linewidth":       0.6,
    "axes.formatter.useoffset":     True,
    "axes.formatter.use_mathtext":  True,
    "axes.formatter.limits":        (0,0),

    "errorbar.capsize":     2,

    "figure.constrained_layout.use":    True,
    'figure.dpi':           300,
    "font.family":          "serif",
    "font.size":            8,

    'grid.alpha':           0.6,
    'grid.linewidth':       0.4,

    "legend.fontsize":      7,
    "lines.linewidth":      0.8,
    "lines.markersize":     2,

    "mathtext.fontset":     "stix",     # serif-like math without LaTeX dependency

    "patch.linewidth":      0.6,

    "savefig.dpi":          300,
    "scatter.marker":       'o',
    "svg.fonttype":         'none',

    "xtick.bottom":         True,
    "xtick.direction":      "in",
    "xtick.labelsize":      7,
    "xtick.major.size":     3,
    "xtick.major.width":    0.6,
    "xtick.minor.bottom":   True,
    "xtick.minor.size":     1.5,
    "xtick.minor.top":      True,
    "xtick.minor.visible":  True,
    "xtick.minor.width":    0.5,
    'xtick.top':            True,

    "ytick.direction":      "in",
    "ytick.labelsize":      7,
    "ytick.left":           True,
    "ytick.major.size":     3,
    "ytick.major.width":    0.6,
    "ytick.minor.left":     True,
    "ytick.minor.right":    True,
    "ytick.minor.size":     1.5,
    "ytick.minor.visible":  True,
    "ytick.minor.width":    0.5,
    "ytick.right":          True,
#   1   2   3   4   5   6   7   8
}

# --- special cases for plt.errorbar ---
errorbar_linewidth = max(mpl_rc['lines.linewidth']-0.2,0.1)

def err_kw(elw:float|None=None, capsz:float|None=None):#-> dict[str,float]:
    """
        usage:
        import config as cfg
        ...
        ax.errorbar(x, y, yerr=dy, **cfg.eb_kw(), capsize=2)
    """
    import matplotlib as mpl
    if elw is None:
        elw=max(mpl_rc.get('lines.linewidth',0.8)-0.2,0.1)
    else:
        elw=elw
    if capsz is None:
        capsize=mpl_rc.get('errorbar.capsize',2)
    else:
        capsize=capsz
    return dict(elinewidth=elw, capthick=elw, capsize=capsize)


# ----------------------------- plt final-export toggle --------------------------------
# outsources text compilations in plots to TeX
export_final:bool = bool(int(os.getenv("SC_EXPORT_FINAL", "0")))  # or set cfg.export_final=True in script

# TeX compilation needs instructions->preamble. Optional: keep your LaTeX preamble in a file in your LaTeX project
LATEX_PREAMBLE_FILE = LATEX_DIR / "__preamble_mpl__.tex"
DEFAULT_PREAMBLE = r"\usepackage{amsmath}\usepackage{siunitx}\usepackage{newtxtext,newtxmath}"

def _load_preamble() -> str:
    try:
        return LATEX_PREAMBLE_FILE.read_text()
    except Exception:
        return DEFAULT_PREAMBLE

# Your existing base rc (rename your current mpl_rc to mpl_rc_base)
mpl_rc_base = mpl_rc
# Extra settings only for final exports
mpl_rc_final = {
    "text.usetex": True,
    "text.latex.preamble": _load_preamble(),
    "pgf.texsystem": "lualatex",   # works well with unicode + modern fonts
    "svg.fonttype": "none",        # selectable text in SVGs
}

@atexit.register
def _reset_export_flag():
    # Prevent “sticking” when working interactively
    global export_final
    export_final = False