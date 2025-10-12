import numpy as np
from numpy.lib import recfunctions as rfn
import logging


# ---------------------------------- Data -------------------------------------
DruckVolt = [
    (989.0, 'V', 0.941954, np.nan),
    (988.0, 'V', 0.940656, np.nan),
    (955.0, 'V', 0.909730, np.nan),
    (815.0, 'V', 0.776586, np.nan),
    (765.0, 'V', 0.728630, np.nan),
    (714.0, 'V', 0.680390, np.nan),
    (661.0, 'V', 0.630479, np.nan),
    (598.0, 'V', 0.569987, np.nan),
    (546.0, 'V', 0.521198, np.nan),
    (506.0, 'V', 0.482754, np.nan),
    (456.0, 'V', 0.434506, np.nan),
    (400.0, 'V', 0.381,     1.0e-3),
    (362.0, 'V', 0.346036, np.nan),
    (322.0, 'V', 0.307280, np.nan),
    (275.0, 'V', 0.262934, np.nan),
    (234.0, 'V', 0.223307, np.nan),
    (183.0, 'V', 0.174954, np.nan),
    (169.0, 'V', 0.161740, np.nan),
    (158.0, 'V', 0.151024, np.nan),
    (143.0, 'V', 0.13,      1.0e-2),
    (126.0, 'V', 0.121080, np.nan),
    (107.0, 'V', 0.102849, np.nan),
    (93.0,  'mV',   90.0769,   np.nan),
    (74.0,  'mV',   71.5692,   np.nan),
    (66.0,  'mV',   63.4452,   np.nan),
    (58.0,  'mV',   56.5072,   np.nan),
    (50.0,  'mV',   48.5128,   np.nan),
    (45.0,  'mV',   43.2957,   np.nan),
    (39.0,  'mV',   37.8806,   np.nan),
    (27.0,  'mV',   26.6272,   np.nan),
    (10.0,  'mV',   10.3379,   np.nan),
    (4.0,   'mV',   4.6009,   np.nan),
    (2.0,   'mV',   2.3171,   np.nan),
    (1.0,   'mV',   1.8209,   np.nan),
]
Dtype = np.dtype([
    ("druck", "f8"),
    ("volts_unit",    "U3"),  # uX with lower case u -> unsigned int, UX with upper case U -> str
    ("volts",   "f8"),
    ("volts_err","f4")
])


# ------------------------------- functions -----------------------------------
def DC_Kenndaten(wqr:str) -> tuple[float,float]:
    """ 
    Aus dem Datenblatt zum benutzten Voltmeter: 
    Genauigkeit: ± (% vom Messwert + % des Bereichs)
    Bereich         1 Jahr (Betriebsdauer seit letzter Kallibrierung)
    100.0000 mV     0.0050 + 0.0035
    1.000000 V      0.0040 + 0.0007

    Parameter:
    - wqr: str, 'mv' oder 'V' -> gibt den Bereich mit dem gemessen wurde

    Return:
    - (Bereich mal relativer Fehler Bereich, relativer Fehler Messwert), 
        NICHT in Prozent [%], in SI-Einheit [V]!
    """
    if wqr.lower() == 'mv':
        return (0.1*0.000050, 0.000035)
    elif wqr.lower() == 'v':
        return (1.0*0.000040, 0.000007)
    raise ValueError("DC_Kennlinie() erwartet str {'mv' | 'V'}")

def format_structured(arr, order=None, floatfmt=".6g", max_rows=None):
    """
        Return a pretty, column-aligned string of a structured array.
    """
    if order is None:
        order = list(arr.dtype.names or [])
    elif isinstance(order, (tuple,np.ndarray)):
        order = list(order)
    elif isinstance(order, str):
        order = [order]
    if not order:
        raise ValueError('format_structured: array has no named fields')
    a = arr[order]  # re-order columns if given
    # convert to strings with basic formatting
    cols = []
    for name in order:
        col = a[name]
        if col.dtype.kind in "f":
            s = [f"{x:{floatfmt}}" for x in col]
        else:
            s = [str(x) for x in col]
        cols.append(s)
    # header + compute widths
    header = list(order)
    widths = [max(len(h), max((len(v) for v in c), default=0)) for h, c in zip(header, cols)]

    def fmt_row(vals):
        return "  ".join(v.rjust(w) for v, w in zip(vals, widths))

    lines = [fmt_row(header), fmt_row(["-" * w for w in widths])]
    n = len(a)
    rows_range = n if max_rows is None else min(n, max_rows)
    for i in range(rows_range):
        lines.append(fmt_row([c[i] for c in cols]))
    if max_rows is not None and n > max_rows:
        lines.append(f"... ({n - max_rows} more rows)")
    return "\n".join(lines)

def log_array(name, arr, *, level=logging.DEBUG):
    """Pretty-print numpy arrays/structured arrays only when the level is enabled."""
    if logger.isEnabledFor(level):
        s = np.array2string(
            arr,
            max_line_width=120,
            threshold=10000,          # show up to 200 elements fully
            floatmode="maxprec",    # stable float printing
            formatter={"float_kind": lambda x: f"{x:.6g}"},
        )
        logger.log(level, "%s:\n%s", name, s)

def fill_errs(x:np.ndarray):
    """ 
        Füllt die nicht manuell eingetragenen Fehler der Spannung auf.

        Aus dem Datenblatt zum benutzten Voltmeter: 
        Genauigkeit:    ± (% vom Messwert + % des Bereichs)
        Bereich         1 Jahr (Betriebsdauer seit letzter Kallibrierung)
        100.0000 mV     0.0050 + 0.0035
        1.000000 V      0.0040 + 0.0007

        NICHT IN PROZENT:
        Genauigkeit:    ± (Bruchteil vom Messwert + Bruchteil vom Bereich)
        100.0000 mv:    0.000050 + 0.000035 = 5.0e-5 + 3.5e-5
        1.000000 V:     0.000040 + 0.000007 = 4.0e-5 + 7.0e-6

        Parameter:
        - x: np.array(n,dtype=[("druck","f8"),("volts_unit","U2"),("volts","f8"),("v_man_er","f4")])

        Return:
        - (Bereich mal relativer Fehler Bereich, relativer Fehler Messwert), 
            NICHT in Prozent [%], in SI-Einheit [V]!
    """
    mask_nan = np.isnan(x['volts_err'])
    mask_mV = (x['volts_unit']=='mV')
    mask_V = ~mask_mV

    x['volts'][mask_mV] *= 1e-3
    x['volts_unit'][mask_mV] = 'V'

    def err_mV(x):
        """ 0.005%=5.0e-5 of reading  +  0.0035%=3.5e-5 of range (100mV=0.1V) """
        return 5.0e-5*x['volts']+3.5e-6

    def err_V(x):
        """ 0.004%=4.0e-5 of reading  +  0.0007%=7.0e-6 of range (1V) """
        return 4.0e-5*x['volts']+7.0e-6

    m1 = mask_nan&mask_mV
    m2 = mask_nan&mask_V
    x['volts_err'][m1] = err_mV(x[m1])
    x['volts_err'][m2] = err_V(x[m2])

def get_druck_errs(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x['druck']*0.03)**2+1)

def druck_si(x:np.ndarray) -> np.ndarray:
    m_mbar = (x['druck_unit']=='mbar')
    x['druck'][m_mbar] *= 1e-3
    x['druck_err'][m_mbar] *=1e-3
    x['druck_unit'][m_mbar] = 'bar'
    return x


# ---------------------- print ONLY when run as a script ----------------------
def _dump(name, arr, *,floatfmt='.6g'):
    if getattr(arr, 'dtype', None) is not None and getattr(arr.dtype,'names',None):
        print(f'{name}:')
        print(format_structured(arr, floatfmt=floatfmt))
    else:
        s = np.array2string(
            np.asarray(arr),
            max_line_width=120,
            threshold=10000,
            floatmode='maxprec',
            formatter={'float_kind': lambda x:f'{x:.6g}'},
        )
        print(f'{name}:',f'{s}',sep='\n')
    print()

# ----------------------------------- main ------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    npDV = np.asarray(DruckVolt, dtype=Dtype)
    log_array("npDV (before)", npDV)

    fill_errs(npDV)
    log_array("npDV (after fill_errs)", npDV)

    barrs = get_druck_errs(npDV)
    log_array("barrs", barrs)

    druck_einheit = np.full(len(barrs), fill_value="mbar", dtype="U4")
    log_array("druck_einheit", druck_einheit)

    npDV_full = rfn.append_fields(
        npDV,
        names=["bar_err", "bar_unit"],
        data=[barrs, druck_einheit],
        dtypes=[barrs.dtype, druck_einheit.dtype],
        asrecarray=False,
        usemask=False,
    )
    log_array("npDV_full (all fields)", npDV_full)

    npDV_full = npDV_full[["druck", "bar_err", "bar_unit", "volts", "volts_err", "volts_unit"]]
    log_array("npDV_full (Columns reordered)", npDV_full, level=logging.INFO)

    npDV_sort = npDV_full[np.argsort(npDV_full['druck'])]
    logger.info("\n%s", format_structured(
        npDV_sort,
        order=["druck", "bar_err", "bar_unit", "volts", "volts_err", "volts_unit"],
        floatfmt=".6g",
    ))


npDV_before             = np.asarray(DruckVolt, dtype=Dtype)
npDV                    = npDV_before.copy()
fill_errs(npDV)
druck_errs              = get_druck_errs(npDV)
druck_einheit           = np.full(len(druck_errs), fill_value='mbar', dtype='U4')

npDV_full_all_fields    = rfn.append_fields(
    npDV,
    names=['druck_err','druck_unit'],
    data=[druck_errs,druck_einheit],
    dtypes=[druck_errs.dtype, druck_einheit.dtype],
    asrecarray=False,
    usemask=False,
)
npDV_full               = npDV_full_all_fields[
    ['volts_unit','volts','volts_err','druck_unit','druck','druck_err',]
]
npDV_si                 = npDV_full.copy()
npDV_si                 = druck_si(npDV_si)

__all__ = [
    'npDV_before', 'npDV', 'druck_errs', 'druck_einheit',
    'npDV_full_all_fields', 'npDV_full', 'npDV_si',
    'format_structured', 'fill_errs', 'get_druck_errs',
]



if __name__ == "__main__":
    _dump("npDV (before)", npDV_before)
    _dump('npDV (after fill_errs)', npDV)
    _dump('druck_errs', druck_errs)
    _dump('druck_einheit', druck_einheit)
    _dump('npDV_full (all fields)', npDV_full_all_fields)
    _dump('npDV_full (column reordered)', npDV_full)
    _dump('npDV (everything in SI-units)', npDV_si)