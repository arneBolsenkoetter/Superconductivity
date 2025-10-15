# ~/path/to/Superconductivity/sc_pthon/src/mySC/callibration.py
# --------------------------------- imports -----------------------------------
from __future__ import annotations

import sys

import config as cfg
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from config import figrect
from mypy.core import Measurement
from its90_table import T_from_p_kpa, p_kpa_from_T, ITS90_STRUCT
from scipy.interpolate import PchipInterpolator

from sc_data.Druck_Spaunung_Korrelation import npDV_si as npDV



this_file_path = Path(__file__).resolve()
print(this_file_path)
this_file_name = this_file_path.stem
print(this_file_name)
# if not cfg.breaker():   sys.exit(0)

# -------------------------------- functions ----------------------------------
def p_from_Up(
    m:float, dm:float, b:float, db:float, covar:float, 
    Up:np.ndarray, Up_err:np.ndarray|None=None
) -> tuple[np.ndarray, np.ndarray|None]:
    p = m*Up + b
    if Up_err is None:
        return p
    return p, np.sqrt((m*Up_err)**2 + (dm*Up)**2 + db**2 + 2*Up*covar)
    # Put this next to your existing code (same module is fine)

def build_T_from_p_interpolator(
    T_tab: np.ndarray,
    p_tab: np.ndarray,
):
    """
    Build a monotone p->T interpolator and its derivative dT/dp
    from an ITS-90 table (T_tab in K, p_tab in SAME units you will query with).
    """
    # Sort by pressure (strictly increasing for vapor pressure)
    idx = np.argsort(p_tab)
    p_sorted = p_tab[idx]
    T_sorted = T_tab[idx]
    # Monotone shape-preserving cubic (no overshoot -> stable inverse)
    T_of_p = PchipInterpolator(p_sorted, T_sorted, extrapolate=False)
    dTdp   = T_of_p.derivative()
    return T_of_p, dTdp, p_sorted.min(), p_sorted.max()

def T_from_p_with_errors(
    p: np.ndarray,
    dp: np.ndarray,
    T_of_p: PchipInterpolator,
    dTdp: PchipInterpolator,
    p_min: float,
    p_max: float,
    clip: bool = True,
):
    """
    Evaluate T(p) and propagate errors: dT = |dT/dp| * dp.
    If clip=True, values slightly outside the table are clipped to [p_min,p_max].
    """
    p_query = p.copy()
    if clip:
        p_query = np.clip(p_query, p_min, p_max)
    else:
        # Optionally warn or mask outside points:
        if np.any((p < p_min) | (p > p_max)):
            raise ValueError("Some pressures fall outside the ITS-90 table range.")
    T  = T_of_p(p_query)
    dT = np.abs(dTdp(p_query)) * dp
    return T, dT


# --------------------------------- tester ------------------------------------
tester = False
if tester:
    # print(npDV['volts'])
    print(npDV)
    exit()



# ------------------------------- mpl config ----------------------------------
cfg.export_final=False
cfg.configure(overrides={'lines.linewidth':0.6})

# get parameters for fit between pressure and voltage (Druck_Spaunung_KOrrelation.py)
# --- linear fit: Pressure = m * Voltage + b (weighted by y-errors) ---
x  = npDV['volts'].astype(float)
y  = npDV['druck'].astype(float)
sy = npDV['druck_err'].astype(float)

# keep only finite points
mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
x, y, sy = x[mask], y[mask], sy[mask]

# Weighted least squares with NumPy (weights = 1/sigma_y)
global m,b,m_err,b_err
(m, b), cov = np.polyfit(x, y, deg=1, w=1.0/sy, cov=True)

var_m = cov[0,0]
var_b = cov[1,1]
covar = cov[0,1]
varco = cov[1,0]
m_err, b_err = np.sqrt(np.diag(cov))

print_cov = False
if print_cov:
    label_w, val_w = 6, 24
    print('')
    print(f"{'var_m':<{label_w}}= {var_m:>{val_w}}{'':^8}"f"{'σ_m ':>{label_w}}= {np.sqrt(var_m):>{val_w}}")
    print(f"{'var_b':<{label_w}}= {var_b:>{val_w}}{'':^8}"f"{'σ_m ':>{label_w}}= {np.sqrt(var_b):>{val_w}}")
    print(f"{'covar':<{label_w}}= {var_b:>{val_w}}{'':^8}"f"{'σ_c ':>{label_w}}= {np.sqrt(var_b):>{val_w}}")
    print(f"{'varco':<{label_w}}= {var_b:>{val_w}}{'':^8}"f"{'σ_v ':>{label_w}}= {np.sqrt(var_b):>{val_w}}")
    print('')
    print(f"Fit: y = m*x + b")              # Fit: y = m*x + b
    print(f"m = {m:.6g} ± {m_err:.2g}")     # m = 1.05119 ± 0.0019
    print(f"b = {b:.6g} ± {b_err:.2g}")     # b = -0.000774967 ± 0.00013
    print(cov)

# --- break ---
# if not cfg.breaker(f'Proceed with {this_file_name}'):   sys.exit(0)

HarryPlotter = False
if HarryPlotter:
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figrect())
    ax.scatter(x=npDV['volts'],y=npDV['druck'],s=1,zorder=3)
    ax.errorbar(
        x=npDV['volts'], y=npDV['druck'],
        xerr=npDV['volts_err'], yerr=npDV['druck_err'],
        fmt='none', **cfg.err_kw(),
        zorder=2
    )
    ax.set_xlabel(r"Voltage [V]")
    ax.set_ylabel(r"Pressure [bar]")

    if fitboi:
        # Optional: add fit line to your existing plot
        xx = np.linspace(x.min(), x.max(), 200)
        yy = m*xx + b
        ax.plot(
            xx, yy,
            lw=1.0, zorder=1, color='red',
            label=(
                "linear fit:\n"
                f"m = {m:.6g} ± {m_err:.2g}\n"
                f"b = {b:.6g} ± {b_err:.2g}"
            ),
        )
        ax.legend()
        out_png = cfg.savefig(fig, "druck_vs_voltage_fitted", "png")
        out_pdf = cfg.savefig(fig, "druck_vs_voltage_fitted")
    else:
        out_png = cfg.savefig(fig, "druck_vs_voltage", "png")
        out_pdf = cfg.savefig(fig, "druck_vs_voltage")

    print('','Saved at', out_png, 'and', out_pdf, sep='\n')

    plt.show()
    plt.close()



# ------------------------ get preassure p_p from Up --------------------------
m1 = Measurement.from_npz(cfg.DATA_DIR/"clean/LHJG__Supraleitung_1.npz")
print(m1.columns)

plot_all = False
if plot_all:
    nrows=5
    fig,ax = plt.subplots(nrows=nrows,ncols=1,figsize=figrect(nrows=nrows))
    ax[0].plot(m1.time, m1.u_ab,label='AB')
    ax[0].plot(m1.time, m1.u_p, label='P')
    ax[0].plot(m1.time, m1.u_probe,label='Probe')
    ax[0].plot(m1.time, m1.u_spule, label='Spule')
    ax[0].legend()

    ax[1].plot(m1.u_ab, m1.u_p, label='P')
    ax[1].plot(m1.u_ab, m1.u_probe, label='Probe')
    ax[1].plot(m1.u_ab, m1.u_spule, label='Spule')
    ax[1].legend()

    ax[2].plot(m1.u_p, m1.u_ab, label='AB')
    ax[2].plot(m1.u_p, m1.u_probe, label='Probe')
    ax[2].plot(m1.u_p, m1.u_spule, label='Spule')
    ax[2].legend()

    ax[3].plot(m1.u_probe, m1.u_ab, label='AB')
    ax[3].plot(m1.u_probe, m1.u_p, label='P')
    ax[3].plot(m1.u_probe, m1.u_spule, label='Spule')
    ax[3].legend()

    ax[4].plot(m1.u_spule, m1.u_ab, label='AB')
    ax[4].plot(m1.u_spule, m1.u_p, label='P')
    ax[4].plot(m1.u_spule, m1.u_probe, label='Probe')
    ax[4].legend()

    plt.show()

print('has u_p_err:', hasattr(m1, 'u_p_err'), type(getattr(m1, 'u_p_err', None)))
# if not cfg.breaker():   sys.exit(0)

p_bar, dp_bar = p_from_Up(m=m, dm=m_err, b=b, db=b_err, covar=covar, Up=m1.u_p, Up_err=m1.u_p_err)

print(p_bar,dp_bar,sep='\n\n')
# danach: T = T_from_p(p_bar)  # per ITS-90/Tabelle/Spline

show_ITS90=False
if show_ITS90:
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize=figrect())
    ax.scatter(x=ITS90_STRUCT['T_K'].astype(float),y=ITS90_STRUCT['p_kPa'].astype(float))
    plt.show()



# --------------------- get temp t_p from preassure p_p -----------------------
# 1) Prepare the table vectors from your struct array
#    Adjust field names to whatever you used:
#    Example assumes ITS90_STRUCT has fields: 'T_K' and 'p_kPa'
T_tab_K   = ITS90_STRUCT['T_K'].astype(float)

# Make sure your p_bar, dp_bar are in the SAME units as p_tab.
# If your table is in kPa and your p_bar is in bar, convert one side.
# Common conversions:
#   1 bar = 100000 Pa
#   kPa -> bar  : multiply by 0.01
#   kPa -> mbar : multiply by 10.0
#   bar -> kPa  : multiply by 100.0
# Here we convert the TABLE to BAR (so it matches p_bar in bar):
p_tab_bar = ITS90_STRUCT['p_kPa'].astype(float) * 0.01

# 2) Build interpolators
Tp, dTp, pmin, pmax = build_T_from_p_interpolator(T_tab_K, p_tab_bar)

# 3) Evaluate T and its uncertainty for your pressures from p_from_Up(...)
#    p_bar, dp_bar already computed in your code:
T_K, dT_K = T_from_p_with_errors(p_bar, dp_bar, Tp, dTp, pmin, pmax, clip=True)

for t,dt in zip(T_K,dT_K):
    print(f"{t:10.3f}  ± {dt:.3f}")


# ------------------------ Allen-Bradley Calibration --------------------------
# Assume you already computed:
# T_K, dT_K from ITS-90 interpolation of p_bar, dp_bar
# and you have U_AB and (optionally) U_AB_err from your Measurement m1

T = T_K.astype(float)
sT = dT_K.astype(float)
U = m1.u_ab.astype(float)
sU = getattr(m1, 'u_ab_err', None)
if sU is None:
    # fallback: use a small constant or estimated noise level (Volts)
    sU = np.full_like(U, 1e-4, dtype=float)

# Optional: restrict to one branch (e.g., below lambda) if you plan separate fits
mask_infty = np.isfinite(T) & np.isfinite(sT) & np.isfinite(U) & np.isfinite(sU)
mask0 = (T<=4.2)
mask_zoom = (1.73<=T)&(T<=2.2)
mask_normalfluid = (2.2<=T)&(T<=4.2)
mask_superfluid = (T<=2.2)

mask = mask_infty&mask_superfluid
T, sT, U, sU = T[mask], sT[mask], U[mask], sU[mask]

plot_AB=True
if plot_AB:
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=figrect())
    ax.errorbar(x=T,xerr=sT,y=U,yerr=sU)
    ax.set_xlabel(r'Temperatur [K]')
    ax.set_ylabel(r'Druck [bar]')
    plt.show()

