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
from its90_table import ITS90_STRUCT
from scipy.odr import ODR, Model, RealData
from scipy.interpolate import PchipInterpolator

from sc_data.Druck_Spaunung_Korrelation import npDV_si as npDV



this_file_path = Path(__file__).resolve()
print(this_file_path)
this_file_name = this_file_path.stem
print(this_file_name)
# if not cfg.confirm():   sys.exit(0)

# -------------------------------- functions ----------------------------------
def p_from_Up(
    m:float, dm:float, b:float, db:float, covar:float, 
    Up:np.ndarray, Up_err:np.ndarray|None=None
) -> tuple[np.ndarray, np.ndarray|None]:
    p = m*Up + b
    if Up_err is None:
        return p, None
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
        clipped = (p < p_min) | (p > p_max)
        if np.any(clipped):
            print(f"[warn] {np.count_nonzero(clipped)} Druckwerte wurden auf den Tabellenbereich geclippt.")
        p_query = np.clip(p_query, p_min, p_max)

    else:
        # Optionally warn or mask outside points:
        if np.any((p < p_min) | (p > p_max)):
            raise ValueError("Some pressures fall outside the ITS-90 table range.")
    T  = T_of_p(p_query)
    dT = np.abs(dTdp(p_query)) * dp
    return T, dT

# ---------- Helper: Laden & Anwenden der U_AB(T)-Kalibrierung ----------
def load_uab_calibration(path) -> tuple[float,float,float,np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return float(z["alpha"]), float(z["beta"]), float(z["gamma"]), z["cov_theta"]

def apply_uab_calibration(U: np.ndarray,
                          sU: np.ndarray | float | None,
                          calib_path: str | Path,
                          mode: str = "clip") -> tuple[np.ndarray, np.ndarray]:
    """
    Rechnet aus U_AB -> T (K) + sigma_T mit gespeicherter Kalibrierung.
    mode: 'strict'  -> ValueError, wenn U außerhalb (alpha, alpha+beta)
          'clip'    -> U wird an die Grenzen geclippt und es wird gewarnt
    """
    alpha, beta, gamma, cov_theta = load_uab_calibration(calib_path)

    U = np.asarray(U, float)
    if sU is None:
        sU = np.zeros_like(U)
    elif np.isscalar(sU):
        sU = np.full_like(U, float(sU))
    else:
        sU = np.asarray(sU, float)

    lo, hi = alpha, alpha + beta

    if mode == "strict":
        if np.any((U <= lo) | (U >= hi)):
            raise ValueError("U outside valid range (alpha, alpha+beta).")
        Uq = U
        sUq = sU
    elif mode == "clip":
        clipped = (U <= lo) | (U >= hi)
        if np.any(clipped):
            print(f"[warn] {np.count_nonzero(clipped)} U_AB-Werte wurden auf ({lo:.6g}, {hi:.6g}) geclippt.")
        Uq = np.clip(U, lo + np.finfo(float).eps, hi - np.finfo(float).eps)
        sUq = sU  # Fehler bleibt
    else:
        raise ValueError("mode must be 'strict' or 'clip'.")

    T, sT = T_from_UAB(Uq, sUq, alpha, beta, gamma, cov_theta)
    return T, sT

def apply_uab_dual(U, sU=None, mode="clip"):
    # Lädt beide Kalibrierungen
    a_s, b_s, g_s, C_s = load_uab_calibration(cfg.DATA_DIR/"calibration_UAB_T_superfluid.npz")
    a_n, b_n, g_n, C_n = load_uab_calibration(cfg.DATA_DIR/"calibration_UAB_T_normalfluid.npz")

    # Optional: Schwelle in T (z. B. 2.17 K) nur für Entscheidung, nicht für Rechnen
    # Praktisch: über U-Grenzen gehen
    lo_s, hi_s = a_s, a_s + b_s
    lo_n, hi_n = a_n, a_n + b_n

    U = np.asarray(U, float)
    if sU is None:
        sU = np.zeros_like(U)
    elif np.isscalar(sU):
        sU = np.full_like(U, float(sU))
    else:
        sU = np.asarray(sU, float)

    # Heuristik: wenn U im Gültigkeitsintervall eines Zweigs liegt, nimm diesen.
    use_s = (U > lo_s) & (U < hi_s)
    use_n = ~use_s & (U > lo_n) & (U < hi_n)

    T_out = np.full_like(U, np.nan, float)
    sT_out = np.full_like(U, np.nan, float)

    if np.any(use_s):
        T_out[use_s], sT_out[use_s] = T_from_UAB(U[use_s], sU[use_s], a_s, b_s, g_s, C_s)
    if np.any(use_n):
        T_out[use_n], sT_out[use_n] = T_from_UAB(U[use_n], sU[use_n], a_n, b_n, g_n, C_n)

    # Option: An rändern „clippen“
    if mode == "clip":
        # superfluid clip
        mask = ~use_s & (U <= lo_s)
        if np.any(mask):
            T_clip, sT_clip = T_from_UAB(np.full(mask.sum(), lo_s+np.finfo(float).eps),
                                         sU[mask], a_s, b_s, g_s, C_s)
            T_out[mask], sT_out[mask] = T_clip, sT_clip
        # normalfluid clip
        mask = ~use_n & (U >= hi_n)
        if np.any(mask):
            T_clip, sT_clip = T_from_UAB(np.full(mask.sum(), hi_n-np.finfo(float).eps),
                                         sU[mask], a_n, b_n, g_n, C_n)
            T_out[mask], sT_out[mask] = T_clip, sT_clip

    return T_out, sT_out


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

y_fit = m*x + b
chi2  = np.sum(((y - y_fit)/sy)**2)
dof   = len(x) - 2

print_cov = False
if print_cov:
    # printing nice terminal outputs:
    # left / right / centered alignment with fixed widths
    # print(f"{name:<18} | {value:>8.3f} | {status:^6}")
    #               ^ left        ^ right, width 8, 3 decimals  ^ center, width 6

    # custom fill characters
    # print(f"{'Header':=^18}")   # =====Header=====

    # # dynamic widths
    # w = 12
    # print(f"{name:<{w}} | {value:>{w}.2f}")
    label_w, val_w = 6, 24
    print('')
    print(f"{'var_m':<{label_w}}= {var_m:>{val_w}}{'':^8}"f"{'σ_m ':>{label_w}}= {np.sqrt(var_m):>{val_w}}")
    print(f"{'var_b':<{label_w}}= {var_b:>{val_w}}{'':^8}"f"{'σ_m ':>{label_w}}= {np.sqrt(var_b):>{val_w}}")
    print(f"{'covar':<{label_w}}= {covar:>{val_w}}{'':^8}"f"{'σ_c ':>{label_w}}= {np.sqrt(abs(covar)):>{val_w}}")
    print(f"{'varco':<{label_w}}= {varco:>{val_w}}{'':^8}"f"{'σ_v ':>{label_w}}= {np.sqrt(abs(varco)):>{val_w}}")
    print('')
    print(f"Fit: y = m*x + b")              # Fit: y = m*x + b
    print(f"m = {m:.6g} ± {m_err:.2g}")     # m = 1.05119 ± 0.0019
    print(f"b = {b:.6g} ± {b_err:.2g}")     # b = -0.000774967 ± 0.00013
    print(cov)
    print('')
    corr_mb = covar / (m_err * b_err)
    print(f"corr(m,b) = {corr_mb:.3f}")

    print(f"chi2/dof = {chi2/dof:.2f}")


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

plot_basic = False
if cfg.confirm('Plot basic data (p_p from Up)?'):
    plot_basic = True
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

if plot_basic:
    print('has u_p_err:', hasattr(m1, 'u_p_err'), type(getattr(m1, 'u_p_err', None)))
    # if not cfg.confirm():   sys.exit(0)

p_bar, dp_bar = p_from_Up(m=m, dm=m_err, b=b, db=b_err, covar=covar, Up=m1.u_p, Up_err=m1.u_p_err)

if plot_basic:
    print(p_bar,dp_bar,sep='\n\n')
    # danach: T = T_from_p(p_bar)  # per ITS-90/Tabelle/Spline

if cfg.confirm('Plot ITS90_STRUCT?'):
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize=figrect())
    ax.scatter(x=ITS90_STRUCT['T_K'].astype(float),y=ITS90_STRUCT['p_kPa'].astype(float))
    plt.show()


# --------------------- get temp t_p from preassure p_p -----------------------
# 1) Prepare the table vectors from your struct array
    # Adjust field names to whatever you used:
    # Example assumes ITS90_STRUCT has fields: 'T_K' and 'p_kPa'
T_tab_K   = ITS90_STRUCT['T_K'].astype(float)

# 1.a) Make sure your p_bar, dp_bar are in the SAME units as p_tab.
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



# --- Diagnose & p-valid auswählen (statt clip=True) ---
print(f"[ITS-90] p_range_table: [{pmin:.6g}, {pmax:.6g}] bar")
print(f"[data ] p_range_data : [{np.nanmin(p_bar):.6g}, {np.nanmax(p_bar):.6g}] bar")

mask_p_valid = (p_bar >= pmin) & (p_bar <= pmax)
print(f"valid p: {mask_p_valid.sum()} / {p_bar.size}  "
      f"(below {np.count_nonzero(p_bar<pmin)}, above {np.count_nonzero(p_bar>pmax)})")

# --- Alle Zeitreihen konsistent auf p-valid kürzen ---
t   = m1.time.astype(float)[mask_p_valid]
U   = m1.u_ab.astype(float)[mask_p_valid]
sU0 = getattr(m1, 'u_ab_err', None)
if sU0 is None:
    sU = np.full_like(U, 1e-4, dtype=float)
else:
    sU = np.asarray(sU0, float)[mask_p_valid]

p_use  = p_bar [mask_p_valid]
dp_use = dp_bar[mask_p_valid]

# --- T(p) ohne Clipping (!) berechnen ---
T_K, dT_K = T_from_p_with_errors(p_use, dp_use, Tp, dTp, pmin, pmax, clip=False)

# --- Ab hier NUR noch diese Arrays verwenden ---
# 1) Gültig/endlich
mask = np.isfinite(T_K) & np.isfinite(dT_K) & np.isfinite(U) & np.isfinite(sU)

# 2) Temperaturfenster superfluid + Lücke um 2.04 K
mask &= (T_K >= 1.80) & (T_K <= 2.15)
mask &= ~((T_K > 2.035) & (T_K < 2.045))

# 3) Transienten-Filter
def deriv(y, x): 
    return np.gradient(y, x)

dUdt = deriv(U,  t)
dTdt = deriv(T_K, t)
thrU = np.nanpercentile(np.abs(dUdt[mask]), 90)
thrT = np.nanpercentile(np.abs(dTdt[mask]), 90)
mask &= (np.abs(dUdt) < thrU) & (np.abs(dTdt) < thrT)

if not np.any(mask):
    raise RuntimeError("Maske leer – Fenster/Filter anpassen (Transienten/MAD).")

# --- ODR mit konsistent geschnittenen Arrays ---
TT  = T_K[mask]; sT  = dT_K[mask]
UU  = U  [mask]; sUU = sU  [mask]

from scipy.odr import ODR, Model, RealData
def uab_model(theta, TT):
    alpha, b1, g1 = theta
    beta, gamma = np.exp(b1), np.exp(g1)
    return alpha + beta*np.exp(-gamma*TT)

alpha0 = np.nanmin(UU)
beta0  = max(np.nanmax(UU) - alpha0, 1e-6)
gamma0 = 1.0
theta0 = (alpha0, np.log(beta0), np.log(gamma0))

# Grob-Fit
data = RealData(TT, UU, sx=sT, sy=sUU)
odr  = ODR(data, Model(uab_model), beta0=theta0)
out  = odr.run()

# MAD-Outlier-Rejection (optional)
U_pred = uab_model(out.beta, TT)
res    = UU - U_pred
mad    = np.median(np.abs(res - np.median(res)))
keep   = np.abs(UU - uab_model(out.beta, TT)) < 3.5*mad

# Finaler Fit auf 'keep'
data = RealData(TT[keep], UU[keep], sx=sT[keep], sy=sUU[keep])
odr  = ODR(data, Model(uab_model), beta0=out.beta)
out  = odr.run()

# Parameter + Kovarianz in Originalparametern
alpha_hat, b1_hat, g1_hat = out.beta
beta_hat,  gamma_hat      = np.exp(b1_hat), np.exp(g1_hat)
cov_trans = out.cov_beta
J = np.array([[1.0, 0.0, 0.0],
              [0.0, beta_hat, 0.0],
              [0.0, 0.0, gamma_hat]], float)
cov_theta = J @ cov_trans @ J.T
alpha, beta, gamma = alpha_hat, beta_hat, gamma_hat

# Checks
Ts = np.linspace(TT[keep].min(), TT[keep].max(), 200)
Us = alpha + beta*np.exp(-gamma*Ts)
assert np.all(np.diff(Us) < 0), "U_AB(T) ist nicht monoton fallend!"

U_hat = alpha + beta*np.exp(-gamma*TT[keep])
chi2  = np.sum(((UU[keep] - U_hat) / sUU[keep])**2)
dof   = len(UU[keep]) - 3
print(f"chi2/dof = {chi2/dof:.2f}")

print("[SF] RMS resid:", np.sqrt(np.mean((UU[keep]-U_hat)**2)))

figure1=plt.figure()
plt.plot(TT[keep], UU[keep] - U_hat, '.')
plt.axhline(0, lw=0.8)
plt.xlabel('T [K]'); plt.ylabel('U_AB - model [V]')
plt.title('Residuals U_AB(T) [SF]')
cfg.savefig(figure1,name='residuals_u_ab(t)_sf')
plt.show()
plt.close()


sig_alpha, sig_beta, sig_gamma = np.sqrt(np.diag(cov_theta))

# Speichern
np.savez(
    cfg.DATA_DIR/"calibration_UAB_T_superfluid.npz",
    alpha=alpha, beta=beta, gamma=gamma,
    cov_theta=cov_theta,
    domain_lo=float(alpha), domain_hi=float(alpha+beta),
    T_range=np.array([TT[keep].min(), TT[keep].max()]),
    meta=np.array(["model: alpha + beta*exp(-gamma*T)", "superfluid masked, ODR"]),
)

print(f"[SF] alpha={alpha:.6g} ± {sig_alpha:.2g}, "
      f"beta={beta:.6g} ± {sig_beta:.2g}, "
      f"gamma={gamma:.6g} ± {sig_gamma:.2g}")



# ==================== Allen-Bradley Kalibrierung: Normalfluid ====================
# Wir setzen auf den bereits berechneten, p-validen Arrays auf:
# t, U, sU, T_K, dT_K sind (wie zuvor) schon auf mask_p_valid geschnitten!

# 1) Masken für den Normalfluid-Bereich
mask_nf = np.isfinite(T_K) & np.isfinite(dT_K) & np.isfinite(U) & np.isfinite(sU)
mask_nf &= (T_K >= 2.20) & (T_K <= 4.18)   # etwas Sicherheitsabstand zur 4.22-K-Tabellengrenze
mask_nf &= ~((T_K > 2.16) & (T_K < 2.19))  # Lambda-Nähe großzügig ausschneiden

# Optionale Transienten-Filter (wie beim superfluiden Bereich)
def deriv(y, x): return np.gradient(y, x)
dUdt = deriv(U,  t)
dTdt = deriv(T_K, t)
if np.any(mask_nf):
    thrU_nf = np.nanpercentile(np.abs(dUdt[mask_nf]), 90)
    thrT_nf = np.nanpercentile(np.abs(dTdt[mask_nf]), 90)
    mask_nf &= (np.abs(dUdt) < thrU_nf) & (np.abs(dTdt) < thrT_nf)

if not np.any(mask_nf):
    raise RuntimeError("Normalfluid-Maske leer – Fenster/Filter anpassen.")

TT  = T_K[mask_nf]; sT  = dT_K[mask_nf]
UU  = U  [mask_nf]; sUU = sU  [mask_nf]

# 2) Modellwahl
# Starte pragmatisch mit demselben 1-Exp-Modell wie im superfluiden Bereich:
from scipy.odr import ODR, Model, RealData
def uab_model(theta, TT):
    alpha, b1, g1 = theta
    beta, gamma = np.exp(b1), np.exp(g1)
    return alpha + beta*np.exp(-gamma*TT)

# Initialwerte robust
alpha0 = np.nanmin(UU)
beta0  = max(np.nanmax(UU) - alpha0, 1e-6)
gamma0 = 0.5  # im Normalfluid oft „flacher“ -> kleinere Start-Gamma ist ok
theta0 = (alpha0, np.log(beta0), np.log(gamma0))

# 3) Grob-Fit → Outlier-Rejection → Final-Fit
data = RealData(TT, UU, sx=sT, sy=sUU)
odr  = ODR(data, Model(uab_model), beta0=theta0)
out  = odr.run()

U_pred = uab_model(out.beta, TT)
res    = UU - U_pred
mad    = np.median(np.abs(res - np.median(res)))
keep   = np.abs(UU - U_pred) < 3.5*mad

data = RealData(TT[keep], UU[keep], sx=sT[keep], sy=sUU[keep])
odr  = ODR(data, Model(uab_model), beta0=out.beta)
out  = odr.run()

# 4) Parameter + Kovarianz in Originalparametern
alpha_hat, b1_hat, g1_hat = out.beta
beta_hat,  gamma_hat      = np.exp(b1_hat), np.exp(g1_hat)
cov_trans = out.cov_beta
J = np.array([[1.0, 0.0, 0.0],
              [0.0, beta_hat, 0.0],
              [0.0, 0.0, gamma_hat]], float)
cov_theta_nf = J @ cov_trans @ J.T
alpha_nf, beta_nf, gamma_nf = alpha_hat, beta_hat, gamma_hat
sig_alpha_nf, sig_beta_nf, sig_gamma_nf = np.sqrt(np.diag(cov_theta_nf))

# 5) Checks & Güte
Ts = np.linspace(TT[keep].min(), TT[keep].max(), 200)
Us = alpha_nf + beta_nf*np.exp(-gamma_nf*Ts)
assert np.all(np.diff(Us) < 0), "U_AB(T) (normalfluid) ist nicht monoton fallend!"

U_hat = alpha_nf + beta_nf*np.exp(-gamma_nf*TT[keep])
rms   = np.sqrt(np.mean((UU[keep] - U_hat)**2))
chi2  = np.sum(((UU[keep] - U_hat)/sT[keep])**2)  # grobe Güte; alternativ /sUU[keep] je nach Vertrauensmaß
dof   = len(UU[keep]) - 3
print(f"[NF] RMS resid: {rms:.3e}, chi2/dof≈{chi2/max(dof,1):.2f}")

figure2=plt.figure()
plt.plot(TT[keep], UU[keep] - U_hat, '.')
plt.axhline(0, lw=0.8)
plt.xlabel('T [K]'); plt.ylabel('U_AB - model [V]')
plt.title('Residuals U_AB(T) [NF]')
cfg.savefig(figure2,name='residual_u_ab(t)_nf')
plt.show()
plt.close()


# 6) Speichern (separat vom superfluiden Zweig)
np.savez(
    cfg.DATA_DIR/"calibration_UAB_T_normalfluid.npz",
    alpha=alpha_nf, beta=beta_nf, gamma=gamma_nf,
    cov_theta=cov_theta_nf,
    domain_lo=float(alpha_nf), domain_hi=float(alpha_nf+beta_nf),
    T_range=np.array([TT[keep].min(), TT[keep].max()]),
    meta=np.array(["model: alpha + beta*exp(-gamma*T)", "normalfluid masked, ODR"]),
)

print(f"[NF] alpha={alpha_nf:.6g} ± {sig_alpha_nf:.2g}, "
      f"beta={beta_nf:.6g} ± {sig_beta_nf:.2g}, "
      f"gamma={gamma_nf:.6g} ± {sig_gamma_nf:.2g}")








# Anwendung:
# calpath = cfg.DATA_DIR/"calibration_UAB_T_superfluid.npz"
# U_new   = m1.u_ab
# sU_new  = getattr(m1, 'u_ab_err', 1e-4)

# T_new, sT_new = apply_uab_calibration(U_new, sU_new, calpath, mode="clip")
