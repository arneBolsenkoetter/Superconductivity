# ~/path/to/Superconductivity/sc_python/src/mypy/critical_field.py
from __future__ import annotations

import csv, os
import numpy as np, sympy as sp
import config as cfg, core
import calibration as calib
import matplotlib.pyplot as plt

from core import Measurement
from config import figrect, FIG_DIR, MYPY
from itertools import islice
from scipy.odr import ODR, RealData, Model
from matplotlib.patches import Ellipse



# ---------------------------- mpl-configuration ------------------------------
cfg.configure()

# ------------------------- physical/lab constants ----------------------------
R=384.5e-6
Ro,Roerr = 40.8e-3,1.0e-3
Ri,Rierr = 21.0e-3,1.0e-3
N=10353
L,Lerr=193.0e-3,3.0e-3


# -------------------------------- functions ----------------------------------
i, n, ro, ri, l = sp.symbols('i n ro ri l', real=True)
a = l/2
f = lambda x: x**2 + sp.sqrt(x**2 + a**2)

Hsym        = sp.Rational(1,2) * n*i/(ro - ri) * sp.log(f(ro)/f(ri))
dH_di_sp    = sp.diff(Hsym, i)
dH_dro_sp   = sp.diff(Hsym, ro)
dH_dri_sp   = sp.diff(Hsym, ri)
dH_dl_sp    = sp.diff(Hsym, l)

H       = sp.lambdify((i, n, ro, ri, l), Hsym, 'numpy')
dH_di   = sp.lambdify((i, n, ro, ri, l), dH_di_sp, 'numpy')
dH_dro  = sp.lambdify((i, n, ro, ri, l), dH_dro_sp, 'numpy')
dH_dri  = sp.lambdify((i, n, ro, ri, l), dH_dri_sp, 'numpy')
dH_dl   = sp.lambdify((i, n, ro, ri, l), dH_dl_sp, 'numpy')

def Herr(I:np.ndarray,I_err:np.ndarray)->np.ndarray:
    return np.sqrt(
        (dH_di(I,N,Ro,Ri,L)*I_err)**2 +
        (dH_dro(I,N,Ro,Ri,L)*Roerr)**2 +
        (dH_dri(I,N,Ro,Ri,L)*Rierr)**2 +
        (dH_dl(I,N,Ro,Ri,L)*Lerr)**2
    )

def _line(beta, x):
    m, b = beta
    return m*x + b

def odr_line(x, y, sx, sy, m0=None, b0=None, maxit=200):
    """
    ODR fit of y = m*x + b with errors in x and y.
    Returns dict(m,b,sm,sb,cov,rchi2,out).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    sx = np.asarray(sx, float); sy = np.asarray(sy, float)

    # avoid zero/neg errors
    tiny = np.finfo(float).tiny
    sx = np.where(sx > 0, sx, tiny)
    sy = np.where(sy > 0, sy, tiny)

    if m0 is None or b0 is None:
        # decent initial guess
        m0, b0 = np.polyfit(x, y, 1, w=1.0/np.where(sy>0, sy, 1.0))

    data  = RealData(x, y, sx=sx, sy=sy)
    model = Model(_line)
    odr   = ODR(data, model, beta0=[m0, b0], maxit=maxit)
    out   = odr.run()

    m, b       = out.beta
    sm, sb     = out.sd_beta
    cov        = out.cov_beta      # parameter covariance (consistent with sd_beta)
    rchi2      = out.res_var       # ~ reduced chi^2
    return dict(m=m, b=b, sm=sm, sb=sb, cov=cov, rchi2=rchi2, out=out)

def odr_prediction_band(xgrid, m, b, cov):
    """
    1σ band from parameter covariance.
    var(ŷ) = [x 1] Σ [x 1]^T
    """
    xgrid = np.asarray(xgrid, float)
    A = np.vstack([xgrid, np.ones_like(xgrid)]).T
    var = np.einsum('ij,jk,ik->i', A, cov, A)
    return m*xgrid + b, np.sqrt(np.maximum(var, 0.0))

def intersection_ground_nether(res_ground, res_nether_inverted, scale_by_rchi2=True,return_cov=False):
    """
        Intersection of:
        grounds (direct):     U = m_g * H + c_g
        nether (inverted):   H = a_n * U + c_n

        Parameters
        ----------
        res_ground : dict   # from odr_line(H, U, sH, sU)
            expects keys: 'm','b','cov','rchi2'
        res_nether_inverted : dict  # from odr_line(U, H, sU, sH)
            expects keys: 'm','b','cov','rchi2'
        scale_by_rchi2 : bool
            If True, scale parameter covariances by reduced chi^2 (recommended).
        return_cov : bool
            If True, also return cov(H*,U*) and the full 2x2 covariance matrix.

        Returns
        -------
        Hs, Us, sH, sU                                  (default)
        OR (if return_cov=True)
        Hs, Us, sH, sU, covHU, Sigma2x
    """
    mg, cg, Cg, rg = res_ground['m'], res_ground['b'], np.asarray(res_ground['cov'], float), res_ground['rchi2']
    a,  cn, Cn, rn = res_nether_inverted['m'], res_nether_inverted['b'], np.asarray(res_nether_inverted['cov'], float), res_nether_inverted['rchi2']

    # Optionally inflate parameter covariances if χ²_ν != 1
    if scale_by_rchi2:
        Cg = Cg * rg
        Cn = Cn * rn

    D = a*mg - 1.0
    if np.isclose(D, 0.0, rtol=0, atol=1e-14):
        raise RuntimeError("Fits are nearly parallel (a*mg ≈ 1); intersection poorly defined.")

    N = - (cn + a*cg)
    Hs = N / D
    Us = mg*Hs + cg

    # --- gradients of H* ---
    dH_dmg  = ((cn + a*cg) * a) / (D*D)
    dH_dcg  = -a / D
    dH_da   = ((-cg)*D - N*mg) / (D*D)
    dH_dcn  = -1.0 / D

    # --- gradients of U* = mg*H* + cg ---
    dU_dmg = Hs + mg*dH_dmg
    dU_dcg = 1.0 + mg*dH_dcg
    dU_da  = mg*dH_da
    dU_dcn = mg*dH_dcn

    # block-diagonal parameter covariance
    C = np.block([
        [Cg,              np.zeros((2,2))],
        [np.zeros((2,2)), Cn            ],
    ])

    JH = np.hstack([dH_dmg, dH_dcg, dH_da, dH_dcn])   # shape (4,)
    JU = np.hstack([dU_dmg, dU_dcg, dU_da, dU_dcn])   # shape (4,)

    varH = float(JH @ C @ JH)
    varU = float(JU @ C @ JU)
    covHU = float(JH @ C @ JU)     # scalar

    sH = np.sqrt(max(varH, 0.0))
    sU = np.sqrt(max(varU, 0.0))

    if return_cov:
        Sigma = np.array([[varH,covHU],[covHU,varU]],float)
        return Hs,Us,sH,sU,covHU,Sigma
    else:
        return Hs, Us, sH, sU

def draw_sigma_ellipse(ax, H0, U0, sH, sU, n_sigma=1.0, **kwargs):
    """
    Axis-aligned ellipse centered at (H0, U0).
    Semi-axes = n_sigma * (sH, sU).
    """
    e = Ellipse(
        (H0, U0),
        width = 2*n_sigma*sH,    # x: 2 * semi-axis
        height= 2*n_sigma*sU,    # y: 2 * semi-axis
        angle = 0,
        **({"facecolor":"none","edgecolor":"k","lw":1.0} | kwargs)
    )
    ax.add_patch(e)

    # # ensure it’s inside the autoscale limits
    # ax.dataLim.update_from_data_xy(np.array([
    #     [H0 - n_sigma*sH, U0 - n_sigma*sU],
    #     [H0 + n_sigma*sH, U0 + n_sigma*sU],
    # ]))
    # ax.autoscale_view()
    return e

def draw_cov_ellipse(ax, H0, U0, Sigma2x2, n_sigma=1.0, **kwargs):
    """
    Draw the 2D covariance ellipse for (H,U).
    Sigma2x2 = [[varH, covHU],
                [covHU, varU]]
    """
    vals, vecs = np.linalg.eigh(Sigma2x2)      # symmetric -> eigh
    vals = np.clip(vals, a_min=0.0, a_max=None)
    order = np.argsort(vals)[::-1]      # sort largest first
    vals, vecs = vals[order], vecs[:, order]

    width  = 2*n_sigma*np.sqrt(vals[0])  # along first eigenvector
    height = 2*n_sigma*np.sqrt(vals[1])
    angle  = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))

    e = Ellipse((H0, U0), width, height, angle=angle,
                **({"facecolor":"none","edgecolor":"k","lw":1.0} | kwargs))
    ax.add_patch(e)

    # # keep autoscale happy
    # ax.dataLim.update_from_data_xy(
    #     np.array([[H0 - width, U0 - height],
    #               [H0 + width, U0 + height]])
    # )
    # ax.autoscale_view()
    return e

def min_run(mask: np.ndarray, k: int = 2) -> np.ndarray:
    """Keep only True-runs of length >= k."""
    if k <= 1:
        return mask
    m = mask.astype(int)
    d = np.diff(np.r_[0, m, 0])
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)
    out = np.zeros_like(mask, bool)
    for s, e in zip(starts, ends):
        if (e - s) >= k:
            out[s:e] = True
    return out


# ----------------- load .npzs into 'Measurement'-instances -------------------
mess: dict[str,Measurement] = {}
for i in range(3,15):
    mess[f'm{i}']=Measurement.from_npz(cfg.DATA_CLEAN/f"LHJG__Supraleitung_{i}.npz")
    print(f'Loaded LHJG_Supraleitung_{i}.npz into mess["m{i}"]')


# ----------------------- masks for different regions -------------------------
temps = [
    (mess['m3'].time>=250.0),
    (mess['m4'].time>=200.0),
    (mess['m5'].time>=150.0),
    (mess['m6'].time>=170.0),
    (mess['m7'].time>=50.0),
    (mess['m8'].time>=50.0),
    (mess['m9'].time>=50.0),
    (mess['m10'].time>=70.0),
    (mess['m11'].time>=50.0),
    (mess['m12'].time>=50.0),
    (mess['m13'].time>=50.0),
    (mess['m14'].time>=50.0),
]
t0s = [
    (mess['m3'].time>=194.0),
    (mess['m4'].time>=8.1),
    (mess['m5'].time>=69.8),
    (mess['m6'].time>=42.1),
    (mess['m7'].time>=21.7),
    (mess['m8'].time>=21.2),
    (mess['m9'].time>=18.5),
    (mess['m10'].time>=20.5),
    (mess['m11'].time>=15.8),
    (mess['m12'].time>=16.2),
    (mess['m13'].time>=18.1),
    (mess['m14'].time>=18.7),
]
grounds = [
    (mess['m3'].u_spule<=7.65e-6)&(mess['m3'].u_probe<=8.0e-5),
    (mess['m4'].u_spule<=7.91e-6)&(mess['m4'].u_probe<=8.0e-5),
    (mess['m5'].u_spule<=8.48e-6)&(mess['m5'].u_probe<=8.0e-5),
    (mess['m6'].u_spule<=1.295e-5)&(mess['m6'].u_probe<=8.0e-5),
    (mess['m7'].u_spule<=2.834e-5)&(mess['m7'].u_probe<=8.0e-5),
    (mess['m8'].u_spule<=5.229e-5)&(mess['m8'].u_probe<=8.0e-5),
    (mess['m9'].u_spule<=7.068e-5)&(mess['m9'].u_probe<=7.880e-5),
    (mess['m10'].u_spule<=9.539e-5)&(mess['m10'].u_probe<=7.837e-5),
    (mess['m11'].u_spule<=12.109e-5)&(mess['m11'].u_probe<=8.0e-5),
    (mess['m12'].u_spule<=12.521e-5)&(mess['m12'].u_probe<=8.0e-5),
    (mess['m13'].u_spule<=14.374e-5)&(mess['m13'].u_probe<=7.969e-5),
    (mess['m14'].u_spule<=15.959e-5)&(mess['m14'].u_probe<=8.0e-5),
]
tops = [
    (mess['m3'].u_spule>=1.500e-5),
    (mess['m4'].u_spule>=1.345e-5),
    (mess['m5'].u_spule>=1.201e-5)  &(mess['m5'].u_probe>=16.750e-5),
    (mess['m6'].u_spule>=1.932e-5)  &(mess['m6'].u_probe>=16.800e-5),
    (mess['m7'].u_spule>=4.107e-5),
    (mess['m8'].u_spule>=5.824e-5),
    (mess['m9'].u_spule>=8.500e-5),
    (mess['m10'].u_spule>=10.492e-5)&(mess['m10'].u_probe>=16.693e-5),
    (mess['m11'].u_spule>=13.414e-5),
    (mess['m12'].u_spule>=13.702e-5)&(mess['m12'].u_probe>=16.720e-5),
    (mess['m13'].u_spule>=15.000e-5)&(mess['m13'].u_probe>=16.881e-5),
    (mess['m14'].u_spule>=16.397e-5)&(mess['m14'].u_probe>=16.963e-5),
]
nether = [
    (mess['m3'].u_probe>=8.000e-5)  &(mess['m3'].u_probe<=16.828e-5),
    (mess['m4'].u_probe>=8.155e-5)  &(mess['m4'].u_probe<=16.858e-5),
    (mess['m5'].u_probe>=8.616e-5)  &(mess['m5'].u_probe<=16.746e-5),
    (mess['m6'].u_probe>=8.000e-5)  &(mess['m6'].u_probe<=16.775e-5),
    (mess['m7'].u_probe>=8.000e-5)  &(mess['m7'].u_probe<=16.629e-5),
    (mess['m8'].u_probe>=8.029e-5)  &(mess['m8'].u_probe<=16.319e-5),
    (mess['m9'].u_probe>=8.000e-5)  &(mess['m9'].u_probe<=16.267e-5),
    (mess['m10'].u_probe>=8.117e-5) &(mess['m10'].u_probe<=16.295e-5),
    (mess['m11'].u_probe>=8.000e-5) &(mess['m11'].u_probe<=16.386e-5),
    (mess['m12'].u_probe>=8.000e-5) &(mess['m12'].u_probe<=16.350e-5),
    (mess['m13'].u_probe>=8.000e-5) &(mess['m13'].u_probe<=16.522e-5),
    (mess['m14'].u_probe>=8.389e-5) &(mess['m14'].u_probe<=16.640e-5),
]
nether2 = [
    None,
    None,
    None,
    None,
    None,
    None,
    (mess['m9'].u_probe>=13.767e-5) &(mess['m9'].u_probe<=16.267e-5),
    (mess['m10'].u_probe>=13.543e-5)&(mess['m10'].u_probe<=16.295e-5),
    (mess['m11'].u_probe>=13.776e-5)&(mess['m11'].u_probe<=16.386e-5),
    (mess['m12'].u_probe>=13.768e-5)&(mess['m12'].u_probe<=16.350e-5),
    (mess['m13'].u_probe>=14.000e-5)&(mess['m13'].u_probe<=16.522e-5),
    (mess['m14'].u_probe>=13.712e-5)&(mess['m14'].u_probe<=16.640e-5),
]
rising = [
    None, 
    None, 
    None,
    None,
    (mess['m7'].u_spule>=3.600e-5),
    (mess['m8'].u_spule>=5.300e-5),
    (mess['m9'].u_spule>=7.600e-5),
    (mess['m10'].u_spule>=10.1e-5),
    (mess['m11'].u_spule>=12.35e-5),
    (mess['m12'].u_spule>=13.2e-5),
    (mess['m13'].u_spule>=14.5e-5),
    (mess['m14'].u_spule>=16.1e-5),
]


# -------------------------------- set flags ----------------------------------
use_temperature_masks=True
if use_temperature_masks:
    masks=temps
    base_string='After temp approx\n constant (t>t_temp)'
    base_name='_temps_masked'
else:
    masks=t0s
    base_string='After sweep of U_spule\n started (t>t0)'
    base_name='_t0s_masked'


include_grounds = True
include_tops    = True
include_nether = True

fit_grounds = True
fit_tops    = True
fit_nether  = True

test_mode           = True
plot_temperatures   = False
plot_intersec       = True


# ------------------------------ main workflow --------------------------------
results = {k: {} for k in mess}

from core import load_its90_table_orig
tab_pascal = load_its90_table_orig(MYPY/'ITS90.csv')
tab_bar=tab_pascal.copy()
names=list(tab_bar.dtype.names)
names[names.index('p_kPa')]='p_bar'
tab_bar.dtype.names=tuple(names)
tab_bar['p_bar']*=1e-2


it = mess.items()
if test_mode:
    it = islice(it, 13)  # set int to the number of items you want to iterate over when test_mode is activated

for i,(key,m) in enumerate(it):
    # -------------------- workflow to get temperatures -----------------------
    t=m.time[masks[i]]
    up = m.u_p[masks[i]]
    uperr = m.u_p_err[masks[i]]
    uab=m.u_ab[masks[i]]
    uaberr=m.u_ab_err[masks[i]]

    from calibration import odr_res
    slope = odr_res['m'];   dslope = odr_res['cov'][0,0]
    offset = odr_res['b'];  doffset = odr_res['cov'][1,1]
    covar = odr_res['cov'][0,1]
    papprox,dpapprox = calib.linear_p_from_Up(
        m=slope,dm=dslope,b=offset,db=doffset,covar=covar,
        Up=up,Up_err=uperr,
    )
    sfmask = (papprox<5.207*1e-2)
    nfmask = (papprox>4.836*1e-2)
    sfpapprox = papprox[sfmask]
    nfpapprox = papprox[nfmask]
    print(len(sfpapprox))
    print(len(nfpapprox))
    print(len(papprox))

    if len(sfpapprox)==len(papprox) and len(nfpapprox)==0:  print('Superfluid');    beta = calib.sfbeta;    cov = calib.sfcov
    else:   print('Normalfluid');   beta = calib.nfbeta;    cov = calib.nfcov

    T,dT = calib.inv_band_from_params_jax(beta,cov,uab,uaberr)
    T = np.array(T)
    dT = np.array(dT)

    meanT=np.mean(T);   medianT=np.median(T)
    idxmin = np.where(T==T.min());  idxmax = np.where(T==T.max())
    Tilo = T[idxmin]-dT[idxmin];     Tihi = T[idxmax]+dT[idxmax]

    results[key]['temp'] = {
        'med':  medianT,
        'men':  meanT,
        'min':  Tilo,
        'max':  Tihi,
        'err':  [Tilo-medianT,Tihi-medianT]
    }

    plot_temperatures = True
    if plot_temperatures and __name__=="__main__":
        fig1,ax1=plt.subplots(nrows=1,ncols=1,figsize=figrect())
        h1=ax1.errorbar(
            x=t,y=papprox,yerr=dpapprox,
            label=r'Vapor-pressure, approximated with linear fitting of $U_P$',
            **cfg.err_kw(elw=0.1),
            fmt='x',ls='',mfc='cyan',mec='cyan',#zorder=5,
            )
        handles = [h1]
        for bar in h1[2]:
            bar.set_alpha(0.7)
        ax1.set_title(key)
        ax1.set_ylabel(r'Pressure $P$ [bar]')
        ax1.set_xlabel(r"Time $t$ [s]")

        tt=np.linspace(t.min(),t.max(),300)

        secax1=ax1.twinx()
        secax1.patch.set_alpha(0.0)
        secax1.set_zorder(ax1.get_zorder())
        ax1.patch.set_alpha(0.0)
        use_legacy_temp=False
        if use_legacy_temp:
            h2 = secax1.errorbar(
                x=x1,y=T,yerr=T_err_asym,
                label=r'ITS90-converted temperature',
                **cfg.err_kw(elw=0.2), fmt='+',ls='',mfc='blue',mec='blue'
                )
            h3=secax1.hlines(medianT,xmin=x1.min()-(x1.max()-x1.min())/20, xmax=x1.max()+(x1.max()-x1.min())/20, color='k', label=r'Temperature Median')
            h4=secax1.hlines(meanT,xmin=x1.min()-(x1.max()-x1.min())/20, xmax=x1.max()+(x1.max()-x1.min())/20, color='orange', label=r'Temperature Mean')
            # secax1.axhline(medianT, xmin=x1.min()-(x1.max()-x1.min())/20, xmax=x1.max()+(x1.max()-x1.min())/20, color='k', label=r'Temperature Median')   <- Axes.axhline thinks xmin,xmax as ratios of plotrange. Plot in data-coordinates:
            secax1.fill_between(tt,medianT-medianT_minus_sigma,medianT+medianT_plus_sigma,alpha=0.3,lw=0.3,color='k')       # <- may have troube with a float instead of an array as yy, Better:
            # secax1.axhspan(medianT-medianT_minus_sigma, medianT+medianT_plus_sigma, alpha=0.3, lw=0.3)
            handles.append([h2,h3,h4])
        else:
            h2=secax1.errorbar(x=t,y=T,yerr=dT,label='Calibrated Temperature',**cfg.err_kw(elw=0.2), fmt='+',ls='',mfc='blue',mec='blue')
            h3=secax1.hlines(medianT,xmin=t.min()-(t.max()-t.min())/20, xmax=t.max()+(t.max()-t.min())/20, color='k', label=r'Temperature Median')
            h4=secax1.hlines(meanT,xmin=t.min()-(t.max()-t.min())/20, xmax=t.max()+(t.max()-t.min())/20, color='orange', label=r'Temperature Mean')
            secax1.fill_between(tt,Tilo,Tihi,alpha=0.3,lw=0.3,color='k')
            handles.extend([h2,h3,h4])
        secax1.set_ylabel(r'Temperature $T$ [K]')

        labels=[h.get_label() for h in handles]
        secax1.legend(handles,labels,loc='upper right',fontsize=6)


    # ----------------------- workflow to get critical field ---------------------------
    get_critical=True
    if get_critical:
        x=m.u_spule[masks[i]]
        x_err=m.u_spule_err[masks[i]]
        y=m.u_probe[masks[i]]
        y_err=m.u_probe_err[masks[i]]

        I_arr=x/R
        I_err=x_err/R
        Hall=np.zeros_like(m.u_spule)
        Hallerr=np.zeros_like(m.u_spule_err)
        Hall[masks[i]]=H(I_arr,N,Ro,Ri,L)
        Hallerr[masks[i]]=Herr(I=I_arr,I_err=I_err)
        Harr=H(I_arr,N,Ro,Ri,L)
        Haerr=Herr(I=I_arr,I_err=I_err)

        rose=np.r_[False,np.diff(m.u_spule)>0]
        rose=min_run(rose,3)                    # 'averages' over 3 consecutive points -> less prone to outliers
        fall=np.r_[False,np.diff(m.u_spule)<0]
        fall=min_run(fall,3)

        if include_grounds or include_nether or include_tops:  _with='_with'
        else:                                                   _with=''

        if fit_grounds or fit_tops: fit_string='_fits';     xx=np.linspace(Harr.min(),Harr.max(),1000)
        else:                       fit_string=''

        maskg=masks[i]&grounds[i]
        xg=m.u_spule[maskg]
        Hg=Hall[maskg]
        yg=m.u_probe[maskg]
        if fit_grounds:
            sHg=Hallerr[maskg]
            syg=m.u_probe_err[maskg]
            resg=odr_line(Hg,yg,sHg,syg)
            print(f"{key:>5}:  U_probe = ({resg['m']:.6g}±{resg['sm']:.2g}) • H + "
                f"({resg['b']:.6g}±{resg['sb']:.2g});  χ^2_ν={resg['rchi2']:.2g}")
            yyg,syyg= odr_prediction_band(xx,resg['m'],resg['b'],resg['cov'])

        maskt=masks[i]&tops[i]
        xt=m.u_spule[maskt]
        Ht=Hall[maskt]
        yt=m.u_probe[maskt]
        if fit_tops:
            sHt=Hallerr[maskt]
            syt=m.u_probe_err[maskt]
            rest=odr_line(Ht,yt,sHt,syt)
            print(f"{key:>5}:  U_probe = ({rest['m']:.6g}±{rest['sm']:.2g}) • H + "
                f"({rest['b']:.6g}±{rest['sb']:.2g});  χ^2_ν={rest['rchi2']:.2g}")
            yyt,syyt= odr_prediction_band(xx,rest['m'],rest['b'],rest['cov'])

        maskn=(masks[i]&nether[i]) if include_nether else masks[i]
        xn=m.u_spule[maskn]
        Hn=Hall[maskn]
        yn=m.u_probe[maskn]
        if rising[i] is None and fit_nether:
            syn=m.u_probe_err[maskn]
            sHn=Hallerr[maskn]
            inv=odr_line(yn,Hn,syn,sHn)
            yyn=np.linspace(y.min(),y.max(),1000)
            xxn,sxxn=odr_prediction_band(yyn,inv['m'],inv['b'],inv['cov'])
            print(f"{key:>5}: H = ({inv['m']:.6g} ± {inv['sm']:.2g}) · U + ({inv['b']:.6g} ± {inv['sb']:.2g}); χ²_ν={inv['rchi2']:.2g}")

        if rising[i] is not None:
            maskr1=(maskn&((rising[i]&~nether2[i])&rose)) if nether2[i] is not None else (maskn&(rising[i]&rose))
            # maskr1=min_run(maskr1,3)
            xr1=m.u_spule[maskr1]
            Hr1=Hall[maskr1]
            yr1=m.u_probe[maskr1]

            maskr2=(maskn&((rising[i]&nether2[i])&rose)) if nether2[i] is not None else np.zeros_like(masks[i])
            # maskr2=min_run(maskr2,3)
            xr2=m.u_spule[maskr2]
            Hr2=Hall[maskr2]
            yr2=m.u_probe[maskr2]

            maskf=maskn&fall
            xf=m.u_spule[maskf]
            Hf=Hall[maskf]
            yf=m.u_probe[maskf]

            if fit_nether:
                syr1=m.u_probe_err[maskr1]
                sHr1=Hallerr[maskr1]
                inv=odr_line(yr1,Hr1,syr1,sHr1)
                yyn=np.linspace(y.min(),y.max(),1000)
                xxn,sxxn=odr_prediction_band(yyn,inv['m'],inv['b'],inv['cov'])
                print(f"{key:>5}: H = ({inv['m']:.6g} ± {inv['sm']:.2g}) · U + ({inv['b']:.6g} ± {inv['sb']:.2g}); χ²_ν={inv['rchi2']:.2g}")

        if fit_grounds and fit_nether:
            interstring='_intersection'
            print("grounds: sm^2 vs cov[0,0] ->", resg['sm']**2, resg['cov'][0,0], "  χ²ν=", resg['rchi2'])
            print("nether: sm^2 vs cov[0,0] ->", inv['sm']**2, inv['cov'][0,0], "  χ²ν=", inv['rchi2'])
            # Draw axes-aligned ellipse:
            # Hstar_ax,Ustar_ax,sH_ax,sU_ax = intersection_ground_nether(res,inv,scale_by_rchi2=False)

            Hstar, Ustar, sH, sU, covHU, Sigma = intersection_ground_nether(resg, inv, scale_by_rchi2=False, return_cov=True)
            print(f"{key}: Intersection at H* = {Hstar:.6g} ± {sH:.2g},  U* = {Ustar:.6g} ± {sU:.2g}")

            results[key]['crit'] = {
                'Hstar':    Hstar,
                'sH':       sH,
                'Ustar':    Ustar,
                'sU':       sU,
                'covHU':    covHU,
            }
        else: interstring=''

        if plot_intersec and __name__=="__main__":
            fig2,ax2=plt.subplots(nrows=1,ncols=1,figsize=figrect())
            ax2.errorbar(x=Harr,xerr=Haerr,y=y,yerr=y_err,label=f'{key}: {base_string}',**cfg.err_kw(),fmt='x',ls='',mfc='cyan',mec='cyan',)
            if include_tops:
                top_string='_tops'
                ax2.scatter(x=Ht,y=yt,label=f'tops {key}',c='orange',zorder=4,marker='+')
                if fit_tops:
                    ax2.plot(xx,yyt,lw=0.8,label=f'{key}: ODR top-fit')
                    ax2.fill_between(xx,yyt-syyt,yyt+syyt,alpha=0.15,linewidth=0.3)
            else: top_string=''
            if include_grounds:
                ground_string='_grounds'
                ax2.scatter(x=Hg,y=yg,label=f'grounds {key}',c='orange',zorder=4,marker='x')
                if fit_grounds:
                    ax2.plot(xx,yyg,lw=0.8,label=f'{key}: ODR ground-fit')
                    ax2.fill_between(xx,yyg-syyg,yyg+syyg,alpha=0.15,linewidth=0.3)
            else: ground_string=''
            if rising[i] is None:
                if include_nether:
                    nether_string='_nether'
                    ax2.scatter(x=Hn,y=yn,label=f'the inbetweeners {key}',zorder=4,c='purple')
                else: nether_string=''
            else:
                if include_nether:
                    nether_string='_nether'
                    ax2.scatter(x=Hr1,y=yr1,label=f'the rising {key}',zorder=4,c='red',marker='^',s=6)
                    if nether2[i] is not None:
                        ax2.scatter(x=Hr2,y=yr2,label=f'The rising 2 {key}',zorder=4,c='pink',marker='^',s=6)
                    ax2.scatter(x=Hf,y=yf,label=f'the falling {key}',zorder=4,c='red',marker='v',s=6)
                else: nether_string=''
            if fit_nether:
                ax2.plot(xxn,yyn,lw=0.8,label=f'{key}: ODR slope-fit')
                ax2.fill_betweenx(yyn,xxn-sxxn,xxn+sxxn,alpha=0.15,linewidth=0.3)
            ax2.legend(loc='best',fontsize=6)
            ax2.set_xlabel(r"Magnetic field at center of coil")
            ax2.set_ylabel(r"$U_{\mathrm{Probe}}$")
            if fit_grounds and fit_nether:
                draw_sigma_ellipse(ax2,Hstar,Ustar,sH,sU,n_sigma=1.0,edgecolor='w',lw=1.0,alpha=1.0,zorder=8)
                draw_cov_ellipse(ax2, Hstar, Ustar, Sigma, n_sigma=1.0, edgecolor='k',zorder=9)
                ax2.plot(Hstar,Ustar,'o',ms=3,mfc='w',mec='k',zorder=8)

    if not test_mode and __name__=="__main__":
        if plot_temperatures:
            fig_name = f'{m.source.stem}{base_name}_temp'
            destiny=FIG_DIR/fig_name;   print('Files saved as: '+str(destiny)+'(.png/.pdf)')
            fig1.savefig(fname=destiny.with_suffix('.pdf'))
            fig1.savefig(fname=destiny.with_suffix('.png'))
        if plot_intersec:
            fig2_name = f'{m.source.stem}{base_name}_mag_field{_with}{ground_string}{top_string}{nether_string}{fit_string}{interstring}'
            destiny2 = FIG_DIR/fig2_name; print('Files saved as: '+cfg.user_stripped(destiny2)+'(.png/pdf)')
            fig2.savefig(fname=destiny2.with_suffix('.pdf'))
            fig2.savefig(fname=destiny2.with_suffix('.png'))
    else: cfg.prRed('Script executed in TEST-MODE -> No images saved, No dict exported')

    plt.show()
    plt.close()

print(results)