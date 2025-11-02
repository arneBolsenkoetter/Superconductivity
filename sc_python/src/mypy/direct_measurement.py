# ~/path/to/Superconductivity/sc_python/src/mypy/direct_measurement.py
from __future__ import annotations

import numpy as np
import config as cfg
import calibration as calib
import critical_field as cfield
import matplotlib.pyplot as plt

from core import Measurement
from config import user_stripped
from pathlib import Path
from calibration import odr_res
from critical_field import min_run
from matplotlib.legend_handler import HandlerTuple

# print(user_stripped(Path(__file__).resolve()))
cfg.configure()
# --- 1) Importing 'LHJG__Supraleitung_15' as Measurement-class ---
m15 = Measurement.from_npz(cfg.DATA_CLEAN/"LHJG__Supraleitung_15.npz")
# --- 1.1) Defining variables for data ---
up = m15.u_p
dup = m15.u_p_err
uab = m15.u_ab
duab = m15.u_ab_err
usample = m15.u_probe
dusample = m15.u_probe_err
# --- 2) Approximating pressure to make informed decision how to calculate temperature ---
p_approx,dp_approx = calib.linear_p_from_Up(
    m=odr_res['m'],dm=odr_res['sm'],b=odr_res['b'],db=odr_res['sb'],covar=odr_res['cov'][0,1],
    Up=up,Up_err=dup,
)
# --- 2.1) Base decision on pressures associated to closest to T_c from ITS90.csv ---
sfmask = (p_approx<5.207e-2)
nfmask = (p_approx>4.836e-2)
bothmask = sfmask&nfmask
# --- 2.1.1) Define maks to filter points into two phases w/ overlap ---
sfonly = sfmask & ~bothmask
nfonly = nfmask & ~bothmask
# --- 2.2) Initialise new array with string allocating point-pressure to phases ---
zuordnung = np.full(p_approx.shape,'undetermined',dtype='U32')
zuordnung[sfonly] = 'Exclusive Superfluid'
zuordnung[nfonly] = 'Exclusive Normalfluid'
zuordnung[bothmask] = 'Super- and Normalfluid'
# print(zuordnung)
# --- 2.3) print list of indices of those points which are within the overlap ---
idx_both = np.where(zuordnung=='Super- and Normalfluid')[0]
idx_list = idx_both.tolist()
# print(idx_list)     # empty: no points in overlap

# --- 3) now, that phase has been analysed, calculate temperatures accordingly ---
T,dT = calib.inv_band_from_params_jax(calib.nfbeta,calib.nfcov,uab,duab)
# print(T)
# --- 3.1) Mask different regions ---
ground_plateau_mask = (T<3.68)
top_mask = (T>3.74)
flanks = (usample>=1.0e-4)&(usample<=1.4e-4)
rising = np.r_[False,np.diff(T)>0]
rising = min_run(rising,3)
falling = np.r_[False,np.diff(T)<0]
falling = min_run(falling,3)
# --- 3.2) ODR: linear fit of 'top'-plateau points ---
top_linear_fit = cfield.odr_line(x=T[top_mask],y=usample[top_mask],sx=dT[top_mask],sy=dusample[top_mask])
xxt = np.linspace(T[flanks].min(),T.max(),512)
yyt,dyyt = cfield.odr_prediction_band(xxt, m=top_linear_fit['m'], b=top_linear_fit['b'], cov=top_linear_fit['cov'])
# --- 3.3) ODR: linear fit of 'falling-flank'-points;   CAVEAT: fitted (Y,X) not (X,Y) due to almost vertical slope ---
falling_linear_fit = cfield.odr_line(x=usample[falling&flanks], y=T[falling&flanks], sx=usample[falling&flanks], sy=dT[falling&flanks])
yyf = np.linspace(usample.min(),usample.max(),512)
xxf,dxxf = cfield.odr_prediction_band(yyf, m=falling_linear_fit['m'], b=falling_linear_fit['b'], cov=falling_linear_fit['cov'])
# --- 3.4) Get intersection of both linear fits ---
Tstar,Ustar,dTstar,dUstar,covStar,sigmaStar = cfield.intersection_ground_nether(top_linear_fit,falling_linear_fit,scale_by_rchi2=False,return_cov=True)

# --- 4) Plot constituents ---
# --- 4.0) Define figure,axis and lists to contain handles or labels ---
afig,aax=plt.subplots(nrows=1,ncols=1,figsize=cfg.figrect())
axitems = []
handles = []
labels  = []
# --- 4.1) Plot all points (T,U_sample) ---
a0 = aax.errorbar(
    x=T,xerr=dT,
    y=usample,yerr=dusample,
    ls='', **cfg.err_kw(elw=0.2),
);  axitems.append(a0)
for bar in a0[2]:   bar.set_alpha(0.3)
handles.append(a0); labels.append(r'Temperatures computed with textit{normal-fluid} parameters')
# a1 = aax.errorbar(
#     x=T[rising&flanks], xerr=dT[rising&flanks],
#     y=usample[rising&flanks], yerr=dusample[rising&flanks],
#     ls='',
# );  axitems.append(a1)
# --- 4.2) plot points of the upper plateau with errors, according fit and confidence band  ---
a2 = aax.errorbar(
    x=T[top_mask],xerr=dT[top_mask],
    y=usample[top_mask],yerr=dusample[top_mask],
    ls='', **cfg.err_kw(elw=0.2),
);  axitems.append(a2)
for bar in a2[2]:  bar.set_alpha(0.5)
handles.append(a2)
labels.append(r'Masked with bounds in $U_\mathrm{Sample}$ and a test (smoothed over 3 consecutive points) if $U_\mathrm{Sample}$ inclreased over time')
a2fit = aax.plot(
    xxt,yyt,
    label='_nolegend_',
);  axitems.append(a2fit)
a2confidenceband = aax.fill_between(
    xxt,yyt-dyyt,yyt+dyyt,
    color=a2fit[0].get_color(), alpha=0.3, lw=0,
);  axitems.append(a2confidenceband)
handles.append((a2fit[0],a2confidenceband))
labels.append(r'ODR-fit $\pm$ 1$\sigma$')
# --- 4.3) plot points inside the falling flank separately with errors, fit and confidence band ---
a3 = aax.errorbar(
    x=T[falling&flanks],xerr=dT[falling&flanks],
    y=usample[falling&flanks],yerr=dusample[falling&flanks],
    ls='', **cfg.err_kw(elw=0.2),
);  axitems.append(a3)
handles.append(a3)
labels.append(r'Masked with $T>3.74\,\mathrm{K}$')
for bar in a3[2]:   bar.set_alpha(0.5)
a3fit = aax.plot(
    xxf,yyf,
    label='_nolegend_',
);  axitems.append(a3fit)
a3confidenceband = aax.fill_betweenx(
    yyf,xxf-dxxf,xxf+dxxf,
    color=a3fit[0].get_color(), alpha=0.3, lw=0,
); axitems.append(a3confidenceband)
handles.append((a3fit[0],a3confidenceband))
labels.append(r'ODR-fit $\pm$ 1$\sigma$')
# --- 4.4) Plot intersection of top- and falling-fit with confidence ellipse ---
a4ellipse = cfield.draw_cov_ellipse(
    aax,Tstar,Ustar,sigmaStar,
    edgecolor='k',zorder=3,
)
a4marker = aax.plot(
    Tstar,Ustar,
    'o',zorder=a4ellipse.get_zorder(),mec='k',mfc='w',
    label='_nolegend_'
)
handles.append((a4ellipse,a4marker[0]))
labels.append(r'Intersection point of linear fits with $1\sigma$-ellipse')

aax.legend(handles, labels, 
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
)

plt.show()
plt.close()