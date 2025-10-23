# ~/path/to/Superconductivity/sc_python/src/mypy/critical_plot.py
from __future__ import annotations

import core
import json
import numpy as np
import config as cfg
import matplotlib.pyplot as plt

from pprint import pprint
from config import figrect
from critical_field import results

cfg.configure()


# ---------------------------------- functions/helpers -------------------------------------
def type_one_crit_field_of_temp(params,T):
    H0,T0 = params
    return H0 * (1 - (T/T0)**2)

pprint(results, width=100, sort_dicts=True, compact=True)

exclude = ['m14']
results = {k: v for (k,v) in results.items() if k not in exclude}
pprint(results,width=100,compact=True)
T = np.asarray([results[key]['temp']['med'] for key in results])
y = T**2
Tmin = np.asarray([results[k]['temp']['min'] for k in results])
Tmax = np.asarray([results[k]['temp']['max'] for k in results])
sT = np.vstack([T-Tmin,Tmax-T])
sy = np.sqrt(2)*sT*T
H = np.asarray([results[k]['crit']['Hstar'] for k in results])
sH = np.asarray([results[k]['crit']['sH'] for k in results])

# --- seed: weighted linear fit of H = a + b*T^2 (weights 1/sH^2) --------------
w = 1.0 / np.asarray(sH, dtype=float)**2
X = np.vstack([np.ones_like(T2), T2]).T
W = np.diag(w)
XTWX = X.T @ W @ X
XTWy = X.T @ W @ y
# Weighted least squares normal equations
WX = X * w[:, None]
beta = np.linalg.lstsq(WX.T @ X, WX.T @ H, rcond=None)[0]
a, b = beta
H0_seed = a
Tc_seed = np.sqrt(-H0_seed / b) if b < 0 and H0_seed > 0 else T.max() * 1.2

tt = np.linspace(0,T2.max(),512)
hh = a+b*tt

#  get first estimates from linear relation:
fig1,ax1 = plt.subplots(nrows=1,ncols=1,figsize=figrect())
a1_1 = ax1.errorbar(x=T2,xerr=sT2,y=H,yerr=sH,ls='')
a1_2 = ax1.plot(tt,hh)
plt.show()



# fig,ax=plt.subplots(nrows=1,ncols=1,figsize=figrect())
# a1 = ax.errorbar(x=T,xerr=sT,y=H,yerr=sH)

# plt.show()
# plt.close()
