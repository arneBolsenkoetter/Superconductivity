# ~/path/to/Superconductivity/sc_python/src/mypy/critical_plot.py
from __future__ import annotations

import core
import json
import numpy as np
import config as cfg
import matplotlib.pyplot as plt

from pprint import pprint
from config import figrect
from scipy.odr import ODR,Model,RealData
from scipy.stats import t as student_t
from critical_field import results

cfg.configure()
test_mode=True

# -------------------------------- functions/helpers -----------------------------------
def H_of_T(params,T):
    H0,T0 = params
    return H0 * (1 - (T/T0)**2)


# ------------------------ data-extraction from calibration.py -------------------------
# pprint(results, width=100, sort_dicts=True, compact=True)
exclude = []#['m14']
results = {k: v for (k,v) in results.items() if k not in exclude}
pprint(results,width=100,compact=True)
T = np.asarray([results[key]['temp']['med'] for key in results])
sT = np.stack([np.squeeze(results[k]['temp']['err']) for k in results],axis=1)
Tmin = np.asarray([np.asarray(results[k]['temp']['min']).item() for k in results],dtype=float)
Tmax = np.asarray([np.asarray(results[k]['temp']['max']).item() for k in results],dtype=float)
Terr = np.vstack([T-Tmin,Tmax-T])
# print(Terr,sT,sep='\n\n');exit()
H = np.asarray([results[k]['crit']['Hstar'] for k in results])
sH = np.asarray([results[k]['crit']['sH'] for k in results])
# -------------- computing secondary X for linear fit to seed H0 --------------
y = T**2
sy = np.vstack([np.sqrt(4*(sT[0,:]*T)**2), np.sqrt(4*(sT[1,:]*T)**2)])


# ------------ seed: weighted linear fit of H = a + b*T^2 (weights 1/sH^2) -------------
w = 1.0 / (np.asarray(sH,float)**2)
X = np.vstack([np.ones_like(y), y]).T
W = np.diag(w)
XTWX = X.T @ W @ X
XTWy = X.T @ W @ y
# Weighted least squares normal equations
WX = X * w[:, None]
beta = np.linalg.lstsq(WX.T @ X, WX.T @ H, rcond=None)[0]
a, b = beta
H0_seed = a
Tc_seed = np.sqrt(-H0_seed / b) if b < 0 and H0_seed > 0 else T.max() * 1.2


# ------------------ fitting a linear function to H over T2 with ODR -------------------
def linear_function(beta,X):
    return beta[0] + beta[1]*X

data =  RealData(y, H, sx=sy.mean(axis=0), sy=sH)  # pick an sx (e.g., symmetrize)
odr =   ODR(data, Model(linear_function), beta0=[a,b])
out =   odr.run()
a,b =   out.beta
cov =   out.cov_beta
resv =  out.res_var
sa,sb = out.sd_beta
H0 = a                  # <------------------------------------------------------------- Seed for H0
sH0 = np.sqrt(cov[0,0])

tt2 = np.linspace(0,y.max(),512)
ft2 = linear_function(out.beta,tt2)
# Jacobian rows [1, x] for each x on the grid
J2 = np.column_stack([np.ones_like(tt2), tt2])
# variance of fitted mean at each x (propagate parameter covariance)
# If your sH are absolute sigmas, cov is already scaled properly by ODR.
# If they are only relative, ODR typically returns cov scaled by resv; either way:
var_mean2 = np.einsum('ij,jk,ik->i', J2, cov, J2)          # shape (512,)
s_mean2   = np.sqrt(var_mean2)
# Optional: 95% CI for the mean (uses normal approx or t with dof = n-2)
n2, p2 = len(H), 2
t952 = student_t.ppf(0.975, df=max(n2-p2, 1))
Hc0_952 = (H0 - t952*sH0, H0 + t952*sH0)
ci68_lo2, ci68_hi2 = ft2 - s_mean2, ft2 + s_mean2
ci95_lo2, ci95_hi2 = ft2 - t952*s_mean2, ft2 + t952*s_mean2
print(f"H_c(0) = {H0:.6g} ± {sH0:.2g} T  (1σ)")
print(f"-T/T0  = {b:.6g} ± {sb:.2g}")
print(f"95% CI: [{Hc0_952[0]:.6g}, {Hc0_952[1]:.6g}] T")
# Optional: prediction band (mean band + residual variance in y-direction).
# With heteroscedastic data, this is an approximation; resv is a sensible global scale.
s_pred2 = np.sqrt(s_mean2**2 + resv)
pb95_lo2, pb95_hi2 = ft2 - t952*s_pred2, ft2 + t952*s_pred2


# ---------------------- fitting a squared function to H_c over T ----------------------
sTsym   = np.maximum(-sT[0,:],sT[1,:])
# print(sT,sTsym,sep='\n\n');exit()
sHsym   = np.asarray(sH,float)
# print(sH);exit()
HTbeta0 = [float(H0),float(np.sqrt(-H0/b))]
HTdata  = RealData(T,H,sx=sTsym,sy=sHsym)
HTodr   = ODR(HTdata,Model(H_of_T),beta0=HTbeta0)
HTout   = HTodr.run()

H0_hat, Tc_hat = HTout.beta
sH0_hat, sTc_hat = HTout.sd_beta
HTcov = HTout.cov_beta      # 2x2 parameter covariance
HTresv = HTout.res_var      # residual variance scale used internally

HTcorr = HTcov / np.sqrt(np.outer(np.diag(HTcov), np.diag(HTcov)))
print("\n--- Nonlinear ODR: H(T) = H0 * (1 - (T/Tc)^2) ---")
print(f"H0  = {H0_hat:.5g} ± {sH0_hat:.2g} T   (1σ)")
print(f"Tc  = {Tc_hat:.5g} ± {sTc_hat:.2g} K   (1σ)")
print("corr(H0,Tc) =", f"{HTcorr[0,1]:+.3f}")
# 95% CI for parameters (rough t-approx, df = N-2)
t95 = student_t.ppf(0.975, df=max(len(T)-2, 1))
print(f"95% CI H0: [{H0_hat - t95*sH0_hat:.6g}, {H0_hat + t95*sH0_hat:.6g}] T")
print(f"95% CI Tc: [{Tc_hat - t95*sTc_hat:.6g}, {Tc_hat + t95*sTc_hat:.6g}] K")
# H_c(0) = 8706.66 ± 1.1e+02 T  (1σ)
# -T/T0  = -615.792 ± 4.5
# 95% CI: [8453.39, 8959.93] T

# --- Nonlinear ODR: H(T) = H0 * (1 - (T/Tc)^2) ---
# H0  = 8707.5 ± 54 T   (1σ)
# Tc  = 3.76 ± 0.0041 K   (1σ)
# corr(H0,Tc) = -0.399
# 95% CI H0: [8585.76, 8829.16] T
# 95% CI Tc: [3.75075, 3.76924] K

tt = np.linspace(0.0, max(T.max()*1.05, Tc_hat*1.05), 512)
ft = H_of_T(HTout.beta, tt)
# Jacobian wrt parameters at each tt:
# ∂H/∂H0 = 1 - (T/Tc)^2
# ∂H/∂Tc = H0 * 2*T^2 / Tc^3
J0 = 1.0 - (tt / Tc_hat) ** 2
J1 = H0_hat * (2.0 * tt**2) / (Tc_hat**3)
J = np.column_stack([J0, J1])
# Var of fitted mean at each tt:
var_mean = np.einsum('ij,jk,ik->i', J, HTcov, J)
s_mean = np.sqrt(np.clip(var_mean, 0, np.inf))
ci68_lo, ci68_hi = ft - s_mean, ft + s_mean
ci95_lo, ci95_hi = ft - t95*s_mean, ft + t95*s_mean
# Optional: 95% prediction band (~ mean band + residual variance). For heteroscedastic data
# this is a global approximation; still useful for visualization.
s_pred = np.sqrt(s_mean**2 + HTresv)
pb95_lo, pb95_hi = ft - t95*s_pred, ft + t95*s_pred


if __name__=="__main__":
    plot_linear=True
    if plot_linear:
        fig1,ax1 = plt.subplots(nrows=1,ncols=1,figsize=figrect())
        a1_1 = ax1.errorbar(x=y,xerr=sy,y=H,yerr=sH,ls='')
        a1_2 = ax1.plot(tt2,ft2)
        ax1.fill_between(tt2, ci95_lo2, ci95_hi2, alpha=0.2, label='95% CI (mean)')
        # If you also want the prediction band, uncomment:
        ax1.fill_between(tt2, pb95_lo2, pb95_hi2, alpha=0.12, label='95% prediction band')
        ax1.set_xlabel(r"Temperature squared $T^2\quad[\mathrm{K}^2]$")
        ax1.set_ylabel(r"Critical Field $H_C\quad[\mathrm{T}]$")
        plt.show()

    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=figrect())
    sT[0,:] *= -1
    a1 = ax.errorbar(
        x=T,xerr=sT,y=H,yerr=sH,
        ls='',
    )
    ax.plot(tt, ft, label='ODR fit')
    ax.fill_between(tt, ci95_lo, ci95_hi, alpha=0.2, label='95% CI (mean)')
    ax.fill_between(tt, pb95_lo, pb95_hi, alpha=0.12, label='95% prediction band')
    plt.show()
    plt.close()
