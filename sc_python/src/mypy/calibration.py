# ~/path/to/Superconductivity/sc_pthon/src/mySC/calibration.py
# --------------------------------- imports --------------------------------------------
from __future__ import annotations

import sys
import core
import numpy as np
import sympy as sp
import config as cfg
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from core import T_from_p_kpa, p_kpa_from_T, ITS90_STRUCT
from scipy import odr
from config import figrect
from pathlib import Path
from mypy.core import Measurement
from scipy.odr import Model, RealData
from scipy.optimize import brentq, curve_fit
from scipy.interpolate import PchipInterpolator
from sc_data.Druck_Spaunung_Korrelation import npDV_si as npDV


# ------------------------------- macro data -------------------------------------------
this_file_path = Path(__file__).resolve()
print(cfg.user_stripped(this_file_path))
this_file_name = this_file_path.stem
print(this_file_name)
# if not cfg.breaker():   sys.exit(0)


# --------------------------------- ITS90 functions ------------------------------------
TABLEIII = cfg.attrmap({# available at https://www.bipm.org/documents/20126/41791796/ITS-90.pdf/b85c434b-16be-4ff1-f8fc-0c93027452d4?version=1.3&t=1593077509109&download=true
    'HE3':      {# 0.65K to 3.2K
        'A0':       1.053477,
        'Ai':       [.980106,.676380,.372692,.151656,-.002263,.006596,.088966,-.004770,-.054943],
        'B':        7.3,
        'C':        4.3,
        'valid_t':  [0.65,3.2],
        'valid_p':  [0.050*1e3,32.01*1e3],
    },
    'HE4_sf':   {# 1.25K to 2.1768K
        'A0':       1.392408,
        'Ai':       [.527153,.166756,.050988,.026514,.001975,-.017976,.005409,.013259,.0,],
        'B':        5.6,
        'C':        2.9,
        'valid_t':  [1.25,2.18],
        'valid_p':  [0.115*1e3,5.082*1e3],
    },
    'HE4_nf':   {# 2.1768K to 5.0K
        'A0':       3.146631,
        'Ai':       [1.357655,.413923,.091159,.016349,.001826,-.004325,-.004973,.0,.0],
        'B':        10.3,
        'C':        1.9,
        'valid_t':  [2.17,5.00],
        'valid_p':  [4.958*1e3,209.97*1e3],
    }
})

def T90(p:np.ndarray, a0:float, ai:np.ndarray, b:float, c:float, units:str|None='Pa', checkmate:bool=False) -> np.ndarray:
    """
        !!! 
        CAUTION: Only valid for the range of 1.25K to 2.1768K and 2.1768K to 5.0K!
        This function (compared to 'T90_over_vapour_pressure') incorporates pressures p in [Pa] OR [bar]
        !!!
        Official documentation:    https://its-90.com/definitions/!
        Computes the temperatures T90 for (an array of) pressures p.

            T90  =  A0 + Σ[Ai*ln(p)*B/C]^i (i=1,i=15)

        Parameters:
        p:      np.ndarray((N,),float),         1D array of pressures
        a0:     float,                          some parameter, apparently on the its-90 website, but couldn't find it
        ai:     np.ndarray((15,),float),        1D array of parameters for the sum over i
        b:      float,                          the same applys as for a0
        c:      float,                          -"-
        units:  string='bar'|'Pa'|'kPa'|'hPa',  units of the pressures parsed hereto T90(p,...)

        Returns:
        T90:    np.ndarray((N,),float),         temperatures corresponding to parsed pressures

        P.S.:   First make sure you have converted your pressures to [Pa], then simply set the flag 'HaveUcheckedtheUnitQuestionmark'=True
        P.P.S:  Unit conversions:
            1bar = 100000Pa <-> 1Pa = 1e-5 bar
            1hPa = 10000Pa  <-> 1Pa = 1e-4 hPa
            1kPa = 1000Pa   <-> 1Pa = 13-3 kPa
    """
    if not checkmate:
        raise ValueError("The pressure parsed to HE4_T90_over_vapour_pressure might have the wrong unit.\nCheck out this function's doc-string (<function>.__doc__) AND its definition to see how you can avoid this captcha.")
    if units=='bar':    p = np.asarray(p,float)*1e5
    elif units=='hPa':  p = np.asarray(p,float)*1e4
    elif units=='kPa':  p = np.asarray(p,float)*1e3
    elif units=='Pa':   p = np.asarray(p,float)
    else:               raise ValueError("ValueError - 'units' must be one of:   'bar' | 'hPa' | 'kPa' | 'Pa'.")
    ai = np.asarray(ai,float)
    # --- actual computation:
    x = (np.log(p)-b)/c
    powers = x[:,None]**np.arange(1,len(ai)+1)
    return a0 + powers @ ai

def linear_p_from_Up(
    m:float, dm:float, b:float, db:float, covar:float, 
    Up:np.ndarray, Up_err:np.ndarray|None=None
    ) -> tuple[np.ndarray, np.ndarray] :
    """
        Linear model 
            p = m*Up + b 
        Returns: 
            (p,σ)

        with:   σ = np.sqrt(var(p)),    var(p) = (m*σU)^2 + (Up*σm)^2 + σb^2 + 2*Up*cov(m,b)
    """
    Up     = np.asarray(Up, float)
    Up_err = np.asarray(Up_err if Up_err is not None else 0.0, float)

    p =     m*Up + b
    var_p = (m*Up_err)**2 + (dm*Up)**2 + db**2 + 2*Up*covar
    dp =    np.sqrt(var_p)
    return p, dp

def interp_with_grid_error(
    xM, yM, X, dX:np.ndarray|None=None,
    err_mode = 'max',
    return_yerr=True,
    ):
    xM = np.asarray(xM,float)
    yM = np.asarray(yM,float)
    if xM.ndim != 1 or yM.ndim != 1 or xM.size != yM.size:
        raise ValueError("xM and yM must be 1D arrays of the same length")
    if xM.size < 2:
        raise ValueError("Need at least two map points for interpolation")
    order = np.argsort(xM)
    xM = xM[order]; yM=yM[order]

    X  = np.asarray(X,float)
    mask = (X>xM.min())&(X<xM.max())
    X = X[mask]
    dX = dX[mask]

    j = np.searchsorted(xM,X,side='right')
    i = j-1

    x0 = xM[i]
    x1 = xM[j]
    y0 = yM[i]
    y1 = yM[j]

    w = x1 - x0
    h = y1 - y0
    dx_left = X-x0

    Y = y0 + dx_left/w * h
    dy_left = Y-y0
    dy_right = y1-Y

    if err_mode=='max':
        dY = np.maximum(dy_left,dy_right)
    elif err_mode=='min':
        dY = np.minimum(dy_left,dy_right)
    elif err_mode=='slope' and dX is not None:
        m = h/w
        dY = dX*m
    elif err_mode=='asym' and dX is not None:
        Xmin = X-dX;    Xplu = X+dX
        jmin = np.searchsorted(xM,Xmin,side='right');   imin=jmin-1
        jplu = np.searchsorted(xM,Xplu,side='right');   iplu=jplu-1
        Ymin = yM[imin] + (Xmin-xM[imin])/(xM[jmin]-xM[imin]) * (yM[jmin]-yM[imin])
        Yplu = yM[iplu] + (Xplu-xM[iplu])/(xM[jplu]-xM[iplu]) * (yM[jplu]-yM[iplu])
        dY = np.vstack([Y-Ymin,Yplu-Y])
    else:
        raise ValueError("err_mode must either be 'max', 'min', 'slope' or 'asym'.")

    return (Y,dY) if return_yerr else (Y,None)


# ------------------------------------- shockley ---------------------------------------
alpha_,beta_,gamma_,T_ = sp.symbols('alpha_ beta_ gamma_ T_',real=True)
shockley = alpha_ + beta_*sp.exp(-gamma_*T_)
dY_dT = sp.diff(shockley,T_)
dY_da = sp.diff(shockley,alpha_)
dY_db = sp.diff(shockley,beta_)
dY_dg = sp.diff(shockley,gamma_)

Y =     sp.lambdify((alpha_,beta_,gamma_,T_),shockley,'numpy')
dY_da = sp.lambdify((alpha_,beta_,gamma_,T_),dY_da,'numpy')
dY_db = sp.lambdify((alpha_,beta_,gamma_,T_),dY_db,'numpy')
dY_dg = sp.lambdify((alpha_,beta_,gamma_,T_),dY_dg,'numpy')


# ------------------------------ functions for odr-fit ---------------------------------
# def mdl_shockley(params,T):
#     alpha,beta,gamma = params
#     return alpha + beta*np.exp((-1)*gamma*T)

# def mdl_jacobean_params(params,T):
#     a,b,g = params
#     e = np.exp(-g*T)
#     J = np.vstack([np.ones_like(T),e,-b*T*e])
#     return J[None,:,:]                          # odr expects shape (1,p,N)

# def mdl_jacobean_variables(params,T):
#     a,b,g = params
#     d = -b*g*np.exp(-g*T)
#     return d[None,None,:]

# def fit_odr(mdl, beta0, T, U, sT, sU, maxit=200):
#     data = odr.RealData(T, U, sx=np.maximum(sT, eps), sy=np.maximum(sU, eps))
#     model = odr.Model(lambda b, x: mdl(b, x))
#     out = odr.ODR(data, model, beta0=beta0, maxit=maxit).run()
#     beta = out.beta
#     cov  = out.cov_beta
#     # compute plain RSS in U for AIC comparison
#     Uhat = mdl(beta, T)
#     rss  = float(np.sum((U - Uhat)**2))
#     k    = len(beta)
#     n    = len(T)
#     aic  = 2*k + n*np.log(rss/n) if rss > 0 else -np.inf
#     return dict(beta=beta, cov=cov, rss=rss, aic=aic, out=out)

def mdl_shockley(params,T,use_theta:bool=True):
    if use_theta:
        a,b,th = params
        g = np.exp(th)
    else:
        alpha,beta,gamma = params
    return a + b*np.exp(-g*T)

def mdl_jacobean_params(params,T,use_theta:bool=True):
    if use_theta:
        a,b,th = params
        g = np.exp(th)
        e = np.exp(-g*T)
        dya = np.ones_like(T)
        dyb = e
        dyth = -b*T*e*g
        return np.vstack([dya,dyb,dyth])[None,:,:]
    else:
        a,b,g = params 
        e = np.exp(-g*T)
        J = np.vstack([np.ones_like(T),e,-b*T*e])
        return J[None,:,:]                          # odr expects shape (1,p,N)

def mdl_jacobean_variables(params,T,use_theta:bool=True):
    if use_theta:
        a,b,th = params
        g = np.exp(th)
    else: a,b,g = params
    a,b,g = params
    d = -b*g*np.exp(-g*T)
    return d[None,None,:]

def inv_shockley(params,U):
    alpha,beta,gamma = params
    return (-1)*np.log((U-alpha)/beta)/gamma

def guess_params(X, Y, log_slope:bool=False):
    X = np.asarray(X,float)
    Y = np.asarray(Y, float)
    a0 = np.percentile(Y,10)
    b0 = max(np.median(Y)-a0,1e-12)
    if log_slope:
        xq = np.quantile(X, [0.2, 0.8])
        yq = np.clip(np.quantile(Y - a0, [0.2, 0.8]), 1e-12, None)
        g0 = abs(np.log(yq[0]/yq[1]) / (xq[1]-xq[0])) if xq[1] > xq[0] else 1.0
    else:
        dx = max(X.max() - X.min(), 1e-12)
        g0 = 1.0/dx
    return np.array([a0,b0,g0],float)


# ------------------------------------- helpers ----------------------------------------
def pltexit():
    plt.close()
    exit()


########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################



# ------------------------------------- tester -----------------------------------------
tester = False
if tester:
    # print(npDV['volts'])
    print(npDV)
    exit()

# ----------------------------------- mpl config ---------------------------------------
cfg.export_final=False
cfg.configure(overrides={'lines.linewidth':0.6})


# get parameters for fit between pressure and voltage (Druck_Spaunung_KOrrelation.py)
# --- linear fit: Pressure = m * Voltage + b (weighted by y-errors) ---
x  = npDV['volts'].astype(float)
sx = npDV['volts_err'].astype(float)
y  = npDV['druck'].astype(float)
sy = npDV['druck_err'].astype(float)

# keep only finite points
mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
x, sx, y, sy = x[mask], sx[mask], y[mask], sy[mask]

# Weighted least squares with NumPy (weights = 1/sigma_y)
# global m,b,var_m,m_err,var_b,b_err,covar,varco
(m, b), cov = np.polyfit(x, y, deg=1, w=1.0/sy, cov=True)

odr_res = core.odr_line(x,y,sx,sy)
xx=np.linspace(x.min(),x.max(),512)
pp,spp=core.odr_prediction_band(xx,odr_res['m'],odr_res['b'],odr_res['cov'])

odr_inv = core.odr_line(y,x,sy,sx)
yy=np.linspace(y.min(),y.max(),512)
vv,svv=core.odr_prediction_band(yy,odr_inv['m'],odr_inv['b'],odr_inv['cov'])

var_m = cov[0,0]
var_b = cov[1,1]
covar = cov[0,1]
varco = cov[1,0]
m_err, b_err = np.sqrt(np.diag(cov))

# for export:
slope=m
slope_var=var_m
slope_err=m_err
offset=b
offset_var=var_b
offset_err=b_err



# -------------------------------------------------- main workflow -----------------------------------------------------
if __name__=="__main__":

    initial_tests = False
    # --- only some checks, from the early stages of this script ---
    if initial_tests:
        print(m)
        print(odr_res['m'])
        print(b)
        print(odr_res['b'])
        print(m_err)
        print(odr_res["sm"])
        print(b_err)
        print(odr_res['sb'])
        print(odr_res['cov'][0,1])

        show_calib=False
        if show_calib:
            calib_fig,calib_axes=plt.subplots(nrows=1,ncols=2,figsize=figrect(ncols=2))

            cal1=calib_axes[0].errorbar(x=x,xerr=sx,y=y,yerr=sy)
            calib_axes[0].plot(xx,pp,zorder=3)
            calib_axes[0].fill_between(xx,pp-spp,pp+spp,alpha=0.3,zorder=2)
            calib_axes[0].set_xlabel(rf'Voltage $U_P$ [V]')
            calib_axes[0].set_ylabel(rf'Pressure $P$ [bar]')

            cal2=calib_axes[1].errorbar(x=y,xerr=sy,y=x,yerr=sx)
            calib_axes[1].plot(yy,vv,zorder=3)
            calib_axes[1].fill_between(yy,vv-svv,vv+svv,alpha=0.3,zorder=2)
            calib_axes[1].set_ylabel(rf'Voltage $U_P$ [V]')
            calib_axes[1].set_xlabel(rf'Pressure $P$ [bar]')
            plt.show()

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
            fitboi=True
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

            print('','Saved at', cfg.user_stripped(out_png), 'and', out_pdf, sep='\n')

            plt.show()
            # plt.close()


    # ------------------------ get preassure p_p from Up -------------------------------
    m1 = Measurement.from_npz(cfg.DATA_DIR/"clean/LHJG__Supraleitung_1.npz")
    print(m1.columns)

    p_bar, dp_bar = linear_p_from_Up(
        Up=m1.u_p, Up_err=m1.u_p_err,
        m=m, dm=m_err, 
        b=b, db=b_err, 
        covar=covar,
    )
    p,dp = linear_p_from_Up(
        Up=m1.u_p,Up_err=m1.u_p_err,
        m=odr_res['m'],dm=odr_res['sm'],
        b=odr_res['b'],db=odr_res['sb'],
        covar=np.sqrt((odr_res['cov'][0,1])**2),
    )
    show_pressure=False
    if show_pressure:
        pressure_fig,pressure_ax=plt.subplots(nrows=1,ncols=2,figsize=figrect(ncols=2))
        f1_bar=pressure_ax[0].errorbar(
            x=m1.u_p,xerr=m1.u_p_err,
            y=p_bar,yerr=dp_bar,
            **cfg.err_kw(),
        )
        f1=pressure_ax[1].errorbar(
            x=m1.u_p,xerr=m1.u_p_err,
            y=p,yerr=dp,
            **cfg.err_kw(elw=0.2,),
        )
        for bar in f1[2]:
            bar.set_alpha(0.3)
        pressure_ax[0].set_xlabel(r'Voltage $U_P$')
        pressure_ax[0].set_ylabel(r'Pressure $P$')
        plt.show()


    # --------------------- get temp t_p from preassure p_p ----------------------------
    # ----- 1) Get temperature-pressure correlation from the ITS90.csv -----
    T_tab_K   = ITS90_STRUCT['T_K'].astype(float)
    p_tab_kPa = ITS90_STRUCT['p_kPa'].astype(float)     # ITS90.csv is in kPa -> conversion:
                                                        # 1 bar = 100000 Pa
                                                        # kPa -> bar  : multiply by 0.01
                                                        # bar -> kPa  : multiply by 100.0
    p_tab_bar = p_tab_kPa * 0.01

    # ----- 1.1)[detour]: fit ITS90 -----
    detour = True
    if detour:
        show_c=True
        if show_c:
            cfig,cax=plt.subplots(nrows=1,ncols=1,figsize=figrect())
            c1 = cax.plot(
                p_tab_bar,T_tab_K,
                label=r'ITS90-calibration',
            )
            xx = np.linspace(p_tab_bar.min(),p_tab_bar.max(),1024)
            whole_range = True
            if whole_range:     xxHE3=xx;   xxHE4sf=xx;     xxHE4nf=xx
            else:
                xxHE3 =     np.linspace(TABLEIII.HE3.valid_p[0], TABLEIII.HE3.valid_p[1],512)*1e-2
                xxHE4sf =   np.linspace(TABLEIII.HE4_sf.valid_p[0], TABLEIII.HE4_sf.valid_p[1],512)*1e-2
                xxHE4nf =   np.linspace(TABLEIII.HE4_nf.valid_p[0], TABLEIII.HE4_nf.valid_p[1],512)*1e-2

            # yyHE3 =     T90(xxHE3, TABLEIII.HE3.A0, TABLEIII.HE3.Ai, TABLEIII.HE3.B, TABLEIII.HE3.C, units='bar', checkmate=True)
            # c2 =        cax.plot(xx,yyHE3,  label=r'$T_{90}$ for $^{3}$HE')
            yyHE4sf =   T90(xxHE4sf, TABLEIII.HE4_sf.A0, TABLEIII.HE4_sf.Ai, TABLEIII.HE4_sf.B, TABLEIII.HE4_sf.C, units='bar', checkmate=True)
            c3 =        cax.plot(xx,yyHE4sf,label=r'$T_{90}$ for $^{4}$HE - superfluid')
            yyHE4nf =   T90(xxHE4nf, TABLEIII.HE4_nf.A0, TABLEIII.HE4_nf.Ai, TABLEIII.HE4_nf.B, TABLEIII.HE4_nf.C, units='bar', checkmate=True)
            c4 =        cax.plot(xx,yyHE4nf,label=r'$T_{90}$ for $^{3}$HE - normalfluid')

            cax.set_ylim(1.0,T_tab_K.max())
            cax.set_xlabel(r'Pressure $P\quad[\mathrm{bar}]$')
            cax.set_ylabel(r'Temperature $T\quad[\mathrm{K}]$')
            cax.legend()
            plt.show()

    # --- defining mask so interpolation doesn't look outside the table ---
    tab_mask = (p>=p_tab_bar.min())&(p<=p_tab_bar.max())
    sf_mask = tab_mask&(p<5.207*1e-2)
    nf_mask = tab_mask&(p>4.836*1e-2)
    # --- applying the masks ---
    p_bar=p_bar[tab_mask];  dp_bar=dp_bar[tab_mask]
    p_sf = p[sf_mask];      dp_sf = dp[sf_mask]
    p_nf = p[nf_mask];      dp_nf = dp[nf_mask]
    p = p[tab_mask];        dp = dp[tab_mask]
    U = m1.u_ab[tab_mask];  dU = m1.u_ab_err[tab_mask]
    # --- interpolating the temperature ---
    dT_method = 'asym'
    T,dT = interp_with_grid_error(p_tab_bar,T_tab_K,p,dp,err_mode=dT_method)
    # --- calculating the temperature ---
    Tsf = T90(p_sf, TABLEIII.HE4_sf.A0, TABLEIII.HE4_sf.Ai, TABLEIII.HE4_sf.B, TABLEIII.HE4_sf.C, units='bar', checkmate=True)
    Tnf = T90(p_nf, TABLEIII.HE4_nf.A0, TABLEIII.HE4_nf.Ai, TABLEIII.HE4_nf.B, TABLEIII.HE4_nf.C, units='bar', checkmate=True)



    # --- little figure of pressure p and voltage U_AB over temperature T ---
    show_d=True
    if show_d:
        dfig,dax = plt.subplots(ncols=1,nrows=1,figsize=figrect())
        dsecax = dax.twinx()
        d1 = dax.errorbar(x=Tsf,y=p_sf,yerr=dp_sf,label=r'$T_{90}$ superfluid')
        d2 = dax.errorbar(x=Tnf,y=p_nf,yerr=dp_nf,label=r'$T_{90}$ normalfluid')
        dsec1 = dsecax.errorbar(x=T,xerr=dT,y=U,yerr=dU,label=r'Voltage $U_\mathrm{AB}$')
        # labels = [dax.get_labels(),dsexax.get_labels()]
        # handles = [dax.get_handles(),dsecax.get_handles()]
        dax.legend()
        # dax.set_xlim(2.0,2.35)
        # dax.set_ylim(0.0,0.1)
        dax.set_xlabel(r'Temperature $T\quad[\mathrm{K}]$')
        dax.set_ylabel(r'Pressure $P(U_\mathrm{P})\quad[\mathrm{bar}]$')
        dsecax.set_ylabel(r'Voltage $U_\mathrm{AB}\quad[\mathrm{V}]$')
        plt.show()
        pltexit()

    # ------------------------ Allen-Bradley Calibration -------------------------------
    if not cfg.confirm():   pltexit()
    # --- defining masks for super-fluid and normal-fluid phases (and additional) ---
    mask_infty = np.isfinite(T) & np.isfinite(dT) & np.isfinite(U) & np.isfinite(dU)
    mask0 = (T<=4.2)
    mask_zoom = (1.73<=T)&(T<=2.2)
    mask_normalfluid = (2.2<=T)&(T<=4.2)
    mask_superfluid = (T>=1.78)&(T<=2.15) 
    # mask_superfluid = ((2.07<=T)&(2.15>=T))     # there is a little secondary (smaller bump at approx 2.05)
    mask = mask_infty&mask0
    # --- apply masks ---
    T_sf, U_sf, sU_sf = T[mask_superfluid],  U[mask_superfluid],  dU[mask_superfluid]
    T_nf, U_nf, sU_nf = T[mask_normalfluid], U[mask_normalfluid], dU[mask_normalfluid]
    # --- special treatment for the errors in T ---
    if dT_method is 'asym': sT_sf = dT[:,mask_superfluid];  sT_nf = dT[:,mask_normalfluid]
    else:                   sT_sf = dT[mask_superfluid];    sT_nf = dT[mask_normalfluid]
    # ODR need symmetric errors:
    sT_sf = np.maximum(sT_sf[0,:],sT_sf[1,:]);  sT_nf = np.maximum(sT_nf[0,:],sT_nf[1,:])

    init_params_sf = guess_params(T_sf,U_sf,log_slope=True);   init_params_nf = guess_params(T_nf,U_nf)

    shockley_model = Model(
        fcn=lambda beta, x: mdl_shockley(beta, x),
        fjacb=mdl_jacobean_params,
        fjacd=mdl_jacobean_variables
        )
    data_sf = RealData(x=T_sf,y=U_sf,sx=sT_sf,sy=sU_sf);    data_nf = RealData(x=T_nf,y=U_nf,sx=sT_nf,sy=sU_nf)
    odr_sf = odr.ODR(data_sf,shockley_model,beta0=init_params_sf,maxit=300)
    odr_nf = odr.ODR(data_nf,shockley_model,beta0=init_params_nf,maxit=300)
    out_sf = odr_sf.run();  out_nf = odr_nf.run()

    asf,bsf,gsf = out_sf.beta;          anf,bnf,gnf = out_nf.beta
    sasf,sbsf,sgsf = out_sf.sd_beta;    sanf,sbnf,sgnf = out_nf.sd_beta
    cov_sf = out_sf.cov_beta;           cov_nf = out_nf.cov_beta
    res_var_sf = out_sf.res_var;        res_var_nf = out_nf.res_var

    print('Scipy.Optimize.Curve_Fit:')
    cf_best_sf,cf_cov_sf = curve_fit(
        lambda T,a,b,g: a+b*np.exp(-g*T),
        T_sf,U_sf,
        )
    cf_best_nf,cf_cov_nf = curve_fit(
        lambda T,a,b,g: a+b*np.exp(-g*T),
        T_nf,U_nf,
        )

    plot_AB=True
    if plot_AB:
        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=figrect(ncols=2))

        sfxx = np.linspace(T[mask_zoom].min(),T[mask_zoom].max(),512)
        sfyy = mdl_shockley(out_sf.beta,sfxx)
        sfcf = mdl_shockley(cf_best_sf,sfxx)
        ax[0].errorbar(x=T[mask_zoom],xerr=dT[:,mask_zoom],y=U[mask_zoom],yerr=dU[mask_zoom])
        ax[0].set_title(r'Superfuid Phase')
        ax[0].set_xlabel(r'Temperatur [K]')
        ax[0].set_ylabel(r'Spannung $U_\mathrm{AB}$')
        ax[0].plot(sfxx,sfyy,zorder=3)
        ax[0].plot(sfxx,sfcf,zorder=3)

        nfxx = np.linspace(T_nf.min(),T_nf.max(),512)
        nfyy = mdl_shockley(out_nf.beta,nfxx)
        nfcf = mdl_shockley(cf_best_nf,nfxx)
        ax[1].errorbar(x=T_nf,xerr=sT_nf,y=U_nf,yerr=sU_nf)
        ax[1].set_title(r'Normalfluid Phase')
        ax[1].set_xlabel(r'Temperatur [K]')
        ax[1].set_ylabel(r'Spannung $U_\mathrm{AB}$')
        ax[1].plot(nfxx,nfyy,zorder=3)
        ax[1].plot(nfxx,nfcf,zorder=3)

        plt.show()

    pltexit()
