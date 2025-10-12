import numpy as np
import config as cfg
import matplotlib.pyplot as plt

from config import figrect
from core import Measurement


mess: dict[str,Measurement] = {}
for i in range(3,16):
    mess[f'm{i}']=Measurement.from_npz(cfg.DATA_DIR/f"clean/LHJG__Supraleitung_{i}.npz")
    print(f'Loaded LHJG_Supraleitung_{i}.npz into mess["m{i}"]')

masks = [
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
masks_ground = [
    (mess['m3'].time>=305.0)&(mess['m3'].time<=315.0),
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
bounds = np.array([(305, 315), (420, 430), (550, 560)])  # shape (n, 2)
# mask = ((time[:, None] >= bounds[:, 0]) & (time[:, None] <= bounds[:, 1])).any(axis=1)


for i,(key,m) in enumerate(mess.items()):
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=figrect())
    u_spule=m.u_spule[masks[i]]
    u_spule_err=m.u_spule_err[masks[i]]
    u_probe=m.u_probe[masks[i]]
    u_probe_err=m.u_probe_err[masks[i]]
    ax.errorbar(x=u_spule,xerr=u_spule_err,y=u_probe,yerr=u_probe_err)
    cfg.savefig(fig,name=f'{m.source}_with_err',ext='png')
    cfg.savefig(fig,name=f'{m.source}_with_err')
    plt.show()
    plt.close()