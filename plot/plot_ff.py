import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from const import MOD_ORIGINS, MOD_SHAPE
from procdata import as_procdata



def plot_flatfield(p):
    '''Flat field used for this run's powders (from /powder/I), with each module's gain written on it
    (p: ProcData or proc file path).'''
    p = as_procdata(p)
    ff_path = p.flatfield_path
    if ff_path is None:
        print(f'{p.path}: no flat field used for the powders')
        return

    with h5py.File(ff_path, 'r') as f:
        ff = f['flatfield'][...].astype(float)
        gains = f['module_gains'][...]
    ff[p.mask == 1] = np.nan   # gaps

    span = np.nanmax(np.abs(ff - 1))
    plt.figure(figsize=(9, 8))
    plt.imshow(ff, cmap='RdBu_r', vmin=1-span, vmax=1+span, interpolation='nearest')
    plt.colorbar(label='relative module gain (data is divided by this)')
    for g, (r, c) in zip(gains, MOD_ORIGINS):
        plt.text(c + MOD_SHAPE[1]/2, r + MOD_SHAPE[0]/2, f'{g:.3f}', ha='center', va='center', fontsize=10,
                 color='white' if abs(g - 1) > 0.6*span else 'black')
    plt.title(f'{p.name}\nflat field: {ff_path.split("/")[-1]}')




if __name__=='__main__':

    proc_path = sys.argv[1]

    plot_flatfield(proc_path)

    plt.show()
