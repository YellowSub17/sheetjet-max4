import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from const import MOD_ORIGINS, MOD_SHAPE
from plot_powders import load_proc



def plot_flatfield(proc_path):
    '''Flat field used for this run's powders (from /powder/I), with each module's gain written on it.'''
    with h5py.File(proc_path, 'r') as f:
        ff_path = f['powder/I'].attrs.get('flatfield') if 'powder/I' in f else None
    if ff_path is None or ff_path == 'None':
        print(f'{proc_path}: no flat field used for the powders')
        return

    with h5py.File(ff_path, 'r') as f:
        ff = f['flatfield'][...].astype(float)
        gains = f['module_gains'][...]
    ff[load_proc(proc_path)['mask'] == 1] = np.nan   # gaps

    span = np.nanmax(np.abs(ff - 1))
    plt.figure(figsize=(9, 8))
    plt.imshow(ff, cmap='RdBu_r', vmin=1-span, vmax=1+span, interpolation='nearest')
    plt.colorbar(label='relative module gain (data is divided by this)')
    for g, (r, c) in zip(gains, MOD_ORIGINS):
        plt.text(c + MOD_SHAPE[1]/2, r + MOD_SHAPE[0]/2, f'{g:.3f}', ha='center', va='center', fontsize=10,
                 color='white' if abs(g - 1) > 0.6*span else 'black')
    plt.title(f'{proc_path.split("/")[-1]}\nflat field: {ff_path.split("/")[-1]}')




if __name__=='__main__':

    proc_path = sys.argv[1]

    plot_flatfield(proc_path)

    plt.show()
