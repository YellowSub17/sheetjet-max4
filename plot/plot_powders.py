import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt




def load_proc(proc_path):
    '''Everything in a proc file: datasets keyed by path, plus root attrs under 'attrs'.'''
    out = {}
    with h5py.File(proc_path, 'r') as f:
        out['attrs'] = dict(f.attrs)
        f.visititems(lambda name, obj: out.__setitem__(name, obj[...]) if isinstance(obj, h5py.Dataset) else None)
    return out


def plot_frame(proc_path, i):
    p = load_proc(proc_path)
    plt.figure(figsize=(8, 5))
    plt.plot(p['powder/q'], p['powder/I'][i])
    plt.xlabel('q')
    plt.ylabel('I')
    plt.title(f'{proc_path.split("/")[-1]}  frame {i}')


def plot_run(proc_path):
    p = load_proc(proc_path)
    q, powders = p['powder/q'], p['powder/I']

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    axs[0].imshow(powders, aspect='auto', origin='lower', extent=(q[0], q[-1], 0, powders.shape[0]))
    axs[0].set_ylabel('frame')
    axs[0].set_title(proc_path.split('/')[-1])

    axs[1].plot(q, np.nanmean(powders, axis=0))
    axs[1].set_xlabel('q')
    axs[1].set_ylabel('mean I')
    fig.tight_layout()


if __name__=='__main__':

    proc_path = sys.argv[1]

    plot_run(proc_path)

    plt.show()
