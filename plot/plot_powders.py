import sys

import numpy as np
import matplotlib.pyplot as plt

from procdata import as_procdata




def plot_frame(p, i):
    '''I(q) of frame i (p: ProcData or proc file path).'''
    p = as_procdata(p)
    plt.figure(figsize=(8, 5))
    plt.plot(p.q, p.I[i])
    plt.xlabel('q')
    plt.ylabel('I')
    plt.title(f'{p.name}  frame {i}')


def plot_run(p):
    '''I(q) against frame, and the run-mean powder (p: ProcData or proc file path).'''
    p = as_procdata(p)
    q, powders = p.q, p.I

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    axs[0].imshow(powders, aspect='auto', origin='lower', extent=(q[0], q[-1], 0, powders.shape[0]))
    axs[0].set_ylabel('frame')
    axs[0].set_title(p.name)

    axs[1].plot(q, np.nanmean(powders, axis=0))
    axs[1].set_xlabel('q')
    axs[1].set_ylabel('mean I')
    fig.tight_layout()


if __name__=='__main__':

    proc_path = sys.argv[1]

    plot_run(proc_path)

    plt.show()
