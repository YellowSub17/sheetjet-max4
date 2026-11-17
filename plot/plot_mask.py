import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from plot_powders import load_proc



# one colour per mask label: good, gaps, edges, seams, beamstop, blob, bad pixels, streaks
LABEL_COLOURS = ['white', 'black', 'tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'magenta', 'tab:purple']



def mask_labels(proc_path):
    '''Label names saved with the mask (e.g. edges-4px), falling back to generic names.'''
    with h5py.File(proc_path, 'r') as f:
        labels = f['mask'].attrs.get('labels')
    if labels is None:
        return ['good', 'gaps', 'edges', 'seams', 'beamstop', 'blob']
    return [l.decode() if isinstance(l, bytes) else str(l) for l in labels]


def plot_mask_values(proc_path):
    '''
        Left: the mask, one colour per label. Right: the run mean, with only the gaps, edges and
        seams (labels 1-3, the same in every run) blanked out, so the pixels the run-specific
        labels mask can be seen. The axes are shared, so zooming one zooms the other.
    '''
    p = load_proc(proc_path)
    labels = mask_labels(proc_path)

    cmap = ListedColormap(LABEL_COLOURS[:len(labels)])
    norm = BoundaryNorm(np.arange(len(labels)+1) - 0.5, len(labels))

    mean = p['mean'].astype(float)
    mean[np.isin(p['mask'], [1, 2, 3])] = np.nan   # negative/overflowing pixels stay, at the ends of the colour scale
    run_specific = p['mask'] >= 4

    fig, axs = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)

    im = axs[0].imshow(p['mask'], cmap=cmap, norm=norm, interpolation='nearest')
    cbar = fig.colorbar(im, ax=axs[0], ticks=range(len(labels)))
    cbar.ax.set_yticklabels([f'{i}: {l}' for i, l in enumerate(labels)])
    axs[0].set_title(f'{proc_path.split("/")[-1]}  mask values')

    # colour scale from the pixels that stay in (99.5th percentile, so strong gradients don't
    # saturate), so the masked ones stand out against it
    im = axs[1].imshow(mean, vmin=0, vmax=np.nanpercentile(mean[~run_specific], 99.5), interpolation='nearest')
    fig.colorbar(im, ax=axs[1])
    axs[1].set_title('run mean (gaps, edges, seams blanked)')


def plot_mask_overlay(proc_path):
    '''Run mean with every mask except the gaps drawn on top (fully opaque).
    Gaps are left as they are (nan in the mean, so they show up white).'''
    p = load_proc(proc_path)
    labels = mask_labels(proc_path)

    mean = p['mean'].astype(float)
    mean[mean < 0] = np.nan

    plt.figure(figsize=(8, 8))
    plt.imshow(mean, vmin=0, vmax=2*np.nanmedian(mean))
    overlay = list(range(2, len(labels)))   # everything but good and gaps
    for label in overlay:
        m = p['mask'] == label
        plt.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([LABEL_COLOURS[label]]), alpha=1,
                   interpolation='nearest')
    plt.plot(p['attrs']['beam_x'], p['attrs']['beam_y'], 'k+', ms=12)
    names = [f'{labels[i]} ({LABEL_COLOURS[i].replace("tab:", "")})' for i in overlay]
    plt.title(f'{proc_path.split("/")[-1]}\n' + '\n'.join(', '.join(names[k:k+3]) for k in range(0, len(names), 3)),
              fontsize=10)





if __name__=='__main__':

    proc_path = sys.argv[1]

    plot_mask_values(proc_path)
    plot_mask_overlay(proc_path)

    plt.show()
