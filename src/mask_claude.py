import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from rundata import RunData


# Convention: True = masked (bad pixel)

# EIGER module layout: 6 x 3 modules of 514 x 1030 px, separated by 36 row / 8 col gaps
MOD_SHAPE = (514, 1030)
MOD_PITCH = (550, 1038)
N_MODS = (6, 3)

# seams between the 256 x 256 chips inside a module (module-local coords, 4 px wide)
SEAM_ROWS = [255]
SEAM_COLS = [255, 513, 771]
SEAM_WIDTH = 4

GAP_VALUE = -2147483648     # inter-module gap / flagged pixel
OVERFLOW_VALUE = 2147483647 # counter overflow

# misbehaving regions picked out by eye (row_start, row_stop, col_start, col_stop), full-detector coords
# streaks hanging below the seam in the bottom-middle module
MANUAL_REGIONS = [
    (3008, 3090, 1810, 1820),
    (3008, 3090, 1873, 1883),
    (3008, 3090, 1937, 1949),
    (3008, 3090, 2001, 2012),
]



def module_origins():
    return [(i*MOD_PITCH[0], j*MOD_PITCH[1]) for i in range(N_MODS[0]) for j in range(N_MODS[1])]


def gap_mask(shape):
    '''Pixels that are not on any module.'''
    mask = np.ones(shape, bool)
    for r, c in module_origins():
        mask[r:r+MOD_SHAPE[0], c:c+MOD_SHAPE[1]] = False
    return mask


def edge_mask(shape, width=3):
    '''Outer `width` pixels of every module (these read high: ~1.8x, 1.2x, 1.1x, 1.04x going inwards).'''
    m = np.zeros(MOD_SHAPE, bool)
    m[:width] = m[-width:] = True
    m[:, :width] = m[:, -width:] = True
    return tile_modules(shape, m)


def seam_mask(shape):
    '''Chip seams inside each module (the low-intensity grid).'''
    m = np.zeros(MOD_SHAPE, bool)
    for r in SEAM_ROWS:
        m[r:r+SEAM_WIDTH] = True
    for c in SEAM_COLS:
        m[:, c:c+SEAM_WIDTH] = True
    return tile_modules(shape, m)


def tile_modules(shape, module_mask):
    mask = np.zeros(shape, bool)
    for r, c in module_origins():
        mask[r:r+MOD_SHAPE[0], c:c+MOD_SHAPE[1]] = module_mask
    return mask


def crazy_mask(data, chunk=20):
    '''Pixels that are flagged, negative or overflowing in ANY frame of the run.'''
    mask = np.zeros(data.shape[1:], bool)
    for i in range(0, data.shape[0], chunk):
        d = data[i:i+chunk]
        mask |= ((d < 0) | (d == OVERFLOW_VALUE)).any(axis=0)
    return mask


def manual_mask(shape, regions=MANUAL_REGIONS):
    mask = np.zeros(shape, bool)
    for r0, r1, c0, c1 in regions:
        mask[r0:r1, c0:c1] = True
    return mask


def run_stats(data, chunk=20):
    '''Per-pixel mean and variance over the run, chunked to keep memory down.'''
    s = np.zeros(data.shape[1:])
    s2 = np.zeros(data.shape[1:])
    for i in range(0, data.shape[0], chunk):
        d = data[i:i+chunk].astype(np.float64)
        s += d.sum(axis=0)
        s2 += (d*d).sum(axis=0)
    n = data.shape[0]
    mean = s/n
    return mean, s2/n - mean**2


def spurious_mask(mean, var, mask, size=9, lo=0.5, hi=1.5, fano_max=3):
    '''Hot / dead / noisy pixels.

    - mean intensity far from the local median (already-masked pixels are excluded
      from the local median so seams and gaps don't drag it around)
    - variance/mean (Fano factor) well above Poisson (~0.8 for normal pixels here)
    '''
    f = np.where(mask, np.nan, mean)
    filled = np.where(mask, np.nanmedian(f), f)
    local = ndimage.median_filter(filled, size=size)
    ratio = f / local

    with np.errstate(divide='ignore', invalid='ignore'):
        fano = var / mean

    return ~mask & ((ratio < lo) | (ratio > hi) | (fano > fano_max))


def intensity_mask(frame, low=50, high=550, beamstop_grow=40, beamstop_min_size=1000):
    '''Run-specific cuts: beamstop (< low, grown to cover the penumbra) and over-exposed (> high).
    Kept separate from the global mask since the beamstop/exposure may change between runs.

    Only low clusters bigger than beamstop_min_size are grown (the beamstop is split into
    several pieces by the gaps/seams); isolated low pixels are masked without growing.'''
    low_px = frame < low
    labels, n = ndimage.label(low_px)
    sizes = ndimage.sum(low_px, labels, range(1, n+1))
    beamstop = np.isin(labels, np.where(sizes >= beamstop_min_size)[0] + 1)
    if beamstop_grow:
        beamstop = ndimage.binary_dilation(beamstop, iterations=beamstop_grow)
    return beamstop | low_px | (frame > high)


def global_mask(run, edge_width=3, spurious_kw=None):
    '''First-pass global mask from one run. Returns (mask, components dict, mean frame).'''
    shape = run.data.shape[1:]
    comps = {
        'gap': gap_mask(shape),
        'edge': edge_mask(shape, edge_width),
        'seam': seam_mask(shape),
        'crazy': crazy_mask(run.data),
        'manual': manual_mask(shape),
    }
    mask = np.logical_or.reduce(list(comps.values()))

    mean, var = run_stats(run.data)
    mean[mask] = np.nan

    comps['spurious'] = spurious_mask(mean, var, mask, **(spurious_kw or {}))
    # mask |= comps['spurious']
    return mask, comps, mean


def plot_mask(frame, comps, vmax=None):
    vmax = vmax or 2*np.nanmedian(frame)
    img = np.clip(np.nan_to_num(frame)/vmax, 0, 1)[..., None].repeat(3, axis=-1)
    colours = {'gap': (0.2, 0.2, 0.2), 'edge': (0, 0.6, 1), 'seam': (0, 0.3, 1),
               'crazy': (1, 0, 1), 'manual': (1, 0.5, 0), 'spurious': (1, 0, 0), 'intensity': (0, 1, 0)}
    for name, m in comps.items():
        img[m] = colours.get(name, (1, 1, 0))
    if 'gap' in comps:
        img[comps['gap']] = colours['gap']
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(', '.join(f'{k}={c}' for k, c in colours.items() if k in comps), fontsize=8)





if __name__=='__main__':

    r = RunData()

    mask, comps, mean = global_mask(r)

    # counts are pixels not already covered by the gaps
    for name, m in comps.items():
        m = m if name == 'gap' else m & ~comps['gap']
        print(f'{name:10s}{m.sum():9d}  {100*m.mean():.2f}%')
    print(f'{"total":10s}{mask.sum():9d}  {100*mask.mean():.2f}%')

    np.save('./mask_global.npy', mask)

    intens = intensity_mask(mean) & ~mask
    print(f'{"intensity":10s}{intens.sum():9d}  {100*intens.mean():.2f}%  (run-specific, not in global mask)')
    np.save('./mask_intensity.npy', intens)

    plot_mask(mean, comps)
    plot_mask(mean, {**comps, 'intensity': intens})

    plt.figure()
    plt.hist(mean[~mask], bins=200)
    plt.yscale('log')

    print(mask.shape)

    plt.show()
