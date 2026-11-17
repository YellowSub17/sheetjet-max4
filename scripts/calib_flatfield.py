'''
Per-module flat field for the JUNGFRAU, from the intensity steps across module boundaries.

For each reference run, the run mean is divided by its azimuthal average (to take out the q
dependence) and the median ratio across every boundary between neighbouring modules is measured
in strips just outside the masked edges. One gain per module is then fitted to all the boundary
ratios (least squares on log ratios, geometric mean fixed to 1). Because each ratio compares
pixels only ~60 px apart, smooth real anisotropy (polarization, sheet absorption) mostly cancels.

The flat field is the median of the per-run gains, written as a full-detector image of factors
to divide the data by (1 in the gaps), plus the per-run fits so drift between runs can be checked.

usage: python scripts/calib_flatfield.py --name water --group water [--date 20261105]
'''
import argparse
import datetime
import warnings

import h5py
import numpy as np

from const import *
from rundata import RunData
from process import find_runs



STRIP_WIDTH = 10   # px, width of the strips compared across each boundary
STRIP_OFFSET = 6   # px, from the module edge (the edge mask is 4 px)
MIN_PX = 20        # boundaries with fewer good columns/rows than this are skipped



def module_pairs():
    '''(i, j, axis) for every pair of neighbouring modules: j is below (axis 0) or right of (axis 1) i.'''
    pairs = []
    for i, (r0, c0) in enumerate(MOD_ORIGINS):
        for j, (r1, c1) in enumerate(MOD_ORIGINS):
            if r1 == r0 + MOD_PITCH[0] and c1 == c0:
                pairs.append((i, j, 0))
            elif c1 == c0 + MOD_PITCH[1] and r1 == r0:
                pairs.append((i, j, 1))
    return pairs


def boundary_strips(res, i, j, axis):
    '''Strips of res either side of the boundary between modules i and j.'''
    (r0, c0), (r1, c1) = MOD_ORIGINS[i], MOD_ORIGINS[j]
    w, off = STRIP_WIDTH, STRIP_OFFSET
    if axis == 0:
        a = res[r0+MOD_SHAPE[0]-off-w:r0+MOD_SHAPE[0]-off, c0:c0+MOD_SHAPE[1]]
        b = res[r1+off:r1+off+w, c1:c1+MOD_SHAPE[1]]
    else:
        a = res[r0:r0+MOD_SHAPE[0], c0+MOD_SHAPE[1]-off-w:c0+MOD_SHAPE[1]-off]
        b = res[r1:r1+MOD_SHAPE[0], c1+off:c1+off+w]
    return a, b


def residual_image(mean, mask, q_map, nq=1000):
    '''mean / azimuthal average at each pixel's q, nan where masked.'''
    good = (mask == 0) & (mean >= 0)
    edges = np.linspace(0, q_map.max() + 1e-6, nq + 1)
    qidx = np.digitize(q_map, edges) - 1
    s = np.bincount(qidx[good], weights=mean[good], minlength=nq)
    c = np.bincount(qidx[good], minlength=nq)
    profile = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(good, mean / profile[qidx], np.nan)


def fit_module_gains(run):
    '''Relative gain per module (geometric mean 1), its uncertainty, and the boundary ratios used.'''
    mask = run.create_run_mask()
    mean = run.read_proc('mean').astype(np.float64)
    res = residual_image(mean, mask, run.q_map)

    rows, logs, ratios = [], [], []
    for i, j, axis in module_pairs():
        a, b = boundary_strips(res, i, j, axis)
        with warnings.catch_warnings():   # columns/rows that are fully masked (e.g. the beamstop)
            warnings.simplefilter('ignore', RuntimeWarning)
            ratio = np.nanmedian(b, axis=axis) / np.nanmedian(a, axis=axis)
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) < MIN_PX:
            continue
        row = np.zeros(len(MOD_ORIGINS))
        row[j], row[i] = 1, -1
        rows.append(row)
        logs.append(np.log(np.median(ratio)))
        ratios.append((i, j, np.median(ratio)))

    # boundary ratios + constraint that the log gains sum to 0
    A = np.array(rows + [np.ones(len(MOD_ORIGINS))])
    y = np.array(logs + [0])
    log_gain, *_ = np.linalg.lstsq(A, y, rcond=None)

    misfit = A[:-1] @ log_gain - y[:-1]
    dof = max(len(logs) - (len(MOD_ORIGINS) - 1), 1)
    cov = np.linalg.pinv(A.T @ A) * (misfit**2).sum() / dof
    gain = np.exp(log_gain)

    return gain, gain*np.sqrt(np.diag(cov)), ratios, misfit.std()


def flatfield_image(gains):
    '''Full-detector image of the module gains (1 outside the modules).'''
    ff = np.ones(DET_SHAPE, np.float32)
    for g, (r, c) in zip(gains, MOD_ORIGINS):
        ff[r:r+MOD_SHAPE[0], c:c+MOD_SHAPE[1]] = g
    return ff


def check_boundaries(run, gains):
    '''Boundary ratios after dividing the run mean by the given module gains.'''
    mask = run.create_run_mask()
    mean = run.read_proc('mean').astype(np.float64) / flatfield_image(gains)
    res = residual_image(mean, mask, run.q_map)
    out = []
    for i, j, axis in module_pairs():
        a, b = boundary_strips(res, i, j, axis)
        with warnings.catch_warnings():   # columns/rows that are fully masked (e.g. the beamstop)
            warnings.simplefilter('ignore', RuntimeWarning)
            ratio = np.nanmedian(b, axis=axis) / np.nanmedian(a, axis=axis)
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) >= MIN_PX:
            out.append(np.median(ratio))
    return np.array(out)




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Per-module flat field from boundary steps in reference runs.')
    parser.add_argument('--name', required=True, help='saved as data/proc/calib/flatfield_{name}.h5')
    parser.add_argument('--group', default='water', help='reference sample group (default water)')
    parser.add_argument('--date', type=int)
    args = parser.parse_args()

    runs, gains, errs, names = [], [], [], []
    for date, group, sub_group, run_number in find_runs(args.date, args.group):
        name = f'{date}/{group}-{sub_group}_{run_number}'
        try:
            run = RunData(date, group, sub_group, run_number, flatfield=None)
            g, e, ratios, misfit = fit_module_gains(run)
        except FileNotFoundError as err:
            print(f'skipping {name}: {err}')
            continue
        print(f'{name}: {len(ratios)} boundaries, fit misfit {100*misfit:.2f}%, gains {g.min():.3f}-{g.max():.3f}')
        runs.append(run); gains.append(g); errs.append(e); names.append(name)

    if not gains:
        raise SystemExit('no reference runs found')

    gains, errs = np.array(gains), np.array(errs)
    ff_gains = np.exp(np.median(np.log(gains), axis=0))
    ff_gains /= np.exp(np.log(ff_gains).mean())   # geometric mean 1

    print('\nflat field (module gains, detector layout, top row first):')
    for k in range(N_MODS[0]):
        print('   ' + '  '.join(f'{ff_gains[N_MODS[1]*k+m]:.3f}' for m in range(N_MODS[1])))
    if len(gains) > 1:
        print(f'spread between runs (max std over modules): {100*gains.std(axis=0).max():.2f}%')

    print('\nboundary ratios before -> after correction (rms deviation from 1):')
    for name, run in zip(names, runs):
        before = check_boundaries(run, np.ones(len(MOD_ORIGINS)))
        after = check_boundaries(run, ff_gains)
        print(f'   {name}: {100*np.sqrt(np.mean(np.log(before)**2)):.2f}% -> {100*np.sqrt(np.mean(np.log(after)**2)):.2f}%')

    path = f'{CALIB_DIR}/flatfield_{args.name}.h5'
    os.makedirs(CALIB_DIR, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('flatfield', data=flatfield_image(ff_gains), compression='gzip')
        f.create_dataset('module_gains', data=ff_gains)
        f.create_dataset('runs/names', data=np.array(names, dtype='S'))
        f.create_dataset('runs/module_gains', data=gains)
        f.create_dataset('runs/module_gain_errors', data=errs)
        f.attrs['description'] = 'per-module relative gain: divide data by flatfield'
        f.attrs['strip_width'] = STRIP_WIDTH
        f.attrs['strip_offset'] = STRIP_OFFSET
        f.attrs['created'] = datetime.datetime.now().isoformat(timespec='seconds')
    print(f'\nsaved {path}')
