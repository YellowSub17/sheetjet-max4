'''
Does the high intensity blob (around row 1747, col 737) change shape within a run or between runs?

The blob sits on top of the sample scattering, and both change brightness from run to run, so a
fixed intensity threshold would mix up "brighter" with "bigger". Instead, per image:
    excess = image - background                 (background = median of the ROI)
    blob   = excess > FRAC * peak excess        (peak = mean of the brightest N_PEAK pixels)
so the blob outline is brightness independent, and the brightness (peak excess) is tracked separately.

Shape is compared with the IoU (overlap / union) of blob masks and the centroid position:
    within a run:   each frame vs the run mean
    between runs:   run means vs each other

usage: python scripts/check_blob.py [--date 20261105] [--group water]
'''
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt

from const import *
from rawdata import RawData
from process import find_runs



ROI = (slice(1650, 1850), slice(620, 860))   # rows, cols around the blob
FRAC = 0.25                                   # blob = excess above this fraction of the peak excess
N_PEAK = 20



def roi_bad_pixels(run):
    '''Gaps/edges/seams (labels 1-3 of the run mask), so the seam through the ROI isn't counted.'''
    return np.isin(run.create_run_mask()[ROI], [1, 2, 3])


def blob_stats(img, bad):
    '''Blob mask and stats for one ROI image.'''
    img = img.astype(np.float64)
    good = ~bad & (img >= 0)
    bg = np.median(img[good])
    excess = np.where(good, img - bg, 0)
    peak = np.sort(excess.ravel())[-N_PEAK:].mean()
    blob = excess > FRAC*peak

    yy, xx = np.nonzero(blob)
    w = excess[blob]
    return blob, dict(
        bg=bg,
        peak=peak,
        area=blob.sum(),
        cy=(yy*w).sum()/w.sum() + ROI[0].start,
        cx=(xx*w).sum()/w.sum() + ROI[1].start,
        flux=w.sum(),
    )


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def check_run(run):
    '''Per-frame stats for one run, plus the run-mean blob.'''
    bad = roi_bad_pixels(run)
    rois = []
    for _, block in run.iter_frames():
        rois.extend(block[(slice(None),) + ROI])
    rois = np.array(rois)

    mean_blob, mean_stats = blob_stats(rois.mean(axis=0), bad)

    frames = []
    for img in rois:
        blob, stats = blob_stats(img, bad)
        stats['iou'] = iou(blob, mean_blob)
        frames.append(stats)
    frames = {k: np.array([f[k] for f in frames]) for k in frames[0]}

    return dict(mean_img=rois.mean(axis=0), mean_blob=mean_blob, mean_stats=mean_stats, frames=frames)


def plot_results(results):
    names = list(results)

    # within each run: frame by frame
    fig, axs = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    for name, res in results.items():
        f = res['frames']
        axs[0].plot(f['iou'], label=name)
        axs[1].plot(f['area'])
        axs[2].plot(f['cx'] - f['cx'].mean())
        axs[3].plot(f['cy'] - f['cy'].mean())
        axs[4].plot(f['peak'])
    for ax, lab in zip(axs, ['IoU with run mean', 'area (px)', 'centroid x - mean (px)', 'centroid y - mean (px)', 'peak excess (counts)']):
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel('frame')
    fig.suptitle(f'blob within each run (threshold {FRAC} x peak excess)')
    fig.tight_layout()

    # between runs: mean images and outlines
    fig, axs = plt.subplots(1, len(names)+1, figsize=(5*(len(names)+1), 5))
    axs = np.atleast_1d(axs)
    extent = (ROI[1].start, ROI[1].stop, ROI[0].stop, ROI[0].start)
    for ax, name in zip(axs, names):
        res = results[name]
        excess = res['mean_img'] - res['mean_stats']['bg']
        ax.imshow(excess / res['mean_stats']['peak'], vmin=0, vmax=1, extent=extent)
        ax.set_title(f'{name}\n(excess / peak)', fontsize=9)
    cols = plt.cm.tab10.colors
    for i, name in enumerate(names):
        axs[-1].contour(results[name]['mean_blob'], levels=[0.5], colors=[cols[i % 10]], extent=extent, origin='upper')
        axs[-1].plot([], [], c=cols[i % 10], label=name)
    axs[-1].set_xlim(extent[:2])
    axs[-1].set_ylim(extent[2:])
    axs[-1].set_aspect('equal')
    axs[-1].legend(fontsize=8)
    axs[-1].set_title('run-mean blob outlines')
    fig.tight_layout()




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Check if the detector blob changes shape within/between runs.')
    parser.add_argument('--date', type=int)
    parser.add_argument('--group')
    args = parser.parse_args()

    results = {}
    for date, group, sub_group, run_number in find_runs(args.date, args.group):
        name = f'{date} {group}-{sub_group}_{run_number}'
        try:
            results[name] = check_run(RawData(date, group, sub_group, run_number))
        except FileNotFoundError as e:
            print(f'skipping {name}: {e}')

    print('\nWithin runs (each frame vs the run mean):')
    print(f'{"run":40s} {"IoU min":>8s} {"IoU mean":>8s} {"area":>12s} {"cx std":>7s} {"cy std":>7s} {"peak":>14s}')
    for name, res in results.items():
        f = res['frames']
        print(f'{name:40s} {f["iou"].min():8.3f} {f["iou"].mean():8.3f} '
              f'{f["area"].mean():6.0f}±{f["area"].std():<5.0f} {f["cx"].std():7.2f} {f["cy"].std():7.2f} '
              f'{f["peak"].mean():7.0f}±{f["peak"].std():<6.0f}')

    print('\nRun means:')
    for name, res in results.items():
        s = res['mean_stats']
        print(f'{name:40s} area {s["area"]:5d}  centroid ({s["cy"]:.1f}, {s["cx"]:.1f})  peak {s["peak"]:.0f}  bg {s["bg"]:.0f}')

    if len(results) > 1:
        print('\nBetween runs (IoU of run-mean blobs):')
        for a, b in itertools.combinations(results, 2):
            print(f'{a} vs {b}: {iou(results[a]["mean_blob"], results[b]["mean_blob"]):.3f}')

    plot_results(results)
    plt.show()
