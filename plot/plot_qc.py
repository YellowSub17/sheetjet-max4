'''
Quality check of every processed run, from the proc files only (no raw data).

Writes, under data/proc/qc/:
    {group}-{sub_group}_{run}.png   mask labels, mask over the run mean, frame totals and the mean powder
    summary.csv                     one row per run: geometry (distance, beam centre, wavelength,
                                    exposure), mask label counts, beamstop extent, frame totals,
                                    saturated pixels, completeness, and the run's row in data/datalog.csv

A PNG is redrawn when its proc file is newer (or with --overwrite); --group limits the PNGs drawn.
summary.csv is always rebuilt for every run.

usage: python plot/plot_qc.py [--group G] [--overwrite]
'''
import argparse
import csv
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from const import DATA_DIR, PROC_DIR
from plot_mask import LABEL_COLOURS
from procdata import ProcData



QC_DIR = f'{PROC_DIR}/qc'
DATALOG = f'{DATA_DIR}/datalog.csv'
DATALOG_COLS = ['Include', 'Sample Type', 'Exposure time (s)', 'Comment']
LOW_FRAC = 0.9      # frames with total intensity below this fraction of the run median are flagged



def proc_files(group=None):
    return sorted(p for p in glob.glob(f'{PROC_DIR}/{group or "*"}/*_proc.h5') if '/calib/' not in p)


def optional(p, name):
    '''A dataset of the proc file, or None if the run has no frame statistics yet.'''
    return p[name] if name in p else None


def low_frames(total):
    return np.where(total < LOW_FRAC * np.median(total))[0]


def plot_run_qc(p, png_path):
    labels = p.labels
    cmap = ListedColormap(LABEL_COLOURS[:len(labels)])
    norm = BoundaryNorm(np.arange(len(labels)+1) - 0.5, len(labels))
    mask = p.mask

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    im = axs[0, 0].imshow(mask, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = fig.colorbar(im, ax=axs[0, 0], ticks=range(len(labels)))
    cbar.ax.set_yticklabels([f'{i}: {l}' for i, l in enumerate(labels)])
    axs[0, 0].set_title('mask labels')

    mean = p.mean.astype(float)
    mean[mean < 0] = np.nan
    axs[0, 1].imshow(mean, vmin=0, vmax=2*np.nanmedian(mean[mask == 0]), interpolation='nearest')
    for label in range(2, len(labels)):
        m = mask == label
        axs[0, 1].imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([LABEL_COLOURS[label]]), interpolation='nearest')
    axs[0, 1].plot(p.attrs['beam_x'], p.attrs['beam_y'], 'k+', ms=12)
    axs[0, 1].set_title('run mean, masks 2+ overlaid')

    total = optional(p, 'frame/total')
    if total is not None:
        low = low_frames(total)
        axs[1, 0].plot(total, lw=0.7)
        axs[1, 0].plot(low, total[low], 'rx', label=f'{len(low)} frames < {LOW_FRAC} x median')
        axs[1, 0].axhline(np.median(total), color='grey', lw=0.5)
        axs[1, 0].legend(fontsize=8)
        axs[1, 1].plot(p.q, np.nanmean(p.I, axis=0))
    else:
        for ax in axs[1]:
            ax.text(0.5, 0.5, 'no frame statistics yet', ha='center', transform=ax.transAxes)
    axs[1, 0].set_xlabel('frame')
    axs[1, 0].set_ylabel('total intensity (good pixels)')
    axs[1, 1].set_xlabel('q (1/A)')
    axs[1, 1].set_ylabel('mean I')
    axs[1, 1].set_title('run-mean powder')

    fig.suptitle(f'{p.date}  {p.name}')
    fig.tight_layout()
    fig.savefig(png_path, dpi=60)
    plt.close(fig)


def read_datalog():
    '''{(date, group, sub_group, run): row} from data/datalog.csv ({} if it isn't there).'''
    if not os.path.exists(DATALOG):
        return {}
    with open(DATALOG, newline='', encoding='utf-8-sig') as f:
        return {(r['Date'].strip(), r['Group'].strip(), r['Sub Group'].strip(), r['Scan'].strip()): r
                for r in csv.DictReader(f)}


def run_geometry(attrs):
    '''Detector distance, beam centre, wavelength and exposure of a run, from the proc file's attrs
    (count_time and n_cycles from the master for proc files written before they were stored).'''
    g = dict(clen_m=f"{float(attrs['clen']):.5f}", beam_x=f"{float(attrs['beam_x']):.1f}",
             beam_y=f"{float(attrs['beam_y']):.1f}", lamb_A=f"{float(attrs['lamb']):.5f}",
             count_time_s=attrs.get('count_time'), n_cycles=attrs.get('n_cycles'))
    if (g['count_time_s'] is None or g['n_cycles'] is None) and os.path.exists(attrs.get('master_path', '')):
        with h5py.File(attrs['master_path'], 'r') as f:
            g['count_time_s'] = f['/entry/instrument/detector/count_time'][()]
            g['n_cycles'] = f['/entry/instrument/detector/number_of_cycles'][()]
    if g['count_time_s'] is not None:
        g['count_time_s'] = f"{float(g['count_time_s']):.4g}"
    return g


def summary_row(p, datalog):
    mask = p.mask
    row = dict(date=p.date, group=p.group, sub_group=p.sub_group, run=p.run, n_images=int(p.attrs.get('n_images', 0)))
    row.update(run_geometry(p.attrs))

    total = optional(p, 'frame/total')
    row['n_frames'] = len(total) if total is not None else ''
    row['complete'] = row['n_frames'] == row['n_images']
    row['data_files'] = p.dset_attrs('mask').get('data_files', '')

    for i, l in enumerate(p.labels):
        row[f'px_{l}'] = int((mask == i).sum())
    rows, cols = np.where(mask == 4)
    row['beamstop_x'] = f'{cols.min()}-{cols.max()}' if len(cols) else ''
    row['beamstop_y'] = f'{rows.min()}-{rows.max()}' if len(rows) else ''

    if total is not None:
        low = low_frames(total)
        row['median_total'] = f'{np.median(total):.4g}'
        row['total_scatter_pct'] = f'{100*total.std()/total.mean():.2f}'
        row['n_low_frames'] = len(low)
        row['low_frames'] = ' '.join(map(str, low[:20])) + (' ...' if len(low) > 20 else '')
        row['n_saturated'] = int(p['frame/n_saturated'].sum())

    log = datalog.get((p.date, p.group, p.sub_group, p.run))
    row['in_datalog'] = log is not None
    for c in DATALOG_COLS:
        row[c] = (log or {}).get(c, '')
    return row




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='QC images and summary.csv for every processed run.')
    parser.add_argument('--group')
    parser.add_argument('--overwrite', action='store_true', help='redraw every PNG')
    args = parser.parse_args()

    os.makedirs(QC_DIR, exist_ok=True)
    datalog = read_datalog()

    # the summary always covers every run; --group only limits which PNGs are (re)drawn
    rows = []
    for path in proc_files():
        try:
            p = ProcData(path)
            png = f'{QC_DIR}/{p.name}.png'
            redraw = args.overwrite or not os.path.exists(png) or os.path.getmtime(png) < os.path.getmtime(path)
            if redraw and args.group in (None, p.group):
                plot_run_qc(p, png)
                print(f'{png}')
            rows.append(summary_row(p, datalog))
        except Exception as e:
            print(f'{path}: FAILED: {e}')

    # union of columns, in order of first appearance (runs differ in mask labels / stats present)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = f'{QC_DIR}/summary.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} runs -> {out}')
