




import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import os



from const import *

def _slice_edges(slices, frac, core_frac, both_sides=True, min_valid=0.5):
    '''
        Edges of a dark core along each 1D slice (rows of `slices`, nan = blanked). The core's floor
        is the median of the 20 lowest values; the edge on each side is where the intensity crosses
        floor + frac x (level outside on that side - floor), with the outside level the 95th
        percentile of that side. Returns (start, end) indices per slice (nan if there is no core,
        i.e. the floor isn't below core_frac x outside); end is all nan if both_sides is False.
    '''
    n, length = slices.shape
    idx = np.arange(length)
    start, end = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        p = slices[i]
        ok = np.isfinite(p)
        if ok.sum() < min_valid*length:
            continue
        p = ndimage.uniform_filter1d(np.interp(idx, idx[ok], p[ok]), 5)
        lo = np.argmin(p)
        if lo < 5:
            continue
        floor = np.median(np.sort(p)[:20])
        out_before = np.percentile(p[:lo], 95)
        out_after = np.percentile(p[lo:], 95) if both_sides else out_before
        if not floor < core_frac * min(out_before, out_after):
            continue
        before = np.nonzero(p[:lo] >= floor + frac*(out_before - floor))[0]
        start[i] = before.max() + 1 if len(before) else 0
        if both_sides:
            after = np.nonzero(p[lo:] >= floor + frac*(out_after - floor))[0]
            end[i] = lo + after.min() - 1 if len(after) else length - 1
    return start, end


# bump when the create_run_mask method changes, so masks saved by an older method are recomputed
MASK_VERSION = 6

# bump when calc_frame_stats changes, so proc files made by an older version are redone
STATS_VERSION = 1


def geometry_maps(beam_x, beam_y, clen, lamb):
    '''Per-pixel x, y (m, from the beam centre), scattering angle 2theta (rad) and q (1/A), for a flat
    detector perpendicular to the beam at distance clen (m).'''
    xx, yy = np.meshgrid((np.arange(DET_SHAPE[1]) - beam_x)*PIXEL_DX, (np.arange(DET_SHAPE[0]) - beam_y)*PIXEL_DX)
    two_theta = np.arctan2(np.hypot(xx, yy), clen)
    return xx, yy, two_theta, 4 * np.pi / lamb * np.sin(two_theta/2)


def cake_index(qidx, phi, n_sec):
    '''Cake bin of each pixel: pairs of powder q bins x n_sec sectors centred on 0, 90, 180, 270 deg.'''
    sec = np.floor((phi / (2*np.pi) + 0.5/n_sec) * n_sec).astype(int) % n_sec
    return (qidx // 2) * n_sec + sec


def hist_edges():
    '''Edges of the raw-value histograms: unit bins centred on -16 ... 1023, then 48 log bins up to 2^24.'''
    return np.concatenate([np.arange(-16.5, 1024), np.geomspace(1023.5, 2**24, 49)[1:]])


class RawData():



    def __init__(self, date = 20261105, group='water', sub_group='longaxis', run_number=4, flatfield=FLATFIELD):

        self.date = date
        self.group = group
        self.sub_group = sub_group
        self.run_number = run_number
        self.flatfield_path = flatfield   # None: no flat field correction
        self._flatfield = None
        self._warned_missing = False
        self.master_path = f'{DATA_DIR}/{self.date}/raw/{group}/{group}-{sub_group}/{group}-{sub_group}_{run_number}_master.h5'
        self.proc_path = proc_path(group, sub_group, run_number)



        ## load relevant stuff from the master h5
        with h5py.File(self.master_path, 'r') as f:
            self.azint_i = f['/entry/azint/dataset'][0,:]
            self.azint_q = f['/entry/azint/bin_to_q'][0,:]
            self.atten = f['/entry/instrument/attenuator/attenuator_transmission'][...]
            self.lamb = f['/entry/instrument/beam/incident_wavelength'][...]
            self.beam_x = f['/entry/instrument/detector/beam_center_x'][...]
            self.beam_y = f['/entry/instrument/detector/beam_center_y'][...]
            self.count_time = f['/entry/instrument/detector/count_time'][...]
            cycles = f.get('/entry/instrument/detector/number_of_cycles')
            self.n_cycles = int(cycles[()]) if cycles is not None else 0   # sub-frames summed per image

            self.n_images = f['/entry/instrument/detector/detectorSpecific/nimages'][...]
            sat = f.get('/entry/instrument/detector/saturation_value')
            self.saturation_value = int(sat[()]) if sat is not None else 2147483646



            self.clen = f['/entry/instrument/detector/distance'][...]

            self.adu_hists = np.zeros((18,2048))
            for module_i in range(18):
                self.adu_hists[module_i,:] = f[f'/entry/instrument/detector/detectorSpecific/adu_histogram/module{module_i}'][:]



        #calculate whatever else you might need
        self.xpts= (np.arange(DET_SHAPE[1]) - self.beam_x)*PIXEL_DX
        self.ypts = (np.arange(DET_SHAPE[0]) - self.beam_y)*PIXEL_DX

        self.xx, self.yy, self.two_theta_map, self.q_map = geometry_maps(self.beam_x, self.beam_y, self.clen, self.lamb)
        self.r_map = np.hypot(self.xx, self.yy)


    def get_mask(self,qmin = 0.1, qmax=3,):
        '''True for every pixel to leave out: run mask, the detector's own pixel_mask and the q range.'''
        mask = np.zeros(DET_SHAPE, bool)

        with h5py.File(self.master_path, 'r') as f:
            pixel_mask = f['/entry/instrument/detector/pixel_mask'][:]

        mask[self.create_run_mask()>0] = True
        mask[pixel_mask>0] = True
        mask[self.q_map<qmin] = True
        mask[self.q_map>qmax] = True

        return mask

    def create_run_mask(self, edge_width=4, seam_width=4,
                        bs_edge_frac=0.5, bs_core_frac=0.5, bs_smooth=11, bs_round=30,
                        blob_thresh=0.1, blob_bg_size=61, blob_join=15, blob_grow=5,
                        bad_thresh=0.2, bad_size=5,
                        streak_thresh=0.05, streak_min_size=20, streak_grow_rows=15, streak_grow_cols=4, streak_bs_gap=30,
                        overwrite=False):
        '''
            Labelled mask for this run (use mask>0 for all masked pixels):
                0: good pixels
                1: gaps between the detector modules
                2: edges-{edge_width}px: edge_width px around the modules
                3: seams-{seam_width}px: seam_width px seams between the chips within the modules
                4: beamstop
                5: blob
                6: bad pixels
                7: streaks
            Where they overlap, the lower label wins (e.g. a dead pixel on a seam stays a seam).

            Gaps, edges and seams come from the module layout in const.py. The label names, with
            the widths filled in, are saved in the 'labels' attribute of /mask.

            Beamstop: found from slices through the run mean (assumes the arm runs horizontally to
            the right of the beam, within the module row containing the beam). In each vertical
            slice (column) the dark core's floor and the level outside it on either side are found;
            the beamstop runs from where the intensity falls below floor + bs_edge_frac x (outside -
            floor) to where it rises back above it (0: just the flat dark core, 0.5: halfway down
            the soft edge). Columns without a dark core (floor not below bs_core_frac x outside) are
            skipped; columns hidden by gaps/seams are interpolated, and the edges are median
            smoothed over bs_smooth columns. The rounded tip is found the same way from horizontal
            slices (rows) around the beam, which trim each row to start where the intensity falls;
            rows at the very top and bottom of the tip miss the dark core, so the corners left there
            are rounded off with an opening by a disk of radius bs_round px.

            Blob: the bright streak in BLOB_ROI, which changes shape between runs. Pixels more than
            blob_thresh above the local background (median filter of blob_bg_size px, recomputed
            with the blob left out). Keeps the cluster with the brightest pixel plus any clusters
            within blob_join px of it (the seam through the ROI cuts off the tail), then grown
            by blob_grow px.

            Bad pixels: anything flagged in the detector's own pixel_mask (dead, noisy, ...), pixels
            whose run mean is negative or overflowing (flagged/overflowing in some frames), and
            pixels more than bad_thresh away from the median of their bad_size x bad_size
            neighbourhood (hot and dead pixels).

            Streaks: stripes a few columns wide running down from the chip seams (a readout
            problem in some chips). Each column is median filtered over 9 rows (to beat down the
            noise) and compared with the median of the 31 columns around it; pixels more than
            streak_thresh above it, in clusters of at least streak_min_size px, are streaks. They are
            grown by streak_grow_rows / streak_grow_cols px to cover the fainter tails. Stripes
            within streak_bs_gap px of the beamstop are ignored: its wavy edge looks like stripes.

            Calculated once and saved to the proc file as /mask (with the run mean); later calls
            with the same parameters load it from there. A saved run mean is reused, so changing
            the parameters doesn't re-read the raw data.
        '''
        params = dict(edge_width=edge_width, seam_width=seam_width,
                      bs_edge_frac=bs_edge_frac, bs_core_frac=bs_core_frac, bs_smooth=bs_smooth, bs_round=bs_round,
                      blob_thresh=blob_thresh, blob_bg_size=blob_bg_size,
                      blob_join=blob_join, blob_grow=blob_grow, bad_thresh=bad_thresh, bad_size=bad_size,
                      streak_thresh=streak_thresh, streak_min_size=streak_min_size,
                      streak_grow_rows=streak_grow_rows, streak_grow_cols=streak_grow_cols, streak_bs_gap=streak_bs_gap,
                      data_files=self.data_files_attr(), version=MASK_VERSION)

        if not overwrite:
            saved = self.read_proc('mask', params)
            if saved is not None:
                return saved

        data_files = dict(data_files=params['data_files'])
        mean = None if overwrite else self.read_proc('mean', data_files)
        if mean is None:
            mean = self.calc_mean(dtype=np.float64)
            self.write_proc('mean', mean.astype(np.float32), dict(data_files, n_frames=self.n_frames()))
        mean = mean.astype(np.float64)

        ## gaps between modules
        module_gaps = np.ones(DET_SHAPE, bool)
        for r, c in MOD_ORIGINS:
            module_gaps[r:r+MOD_SHAPE[0], c:c+MOD_SHAPE[1]] = False

        ## edges around modules
        module_edges = np.zeros(DET_SHAPE, bool)
        for r, c in MOD_ORIGINS:
            module_edges[r:r+edge_width, c:c+MOD_SHAPE[1]] = True
            module_edges[r+MOD_SHAPE[0]-edge_width:r+MOD_SHAPE[0], c:c+MOD_SHAPE[1]] = True

            module_edges[r:r+MOD_SHAPE[0], c:c+edge_width] = True
            module_edges[r:r+MOD_SHAPE[0], c+MOD_SHAPE[1]-edge_width:c+MOD_SHAPE[1]] = True

        ## seams within modules
        module_seams = np.zeros(DET_SHAPE, bool)
        for r, c in MOD_ORIGINS:
            for r_seam in SEAM_ROWS:
                module_seams[r+r_seam:r+r_seam+seam_width, c+edge_width:c+MOD_SHAPE[1]-edge_width] = True

            for c_seam in SEAM_COLS:
                module_seams[r+edge_width:r+MOD_SHAPE[0]-edge_width, c+c_seam:c+c_seam+seam_width] = True

        detector = module_gaps | module_edges | module_seams

        ## beamstop
        blank = detector | (mean < 0)
        img = np.where(blank, np.nan, mean)
        bx, by = int(round(float(self.beam_x))), int(round(float(self.beam_y)))
        mod_row = next(r for r, c in MOD_ORIGINS if r <= by < r + MOD_SHAPE[0])
        y_lo, y_hi = mod_row, mod_row + MOD_SHAPE[0]

        # vertical slices: top/bottom edge in every column
        top, bottom = _slice_edges(img[y_lo:y_hi].T, bs_edge_frac, bs_core_frac, both_sides=True)
        top, bottom = top + y_lo, bottom + y_lo
        have = np.isfinite(top)
        cols = np.arange(DET_SHAPE[1])
        span = (cols >= cols[have].min()) & (cols <= cols[have].max())
        for edge in (top, bottom):
            edge[span] = np.interp(cols[span], cols[have], edge[have])
            edge[span] = ndimage.median_filter(edge[span], bs_smooth)
        rows = np.arange(DET_SHAPE[0])[:, None]
        with np.errstate(invalid='ignore'):
            beamstop = (rows >= top[None, :]) & (rows <= bottom[None, :])

        # horizontal slices around the beam: each row starts where the intensity falls (the tip)
        x_lo, x_hi = max(bx - 275, 0), min(bx + 325, DET_SHAPE[1])
        left, _ = _slice_edges(img[y_lo:y_hi, x_lo:x_hi], bs_edge_frac, bs_core_frac, both_sides=False)
        found = np.isfinite(left)
        if found.sum() > bs_smooth:
            left[found] = ndimage.median_filter(left[found], bs_smooth)
        for r in np.nonzero(found)[0]:
            beamstop[y_lo + r, :x_lo + int(left[r])] = False

        # round off the corners (opening with a disk, on a crop padded so the detector edge doesn't cut the arm)
        if bs_round and beamstop.any():
            rr, cc = np.nonzero(beamstop)
            r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            y, x = np.ogrid[-bs_round:bs_round+1, -bs_round:bs_round+1]
            crop = np.pad(beamstop[r0:r1, c0:c1], bs_round + 1, mode='edge')
            opened = ndimage.binary_opening(crop, structure=x**2 + y**2 <= bs_round**2)
            beamstop[r0:r1, c0:c1] = opened[bs_round+1:-(bs_round+1), bs_round+1:-(bs_round+1)]

        ## blob
        roi = mean[BLOB_ROI]
        good = (roi >= 0) & ~detector[BLOB_ROI]
        filled = np.where(good, roi, np.median(roi[good]))

        bg = ndimage.median_filter(filled, size=blob_bg_size)
        hot = good & (roi/bg - 1 > blob_thresh)
        bg = ndimage.median_filter(np.where(hot, bg, filled), size=blob_bg_size)   # again, without the blob
        hot = good & (roi/bg - 1 > blob_thresh)

        labels, n = ndimage.label(hot)
        peak = np.unravel_index(np.argmax(np.where(good, roi, 0)), roi.shape)
        near = ndimage.binary_dilation(labels == labels[peak], iterations=blob_join)
        keep = np.unique(labels[near & hot])

        blob = np.zeros(DET_SHAPE, bool)
        blob[BLOB_ROI] = ndimage.binary_dilation(np.isin(labels, keep[keep > 0]), iterations=blob_grow)

        ## bad pixels
        with h5py.File(self.master_path, 'r') as f:
            pixel_mask = f['/entry/instrument/detector/pixel_mask'][:]

        valid = (mean >= 0) & (mean < 1e6)
        good = ~detector & ~beamstop & ~blob & (pixel_mask == 0) & valid

        # masked pixels take the value of the nearest good pixel, so the local medians near a
        # masked region (e.g. just outside the beamstop) only see real neighbouring values
        _, nearest = ndimage.distance_transform_edt(~good, return_indices=True)
        filled = mean[nearest[0], nearest[1]]

        local = ndimage.median_filter(filled, size=bad_size)
        with np.errstate(divide='ignore', invalid='ignore'):
            bad = (pixel_mask > 0) | ~valid | (good & (np.abs(mean/local - 1) > bad_thresh))

        ## streaks
        vertical = ndimage.median_filter(filled, size=(9, 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            stripes = good & (vertical / ndimage.median_filter(vertical, size=(1, 31)) - 1 > streak_thresh)
        stripes &= ndimage.distance_transform_edt(~beamstop) > streak_bs_gap
        labels, n = ndimage.label(stripes)
        sizes = ndimage.sum(stripes, labels, range(1, n+1))
        streaks = np.isin(labels, np.where(sizes >= streak_min_size)[0] + 1)
        streaks = ndimage.binary_dilation(streaks, structure=np.ones((2*streak_grow_rows+1, 2*streak_grow_cols+1), bool))

        ## labelled mask: later labels only go on pixels that are still good
        mask = np.zeros(DET_SHAPE, np.uint8)
        for label, m in [(1, module_gaps), (2, module_edges), (3, module_seams), (4, beamstop), (5, blob),
                         (6, bad), (7, streaks)]:
            mask[m & (mask == 0)] = label

        labels = ['good', 'gaps', f'edges-{edge_width}px', f'seams-{seam_width}px', 'beamstop', 'blob',
                  'bad pixels', 'streaks']
        self.write_proc('mask', mask, dict(params, labels=labels))

        return mask

    def data_files(self):
        '''
            Names of the /entry/data/data_* datasets whose linked files are present. Runs that are
            only partly downloaded use the files that are there (with a warning naming the missing
            ones); only a run with no data files at all raises an error.
        '''
        present, missing = [], []
        with h5py.File(self.master_path, 'r') as f:
            for name in sorted(k for k in f['/entry/data'] if k.startswith('data_')):
                link = f['/entry/data'].get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    path = os.path.join(os.path.dirname(self.master_path), link.filename)
                    if not os.path.exists(path):
                        missing.append(link.filename)
                        continue
                present.append(name)

        if not present:
            raise FileNotFoundError(f'{self.master_path}: none of its data files are present ({", ".join(missing)})')
        if missing and not self._warned_missing:
            print(f'warning: {self.master_path}: missing {", ".join(missing)}, using {", ".join(present)} only')
            self._warned_missing = True
        return present

    def data_files_attr(self):
        '''The data files used, as stored with the proc outputs (so a later full download is noticed).'''
        return ','.join(self.data_files())

    def iter_frames(self, chunk=20):
        '''Yields (first frame index, block of frames) across all data files of the run.'''
        start = 0
        names = self.data_files()
        with h5py.File(self.master_path, 'r') as f:
            for name in names:
                d = f[f'/entry/data/{name}']
                for i in range(0, d.shape[0], chunk):
                    block = d[i:i+chunk]
                    yield start, block
                    start += block.shape[0]

    def get_data(self):
        '''Whole run in memory: only for small runs, use iter_frames for big ones.'''
        return np.concatenate([block for _, block in self.iter_frames(chunk=100)])

    def calc_mean(self, dtype=np.float32):
        '''Run mean of the raw frames (no flat field, so calib_flatfield.py can fit it).'''
        total = np.zeros(DET_SHAPE, np.float64)
        n = 0
        for _, block in self.iter_frames():
            total += block.sum(axis=0, dtype=np.float64)
            n += block.shape[0]
        return (total / n).astype(dtype)

    def write_proc(self, name, data, attrs=None):
        '''Write (or replace) a dataset in this run's proc file.'''
        os.makedirs(os.path.dirname(self.proc_path), exist_ok=True)
        with h5py.File(self.proc_path, 'a') as f:
            # run metadata, refreshed on every write so older proc files pick up new fields
            f.attrs['master_path'] = self.master_path
            f.attrs['n_images'] = self.n_images
            f.attrs['beam_x'] = self.beam_x
            f.attrs['beam_y'] = self.beam_y
            f.attrs['clen'] = self.clen
            f.attrs['lamb'] = self.lamb
            f.attrs['count_time'] = self.count_time
            f.attrs['n_cycles'] = self.n_cycles
            if name in f:
                del f[name]
            dset = f.create_dataset(name, data=data, compression='gzip')
            for k, v in (attrs or {}).items():
                dset.attrs[k] = v

    def read_proc(self, name, attrs=None):
        '''Load a dataset from the proc file, or None if it's missing or was made with different attrs.'''
        if not os.path.exists(self.proc_path):
            return None
        with h5py.File(self.proc_path, 'r') as f:
            if name not in f:
                return None
            dset = f[name]
            for k, v in (attrs or {}).items():
                if k not in dset.attrs or dset.attrs[k] != v:
                    return None
            return dset[...]



    def get_flatfield(self):
        '''Per-module flat field to divide the data by (all ones if flatfield=None).'''
        if self._flatfield is None:
            if self.flatfield_path is None:
                self._flatfield = np.ones(DET_SHAPE, np.float32)
            elif not os.path.exists(self.flatfield_path):
                raise FileNotFoundError(f'flat field {self.flatfield_path} not found: make it with '
                                        'calib_flatfield.py, or use flatfield=None')
            else:
                with h5py.File(self.flatfield_path, 'r') as f:
                    self._flatfield = f['flatfield'][...]
        return self._flatfield

    def _powder_bins(self, nq, qmin, qmax, n_sec):
        '''Good pixels in the powder q range (flat indices), their q bin, cake bin and flat field.'''
        mask = self.get_mask(qmin=qmin, qmax=qmax)
        edges = np.linspace(qmin, qmax, nq + 1)
        qidx = np.digitize(self.q_map, edges).ravel() - 1         # -1 or nq = out of range
        sel = np.where(~mask.ravel() & (qidx >= 0) & (qidx < nq))[0]
        qidx = qidx[sel]

        # cake: nq/2 q bins (pairs of powder bins) x n_sec sectors centred on 0, 90, 180, 270 deg
        cidx = cake_index(qidx, np.arctan2(self.yy, self.xx).ravel()[sel], n_sec)

        return sel, qidx, cidx, 0.5*(edges[1:] + edges[:-1]), self.get_flatfield().ravel()[sel]

    def calc_frame_stats(self, nq=256, qmin=0.3, qmax=2.3, n_sec=16, nq_hist=64):
        '''
            Everything per frame, in one pass through the run. Saturated pixels (>= the detector's
            saturation value) are left out of the intensities; there is no other intensity cut.

            Flat-fielded, good pixels in [qmin, qmax]:
                powder_I, powder_std, powder_counts   (n, nq)   mean, std and pixel count per q bin
                cake_I                                (n, nq/2, n_sec) mean per q bin and sector
                total                                 (n,)      summed intensity
            Raw values, all good pixels:
                hist_frame                            (n, nbins) histogram per frame (edges: hist_edges)
                hist_q                                (nq_hist, nbins) run histogram per q bin (edges: hist_q_edges)
                max                                   (n,)      largest unsaturated value
            Raw values, every pixel:
                n_saturated                           (n,)
                std_image, max_image                  (H, W)    per-pixel std and max over the run
        '''
        sel, qidx, cidx, q, flatfield = self._powder_bins(nq, qmin, qmax, n_sec)
        nc = (nq // 2) * n_sec

        # histograms: all good pixels, any q
        hsel = np.where(~self.get_mask(qmin=0, qmax=np.inf).ravel())[0]
        edges = hist_edges()
        nb = len(edges) + 1
        hist_q_edges = np.linspace(0, self.q_map.max() * (1 + 1e-6), nq_hist + 1)
        hq = (np.digitize(self.q_map.ravel()[hsel], hist_q_edges) - 1) * nb

        n = self.n_frames()
        out = dict(powder_I=np.full((n, nq), np.nan, np.float32), powder_std=np.full((n, nq), np.nan, np.float32),
                   powder_counts=np.zeros((n, nq), np.int32), cake_I=np.full((n, nq // 2, n_sec), np.nan, np.float32),
                   total=np.zeros(n), hist_frame=np.zeros((n, nb), np.int32), max=np.zeros(n, np.int64),
                   n_saturated=np.zeros(n, np.int64))
        hist_q = np.zeros(nq_hist * nb, np.int64)
        total_img = np.zeros(DET_SHAPE, np.float64)
        sq_img = np.zeros(DET_SHAPE, np.float64)
        max_img = np.full(DET_SHAPE, np.iinfo(np.int32).min, np.int64)

        for start, block in self.iter_frames():
            for j, frame in enumerate(block):
                i = start + j
                flat = frame.ravel()

                raw = flat[sel]
                ok = raw < self.saturation_value
                img = raw[ok].astype(np.float64) / flatfield[ok]
                s = np.bincount(qidx[ok], img, nq)
                s2 = np.bincount(qidx[ok], img**2, nq)
                c = np.bincount(qidx[ok], minlength=nq)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean = s / c
                    out['powder_I'][i] = mean
                    out['powder_std'][i] = np.sqrt(np.maximum(s2 / c - mean**2, 0))
                    out['cake_I'][i] = (np.bincount(cidx[ok], img, nc) / np.bincount(cidx[ok], minlength=nc)
                                        ).reshape(nq // 2, n_sec)
                out['powder_counts'][i] = c
                out['total'][i] = img.sum()

                raw = flat[hsel]
                # integer values: unit bins by offset (bin k holds k-17), only the rare large values searched
                h = np.clip(raw, -17, 1024) + 17
                big = raw >= 1024
                h[big] = np.searchsorted(edges, raw[big], side='right')
                out['hist_frame'][i] = np.bincount(h, minlength=nb)
                hist_q += np.bincount(hq + h, minlength=nq_hist * nb)
                out['max'][i] = raw[raw < self.saturation_value].max(initial=0)

                out['n_saturated'][i] = (flat >= self.saturation_value).sum()
                f64 = frame.astype(np.float64)
                total_img += f64
                sq_img += f64**2
                np.maximum(max_img, frame, out=max_img)

        mean_img = total_img / n
        out['std_image'] = np.sqrt(np.maximum(sq_img / n - mean_img**2, 0)).astype(np.float32)
        out['max_image'] = max_img.astype(np.int32)
        out.update(q=q, cake_q=0.5 * (q[0::2] + q[1::2]), cake_phi=np.arange(n_sec) * 360 / n_sec,
                   cake_counts=np.bincount(cidx, minlength=nc).reshape(nq // 2, n_sec),
                   hist_edges=edges, hist_q=hist_q.reshape(nq_hist, nb), hist_q_edges=hist_q_edges)
        return out

    def calc_powders(self, nq = 256, qmin = 0.3, qmax=2.3, run_mean=False):
        '''
            Azimuthally averaged I(q) for every frame, or for the run mean if run_mean=True.
            Returns (powders, q, counts). Per frame, this is the powder part of calc_frame_stats.
        '''
        if not run_mean:
            out = self.calc_frame_stats(nq, qmin, qmax)
            return out['powder_I'], out['q'], out['powder_counts']

        sel, qidx, _, q, flatfield = self._powder_bins(nq, qmin, qmax, n_sec=16)
        img = self.calc_mean(np.float64).ravel()[sel] / flatfield
        ok = np.isfinite(img)
        counts = np.bincount(qidx[ok], minlength=nq)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.bincount(qidx[ok], img[ok], nq) / counts, q, counts

    def save_frame_stats(self, nq=256, qmin=0.3, qmax=2.3, n_sec=16, nq_hist=64):
        '''calc_frame_stats into the proc file (see the README for the layout).'''
        params = dict(nq=nq, qmin=qmin, qmax=qmax, n_sec=n_sec, nq_hist=nq_hist)
        out = self.calc_frame_stats(**params)

        # written last, so an interrupted run has no /powder/I and is redone
        self.write_proc('powder/q', out['q'])
        self.write_proc('powder/counts', out['powder_counts'])
        self.write_proc('powder/std', out['powder_std'])
        self.write_proc('cake/I', out['cake_I'], dict(n_sec=n_sec))
        self.write_proc('cake/q', out['cake_q'])
        self.write_proc('cake/phi', out['cake_phi'], dict(units='deg, 0 = +x (right), 90 = +y (down), sector centres'))
        self.write_proc('cake/counts', out['cake_counts'])
        self.write_proc('hist/frame', out['hist_frame'], dict(values='raw (not flat-fielded), good pixels'))
        self.write_proc('hist/edges', out['hist_edges'], dict(note='bin k counts values in [edges[k-1], edges[k]); '
                                                                     'bin 0 is below edges[0], the last bin is >= edges[-1]'))
        self.write_proc('hist/q', out['hist_q'])
        self.write_proc('hist/q_edges', out['hist_q_edges'])
        self.write_proc('frame/total', out['total'])
        self.write_proc('frame/max', out['max'])
        self.write_proc('frame/n_saturated', out['n_saturated'])
        self.write_proc('std', out['std_image'])
        self.write_proc('max', out['max_image'])
        self.write_proc('powder/I', out['powder_I'], dict(params, flatfield=str(self.flatfield_path),
                                                            saturation_value=self.saturation_value,
                                                            data_files=self.data_files_attr(), version=STATS_VERSION))
        return out

    def n_frames(self):
        with h5py.File(self.master_path, 'r') as f:
            return sum(f[f'/entry/data/{name}'].shape[0] for name in self.data_files())























