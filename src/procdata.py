'''
Reading a run's processed file (written by rawdata.RawData) and fast post-processing of it, such
as the solid-angle and polarization corrections. Nothing here touches the raw data.
'''
import os

import h5py
import numpy as np

from const import *
from rawdata import geometry_maps, cake_index



# generic label names, for proc files written before the names were stored with the mask
DEFAULT_LABELS = ['good', 'gaps', 'edges', 'seams', 'beamstop', 'blob', 'bad pixels', 'streaks']



def pixel_factors(xx, yy, two_theta, clen, solid_angle=True, polarization=True, pol_frac=0.99):
    '''
        Factor each pixel's measured intensity carries from the geometry, to divide out:
            solid angle:   pixel area / clen^2 * cos^3(2theta) (sr), so corrected intensities are per
                           steradian and comparable between detector distances
            polarization:  pol_frac*(1 - sin^2(2theta) cos^2(phi)) + (1-pol_frac)*(1 - sin^2(2theta) sin^2(phi)),
                           for a beam polarized along the detector x axis (horizontal) by pol_frac
    '''
    f = np.ones(two_theta.shape)
    if solid_angle:
        f *= PIXEL_DX**2 / clen**2 * np.cos(two_theta)**3
    if polarization:
        s2 = np.sin(two_theta)**2
        phi = np.arctan2(yy, xx)
        f *= pol_frac * (1 - s2*np.cos(phi)**2) + (1 - pol_frac) * (1 - s2*np.sin(phi)**2)
    return f



def as_procdata(p):
    '''A ProcData from a ProcData or a proc file path.'''
    return p if isinstance(p, ProcData) else ProcData.from_path(p)



class ProcData():
    '''
        One run's proc file. Datasets are read when first asked for and then kept, so the large
        per-pixel images (/std, /max) cost nothing unless used:

            p = ProcData('water', 'longaxis', 4)
            p['powder/I'], p.mask, p.attrs['clen']
            c = p.corrected()        # solid angle and polarization corrected powders and cake
    '''

    def __init__(self, group=None, sub_group=None, run_number=None, path=None):
        '''The run's proc file (data/proc/{group}/{group}-{sub_group}_{run_number}_proc.h5), or the
        file at path (see from_path).'''
        if path is None:
            if None in (group, sub_group, run_number):
                raise TypeError('give group, sub_group and run_number (or path=...)')
            path = proc_path(group, sub_group, run_number)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found: process the run first (scripts/process.py)')
        self.path = os.path.abspath(path)
        with h5py.File(path, 'r') as f:
            self.attrs = dict(f.attrs)
            self.datasets = []
            f.visititems(lambda name, obj: self.datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        self._data = {}
        self._geometry = {}

    @classmethod
    def from_path(cls, path):
        '''Open a proc file by its path, e.g. one given on the command line.'''
        return cls(path=path)

    def __repr__(self):
        return f'ProcData({self.path!r})'

    def __contains__(self, name):
        return name in self.datasets

    def __getitem__(self, name):
        if name not in self._data:
            if name not in self.datasets:
                raise KeyError(f'{name} not in {self.path}')
            with h5py.File(self.path, 'r') as f:
                self._data[name] = f[name][...]
        return self._data[name]

    def dset_attrs(self, name):
        '''Attributes of one dataset (e.g. the parameters it was made with).'''
        with h5py.File(self.path, 'r') as f:
            return dict(f[name].attrs)


    # run id: {group}-{sub_group}_{run}_proc.h5 in the {group} folder, date from the raw master path
    @property
    def name(self):
        return os.path.basename(self.path)[:-len('_proc.h5')]

    @property
    def group(self):
        return os.path.basename(os.path.dirname(self.path))

    @property
    def sub_group(self):
        return self.name.rsplit('_', 1)[0][len(self.group)+1:]

    @property
    def run(self):
        return self.name.rsplit('_', 1)[1]

    @property
    def date(self):
        return self.attrs['master_path'].split('/raw/')[0].split('/')[-1]


    @property
    def mask(self):
        return self['mask']

    @property
    def labels(self):
        '''Mask label names (e.g. edges-4px), from the mask's attrs.'''
        labels = self.dset_attrs('mask').get('labels')
        if labels is None:
            return DEFAULT_LABELS[:int(self.mask.max()) + 1]
        return [l.decode() if isinstance(l, bytes) else str(l) for l in labels]

    @property
    def mean(self):
        return self['mean']

    @property
    def q(self):
        return self['powder/q']

    @property
    def I(self):
        return self['powder/I']

    @property
    def flatfield_path(self):
        '''Flat field the powders were divided by (None if none).'''
        if 'powder/I' not in self:
            return None
        path = self.dset_attrs('powder/I').get('flatfield')
        return None if path in (None, 'None') else path

    def flatfield(self):
        '''The flat field image the powders were divided by (ones if none).'''
        if self.flatfield_path is None:
            return np.ones(DET_SHAPE)
        with h5py.File(self.flatfield_path, 'r') as f:
            return f['flatfield'][...]

    def geometry(self, centre=None):
        '''(xx, yy, two_theta, q) per pixel (m, m, rad, 1/A) from the run's geometry, or about
        another beam centre (x, y) in pixels.'''
        centre = tuple(centre) if centre is not None else (float(self.attrs['beam_x']), float(self.attrs['beam_y']))
        if centre not in self._geometry:
            self._geometry[centre] = geometry_maps(*centre, float(self.attrs['clen']), float(self.attrs['lamb']))
        return self._geometry[centre]


    def corrected(self, solid_angle=True, polarization=True, pol_frac=0.99):
        '''
            Per-frame powders and cake corrected for solid angle and polarization (see pixel_factors).
            The factor of each bin is the mean factor of the good pixels in it, from the saved mask
            and the run's geometry, and the bin mean is divided by it (exact when the anisotropy
            within a bin is the polarization itself).

            Returns a dict: q, I (frames, nq), factor (nq,), and cake_q, cake_phi, cake_I (frames,
            nq/2, n_sec), cake_factor (nq/2, n_sec) when the proc file has a cake.
        '''
        xx, yy, two_theta, q_map = self.geometry()
        p = self.dset_attrs('powder/I')
        nq, qmin, qmax = int(p['nq']), float(p['qmin']), float(p['qmax'])
        out = dict(q=self.q, I=self.I)
        if 'cake/I' in self:
            out.update(cake_q=self['cake/q'], cake_phi=self['cake/phi'], cake_I=self['cake/I'])

        fac = pixel_factors(xx, yy, two_theta, float(self.attrs['clen']), solid_angle, polarization, pol_frac).ravel()
        qidx = np.digitize(q_map, np.linspace(qmin, qmax, nq + 1)).ravel() - 1
        sel = np.where((self.mask.ravel() == 0) & (qidx >= 0) & (qidx < nq))[0]
        qidx, fac = qidx[sel], fac[sel]

        out['factor'] = np.bincount(qidx, fac, nq) / np.bincount(qidx, minlength=nq)
        out['I'] = out['I'] / out['factor']
        if 'cake_I' in out:
            n_sec = out['cake_I'].shape[2]
            nc = (nq // 2) * n_sec
            cidx = cake_index(qidx, np.arctan2(yy, xx).ravel()[sel], n_sec)
            with np.errstate(invalid='ignore', divide='ignore'):
                out['cake_factor'] = (np.bincount(cidx, fac, nc) / np.bincount(cidx, minlength=nc)).reshape(nq // 2, n_sec)
            out['cake_I'] = out['cake_I'] / out['cake_factor']
        return out
