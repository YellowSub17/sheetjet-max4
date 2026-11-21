'''
Anisotropy of the run means, from proc files only (no raw data).

Figures:
    1. azimuthal harmonics (cos, sin, cos2, sin2 of phi, relative to I0) against q for every run,
       with the expected horizontal-polarization cos2 term, plus I0 and the cos term against radius
    2. sample vs gas: ring peak q against azimuth, I(phi) at a few q, and the I(q, phi)/I(q) cakes
    3. mean / azimuthal average images, sample and gas side by side

Sample runs are centred on their water ring (fitted here); gas runs have no ring and use the
beam centre from their master file. phi = 0 is +x (right), 90 deg is +y (down in the image).

usage: python plot/plot_aniso.py SAMPLE_PROC [SAMPLE_PROC ...] --gas GAS_PROC [GAS_PROC ...]
'''
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

from const import PIXEL_DX
from procdata import as_procdata



QE_HARM = np.linspace(0.3, 2.3, 81)     # q bins for the harmonics
QE_CAKE = np.linspace(1.5, 2.3, 41)     # q bins for the cake around the water ring
N_SEC = 36                              # azimuthal sectors in the cake



class Aniso:
    '''Flat-fielded run mean, good pixels and geometry for one proc file, about a given centre.'''

    def __init__(self, p, centre=None):
        self.p = as_procdata(p)
        self.name = self.p.name
        self.clen, self.lamb = float(self.p.attrs['clen']), float(self.p.attrs['lamb'])
        self.img = self.p.mean.astype(np.float64) / self.p.flatfield()
        self.mask = self.p.mask
        self.good = (self.mask == 0) & np.isfinite(self.img) & (self.img >= 0)
        self.set_centre(centre)

    def set_centre(self, centre=None):
        '''Geometry about centre (x, y) in pixels; None = the beam centre of the run.'''
        xx, yy, _, self.q = self.p.geometry(centre)
        self.centre = (float(self.p.attrs['beam_x']), float(self.p.attrs['beam_y'])) if centre is None else tuple(centre)
        self.r = np.hypot(xx, yy) / PIXEL_DX
        self.phi = np.arctan2(yy, xx)

    def harmonics(self, qe=QE_HARM):
        '''Least-squares fit of I = a0 + c1 cos + s1 sin + c2 cos2 + s2 sin2 in each q bin, on good
        pixels (so gaps don't bias it). Returns q, I0 and (c1, s1, c2, s2)/I0.'''
        g = self.good
        k = np.digitize(self.q[g], qe) - 1
        ok = (k >= 0) & (k < len(qe)-1)
        k, phi, I = k[ok], self.phi[g][ok], self.img[g][ok]
        B = np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi), np.cos(2*phi), np.sin(2*phi)])
        n = len(qe) - 1
        AtA, Atb = np.zeros((n, 5, 5)), np.zeros((n, 5))
        for i in range(5):
            Atb[:, i] = np.bincount(k, B[i]*I, n)
            for j in range(i, 5):
                AtA[:, i, j] = AtA[:, j, i] = np.bincount(k, B[i]*B[j], n)
        c = np.linalg.solve(AtA, Atb[..., None])[..., 0]
        return 0.5*(qe[1:] + qe[:-1]), c[:, 0], c[:, 1:] / c[:, :1]

    def cake(self, qe=QE_CAKE, nsec=N_SEC, min_px=30):
        '''Mean intensity in (q, phi) bins; nan where a bin has fewer than min_px good pixels.'''
        g = self.good
        k = np.digitize(self.q[g], qe) - 1
        s = ((self.phi[g] + np.pi) / (2*np.pi) * nsec).astype(int) % nsec
        ok = (k >= 0) & (k < len(qe)-1)
        n = len(qe) - 1
        S = np.bincount(k[ok]*nsec + s[ok], self.img[g][ok], n*nsec).reshape(n, nsec)
        C = np.bincount(k[ok]*nsec + s[ok], minlength=n*nsec).reshape(n, nsec)
        phic = np.degrees(-np.pi + (np.arange(nsec) + 0.5) * 2*np.pi/nsec)
        return 0.5*(qe[1:] + qe[:-1]), phic, np.where(C > min_px, S/np.maximum(C, 1), np.nan)

    def relative_image(self):
        '''Mean divided by its azimuthal average at each radius (gaps, edges, seams nan).'''
        g = self.good
        rr = self.r.astype(int)
        prof = np.bincount(rr[g], self.img[g], rr.max()+1) / np.maximum(np.bincount(rr[g], minlength=rr.max()+1), 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            rel = self.img / prof[rr]
        rel[np.isin(self.mask, [1, 2, 3]) | ~np.isfinite(self.img)] = np.nan
        return rel


def ring_peak_radii(a, nsec=72, rlo=850, rhi=1450, half=120):
    '''(phi, peak radius in px) for each sector where the ring is well covered: parabola through the
    radial profile around its maximum.'''
    g = a.good & (a.r > rlo) & (a.r < rhi)
    sec = ((a.phi[g] + np.pi) / (2*np.pi) * nsec).astype(int) % nsec
    rb = (a.r[g] - rlo).astype(int) // 4
    nb = (rhi - rlo)//4 + 1
    S = np.bincount(sec*nb + rb, a.img[g], nsec*nb).reshape(nsec, nb)
    C = np.bincount(sec*nb + rb, minlength=nsec*nb).reshape(nsec, nb)
    rc = rlo + 4*np.arange(nb) + 2
    pts = []
    for i in range(nsec):
        ok = C[i] > 20
        if ok.mean() < 0.8:
            continue
        prof = np.where(ok, S[i]/np.maximum(C[i], 1), np.nan)
        j = np.nanargmax(prof)
        w = ok & (abs(rc - rc[j]) < half)
        if w.sum() < 30:
            continue
        p = np.polyfit(rc[w] - rc[j], prof[w], 2)
        if p[0] < 0:
            pts.append((-np.pi + (i+0.5)*2*np.pi/nsec, rc[j] - p[1]/(2*p[0])))
    return np.array(pts)


def fit_ring_centre(a, n_iter=4):
    '''Centre of the circle through the ring peak in each sector, re-sectored about each new centre.'''
    for _ in range(n_iter):
        pts = ring_peak_radii(a)
        x = a.centre[0] + pts[:, 1]*np.cos(pts[:, 0])
        y = a.centre[1] + pts[:, 1]*np.sin(pts[:, 0])
        A = np.stack([2*x, 2*y, np.ones_like(x)], 1)
        (x0, y0, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
        a.set_centre((x0, y0))
    R = np.sqrt(c + x0**2 + y0**2)
    return (x0, y0), R, (np.hypot(x - x0, y - y0) - R).std()


def ring_peak_q(q, cake, half=6):
    '''Ring peak q in each sector of a cake (parabola around the maximum); nan for poorly covered sectors.'''
    pk = np.full(cake.shape[1], np.nan)
    for j in range(cake.shape[1]):
        p = cake[:, j]
        if np.isnan(p).mean() > 0.2:
            continue
        i = np.nanargmax(p)
        w = slice(max(i-half, 0), i+half+1)
        ok = np.isfinite(p[w])
        c = np.polyfit(q[w][ok] - q[i], p[w][ok], 2)
        pk[j] = q[i] - c[1]/(2*c[0])
    return pk



def plot_harmonics(runs):
    titles = ['cos phi  (right - left)', 'sin phi  (bottom - top)', 'cos 2phi  (horizontal - vertical)',
              'sin 2phi  (diagonals)']
    fig, axs = plt.subplots(3, 2, figsize=(13, 12))
    axs = axs.ravel()
    for a in runs:
        q, I0, rel = a.harmonics()
        label = f'{a.name}  ({a.centre[0]:.0f}, {a.centre[1]:.0f})'
        for j in range(4):
            axs[j].plot(q, 100*rel[:, j], label=label)
        axs[4].plot(q, I0/np.nanmax(I0), label=a.name)
        r = a.clen*np.tan(2*np.arcsin(q*a.lamb/(4*np.pi))) / PIXEL_DX
        axs[5].plot(r, 100*rel[:, 0], label=a.name)

    s2 = np.sin(2*np.arcsin(q*runs[0].lamb/(4*np.pi)))**2
    axs[2].plot(q, -100*(s2/2)/(1 - s2/2), 'k--', label='horizontal polarization (expected)')

    for j in range(4):
        axs[j].set_title(titles[j] + '  [%]')
        axs[j].set_xlabel('q (1/A)')
        axs[j].axhline(0, color='grey', lw=0.5)
    axs[0].legend(fontsize=8)
    axs[2].legend(fontsize=8)
    axs[4].set_title('I0 (normalised)')
    axs[4].set_xlabel('q (1/A)')
    axs[4].legend(fontsize=8)
    axs[5].set_title('cos phi against detector radius  [%]')
    axs[5].set_xlabel('r (px)')
    axs[5].axhline(0, color='grey', lw=0.5)
    axs[5].legend(fontsize=8)
    fig.tight_layout()


def plot_ring_azimuth(sample, gas):
    qs, phic, cs = sample.cake()
    qg, _, cg = gas.cake()
    pk = ring_peak_q(qs, cs)

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs = axs.ravel()
    axs[0].plot(phic, pk, 'o-')
    axs[0].set_title(f'{sample.name}: ring peak q against azimuth')
    axs[0].set_ylabel('peak q (1/A)')

    for qq, col in [(1.6, 'tab:purple'), (1.93, 'tab:blue'), (2.2, 'tab:cyan')]:
        i = np.argmin(abs(qs - qq))
        axs[1].plot(phic, cs[i]/np.nanmean(cs[i]), '-', color=col, label=f'{sample.name} q={qs[i]:.2f}')
        i = np.argmin(abs(qg - qq))
        axs[1].plot(phic, cg[i]/np.nanmean(cg[i]), '--', color=col, label=f'{gas.name} q={qg[i]:.2f}')
    axs[1].set_title('I(phi) / mean   (solid sample, dashed gas)')
    axs[1].legend(fontsize=7, ncol=2)

    for ax, (a, q, c) in zip(axs[2:], [(sample, qs, cs), (gas, qg, cg)]):
        im = ax.imshow(c/np.nanmean(c, 1, keepdims=True), aspect='auto', extent=[-180, 180, q[-1], q[0]],
                       vmin=0.9, vmax=1.1, cmap='RdBu_r')
        ax.set_title(f'{a.name}  I(q, phi) / I(q)')
        ax.set_ylabel('q (1/A)')
        fig.colorbar(im, ax=ax)
    for ax in axs:
        ax.set_xlabel('phi (deg, 0 = right, 90 = down)')
    fig.tight_layout()


def plot_relative_images(runs, radii=(600, 900, 1100, 1300)):
    fig, axs = plt.subplots(1, len(runs), figsize=(7*len(runs), 7), sharex=True, sharey=True, squeeze=False)
    for ax, a in zip(axs[0], runs):
        im = ax.imshow(a.relative_image(), vmin=0.85, vmax=1.15, cmap='RdBu_r', interpolation='nearest')
        ax.plot(*a.centre, 'k+', ms=15)
        for R in radii:
            ax.add_patch(plt.Circle(a.centre, R, fill=False, lw=0.5, color='k', ls=':'))
        ax.set_title(f'{a.name}: mean / azimuthal average\n(circles r = {", ".join(map(str, radii))} px)')
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8)




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Anisotropy of sample and gas run means.')
    parser.add_argument('samples', nargs='+', help='proc files of sample runs (centred on their ring)')
    parser.add_argument('--gas', nargs='+', required=True, help='proc files of gas runs (master beam centre)')
    args = parser.parse_args()

    warnings.simplefilter('ignore', RuntimeWarning)

    samples = [Aniso(p) for p in args.samples]
    for a in samples:
        master = a.centre
        centre, R, rms = fit_ring_centre(a)
        print(f'{a.name}: ring centre ({centre[0]:.1f}, {centre[1]:.1f}), master ({master[0]:.1f}, {master[1]:.1f}), '
              f'ring radius {R:.1f} px, rms {rms:.1f} px')
    gases = [Aniso(p) for p in args.gas]

    plot_harmonics(samples + gases)
    plot_ring_azimuth(samples[0], gases[0])
    plot_relative_images([samples[0], gases[0]])

    plt.show()
