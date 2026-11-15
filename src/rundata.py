




import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt




from const import *

class RunData():



    def __init__(self, date = 20261105, group='GoldNR_1X', sub_group='longaxis', run_number=6):

        self.date = date
        self.group = group
        self.sub_group = sub_group
        self.run_number = run_number
        self.master_path = f'{DATA_DIR}/{self.date}/raw/{group}/{group}-{sub_group}/{group}-{sub_group}_{run_number}_master.h5'



        ## load relevant stuff from the master h5
        with h5py.File(self.master_path, 'r') as f:
            self.azint_i = f['/entry/azint/dataset'][0,:]
            self.azint_q = f['/entry/azint/bin_to_q'][0,:]
            self.atten = f['/entry/instrument/attenuator/attenuator_transmission'][...]
            self.lamb = f['/entry/instrument/beam/incident_wavelength'][...]
            self.beam_x = f['/entry/instrument/detector/beam_center_x'][...]
            self.beam_y = f['/entry/instrument/detector/beam_center_y'][...]
            self.count_time = f['/entry/instrument/detector/count_time'][...]

            self.n_images = f['/entry/instrument/detector/detectorSpecific/nimages'][...]

            self.clen = f['/entry/instrument/detector/distance'][...]

            self.adu_hists = np.zeros((18,2048))
            for module_i in range(18):
                self.adu_hists[module_i,:] = f[f'/entry/instrument/detector/detectorSpecific/adu_histogram/module{module_i}'][:]





        #calculate whatever else you might need
        self.xpts= (np.arange(DET_SHAPE[1]) - self.beam_x)*PIXEL_DX
        self.ypts = (np.arange(DET_SHAPE[0]) - self.beam_y)*PIXEL_DX

        self.xx, self.yy = np.meshgrid(self.xpts, self.ypts)

        self.r_map = np.hypot(self.xx, self.yy)
        self.two_theta_map = np.arctan2(self.r_map, self.clen)
        self.q_map = 4 * np.pi / self.lamb * np.sin(self.two_theta_map/2)


    def get_mask(self,qmin = 0.1, qmax=3,):
        mask = np.zeros(DET_SHAPE, bool)

        global_mask = np.load(f'{DATA_DIR}/global_mask.npy')
        with h5py.File(self.master_path, 'r') as f:
            run_mask = f['/entry/instrument/detector/pixel_mask'][:]

        mask[global_mask>0] = True
        mask[run_mask>0] = True
        mask[self.q_map<qmin] = True
        mask[self.q_map>qmax] = True

        return mask

    def get_data(self):
        with h5py.File(self.master_path, 'r') as f:
            data =  f['/entry/data/data_000001'][:]
        return data


    def calc_hist(self):
        mask = self.get_mask()
        data = self.get_data()






    def calc_powders(self, nq = 256, qmin = 0.1, qmax=2.3, inten_min=0, inten_max=650, run_mean=False):

        mask = self.get_mask(qmin=qmin, qmax=qmax)
        data = self.get_data()


        if run_mean:
            data = np.mean(data, axis=0, dtype=np.float64)[np.newaxis]

        n = data.shape[0]

        edges = np.linspace(qmin, qmax, nq + 1)
        qidx = np.digitize(self.q_map, edges).ravel() - 1         # -1 or nq = out of range
        qcen = 0.5 * (edges[1:] + edges[:-1])
        mask_flat = mask.ravel()


        powders = np.full((n, nq), np.nan)

        for i in range(n):
            img = data[i].ravel().astype(np.float32)
            keep = (~mask_flat
                    & np.isfinite(img)
                    & (img >= inten_min) & (img <= inten_max)
                    & (qidx >= 0) & (qidx < nq))
            s = np.bincount(qidx[keep], weights=img[keep], minlength=nq)
            c = np.bincount(qidx[keep], minlength=nq)
            np.divide(s, c, out=powders[i], where=c > 0)


        if run_mean:
            return powders[0], qcen, c   # 1D pattern for the averaged image
        else:
            return powders, qcen, c


























if __name__=='__main__':


    r =  RunData()


    q_rois = [0.314, 0.544, 0.647, 1.098, 1.547, 1.97, 2.3]

    cols= 'rgbcmyk'

    p, q ,c = r.calc_powders(run_mean=True)
    plt.figure(figsize=(8,8))
    plt.plot(q, c)
    plt.vlines(q_rois, ymin=0, ymax=np.max(c), colors = list(cols), linestyle='dashed')


    plt.figure(figsize=(8,8))
    plt.plot(q, p)
    plt.vlines(q_rois, ymin=0, ymax=np.max(p), colors = list(cols), linestyle='dashed')



    p, q ,c = r.calc_powders()
    plt.figure(figsize=(8,8))
    for i in range(0, p.shape[0], 10):
        plt.plot(q, p[i])
    plt.vlines(q_rois, ymin=0, ymax=np.max(p), colors = list(cols), linestyle='dashed')






    d = r.get_data()

    mean = np.mean(d, axis=0, dtype=np.float64)


    m = r.get_mask()
    mean[m] = np.nan


    q_circle = 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean, origin='upper', cmap='viridis', vmin=0, vmax=550)


    ax.contour(r.q_map, levels=q_rois, colors=list(cols), linewidths=1.5, linestyles='dashed')
    ax.plot(r.beam_x, r.beam_y, 'r+', ms=12)
    plt.show()
























