




import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt



from const import DATA_DIR

class RunData():



    def __init__(self, date = 20261105, group='GoldNR_1X', sub_group='longaxis', run_number=6):

        self.date = date
        self.group = group
        self.sub_group = sub_group
        self.run_number = run_number
        self.master_path = f'{DATA_DIR}/{self.date}/raw/{group}/{group}-{sub_group}/{group}-{sub_group}_{run_number}_master.h5'


        with h5py.File(self.master_path, 'r') as f:
            self.data =  f['/entry/data/data_000001'][:]









if __name__=='__main__':

    r =  RunData()


    # np.save('./mean.npy', r.data.mean(axis=0))

    # np.save('./frame0.npy', r.data[0])
    # np.save('./frame1.npy', r.data[1])
    # np.save('./frame5.npy', r.data[5])
    # np.save('./frame10.npy', r.data[10])
    # np.save('./frame100.npy', r.data[100])

    # frame = np.load('./frame100.npy').astype(np.float32)
    frame = np.load('./mean.npy').astype(np.float32)

    frame[frame==-2147483648] = np.nan

    frame_median = np.nanmedian(frame)

    plt.figure()
    plt.hist(frame[~np.isnan(frame)].flatten(), range=(0, frame_median*3), bins=100)


    plt.figure()
    plt.imshow(frame, vmin=0, vmax=frame_median*3)

    plt.figure()
    plt.imshow(frame>500)


    # for frame_i in [1, 5, 10, 100]:
        # frame_iter = np.load(f'./frame{frame_i}.npy').astype(int)
        # plt.figure()
        # plt.imshow(frame==frame_iter)











    # frame = np.load('./mean.npy')

    # print(np.where(r.data< -1))

    # frame[frame>1000] = 0
    # frame[frame<0] = 0

#     plt.figure()
    # plt.imshow(frame)
    # plt.colorbar()



    # plt.figure()
    # plt.hist(frame, bins=100)




    # plt.figure()
    # plt.imshow(r.data[0] < -1, vmin=0, vmax=400)

    # # plt.figure()
    # # plt.imshow(r.data.mean(axis=0), vmin=0, vmax=400)

    # # plt.figure()
    # # plt.imshow(r.data[0]-r.data[100], vmin=0, vmax=10)

    # plt.figure()
    # plt.hist(r.data[0].flatten())



    plt.show()


















