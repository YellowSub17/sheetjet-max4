


import matplotlib.pyplot as plt
import numpy as np

from const import *






class MaskData:

    def __init__(self):

        '''
            0: Good pixels.
            1: Bad pixels between modules of the detector
        '''
        self.module_gaps = np.ones(DET_SHAPE, bool)
        for r, c in MOD_ORIGINS:
            self.module_gaps[r:r+MOD_SHAPE[0], c:c+MOD_SHAPE[1]] = False

        self.module_edges = np.zeros(DET_SHAPE, bool)
        for r, c in MOD_ORIGINS:
            self.module_edges[r:r+EDGE_WIDTH, c:c+MOD_SHAPE[1]] = True
            self.module_edges[r+MOD_SHAPE[0]-EDGE_WIDTH:r+MOD_SHAPE[0], c:c+MOD_SHAPE[1]] = True

            self.module_edges[r:r+MOD_SHAPE[0], c:c+EDGE_WIDTH] = True
            self.module_edges[r:r+MOD_SHAPE[0], c+MOD_SHAPE[1]-EDGE_WIDTH:c+MOD_SHAPE[1]] = True

        self.module_seams = np.zeros(DET_SHAPE, bool)

        for r, c in MOD_ORIGINS:
            for r_seam in SEAM_ROWS:
                self.module_seams[r+r_seam:r+r_seam+SEAM_WIDTH, c+EDGE_WIDTH:c+MOD_SHAPE[1]-EDGE_WIDTH] = True

            for c_seam in SEAM_COLS:
                self.module_seams[r+EDGE_WIDTH:r+MOD_SHAPE[0]-EDGE_WIDTH, c+c_seam:c+c_seam+SEAM_WIDTH] = True


        # #these pixels seem to be == 2^31 in every frame
        # self.dead_loc = (np.array([  33,   98,  185,  242,  711,  792,  796,  813, 1031, 1360, 1418,
           # 1420, 1909, 1913, 1925, 2355, 2355, 2356, 2368, 2434, 2445, 2509,
           # 2646, 2664, 2837, 2837, 2838, 2845, 2845, 2848, 2859, 2859, 2859,
           # 2991, 3211, 3212]), np.array([2681, 1692, 2554,  168,  837,  686,  603, 1559,  708,  379,  119,
            # 119, 2278, 2828, 2157, 1368, 1371, 1368, 1259, 1166, 2936, 1365,
            # 948,  939, 1406, 1407, 1409, 1418, 1419, 1424, 1404, 1405, 1408,
           # 1337, 1058, 1056]))

        # self.dead_pixels = np.zeros(DET_SHAPE, bool)
        # self.dead_pixels[self.dead_loc] = True


        self.mask = self.module_gaps + 2*self.module_edges + 3*self.module_seams # + 4*self.dead_pixels

















if __name__=='__main__':




    m = MaskData()

    plt.figure()
    plt.imshow(m.module_gaps)

    plt.figure()
    plt.imshow(m.module_edges)

    plt.figure()
    plt.imshow(m.module_seams)

    plt.figure()
    plt.imshow(m.module_gaps+2*m.module_edges +3*m.module_seams)

    np.save(f'{DATA_DIR}/global_mask.npy', m.mask)



    plt.show()













