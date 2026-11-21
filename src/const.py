



import os

# data tree: the data/ folder of this package (next to src/)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PROC_DIR = f'{DATA_DIR}/proc'
CALIB_DIR = f'{PROC_DIR}/calib'


def proc_path(group, sub_group, run_number):
    '''Processed file of a run (written by rawdata.RawData, read by procdata.ProcData).'''
    return f'{PROC_DIR}/{group}/{group}-{sub_group}_{run_number}_proc.h5'

# per-module flat field divided out of the powders (made by calib_flatfield.py); None = no correction
FLATFIELD = f'{CALIB_DIR}/flatfield_water.h5'



DET_SHAPE = (3264, 3106)
MOD_SHAPE = (514, 1030)
MOD_PITCH = (550, 1038)
N_MODS = (6,3)
MOD_GAP_ROWS = 36
MOD_GAP_COLS = 8

# seams between the 256 x 256 chips inside a module (module-local coords; width set in RawData.create_run_mask)
SEAM_ROWS = [255]
SEAM_COLS = [255, 513, 771]

MOD_ORIGINS = [(i*MOD_PITCH[0], j*MOD_PITCH[1]) for i in range(N_MODS[0]) for j in range(N_MODS[1])]


PIXEL_DX = 75e-6

# region (rows, cols) containing the bright streak/blob, masked per run in RawData.create_run_mask
BLOB_ROI = (slice(1600, 1900), slice(560, 920))


