




DATA_DIR = '/Users/pat/Documents/work/sheetjet-max4/data'



DET_SHAPE = (3264, 3106)
MOD_SHAPE = (514, 1030)
MOD_PITCH = (550, 1038)
N_MODS = (6,3)
MOD_GAP_ROWS = 36
MOD_GAP_COLS = 8

# seams between the 256 x 256 chips inside a module (module-local coords, 4 px wide)
SEAM_ROWS = [255]
SEAM_COLS = [255, 513, 771]
SEAM_WIDTH = 4

MOD_ORIGINS = [(i*MOD_PITCH[0], j*MOD_PITCH[1]) for i in range(N_MODS[0]) for j in range(N_MODS[1])]

EDGE_WIDTH = 4

PIXEL_DX = 75e-6


