# sheetjet-max4

Analysis of liquid-sheet-jet X-ray scattering data from the PSI JUNGFRAU 9M detector at MAX IV.
For every run this builds a labelled detector mask (gaps, module edges, chip seams, beamstop, blob,
bad pixels, streaks), applies a per-module flat field, and stores the azimuthally integrated powder
pattern I(q) of every frame in one processed h5 file per run. The plotting scripts work from those
processed files only.


## Install

Requires Python >= 3.9.

```
git clone <repo url> sheetjet-max4
cd sheetjet-max4
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs numpy, scipy, h5py, hdf5plugin (needed to read the compressed JUNGFRAU data) and
matplotlib, and makes `src/rundata.py` and `src/const.py` importable from anywhere, so the scripts in
`scripts/` and `plot/` run from any folder. The install is editable: changes to `src/` take effect
without reinstalling. Activate the environment (`source .venv/bin/activate`) in each new shell.

A conda environment works too: activate it and run `pip install -e .` inside it.

The `data/` folder is not in the repo (it is in `.gitignore`), so make it and put the raw data in
it, following the layout under [Data](#data):

```
mkdir -p data/20261105/raw
# copy or download the runs into data/20261105/raw/{group}/{group}-{sub_group}/
```

On a fresh copy there is no flat field yet (`data/proc/calib/flatfield_water.h5`), and the powders
need it, so make it before the first processing run:

```
python scripts/calib_flatfield.py --name water --group water
python scripts/process.py --date 20261105
```


## Data

All data lives in the `data/` folder of the package (next to `src/`). It is not tracked by git:
put the raw data there yourself; the processed output is written there too. Expected layout:

```
data/{date}/raw/{group}/{group}-{sub_group}/{group}-{sub_group}_{run}_master.h5
data/{date}/raw/{group}/{group}-{sub_group}/{group}-{sub_group}_{run}_data_000001.h5   (, _000002, ...)

data/proc/{group}/{group}-{sub_group}_{run}_proc.h5            processed output (one per run)
data/proc/calib/flatfield_water.h5                              per-module flat field
```

e.g. `data/20261105/raw/water/water-longaxis/water-longaxis_4_master.h5`.

A run with some data files missing is processed with the files present (with a warning) and is
redone automatically once the rest arrive.

### Proc file contents

Everything per frame comes from one pass through the raw data. Intensities are flat-fielded and use
only good pixels (`mask == 0`); saturated pixels are left out, with no other intensity cut. The
histograms use the raw values instead. The parameters used are stored as attributes (on `/mask`
and `/powder/I`), and a run is redone when they, the data files present or the code version change.

| Dataset | Shape | |
|---|---|---|
| `/mask` | (H, W) uint8 | labelled mask (see [Mask labels](#mask-labels)) |
| `/mean`, `/std`, `/max` | (H, W) | per-pixel mean, standard deviation and maximum over the run (raw) |
| `/powder/I`, `/powder/std` | (frames, 256) | mean and std of the pixels in each q bin (q 0.3-2.3) |
| `/powder/counts` | (frames, 256) | pixels per q bin |
| `/powder/q` | (256,) | bin centres (1/A) |
| `/cake/I` | (frames, 128, 16) | mean I(q, phi): 128 q bins x 16 azimuthal sectors |
| `/cake/q`, `/cake/phi`, `/cake/counts` | | bin centres (phi in deg: 0 = right, 90 = down) and pixels per bin |
| `/hist/frame` | (frames, 1090) | histogram of raw pixel values per frame, all good pixels |
| `/hist/q` | (64, 1090) | the same summed over the run, in 64 q bins (edges `/hist/q_edges`) |
| `/hist/edges` | (1089,) | unit bins for values -16 to 1023, then 48 log bins up to 2^24; bin 0 is below the first edge, the last bin above the last |
| `/frame/total` | (frames,) | total intensity of the good pixels in the q range |
| `/frame/max`, `/frame/n_saturated` | (frames,) | largest unsaturated pixel value, number of saturated pixels |


## Layout

| Path | |
|---|---|
| `src/const.py` | detector geometry, data paths |
| `src/rundata.py` | `RunData`: reads a run, builds the run mask, computes and saves the powders |
| `scripts/process.py` | processes every run (mask, powders and frame statistics), skipping runs already done |
| `scripts/calib_flatfield.py` | fits the per-module flat field from the steps across module boundaries |
| `scripts/slurm_process.sh`, `scripts/submit_all.sh` | SLURM job for one group, and the commands to submit one job per group |
| `scripts/check_blob.py` | checks whether the bright blob changes shape within or between runs |
| `plot/plot_powders.py` | powders of a run: I(q) against frame, and the mean |
| `plot/plot_mask.py` | mask labels next to the run mean, and the mask overlaid on the mean |
| `plot/plot_ff.py` | flat field used for a run, with each module's gain |
| `plot/plot_aniso.py` | azimuthal anisotropy of sample and gas runs |
| `plot/plot_qc.py` | QC image per run and `summary.csv` of all runs, in `data/proc/qc/` |


## Usage

Process all runs of a date (or one group), from the repo root:

```
python scripts/process.py --date 20261105
python scripts/process.py --date 20261105 --group water
python scripts/process.py --date 20261105 --overwrite      # redo everything
python scripts/process.py                                   # every date
```

On a SLURM cluster, run one job per group. From the repo root (with the flat field already made):

```
bash scripts/submit_all.sh            # prints an sbatch command for every group in data/*/raw/
bash scripts/submit_all.sh | bash     # submits them; logs go to logs/
sbatch --job-name=proc_water scripts/slurm_process.sh water     # a single group
```

Edit the `#SBATCH` lines in `scripts/slurm_process.sh` (time, memory, partition, account) for your
cluster.

Make the flat field from the water runs, then reprocess so the powders use it:

```
python scripts/calib_flatfield.py --name water --group water
python scripts/process.py --date 20261105 --overwrite
```

Check the masks and frames of every processed run: one image per run and a table of all runs
(mask label counts, beamstop extent, low-intensity frames, saturated pixels, completeness, and the
run's row of `data/datalog.csv`) in `data/proc/qc/`:

```
python plot/plot_qc.py
```

Plot from a proc file:

```
P=data/proc/water/water-longaxis_4_proc.h5
python plot/plot_powders.py $P
python plot/plot_mask.py $P
python plot/plot_ff.py $P
python plot/plot_aniso.py $P --gas data/proc/gas_background/gas_background-gas_background_1_proc.h5
```

In Python:

```python
from rundata import RunData

run = RunData(20261105, 'water', 'longaxis', 4)
mask = run.create_run_mask()      # loads the saved mask if its parameters match
bad = mask > 0
```


## Mask labels

| Value | Label |
|---|---|
| 0 | good |
| 1 | gaps between modules |
| 2 | module edges (`edges-{n}px`) |
| 3 | chip seams (`seams-{n}px`) |
| 4 | beamstop |
| 5 | blob |
| 6 | bad pixels |
| 7 | streaks |

Use `mask > 0` for all masked pixels. The label names are also stored in the `labels` attribute of
`/mask`.
