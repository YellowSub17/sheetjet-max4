# sheetjet-max4

Analysis of liquid-sheet-jet X-ray scattering data from the PSI JUNGFRAU 9M detector at MAX IV.
For every run this builds a labelled detector mask (gaps, module edges, chip seams, beamstop, blob,
bad pixels, streaks), applies a per-module flat field, and stores the azimuthally integrated powder
pattern I(q) of every frame in one processed h5 file per run. The plotting scripts work from those
processed files only.


## Install

Requires Python >= 3.10.

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

Each proc file holds `/mean` (run mean image), `/mask` (labelled mask), and `/powder/I`,
`/powder/q`, `/powder/counts` (per-frame powders), with the parameters used stored as attributes.


## Layout

| Path | |
|---|---|
| `src/const.py` | detector geometry, data paths |
| `src/rundata.py` | `RunData`: reads a run, builds the run mask, computes and saves the powders |
| `scripts/process.py` | processes every run (mask and powders), skipping runs already done |
| `scripts/calib_flatfield.py` | fits the per-module flat field from the steps across module boundaries |
| `scripts/check_blob.py` | checks whether the bright blob changes shape within or between runs |
| `plot/plot_powders.py` | powders of a run: I(q) against frame, and the mean |
| `plot/plot_mask.py` | mask labels next to the run mean, and the mask overlaid on the mean |
| `plot/plot_ff.py` | flat field used for a run, with each module's gain |
| `plot/plot_aniso.py` | azimuthal anisotropy of sample and gas runs |


## Usage

Process all runs of a date (or one group), from the repo root:

```
python scripts/process.py --date 20261105
python scripts/process.py --date 20261105 --group water
python scripts/process.py --date 20261105 --overwrite      # redo everything
```

Make the flat field from the water runs, then reprocess so the powders use it:

```
python scripts/calib_flatfield.py --name water --group water
python scripts/process.py --date 20261105 --overwrite
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
