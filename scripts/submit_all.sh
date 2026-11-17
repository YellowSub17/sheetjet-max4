#!/bin/bash
# Print an sbatch command for every group found under data/*/raw/ (one SLURM job per group).
# Nothing is submitted: check the list, then pipe it to bash.
#
# usage (from the repo root):
#     bash scripts/submit_all.sh                  # print the commands
#     bash scripts/submit_all.sh | bash           # submit them
#     bash scripts/submit_all.sh --overwrite      # extra arguments are passed on to process.py

cd "$(dirname "$0")/.."

if [ ! -f data/proc/calib/flatfield_water.h5 ]; then
    echo "# no flat field yet: run this first (and let it finish):" >&2
    echo "#     python scripts/calib_flatfield.py --name water --group water" >&2
fi

mkdir -p logs   # SLURM writes the job logs here, and won't create the folder itself

for dir in data/*/raw/*/; do
    group=$(basename "$dir")
    echo "sbatch --job-name=proc_$group scripts/slurm_process.sh $group $*"
done | sort -u
