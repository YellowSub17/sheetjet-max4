#!/bin/bash
# Process every run of one group (mask, powders, frame statistics) as a SLURM job.
#
# usage (from the repo root, so the job finds the code and .venv):
#     sbatch --job-name=proc_GROUP scripts/slurm_process.sh GROUP [--overwrite]
#
# scripts/submit_all.sh prints this command for every group.
#
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
# fill in if your cluster needs them:
##SBATCH --partition=
##SBATCH --account=

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch scripts/slurm_process.sh GROUP [--overwrite]" >&2
    exit 1
fi
GROUP=$1
shift

# sbatch runs a copy of this script, so locate the repo from where the job was submitted
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate

echo "group $GROUP on $(hostname), started $(date)"
python -u scripts/process.py --group "$GROUP" "$@"
echo "finished $(date)"
