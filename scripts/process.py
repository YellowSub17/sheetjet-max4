import argparse
import glob
import os
import time
import traceback


from const import *
from rundata import RunData




def find_runs(date=None, group=None):
    '''(date, group, sub_group, run_number) for every master file under DATA_DIR/{date}/raw/'''
    pattern = f'{DATA_DIR}/{date or "*"}/raw/{group or "*"}/*/*_master.h5'
    runs = []
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        group_i = os.path.basename(os.path.dirname(run_dir))
        sub_group = os.path.basename(run_dir)[len(group_i)+1:]          # {group}-{sub_group}
        run_number = int(os.path.basename(path).split('_')[-2])         # {group}-{sub_group}_{run}_master.h5
        date_i = int(path.split('/raw/')[0].split('/')[-1])
        runs.append((date_i, group_i, sub_group, run_number))
    return runs


def process_run(run, overwrite=False):
    # redone when the data files present changed (e.g. the rest of a partly downloaded run arrived)
    data_files = dict(data_files=run.data_files_attr())

    if overwrite or run.read_proc('mask', data_files) is None:
        m = run.create_run_mask(overwrite=overwrite)
        print(f'    mask: {(m>0).sum()} px')

    if overwrite or run.read_proc('powder/I', data_files) is None:
        powders, _, _ = run.save_powders()
        print(f'    powders: {powders.shape}')




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Beamstop masks and per-frame powders for every run.')
    parser.add_argument('--date', type=int)
    parser.add_argument('--group')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    runs = find_runs(args.date, args.group)
    print(f'{len(runs)} runs in {DATA_DIR}')

    failed = []
    for date, group, sub_group, run_number in runs:
        print(f'{date} {group}-{sub_group}_{run_number}')
        t0 = time.time()
        try:
            run = RunData(date, group, sub_group, run_number)
            process_run(run, overwrite=args.overwrite)
            print(f'    done in {time.time()-t0:.0f} s -> {run.proc_path}')
        except Exception as e:
            print(f'    FAILED: {e}')
            if not isinstance(e, FileNotFoundError):
                traceback.print_exc()
            failed.append((date, group, sub_group, run_number, str(e)))

    print(f'\n{len(runs)-len(failed)}/{len(runs)} runs processed')
    for f in failed:
        print('  failed:', *f)
