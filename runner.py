import itertools
import os
import subprocess
import sys
import asyncio
import copy
import numpy as np
import glob
import shutil
from pathlib import Path


local = '--local' in sys.argv

GPUS = [0, 1, 2, 3]
MULTIPLEX = 2

PROJECT_NAME = 'sac_dists'
CODE_DIR = '..'
excluded_flags = []


basename = 'sactrunc_v4_cartbalance'

grid = [
    {
        '_main': ['train.py'],
        'seed': list(range(4)),
        'env': ['cartpole_balance'],
    }
]


def construct_varying_keys(grids):
    all_keys = set().union(*[g.keys() for g in grids])
    merged = {k: set() for k in all_keys}
    for grid in grids:
        for key in all_keys:
            grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
            merged[key] = merged[key].union(grid_key_value)
    varying_keys = {key for key in merged if len(merged[key]) > 1}
    return varying_keys


def construct_jobs(grids):
    jobs = []
    for grid in grids:
        individual_options = [[{key: value} for value in values]
                              for key, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        jobs += [{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options]
    return jobs


def construct_job_string(job, name, source_dir=''):
    """construct the string to execute the job"""
    flagstring = f"python -u {source_dir}{job['_main']}"
    for flag in job:
        if flag not in excluded_flags and not flag.startswith('_'):
            flagstring = flagstring + " " + flag + "=" + str(job[flag])
    return flagstring + ' experiment=' + name


def construct_name(job, varying_keys):
    """construct the job's name out of the varying keys in this sweep"""
    job_name = basename
    for flag in job:
        if flag in varying_keys and not flag.startswith('_'):
            job_name = job_name + "_" + flag + str(job[flag])
    return job_name


def copy_job_source(target_dir):
    # exclude the results dir since it's large and scanning takes forever
    # note that this syntax is extremely dumb!
    # [!r] will exclude every directory that starts with 'r'
    patterns = [
        '*.xml', '[!re]*/**/*.xml',
        '*.py', '[!re]*/**/*.py',
        '*.yaml', '[!re]*/**/*.yaml',
    ]

    for pattern in patterns:
        for f in Path('.').glob(pattern):
            target_path = f'{target_dir}{f}'
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # print(f"Copying {f} to {target_path}.")
            shutil.copy(f, target_path)


def run_job_slurm(job):
    # construct job name
    job_name = construct_name(job, varying_keys)

    # create slurm dirs if needed
    slurm_log_dir = 'slurm_logs'
    slurm_script_dir = 'slurm_scripts'
    os.makedirs(slurm_script_dir, exist_ok=True)
    os.makedirs(slurm_log_dir, exist_ok=True)

    # copy code to a temp directory
    true_source_dir = '.'
    job_source_dir = f'{CODE_DIR}/{PROJECT_NAME}-clones/{job_name}/'
    os.makedirs(job_source_dir, exist_ok=True)
    copy_job_source(job_source_dir)

    # make the job command
    job_string = construct_job_string(job, job_name, source_dir=job_source_dir)

    # write a slurm script
    slurm_script_path = f'{slurm_script_dir}/{job_name}.slurm'
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write(f"#SBATCH --job-name={job_name}\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write(f"#SBATCH --output=slurm_logs/{job_name}.out\n")
        slurmfile.write(f"#SBATCH --error=slurm_logs/{job_name}.err\n")
        slurmfile.write("#SBATCH --export=ALL\n")
        # slurmfile.write("#SBATCH --signal=USR1@600\n")
        slurmfile.write("#SBATCH --time=2-00\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH --mem=32gb\n")

        slurmfile.write("#SBATCH -c 4\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")
        slurmfile.write("#SBATCH --constraint=turing|volta\n")
        slurmfile.write("#SBATCH --exclude=lion[1-26]\n")
        slurmfile.write("#SBATCH --exclude=vine[3-14]\n")

        slurmfile.write("cd " + true_source_dir + '\n')
        slurmfile.write(f"{job_string} &\n")
        slurmfile.write("wait\n")

    # run the slurm script
    print("Dispatching `{}`".format(job_string))
    os.system(f'sbatch {slurm_script_path} &')


async def run_job(gpu_id, job):
    job_name = construct_name(job, varying_keys)
    job_string = construct_job_string(job, job_name)
    job_string = job_string + " experiment=" + job_name

    print("Dispatching `{}`".format(job_string))
    env = {
        **os.environ,
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    }
    proc = await asyncio.create_subprocess_shell(job_string, env=env)
    stdout, stderr = await proc.communicate()


async def worker_fn(gpu_id, queue):
    while True:
        job = await queue.get()
        await run_job(gpu_id, job)
        queue.task_done()


async def main():
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    n_parallel = MULTIPLEX * len(GPUS)
    workers = []
    for i in range(n_parallel):
        gpu_id = GPUS[i % len(GPUS)]
        worker = asyncio.create_task(worker_fn(gpu_id, queue))
        workers.append(worker)

    await queue.join()
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def slurm_main():
    for job in jobs:
        run_job_slurm(job)

if __name__ == '__main__':
    jobs = construct_jobs(grid)
    varying_keys = construct_varying_keys(grid)
    if local:
        asyncio.run(main())
    else:
        slurm_main()
