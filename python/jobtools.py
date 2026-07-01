"""
JOB MANAGEMENT UTILITY SCRIPT - DO NOT MODIFY

This script provides a robust framework for managing, executing, and tracking computational jobs
across different engines (loop, dask, joblib, slurm). It handles job registration, parameter hashing,
parallel execution, and result caching.

IMPORTANT:
---------
THIS IS A CORE UTILITY SCRIPT. DO NOT MODIFY IT UNLESS YOU ARE CERTAIN OF THE IMPACT.
Changes to this script may break existing job pipelines and workflows.

If you need to customize job behavior, create a new script that imports these functions
rather than modifying this file directly.

Features:
---------
- Job registration and retrieval system
- Automatic path generation with parameter hashing
- Support for multiple execution engines (sequential, parallel, SLURM)
- Result caching with NetCDF files
- Progress tracking for job lists
"""

import os
import sys
import stat
from pathlib import Path
import json
import time
import string
import inspect
import joblib
import pandas as pd
import xarray as xr
import subprocess
import random

# Global job registry
job_list = {}

# Check for Dask availability
try:
    from dask.distributed import Client
    HAVE_DASK = True
except ImportError:
    HAVE_DASK = False

def register_job(job):
    """
    Register a new job in the global job list.

    Parameters
    ----------
    job : Job
        Job object to register. Must have a unique job_name attribute.

    Notes
    -----
    Will raise an AssertionError if a job with the same name already exists.
    """
    global job_list
    assert job.job_name not in job_list, f"Job '{job.job_name}' already registered"
    job_list[job.job_name] = job

def retrieve_job(job_name):
    """
    Retrieve a previously registered job.

    Parameters
    ----------
    job_name : str
        Name of the job to retrieve.

    Returns
    -------
    Job
        The registered Job object.
    """
    return job_list[job_name]

def get_path(base_folder, job_name, params):
    """
    Generate a unique save path for a job based on its parameters.

    Creates a directory structure: base_folder/job_name/hash/
    and saves parameters to a JSON file in that directory.

    Parameters
    ----------
    base_folder : str or Path
        Base directory for job outputs.
    job_name : str
        Name of the job.
    params : dict
        Parameters that define the job configuration.

    Returns
    -------
    Path
        Full path to the job's output directory.
    """
    # Create a unique hash from job parameters
    hash = joblib.hash(params)
    save_path = Path(base_folder) / job_name / hash

    # Create directory and save parameters if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        with open(save_path / '__params__.json', mode='w') as f:
            json.dump(params, f, indent=4)

    return save_path

def _run_one_job_task(base_folder, job_name, params, func, keys):
    """
    Internal function to execute a single job task.

    Parameters
    ----------
    base_folder : str or Path
        Base directory for job outputs.
    job_name : str
        Name of the job.
    params : dict
        Job parameters.
    func : callable
        Function to execute for the job.
    keys : tuple
        Arguments to pass to the function.
    """
    job = Job(base_folder, job_name, params, func)
    job.compute(keys)

# Template for SLURM submission scripts
_slurm_script = """#! {python}
import sys
sys.path.append("{module_folder}")
from jobtools import _run_one_job_task

from {module_name} import {job_instance_name} as job

job.compute({keys}, force_recompute={force_recompute})
"""

def compute_job_list(job, list_keys, force_recompute=True, engine='loop', **engine_kargs):
    """
    Execute multiple job tasks sequentially or in parallel.

    Parameters
    ----------
    job : Job
        Job object to execute.
    list_keys : list
        List of argument tuples to pass to the job function.
    force_recompute : bool, optional
        If False, skip tasks that have already been computed. Default is True.
    engine : str, optional
        Execution engine to use. Options:
        - 'loop': Sequential execution (default)
        - 'dask': Parallel execution using Dask
        - 'joblib': Parallel execution using Joblib
        - 'slurm': Submit jobs to SLURM cluster
    **engine_kargs : dict
        Additional arguments for the selected engine.
    """
    if not force_recompute:
        # Filter out already completed tasks
        cleaned_list_key = []
        for keys in list_keys:
            output_filename = job.get_filename(*keys)
            if not os.path.exists(output_filename):
                cleaned_list_key.append(keys)
            else:
                print(job.job_name, 'already processed', keys)
        list_keys = cleaned_list_key

    t0 = time.perf_counter()

    if engine == 'loop':
        # Sequential execution
        for keys in list_keys:
            job.compute(keys, force_recompute=force_recompute)

    elif engine == 'dask':
        # Parallel execution with Dask
        client = Client(**engine_kargs)
        tasks = []
        for keys in list_keys:
            task = client.submit(_run_one_job_task, job.base_folder, job.job_name, job.params, job.func, keys)
            tasks.append(task)
        for task in tasks:
            task.result()

    elif engine == 'joblib':
        # Parallel execution with Joblib
        n_jobs = engine_kargs['n_jobs']
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_run_one_job_task)(job.base_folder, job.job_name, job.params, job.func, keys)
            for keys in list_keys
        )

    elif engine == 'slurm':
        # Submit jobs to SLURM cluster
        rand_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

        slurm_script_folder = Path('.') / 'slurm_scripts'
        slurm_script_folder = slurm_script_folder.absolute()
        slurm_script_folder.mkdir(exist_ok=True)

        for i, keys in enumerate(list_keys):
            # Handle different key formats
            if isinstance(keys, str):
                keys = (keys,)
            elif not isinstance(keys, tuple):
                raise ValueError(f"Keys must be string or tuple, got {type(keys)}")

            # Create unique script name
            key_txt = '_'.join([str(e) for e in keys])
            script_name = slurm_script_folder / f'{job.job_name}_{key_txt}_{rand_name}_{i}.py'
            output_name = slurm_script_folder / f'{job.job_name}_{key_txt}_{rand_name}_{i}.out'

            print()
            print('###', script_name.stem, '###')

            # Get module name from job function file
            module_name = Path(inspect.getfile(job.func)).stem

            # Write SLURM script
            with open(script_name, 'w') as f:
                slurm_script = _slurm_script.format(
                    python=sys.executable,
                    module_folder=Path('.').absolute(),
                    module_name=module_name,
                    job_instance_name=job.job_name+'_job',
                    keys=', '.join(f'"{k}"' for k in keys),
                    force_recompute=force_recompute,
                )
                f.write(slurm_script)
                os.fchmod(f.fileno(), mode=stat.S_IRWXU)  # Make script executable

            # Prepare SLURM command
            slurm_params = engine_kargs.get('slurm_params', {'cpus-per-task':'1', 'mem':'1G'})
            cmd = ['sbatch']
            cmd += [f'--{key}={value}' for key, value in slurm_params.items()]
            cmd += [f'--output={str(output_name)}', str(script_name)]

            print(cmd)
            proc = subprocess.Popen(cmd)
            proc.wait()

    t1 = time.perf_counter()
    print(job.job_name, 'Total time {:.3f}'.format(t1-t0))

class Job:
    """
    Job class for managing computational tasks.

    Attributes
    ----------
    base_folder : Path
        Base directory for job outputs.
    job_name : str
        Name of the job.
    params : dict
        Parameters for the job.
    save_path : Path
        Directory where job results will be saved.
    func : callable
        Function to execute for the job.
    """
    def __init__(self, base_folder, job_name, params, func):
        self.base_folder = base_folder
        self.job_name = job_name
        self.params = params
        self.save_path = get_path(base_folder, job_name, params)
        self.func = func

    def _make_keys(self, *args):
        """
        Convert input arguments to a standardized tuple of keys.

        Parameters
        ----------
        *args : variable
            Input arguments to convert.

        Returns
        -------
        tuple
            Standardized keys for the job.
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                keys = (arg,)
            elif isinstance(arg, tuple):
                keys = arg
            elif isinstance(arg, list):
                keys = tuple(arg)
            else:
                raise ValueError('Keys must be string, tuple, or list')
        elif len(args) > 1:
            keys = args
        else:
            raise ValueError('At least one key must be provided')
        return keys

    def get_filename(self, *args):
        """
        Generate output filename for a job with given keys.

        Parameters
        ----------
        *args : variable
            Keys for the job.

        Returns
        -------
        Path
            Full path to the output NetCDF file.
        """
        keys = self._make_keys(*args)
        filename = self.save_path / ('_'.join(str(k) for k in keys) + '.nc')
        return filename

    def is_job_done(self, *args):
        """
        Check if a job with given keys has already been computed.

        Parameters
        ----------
        *args : variable
            Keys for the job.

        Returns
        -------
        bool
            True if the job output file exists, False otherwise.
        """
        filename = self.get_filename(*args)
        return os.path.exists(filename)

    def get(self, *args, compute=True):
        """
        Get the result of a job, computing it if necessary.

        Parameters
        ----------
        *args : variable
            Keys for the job.
        compute : bool, optional
            If True and the job hasn't been computed, compute it now. Default is True.

        Returns
        -------
        xr.Dataset or None
            The job result as an xarray Dataset, or None if computation failed.
        """
        filename = self.get_filename(*args)
        if not os.path.exists(filename) and compute:
            ds = self.compute(*args)
            return ds
        ds = xr.open_dataset(filename)
        return ds

    def compute(self, *args, force_recompute=False):
        """
        Compute a job with given keys.

        Parameters
        ----------
        *args : variable
            Keys for the job.
        force_recompute : bool, optional
            If True, recompute even if output file exists. Default is False.

        Returns
        -------
        xr.Dataset or None
            The job result as an xarray Dataset, or None if computation failed.
        """
        keys = self._make_keys(*args)
        output_filename = self.get_filename(*args)

        # Skip if already computed and not forcing recomputation
        if not force_recompute and os.path.exists(output_filename):
            print(self.job_name, 'already processed', keys)
            return

        print(self.job_name, 'is processing', keys)
        try:
            ds = self.func(*keys, **self.params)
        except Exception:
            print('Error processing', self.job_name, keys)
            return None

        # Save results if computation succeeded
        if ds is not None:
            try:
                ds.to_netcdf(output_filename)
            except PermissionError:
                # Handle case where parallel jobs try to write simultaneously
                print('Error: Permission denied when writing output file')
                return None

        return ds

def get_job_done(job_list, run_keys):
    """
    Generate a DataFrame showing the completion status of jobs.

    Parameters
    ----------
    job_list : list
        List of Job objects to check.
    run_keys : list
        List of keys to check for each job.

    Returns
    -------
    pd.DataFrame
        DataFrame with boolean values indicating whether each job is done for each key.
    """
    names = [job.job_name for job in job_list]
    df = pd.DataFrame(index=run_keys, columns=names)

    for job in job_list:
        for run_key in run_keys:
            df.at[run_key, job.job_name] = job.is_job_done(run_key)

    # Ensure boolean dtype
    for name in names:
        df[name] = df[name].values.astype(bool)

    return df