from configuration import *
from dataio import read_eeg

import jobtools
from params import trimming_params
import xarray as xr

def get_start_end(data, sampling_rate, run_key, **p):
    """
    Compute start and end indices for segmenting time-series data based on a specified method.

    Parameters
    ----------
    data : ndarray
        Input time-series data (2D array with shape [timepoints, channels]).
    sampling_rate : int or float
        Sampling rate of the data (in Hz).
    run_key : str
        Identifier for the run, used to extract the phase (e.g., 'sub-01_Repos').
    **p : dict
        Additional parameters for the method, including:
        - 'method' : str
            Method to compute start/end indices. Options:
            - 'centered': Center a window around the middle of the data.
            - 'delay': Extract a window starting at `p['start']` and ending at `p['end']`, then trim to `p['window_duration']`.
            - 'conservative': Ensure the window fits within the data bounds.
            - 'both': Use fixed start and end times from `p['start']` and `p['end']`.
            - (default): Use the entire data range.
        - 'window_duration' : dict
            Dictionary mapping phases (e.g., 'Repos') to window durations (in seconds).
        - 'start' : float
            Start time (in seconds) for methods like 'delay' or 'both'.
        - 'end' : float
            End time (in seconds) for methods like 'delay' or 'both'.

    Returns
    -------
    start : int
        Starting index (in samples) for the segment.
    end : int
        Ending index (in samples) for the segment.

    Notes
    -----
    - For 'centered', the window is centered in the data, with duration from `p['window_duration'][phase]`.
    - For 'delay', the data is first trimmed to `[start, end]`, then further trimmed to match `window_duration`.
    - For 'conservative', the window is adjusted to ensure it fits within the data bounds.
    - For 'both', the start and end are directly converted from seconds to samples.
    - For unrecognized methods, the entire data range is returned.
    """
    method = p['method']
    sampling_rate = int(sampling_rate)

    # Extract phase from run_key (e.g., 'run_E1' -> 'E1')
    phase = run_key.split('_')[1]

    # Get window duration for the phase (used by 'centered', 'delay', 'conservative')
    if method in ['centered', 'delay', 'conservative']:
        duration = p['window_duration'][phase]

    # Centered: Window around the middle of the data
    if method == 'centered':
        duration_samples = int(duration * sampling_rate)
        center = int(data.shape[0] / 2)
        start = max(0, int(center - duration_samples // 2))
        end = min(data.shape[0], int(center + duration_samples // 2))

    # Both: Fixed start and end times
    elif method == 'both':
        start = max(0, int(sampling_rate * p['start']))
        end = min(data.shape[0], int(sampling_rate * p['end']))

    # Delay: Trim data to [start, end], then trim to window_duration
    elif method == 'delay':
        start = max(0, int(sampling_rate * p['start']))
        end = min(data.shape[0], int(sampling_rate * p['end']))
        trimmed = data[start:data.shape[0] - end, :]

        end_window = sampling_rate * duration
        trimmed = trimmed[:end_window, :]
        # Update start and end to reflect the trimmed window
        start = start
        end = start + int(end_window)

    # Conservative: Ensure window fits within data bounds
    elif method == 'conservative':
        sample_duration = int(sampling_rate * duration)
        n = data.shape[0]
        start = max(0, min(n - sample_duration, int(sampling_rate * p['start'])))
        end = start + sample_duration

    # Default: Use entire data range
    else:
        start = 0
        end = data.shape[0]

    return start, end


def compute_trimming(run_key, **p):
    """
    Compute trimming indices and times for EEG data based on a specified method.

    This function reads EEG data for a given run, calculates the start and end indices
    for trimming using `get_start_end`, and returns the results as an xarray Dataset
    with both sample indices and corresponding times in seconds.

    Parameters
    ----------
    run_key : str
        Unique identifier for the run (e.g., 'sub-01_E1').
    **p : dict
        Additional parameters passed to `get_start_end` to determine the trimming method.
        See `get_start_end` for details on supported methods and parameters.

    Returns
    -------
    xr.Dataset
        A Dataset containing:
        - 'start_index': Starting sample index for the trimmed segment.
        - 'end_index': Ending sample index for the trimmed segment.
        - 'sampling_rate': Sampling rate (Hz) of the EEG data.
        - 'start_time': Starting time (in seconds) of the trimmed segment.
        - 'end_time': Ending time (in seconds) of the trimmed segment.
    """
    # Read EEG data for the specified run
    resp, sr = read_eeg(run_key)

    # Compute start and end indices based on the trimming method
    start, end = get_start_end(resp, sr, run_key, **p)

    # Create an xarray Dataset to store trimming information
    ds = xr.Dataset()
    ds['start_index'] = start
    ds['end_index'] = end
    ds['sampling_rate'] = sr
    ds['start_time'] = start / sr  # Convert start index to time (seconds)
    ds['end_time'] = end / sr        # Convert end index to time (seconds)

    return ds


def test_compute_trimming():
    """
    Test the `compute_trimming` function with a sample run key and predefined trimming parameters.

    This function demonstrates how to use `compute_trimming` by applying it to a specific run
    (e.g., 'sub-01_Repos') and printing the resulting xarray Dataset containing trimming indices and times.

    Notes
    -----
    - Uses a hardcoded `run_key` ('sub-01_Repos') for testing.
    - Assumes `trimming_params` is a dictionary of parameters defined elsewhere (e.g., in `configuration.py`).
    - Prints the output Dataset to verify the trimming results.
    """
    run_key = 'sub-01_Repos'  # Example run key for testing
    ds = compute_trimming(run_key, **trimming_params)  # Compute trimming indices and times
    print(ds)  # Print the resulting Dataset


trimming_job = jobtools.Job(precomputedir, 'trimming', trimming_params, compute_trimming)
jobtools.register_job(trimming_job)


if __name__ == "__main__":

    test_compute_trimming()
