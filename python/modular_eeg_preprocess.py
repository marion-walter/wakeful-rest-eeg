"""
MODULAR EEG PREPROCESSING PIPELINE

This script implements a comprehensive, modular pipeline for preprocessing EEG data. 
It provides a flexible framework for applying various preprocessing steps in a configurable order, 
with support for parallel execution and job management.

KEY FEATURES:
-------------
1. MODULAR PREPROCESSING:
   - Each preprocessing step (trimming, detrending, filtering, artifact removal, etc.)
     is implemented as a separate function
   - Steps can be combined in any order through configuration parameters
   - Supports multiple applications of the same step with different parameters

2. ARTIFACT CORRECTION METHODS:
   - EOG artifacts: Multiple ICA-based approaches (standard ICA, KSSA-ICA, CWT-MSSA-ICA,
     and rectified variants)
   - ECG artifacts: ICA and CWT-MSSA-ICA methods
   - EMG artifacts: RMS-based detection and correction
   - Large artifacts: SSA-based correction

3. JOB MANAGEMENT:
   - Integration with jobtools for parallel execution
   - Support for multiple execution engines (loop, dask, joblib, SLURM)
   - Automatic caching of results with parameter hashing
   - Progress tracking and visualization

4. VISUALIZATION:
   - Interactive Plotly visualizations for quality control
   - Comparison plots between preprocessing stages
   - PSD ratio plots for artifact correction evaluation

5. FLEXIBLE CONFIGURATION:
   - Parameters defined in separate configuration files
   - Support for subject-specific and session-specific settings

USAGE:
-----
The pipeline can be run in several ways:
1. Test mode: Run on a single subject with test_modular_eeg_preprocessing()
2. Full processing: Run compute_all() to process all subjects/sessions
3. Custom processing: Call compute_modular_eeg_preprocessing() with custom parameters

NOTES:
------
- The script is designed to be imported as a module, with job registration
  happening at import time
- Preprocessing parameters are defined in params.py
- Results are saved as xarray Datasets with full provenance tracking
"""


# Necessary imports
from configuration import *


# Base imports
from pathlib import WindowsPath
import xarray as xr
import pandas as pd
import numpy as np
import jobtools
from joblib import Parallel, delayed
import os
import plotly.express as px
from pathlib import Path


# Specific imports used several times
import scipy.signal as signal
import mne


# Intra package imports
from dataio import read_eeg, read_ecg
from utils import *
from artifact_algorithms import *
from SSA import embedding
from params import modular_eeg_preprocessing_params
from compute_trimming import trimming_job


from params import subject_keys



############################################################################################
#                               PREPROCESSING
############################################################################################

def correct_channel_names(data, run_key, channels, **p):
    """
    Correct channel name inconsistencies based on subject ID.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    run_key : str
        Run identifier, e.g., ``'sub-01_Repos'``.
    channels : list of str or np.ndarray
        List of channel names.
    **p : dict
        Additional keyword arguments (unused).

    Returns
    -------
    data : np.ndarray
        Unmodified data array.
    channels : list of str or np.ndarray
        Corrected channel names. For subjects with ID < 31 and not equal to 23,
        channel ``'FCz'`` is replaced by ``'Iz'``.
    """

    print('___________________________________Correcting channel names___________________________________')
    sub_id = int(('').join([a for a in run_key.split('_')[0] if a.isdigit()]))
    if sub_id < 31 and sub_id != 23:
        channels[list(channels).index('FCz')] = 'Iz'
    return data, channels

def subset(data, channels, **p):
    """
    Subset a data array along specified dimensions.

    Parameters
    ----------
    data : np.ndarray
        Multi-dimensional data array (e.g., time x channels).
    channels : list of str or np.ndarray
        Channel names corresponding to the channel dimension.
    **p : dict
        Dictionary of slicing specifications. Keys can be:
        - ``'channel'`` : list, tuple, or array
            Indices or names of channels to select.
        - ``'time'`` : tuple or array
            Indices or range to select along the time dimension.
        - int
            Any other dimension index (as a string) to slice explicitly.

        Values can be:
        - tuple
            Start and end indices (converted to `slice`).
        - list, np.ndarray
            Explicit indices to select.
        - list of str (for 'channel')
            Channel names to select.
        - tuple of str (for 'channel')
            Start and end indices based on channel names (converted to `slice`).

    Returns
    -------
    data_subset : np.ndarray
        Subset of the input data.
    channels_subset : np.ndarray
        Subset of channel names after indexing.

    Notes
    -----
    - ``time`` is assumed to be the first dimension.
    - ``channel`` is assued to be the second dimension.
    - Other dimensions can be accessed by their integer index (as string keys).
    """
    print('___________________________________Subsetting___________________________________')


    slicing_dict = p.copy()

    index = [slice(None)]*data.ndim

    for dimension, slicing in slicing_dict.items():
        if dimension == 'channel':
            if any([type(sl)==str for sl in slicing]):
                slicing_channels = type(slicing)([list(channels).index(channel) for channel in slicing])
            else:
                slicing_channels = slicing

            if isinstance(slicing_channels, tuple):
                index_channels = slice(*slicing_channels)  
            else:
                index_channels = np.array(slicing_channels)

            index[1] = index_channels
            channels = channels[index_channels]

        elif dimension == 'time':
            if isinstance(slicing, tuple):
                index[0] = slice(*slicing)  
            else:
                index[0] = np.array(slicing)
        else:
            if isinstance(slicing, tuple):
                index[int(dimension)] = slice(*slicing)  
            else:
                index[int(dimension)] = np.array(slicing)

    index = tuple(index)

    # print("Input shape:", data.shape)
    # print("Index used:", index)
    # print("Output shape:", data[index].shape)

    return data[index], channels





# Rescaling
############

def scale_eeg(data, **p):
    """
    Rescale EEG data by a specified factor.

    Parameters
    ----------
    data : np.ndarray
        EEG data array, shape typically (samples, channels).
    **p : dict
        Keyword arguments, expects:
        - rescaling_factor : float
            Factor by which to multiply the EEG data.

    Returns
    -------
    np.ndarray
        Rescaled EEG data as float.

    """
    
    print('___________________________________Rescaling by a factor of  :', p['rescaling_factor'], '___________________________________')
    # Rescale signal
    return (data*p['rescaling_factor']).astype(float)

# Trimming
#############

def trim_eeg(data, sampling_rate, run_key=None, **p):
    """
    Trim EEG data based on various methods and session parameters.

    Parameters
    ----------
    data : np.ndarray
        EEG data array with shape (samples, channels).
    sampling_rate : int or float
        Sampling rate of the EEG data in Hz.
    run_key : str, optional
        Identifier string for the EEG run/session (default is None).
    **p : dict
        Keyword arguments controlling trimming behavior. Expected keys include:
        - method : str
            Trimming method to use. Supported methods are:
            
            - 'centered':
              Trims a window centered around the middle of the data. The window duration
              is defined by `window_duration` for the session in seconds. Useful for 
              selecting a fixed-length segment centered in the recording.

            - 'both':
              Trims data starting at `start` seconds and removes `end` seconds from the 
              end of the data. The start and end are offsets in seconds. This method is 
              useful for removing unwanted initial and final segments.

            - 'delay':
              Trims data starting at `start` seconds, then removes `end` seconds from the 
              end, and finally restricts the trimmed data to a duration defined by 
              `window_duration`. This method is used for delayed segment extraction with 
              fixed length.

            - 'conservative':
              Extracts a segment of fixed duration (`window_duration` in seconds) starting 
              at `start` seconds, ensuring the segment fits within the total data length 
              conservatively. Useful to avoid boundary issues when trimming near the end.

        - session_adapted : bool
            Whether to adapt trimming based on session-specific parameters, such as trigger 
            times and margins.
        - interpolate_triggers : bool
            Whether to interpolate missing triggers (used if session_adapted).
        - window_duration : dict
            Duration window for trimming in seconds, keyed by session name.
        - start : float
            Start time in seconds for trimming (used in some methods).
        - end : float
            End time in seconds for trimming (used in some methods).

    Returns
    -------
    np.ndarray
        Trimmed EEG data subset.

    Notes
    -----
    If `session_adapted` is True and the run_key corresponds to specific conditions 
    (i.e., not 'Repos', 'Distraction', or 'baseline'), the function uses triggers and 
    session margins to define trimming boundaries.
    """

    method = p['method']
    print('___________________________________Trimming method :', method, '___________________________________')

    sampling_rate = int(sampling_rate)
    before = data.shape[0]/sampling_rate/60
    
    ds_trim = trimming_job.get(run_key)
    start = ds_trim["start_index"].values
    end = ds_trim["end_index"].values
    print(start, end)

    trimmed = data[start:end]

    after = trimmed.shape[0]/sampling_rate/60
    print(f'___________________________________Before/After trim :{before}| {after}___________________________________')

    return trimmed


@keying('run_key')
def trimming_adjustment(run_key, indices, sampling_rate=1, trimming_params=None, start_trimming = None, original_sampling_rate = 500):
    """
    Adjust indices of EEG triggers to account for trimming applied during preprocessing.

    Parameters
    ----------
    run_key : str
        Identifier string of the EEG run/session.
    indices : np.ndarray
        Array of sample indices (e.g. event or time indices) before trimming.
    sampling_rate : int or float, optional
        Sampling rate to which indices should be adjusted (default is 1).
    trimming_params : dict, optional
        Parameters used during trimming, including session parameters and margins.
    start_trimming : int or None, optional
        If provided, directly subtract this start trimming offset from indices.
    original_sampling_rate : int, optional
        Original sampling rate before downsampling or resampling (default is 500).

    Returns
    -------
    np.ndarray
        Adjusted indices aligned with trimmed EEG data.

    """
    
    if sampling_rate is None or (start_trimming is None and (run_key.split('_')[1][0] in ['E', 'R'] and trimming_params is not None)):
        sigs_eeg, sr, _ = read_eeg(run_key)

        sigs_eeg = sigs_eeg.astype(float)

    if sampling_rate is None:
        sampling_rate = sr

    if start_trimming is not None:
        return indices - start_trimming
    
    else:
        if trimming_params is None:
            return indices
        else:
            monotonic_indices = (np.arange(len(sigs_eeg))*1/sampling_rate)[:,np.newaxis]
            new_zero = trim_eeg(monotonic_indices, sampling_rate=sr, run_key=run_key, **trimming_params)
            if sampling_rate == 1:
                new_zero = new_zero.astype(int)
            return indices - new_zero[0]

# Detrending
#############

def detrend_eeg(data, sampling_rate, run_key = None, channels= None, **p):
    """
    Detrend EEG signals using a specified method.

    Parameters
    ----------
    data : np.ndarray
        EEG data array, shape (n_samples, n_channels).
    sampling_rate : float
        Sampling rate of the EEG data in Hz.
    run_key : str, optional
        Identifier for the EEG run/session (used in some methods).
    channels : list of int or None, optional
        List of channel indices to consider for certain detrending methods.
    **p : dict
        Parameters controlling detrending. Must include key ``name`` specifying method:
        
        - ``'linear'``:
          Removes a linear trend from each channel using
          :func:`scipy.signal.detrend`.

        - ``'robust_linear'``:
          Uses robust polynomial detrending via ``mg.detrend.detrend``, with keys:
            * ``order`` : polynomial order
            * ``threshold`` : outlier rejection threshold
            * ``n_iter`` : number of iterations

        - ``'trend_filter'``:
          Applies :func:`trend_filtering` for L₁ trend removal with keys:
            * ``vlambda`` : regularization strength
            * ``downsample`` : downsampling factor
          
        - ``'highpass'``:
          High-pass filters the data using MNE's :func:`mne.filter.filter_data`
          or step removal if ``step_removal=True`` with parameters:
            * ``cutoff_frequency`` : cutoff in Hz
            * ``time_window`` : window for variation detection
            * ``energy_threshold`` : threshold for variation energy
            * ``peak_to_peak_threshold`` : threshold for variation amplitude

    Returns
    -------
    np.ndarray
        Detrended EEG data, shape (n_samples, n_channels).

    Notes
    -----
    If no valid method is specified, returns the input data cast to float.
    """

    method = p['name']

    print('___________________________________Detrending method :', method, '___________________________________')

    if method == 'linear':
        detrended = signal.detrend(data, type='linear', axis=0)

    elif method == 'robust_linear':
        import meegkit as mg
        detrended, _, _ = mg.detrend.detrend(data, p['order'], w = np.ones(data.shape), basis='polynomials', threshold=p['threshold'], n_iter=p['n_iter'], show=False)

    elif method == 'trend_filter':
        detrended = trend_filtering(data, vlambda = p['vlambda'], downsample = p['downsample'], sampling_rate = sampling_rate )

    elif method == 'highpass':
        # sos = signal.butter(p['order'] , p['cutoff_frequency'] , btype='high', output='sos', fs = sampling_rate)
        # detrended = signal.sosfilt(sos, data, axis = 0)
        if p['step_removal']:
            detrended = correct_important_variations(data, time_window = p['time_window'],
                                                              energy_threshold = p['energy_threshold'], 
                                                              peak_to_peak_threshold = p['peak_to_peak_threshold'],
                                                              sampling_rate=int(sampling_rate), high_pass_frequency= p['cutoff_frequency'],
                                                              run_key=run_key, channels = channels )
        else:
            detrended = mne.filter.filter_data(data.astype(float).T , sampling_rate,  p['cutoff_frequency'], None).T
    
    else:
        print('-------------------------------No correct method provided, returning undetrended data-------------------------------')
        detrended = data.astype(float)
        

    return detrended.astype(float)

def trend_filtering(data, vlambda = 0.001, downsample = 5000, sampling_rate = 500):
    """
    Apply L₁ trend filtering to each channel of an array.

    Parameters
    ----------
    data : np.ndarray
        EEG data, shape (n_samples, n_channels).
    vlambda : float, optional
        Regularization parameter controlling smoothness of the estimated trend.
    downsample : int, optional
        Step size for downsampling before optimization to reduce computation.
    sampling_rate : float, optional
        Sampling rate of the data in Hz.

    Returns
    -------
    np.ndarray
        Detrended data array, same shape as input.

    See Also
    --------
    trend_filtering_one_channel : Core computation for a single channel.
    """
    detrended = np.empty(data.shape)

    for i in range(data.shape[1]):
        detrended[:,i] = trend_filtering_one_channel(data[:,i], vlambda = vlambda, downsample = downsample, sampling_rate = sampling_rate)
    
    return detrended

def trend_filtering_one_channel(data, vlambda = 0.001, downsample = 5000, sampling_rate = 500, return_trend = False):
    """
    Apply L₁ trend filtering to a single channel of EEG data.

    Parameters
    ----------
    data : np.ndarray
        Single-channel EEG data, shape (n_samples,).
    vlambda : float, optional
        Regularization parameter controlling smoothness of the estimated trend.
    downsample : int, optional
        Step size for downsampling before optimization.
    sampling_rate : float, optional
        Sampling rate of the data in Hz.
    return_trend : bool, optional
        If True, return the estimated trend instead of the detrended data.

    Returns
    -------
    np.ndarray
        If ``return_trend`` is False, detrended signal of same shape as input.
        If True, estimated trend of same shape as input.

    Raises
    ------
    Exception
        If the convex optimization solver does not converge.

    Notes
    -----
    The optimization problem solved is:

    .. math::

        \\min_x \\frac{1}{2} \\|y - x\\|_2^2 + \\lambda \\|D x\\|_1

    where :math:`D` is the second-order difference matrix.
    """
    # Downsample data to avoid low frequency filtering and excessive computation
    y = data[::downsample]
    n = y.size
    times = np.arange(data.shape[0])/sampling_rate

    D = build_second_difference_matrix(n)

    # Solve problem using convex optimization
    import cvxpy as cpx
    x = cpx.Variable(shape=n)
    obj = cpx.Minimize(0.5 * cpx.sum_squares(y - x) + vlambda * cpx.norm(D@x, 1))
    prob = cpx.Problem(obj)

    import cvxopt as cvxopt
    prob.solve(solver=cpx.CVXOPT, verbose=False)
    # print('Solver status: {}'.format(prob.status))

    # Check for error.
    if prob.status != cpx.OPTIMAL:
        raise Exception("Solver did not converge!")
    # print("optimal objective value: {}".format(obj.value))

    # Interpolate the fit to upsample
    values = x.value
    detrended = data - np.interp(times, times[::downsample], values)
    if return_trend :
        return np.interp(times, times[::downsample], values)
    else:
        return detrended

def build_second_difference_matrix(n):
    """
    Construct a forward second-order finite difference matrix.

    Parameters
    ----------
    n : int
        Length of the input signal.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (n-2, n) representing the second difference operator.

    """
    from scipy.sparse import spdiags
    e = np.ones((1, n))
    D = spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)

    return D

def correct_important_variations(data, time_window, energy_threshold, peak_to_peak_threshold, sampling_rate,
                                 high_pass_frequency, verify_plot=False, run_key = None, channels = None):
    """
    Detect and correct some important signal variations in EEG data.

    This function identifies high-energy or high-amplitude variations in EEG signals
    and corrects them using localized trend removal and high-pass filtering.
    The detection is performed on a downsampled, embedded version of the data,
    and corrections are applied to selected time windows.

    Parameters
    ----------
    data : np.ndarray
        EEG data array of shape (n_samples, n_channels).
    time_window : int
        Window length (in samples of the downsampled signal) for embedding when
        computing features for variation detection.
    energy_threshold : float
        Threshold for the normalized energy feature above which a window is marked for correction.
    peak_to_peak_threshold : float
        Threshold for the normalized peak-to-peak feature above which a window is marked for correction.
    sampling_rate : float
        Sampling rate of the EEG data in Hz.
    high_pass_frequency : float
        Cutoff frequency in Hz for the high-pass filter applied after correction.
    verify_plot : bool, optional
        If True, generate and save plots showing the original, mask, filtered, and corrected signals
        for each channel.
    run_key : str, optional
        Identifier string for the EEG run/session, used in plot filenames.
    channels : list of str or None, optional
        List of channel names corresponding to the columns of `data`.

    Returns
    -------
    np.ndarray
        Corrected EEG data array, shape (n_samples, n_channels).

    Notes
    -----
    **Detection process:**
    1. The data are mean-centered per channel.
    2. Each channel is downsampled by a factor of 5 and embedded in a lagged coordinate space.
    3. Two features are computed for each window: energy and peak-to-peak amplitude.
    4. Features are normalized by their median values, and windows exceeding either threshold
       are marked for correction.

    **Correction process:**
    - For each detected window, a margin of 10 seconds is added on both sides.
    - The local trend is estimated using :func:`trend_filtering_one_channel` and subtracted.
    - The resulting signal is high-pass filtered using MNE's :func:`mne.filter.filter_data`.

    **Plotting:**
    - If `verify_plot=True`, a PNG plot is saved for each channel showing:
      - Original EEG (downsampled for display)
      - Binary mask indicating detected variations
      - High-pass filtered signal
      - Corrected signal after trend removal and filtering

    See Also
    --------
    trend_filtering_one_channel : Performs L₁ trend filtering for a single channel.
    embedding : Converts a 1D signal into an embedded (lagged) matrix.
    energy : Computes signal energy per window.
    peak_to_peak : Computes peak-to-peak amplitude per window.
    find_windows : Converts a binary mask into time window boundaries.
    """

    centered = center_eeg(data)


    def process_channel(k):
        print(f'Correcting channel {k} : {channels[k]}')
        
        X = embedding(centered[::5, k], window_length=time_window)
        multivariate_matrix = X - np.mean(X, axis=0)
        feature_matrix = np.empty((2, multivariate_matrix.shape[1]))

        for i, f in enumerate([energy, peak_to_peak]):
            feature_matrix[i, :] = f(multivariate_matrix)
            feature_matrix[i, :] /= np.median(feature_matrix[i, :], axis=0)

        mask = ((feature_matrix[0, :] > energy_threshold) + 
                (feature_matrix[1, :] > peak_to_peak_threshold)).astype(int)
        
        windows = find_windows(mask, sr = int(sampling_rate/5))
        filtered = mne.filter.filter_data(centered[:, k].astype(float).copy(), sampling_rate, high_pass_frequency, None, verbose=False)
        corrected = filtered.copy()
        for idx in range(len(windows)):
            side = 10
            step_a = int(max(windows[idx][0]-side, 0)*sampling_rate)
            step_b = int(min(windows[idx][1]+side, data.shape[0]*1/sampling_rate)*sampling_rate)
            X_trend = centered[:, k][step_a: step_b]
            median = pd.Series(np.pad(X_trend, int(sampling_rate), mode='reflect')).rolling(int(sampling_rate), center=True).median().values[int(sampling_rate):-int(sampling_rate)]
            trend_filter = trend_filtering_one_channel(median, vlambda=0.01, downsample=sampling_rate, return_trend=True)
            corrected[step_a: step_b] = mne.filter.filter_data(X_trend - trend_filter, sampling_rate, high_pass_frequency, None, verbose=False)


        if verify_plot:
            times = np.arange(centered.shape[0]) * 1 / sampling_rate
            t = np.linspace(0, centered.shape[0] * 1 / sampling_rate, mask.shape[0])

            df_plot = pd.concat([
                pd.DataFrame({'Time': times[::5], 'Type': 'EEG', 'EEG': centered[::5, k]}),
                pd.DataFrame({'Time': t, 'Type': 'Mask', 'EEG': mask * np.max(np.abs(centered[::5, k]))}),
                pd.DataFrame({'Time': times[::5], 'Type': 'Filtered', 'EEG': filtered[::5]}),
                pd.DataFrame({'Time': times[::5], 'Type': 'Corrected', 'EEG': corrected[::5]}),
            ])

            name = f'{run_key}_{channels[k]}.png'
            fig = px.line(df_plot, x='Time', y='EEG', color='Type', title=name, width = 1200, height = 600)
            fig.write_image(figures_path / 'Steps' / name)
        
        return corrected
    
    with set_num_threads(1):
        num_cores =  max(1, os.cpu_count() - 1)
        corrected_signals = Parallel(n_jobs=num_cores)(delayed(process_channel)(k) for k in range(centered.shape[1]))

    return np.column_stack(corrected_signals).astype(float)

# Centering
############

def center_eeg(data, **p):
    """
    Center EEG signals by removing the mean from each channel.

    This function subtracts the mean value of each channel from the 
    signal, ensuring the data is centered around zero. 

    Parameters
    ----------
    data : ndarray of shape (n_times, n_channels)
        Input EEG data, where rows correspond to time points and 
        columns correspond to channels.
    **p : dict, optional
        Additional parameters (not used).

    Returns
    -------
    centered : ndarray of shape (n_times, n_channels)
        Mean-centered EEG data.

    Notes
    -----
    - The mean is computed independently for each channel.
    - This step is purely linear and does not affect signal frequency 
      content.

    """
    print('___________________________________Mean centering___________________________________')
    centered = data - np.mean(data, axis = 0)
    return centered 

# Notch filtering
############

def notch_filter_eeg(data, sampling_rate, **p):
    """
    Apply a notch filter to EEG data to remove specific frequency components.

    This function removes narrow-band noise (e.g., powerline interference) 
    at the specified notch frequency or frequencies using MNE's filtering 
    utilities.

    Parameters
    ----------
    data : ndarray of shape (n_times, n_channels)
        Input EEG data, where rows correspond to time points and columns 
        correspond to channels.
    sampling_rate : float
        Sampling rate of the EEG signal in Hz.
    **p : dict
        Filter parameters:
        
        - ``notch_frequency`` : float or list of float
            Frequency (or list of frequencies) to remove.
        - ``method`` : {'fft', 'iir'}
            Filtering method to use.
        - ``iir_params`` : dict, optional
            Additional IIR filter parameters if method is 'iir'.
        - ``phase`` : {'zero', 'zero-double', 'minimum'}
            Phase response type.

    Returns
    -------
    notch_filtered : ndarray of shape (n_times, n_channels)
        EEG data after notch filtering.

    Notes
    -----
    - Notch filtering is useful for removing mains noise at 50/60 Hz 
      and its harmonics.

    """
    print(f'___________________________________Notch filtering at {p["notch_frequency"]}___________________________________')
    if p['method'] == 'iir':
        iir_params = p['iir_params']
    else:
        iir_params = None

    notch_filtered = mne.filter.notch_filter(data.T, sampling_rate , p["notch_frequency"], method= p["method"], iir_params=iir_params, phase=p["phase"], verbose = False)

    return notch_filtered.T

# Filtering
############

def filter_eeg(data, sampling_rate, **p):
    """
    Apply low-pass and/or high-pass filtering to EEG data.

    This function applies band-pass, low-pass, or high-pass filtering 
    to EEG signals using MNE's filtering utilities.

    Parameters
    ----------
    data : ndarray of shape (n_times, n_channels)
        Input EEG data, where rows correspond to time points and columns 
        correspond to channels.
    sampling_rate : float
        Sampling rate of the EEG signal in Hz.
    **p : dict
        Filter parameters:
        
        - ``low_cutoff`` : float or None
            Lower frequency bound for filtering in Hz.
        - ``high_cutoff`` : float or None
            Upper frequency bound for filtering in Hz.
        - ``method`` : {'fft', 'iir'}
            Filtering method to use.
        - ``iir_params`` : dict, optional
            Additional IIR filter parameters if method is 'iir'.
        - ``phase`` : {'zero', 'zero-double', 'minimum'}
            Phase response type.

    Returns
    -------
    filtered_eeg : ndarray of shape (n_times, n_channels)
        EEG data after filtering.

    Notes
    -----
    - Can be used for high-pass, low-pass, or band-pass filtering 
      depending on cutoff values.

    """
    '''
    Lowpass or/and highpass filters the signal
    '''
    print(f'___________________________________Filtering between {p["low_cutoff"]} Hz and {p["high_cutoff"]} Hz___________________________________')
    if p['method'] == 'iir':
        iir_params = p['iir_params']
    else:
        iir_params = None

    filtered_eeg = mne.filter.filter_data(data.T ,sampling_rate, p['low_cutoff'], p['high_cutoff'],
                                method = p['method'], iir_params=iir_params, phase=p['phase'], verbose = False)
    
    return filtered_eeg.T

# Compute ICA
#############

def compute_ICA(data, channels, sampling_rate, **p):
    """
    Compute Independent Component Analysis (ICA) decomposition on EEG data.

    This function converts NumPy EEG data into an MNE `Raw` object, applies
    montage and filtering, optionally re-references the data, and performs
    ICA decomposition using parameters passed in `p`.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        EEG data matrix where rows are time points and columns are channels.
    channels : list of str
        Names of EEG channels, corresponding to the columns of `data`.
    sampling_rate : float
        Sampling rate of the EEG data in Hz.
    **p : dict
        Additional parameters controlling preprocessing and ICA:
        
        - ``filter_signal`` : tuple of (low_freq, high_freq)
            Band-pass filter range in Hz.
        - ``set_reference`` : str or None
            Reference scheme (e.g., 'average') or None to keep as is.
        - ``use_torch`` : bool
            Whether to use PyTorch backend for ICA fitting.
        - ``n_components`` : int or None
            Number of ICA components to estimate.
        - ``random_state`` : int or None
            Seed for ICA reproducibility.
        - ``method`` : {'fastica', 'picard', 'infomax'}
            ICA fitting method.
        - ``fit_params`` : dict
            Additional parameters for ICA fit.
        - ``max_iter`` : int
            Maximum iterations for ICA algorithm.

    Returns
    -------
    data : ndarray of shape (n_samples, n_channels)
        Original EEG data (unchanged).
    ica : mne.preprocessing.ICA
        The fitted ICA object.

    Notes
    -----
    - Uses the `standard_1020` electrode montage.
    - If ``use_torch`` is True, MNE will use the Torch-based ICA solver.

    See Also
    --------
    mne.preprocessing.ICA : ICA implementation in MNE.
    """

    raw = numpy_to_mne(data.copy().T, channel_names=channels, sampling_rate=sampling_rate)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    filter_frequencies = p['filter_signal']
    raw.filter(filter_frequencies[0], filter_frequencies[1])

    if p['set_reference'] is not None:
        raw.set_eeg_reference(p['set_reference'])

    print(f'___________________________________Computing ICA___________________________________')

    if p['use_torch']:
        mne.utils.set_config("MNE_ICA_USE_TORCH", "true")

    ica = mne.preprocessing.ICA(n_components=p['n_components'], random_state=p['random_state'],
                                    method=p['method'], fit_params=p['fit_params'], max_iter=p['max_iter'])
    
    ica.fit(raw, verbose=True, )
    
    return data, ica

def ica_component_selection(raw, ica, run_key=None, get_labels = False, **p):
    """
    Select ICA components to exclude from EEG data based on different strategies.

    This function supports manual selection, automatic ICLabel classification,
    and ECG artifact detection methods for identifying artifactual ICA components.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA object corresponding to `raw`.
    run_key : str or None, optional
        Identifier for the dataset/run (used for ECG reading).
    get_labels : {False, 1, 2}, default=False
        Output mode:
        - ``False`` : Return indices of components to exclude.
        - ``1`` : Return component labels only.
        - ``2`` : Return tuple ``(to_be_corrected, labels_corrected)``.
    **p : dict
        Processing and selection parameters:
        
        - ``ICA_params`` : dict
            Parameters for filtering and referencing before ICA:
            - ``filter_signal`` : tuple of (low_freq, high_freq)
            - ``set_reference`` : str or None
        - ``selection_method`` : {'manual', 'iclabel', 'mne_bad_ecg'}
            Method to identify components to remove.
        - ``automatic_label`` : str
            ICLabel label for automatic rejection in 'iclabel' mode.
        - ``ecg_threshold`` : float
            Threshold for ECG artifact correlation detection.
        - ``trimming_parameters`` : dict
            Parameters for trimming ECG data (passed to `trim_eeg`).

    Returns
    -------
    list of int or list of str or tuple
        Depending on `get_labels`:
        - List of component indices to exclude.
        - List of component labels (if get_labels=1).
        - Tuple (to_be_corrected, labels_corrected) if get_labels=2.

    Notes
    -----
    - **manual**:
        Prompts user to manually inspect ICA sources and components.
    - **iclabel**:
        Uses ICLabel classification with re-referencing to average
        and component correspondence matching.
    - **mne_bad_ecg**:
        Detects ECG artifacts via correlation with ECG channel.

    See Also
    --------
    mne.preprocessing.ICA.exclude : Attribute storing indices to exclude.
    label_components : Function to classify ICA components.
    trim_eeg : Function to trim ECG/EEG data segments.
    """
    from mne_icalabel import label_components

    filter_frequencies = p['ICA_params']['filter_signal']
    filtered_raw = raw.copy()
    filtered_raw.filter(filter_frequencies[0], filter_frequencies[1])

    if p['ICA_params']['set_reference'] is not None:
        filtered_raw.set_eeg_reference(p['ICA_params']['set_reference'])
    
    to_be_corrected = []

    if p['selection_method'] == 'manual':
        ic_labels = label_components(filtered_raw , ica, method="iclabel")
        print('Unrereferenced labels', ic_labels["labels"])
        fig_1 = ica.plot_sources(filtered_raw , show=True)
        fig_2 = ica.plot_components(show=True)
        to_be_corrected = ica.exclude
        ica.exclude = []

    elif p['selection_method'] == 'iclabel':
        # Computing ICA on average reference signal
        ica_avg = ica.copy()
        avg_raw = filtered_raw.copy().set_eeg_reference()
        ica_avg.fit(avg_raw)

        # Rereferencing original components to average
        ica_transformed = ica.copy()
        ica_transformed.pca_components_[: ica_transformed.n_components_] -= np.mean(ica_transformed.pca_components_[: ica_transformed.n_components_],axis=1)[:, np.newaxis]

        # Computing correspondences
        from sklearn.metrics.pairwise import cosine_similarity
        cosines_all = cosine_similarity(ica_avg.get_components().T,
                                 ica_transformed.get_components().T)
        cosines_all_maxima = np.max(np.abs(cosines_all), axis = 0)
        correspondences = np.argmax(np.abs(cosines_all), axis = 0)
        # correspondences = [c for i,c in enumerate(correspondences) if cosines_all_maxima[i]>0.7]
        correspondences = [int(c) if cosines_all_maxima[i]>0.7 else -1 for i,c in enumerate(correspondences) ]

        # Computing labels
        labels_avg = label_components(avg_raw , ica_avg, method="iclabel")["labels"]
        labels_original = label_components(filtered_raw , ica, method="iclabel")["labels"]
        # labels_corrected = [labels_avg[i] if i in correspondences else labels_original[i] for i in range(len(labels_original)) ]
        labels_corrected = [labels_avg[correspondences[i]] if correspondences[i] != -1 else labels_original[i] for i in range(len(labels_original)) ]
        print(labels_corrected)
        to_be_corrected = [i for i, label in enumerate(labels_corrected) if label==p['automatic_label']]

        # Adjusting labels based on average reference ICA

    
    elif p['selection_method'] == 'mne_bad_ecg':
        ecg, sr = read_ecg(run_key)
        ecg = trim_eeg(ecg.reshape(-1,1), sr, run_key, **p['trimming_parameters']).T

        raw_ecg = numpy_to_mne(ecg, ['ECG'], sr, 'ecg')
        raw_ecg.set_montage(None)
        raw_plus_ecg = raw.copy()
        raw_plus_ecg.add_channels([raw_ecg])
        raw_plus_ecg.filter(1,100)
        raw_plus_ecg.set_eeg_reference('average')
        ecg_index, ecg_scores = ica.find_bads_ecg(raw_plus_ecg, ch_name='ECG', method = 'correlation', measure='correlation', threshold = p['ecg_threshold'])
        print([(i, ecg_scores[i]) for i in  ecg_index])
        to_be_corrected = ecg_index


    selection_method = p['selection_method']
    print(f'___________________________________Selected components : {to_be_corrected} with method {selection_method}___________________________________')

    if get_labels == 1:
        return labels_corrected
    
    elif get_labels == 2:
        return to_be_corrected, labels_corrected
    else:
        return to_be_corrected

# EOG Artefact removal
###################

def remove_eog_artifacts(data, ica, sampling_rate, channels= None, save=None, **p):
    """
    Remove EOG artifacts from EEG data using various ICA-based correction methods.

    This function orchestrates the removal of ocular artifacts from EEG signals
    by applying a selected Independent Component Analysis (ICA) correction method.
    The method is chosen via the parameter ``p['name']`` and can include standard ICA,
    KSSA-ICA, CWT-MSSA-ICA, or rectified variants.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        EEG data in NumPy array format. Each column corresponds to a channel.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA object used for component identification and removal.
    sampling_rate : float
        Sampling frequency of the EEG data in Hz.
    channels : list of str, optional
        List of channel names corresponding to the columns in ``data``.
        Required for visualization.
    save : str or Path, optional
        File path (without extension) to save visualization results as HTML.
    **p : dict
        Additional parameters controlling:
        - ``name`` : str
            Correction method. Options: ``'ICA'``, ``'KSSA_ICA'``,
            ``'CWT_MSSA_ICA'``, ``'Rectified_CWT_MSSA_ICA'``,
            ``'Rectified_V2_CWT_MSSA_ICA'``, ``'Rectified_V3_CWT_MSSA_ICA'``.
        - Method-specific settings (see corresponding correction functions).

    Returns
    -------
    eog_corrected : ndarray of shape (n_samples, n_channels)
        EEG data after EOG artifact removal.

    Notes
    -----
    This is a high-level wrapper that delegates the actual artifact correction
    to specialized functions depending on the chosen method.
    """

    method = p['name']
    print(f'___________________________________Removing EOG artifacts using {method}___________________________________')

    raw = numpy_to_mne(data.T, channel_names=channels, sampling_rate=sampling_rate)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    to_be_corrected, labels = ica_component_selection(raw, ica, get_labels=2, **p)

    if len(to_be_corrected) > 0 :
        if method == 'ICA':
            eog_corrected = remove_from_ICA(raw, ica, to_be_corrected, channels, **p)

        elif method == 'KSSA_ICA':
            eog_corrected = KSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels, save=save, **p)    
        
        elif method == 'CWT_MSSA_ICA':
            eog_corrected = CWT_MSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels, save=save, **p)   
        
        elif method == 'Rectified_CWT_MSSA_ICA':
            eog_corrected = rectified_CWT_MSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels, save=save, **p)   

        elif method == 'Rectified_V2_CWT_MSSA_ICA':
            eog_corrected = rectified_V2_CWT_MSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels, save=save, **p)   
        
        elif method == 'Rectified_V3_CWT_MSSA_ICA':
            eog_corrected = rectified_V3_CWT_MSSA_ICA_correction(raw, ica, labels, sampling_rate, channels, save=save, **p) 


    else:
        eog_corrected = data

    return eog_corrected 

def remove_from_ICA(raw, ica, to_be_corrected, channels = None, visualize=True, return_ica=False, **p):
    """
    Remove selected ICA components from EEG data.

    This function removes specified ICA components from an MNE Raw object and,
    if enabled, visualizes the before-and-after comparison for selected channels.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA object used for component removal.
    to_be_corrected : list of int
        Indices of ICA components to remove.
    channels : list of str, optional
        List of channel names for visualization. Required if ``visualize=True``.
    visualize : bool, default=True
        If True, displays a time series comparison of raw and ICA-corrected signals
        for a subset of channels.
    return_ica : bool, default=False
        If True, returns the ICA object after modification.
        (Currently unused in the function.)
    **p : dict
        Additional parameters (unused in current implementation).

    Returns
    -------
    corrected_data : ndarray of shape (n_samples, n_channels)
        ICA-corrected EEG data.

    Notes
    -----
    The visualization compares raw and corrected data for the channels
    indexed by ``[0, 22, 53, 46]``.
    """

    ica.exclude = to_be_corrected
    raw_ica = raw.copy()
    ica.apply(raw_ica)

    if visualize:
        channel_idx = [0, 22, 53, 46]
        t = raw.times[::5]
        df = pd.DataFrame()
        for i in channel_idx :
            df = pd.concat([df,
                pd.DataFrame({'Time' : t, 'Type' : 'Raw', 'Channel': channels[i], 'Amplitude'  : raw.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'ICA', 'Channel':  channels[i], 'Amplitude'  : raw_ica.get_data()[i,::5]}),
            ])
        f = px.line(df, x='Time', y='Amplitude', color='Type', facet_row='Channel', height = 300*len(channel_idx))
        f.show()

    return raw_ica.get_data().T

def KSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels = None, visualize=True, save=None,  **p):
    """
    Apply K-means Singular Spectrum Analysis (KSSA) correction to selected ICA components.

    This method decomposes selected ICA components using K-means SSA, reconstructs
    artifact-free components, and reinserts them into the EEG data. Optionally,
    the process can be visualized and saved.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA object.
    to_be_corrected : list of int
        ICA component indices to correct.
    sampling_rate : float
        Sampling frequency of the EEG data in Hz.
    channels : list of str, optional
        Channel names for visualization.
    visualize : bool, default=True
        If True, visualizes a comparison of raw, standard ICA, and KSSA-ICA corrected signals.
    save : str or Path, optional
        Path to save visualization as HTML.
    **p : dict
        Additional parameters controlling KSSA decomposition:
        - ``window_length_seconds`` : float
            SSA window length in seconds.
        - ``kmeans_cluster_count`` : int
            Number of clusters for K-means in SSA space.
        - ``fractal_dimension_threshold`` : float
            Threshold for fractal dimension filtering.
        - ``ssa_threshold`` : float
            Threshold for SSA component selection.
        - ``ICA_settings`` : dict
            Settings for ICA (e.g., ``random_state``).

    Returns
    -------
    corrected_data : ndarray of shape (n_samples, n_channels)
        EEG data after KSSA-ICA correction.

    Notes
    -----
    The correction is applied only to the selected ICA components. Non-selected
    components remain unchanged.
    """

    source_signal = ica.get_sources(raw)

    print('Starting KSSA corrections')
    def get_correction(k):
        single_channel_signal = source_signal.get_data()[k]

        corrected, template = kmeans_ssa(single_channel_signal, window_length = p['window_length_seconds'],
                                          sampling_rate=sampling_rate, kmeans_cluster_count = p['kmeans_cluster_count'], 
                                          fractal_dimension_threshold = p['fractal_dimension_threshold'], ssa_threshold = p['ssa_threshold'],
                                          random_state = p['ICA_settings']['random_state'])
        
        correction = corrected*template
        return correction


    num_cores =  max(1, os.cpu_count() - 1)
    corrected_signals = Parallel(n_jobs=num_cores)(delayed(get_correction)(k) for k in to_be_corrected)
    corrected_signals = np.column_stack(corrected_signals).T.astype(float)
    print('Corrections computed')

    if visualize:

        simple_ica, kssa_ica  = apply_modified(ica, raw, corrected_signals, to_be_corrected, include_ica=True)

        channel_idx = [0, 22, 53, 46]
        t = raw.times[::5]
        df = pd.DataFrame()
        for i in channel_idx :
            df = pd.concat([df,
                pd.DataFrame({'Time' : t, 'Type' : 'Raw', 'Channel': channels[i], 'Amplitude'  : raw.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'ICA', 'Channel':  channels[i], 'Amplitude'  : simple_ica.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'KSSA-ICA', 'Channel':  channels[i], 'Amplitude'  : kssa_ica.get_data()[i,::5]}),
            ])
        f = px.line(df, x='Time', y='Amplitude', color='Type', facet_row='Channel', height = 300*len(channel_idx), title=f'Sample KSSA-ICA results {save}')
        f.show()
        if save is not None:
            save = save + '.html'
            f.write_html(figures_path / 'ICA' / 'ICA_KSSA' / save)




    else:
        kssa_ica  = apply_modified(ica, raw, corrected_signals, to_be_corrected, include_ica=False)

    return kssa_ica.get_data().T

def CWT_MSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels = None, visualize=True, save=None,  **p):
    """
    Apply Continuous Wavelet Transform (CWT) and Multistage Singular Spectrum Analysis (MSSA)
    corrections to ICA components, optionally visualizing the results.

    This function extracts ICA source signals from the given `raw` data, applies
    the `CWT_MSSA` artifact correction method to selected components, and reconstructs
    the corrected EEG signal. Optionally, it visualizes and/or saves comparison
    plots between raw, ICA, and corrected signals.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data object from MNE.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA object containing the decomposition of `raw`.
    to_be_corrected : list of int
        List of ICA component indices to be corrected using CWT-MSSA.
    sampling_rate : float
        Sampling rate of the EEG data in Hz.
    channels : list of str, optional
        List of channel names, used for labeling plots. If None, channel labels are omitted.
    visualize : bool, default=True
        If True, generates interactive plots comparing raw, ICA, and corrected data.
    save : str or None, default=None
        Base filename (without extension) for saving the visualization as an HTML file.
        If None, no file is saved.
    **p : dict
        Additional parameters passed to the `CWT_MSSA` function, including:
        - wavelets_frequency_spacing
        - wavelets_min_frequency
        - wavelets_max_frequency
        - filter_f_min
        - filter_f_max
        - metric_function
        - metric_factor
        - margins
        - widths
        - ssa_window_seconds
        - first_ssa_trajectories_indices
        - second_ssa_downsampling
        - second_ssa_window_seconds
        - second_ssa_trajectories_indices
        - return_unbias
        - return_corrected

    Returns
    -------
    ndarray of shape (n_times, n_channels)
        The reconstructed EEG data after applying CWT-MSSA corrections.

    Notes
    -----
    - Uses `plotly.express` for visualization if `visualize=True`.
    - This function is designed for offline batch processing and may be slow for large datasets.

    See Also
    --------
    rectified_CWT_MSSA_ICA_correction : Similar method with rectification preprocessing.
    """

    source_signal = ica.get_sources(raw)

    print('Starting CWT MSSA corrections')
    def get_correction(k):
        single_channel_signal = source_signal.get_data()[k]

        if base_folder == Path('/mnt/data/RelaxCons/'):
            show = False
        else:
            show = True
    
        correction = CWT_MSSA( single_channel_signal, sampling_rate,
                                        wavelets_frequency_spacing = p['wavelets_frequency_spacing'] , wavelets_min_frequency = p['wavelets_min_frequency'],
                                        wavelets_max_frequency = p['wavelets_max_frequency'], filter_f_min = p['filter_f_min'], 
                                        filter_f_max = p['filter_f_max'], metric_function=p['metric_function'], metric_factor=p['metric_factor'],
                                        margins = p['margins'], widths = p['widths'], ssa_window_seconds = p['ssa_window_seconds'],
                                        first_ssa_trajectories_indices = p['first_ssa_trajectories_indices'], second_ssa_downsampling = p['second_ssa_downsampling'], 
                                        second_ssa_window_seconds = p['second_ssa_window_seconds'], second_ssa_trajectories_indices =p['second_ssa_trajectories_indices'],
                                        return_unbias = p['return_unbias'], return_corrected = p['return_corrected'], show=show, save = None, details= True,
                                        )
        
        
        
        return correction


    # num_cores =  max(1, os.cpu_count() - 1)
    # corrected_signals = Parallel(n_jobs=num_cores)(delayed(get_correction)(k) for k in to_be_corrected)
    # corrected_signals = np.column_stack(corrected_signals).T.astype(float)
    corrected_signals = []
    print('Corrections computed')
    for k in to_be_corrected :
        corrected_signals.append(get_correction(k))
    corrected_signals = np.column_stack(corrected_signals).T.astype(float)

    if base_folder == Path('/mnt/data/RelaxCons/'):
        show = False
    else:
        show = True

    if visualize and show:

        simple_ica, cwt_mssa_ica  = apply_modified(ica, raw, corrected_signals, to_be_corrected, include_ica=True)

        channel_idx = [0, 22, 53, 46]
        t = raw.times[::5]
        df = pd.DataFrame()
        for i in channel_idx :
            df = pd.concat([df,
                pd.DataFrame({'Time' : t, 'Type' : 'Raw', 'Channel': channels[i], 'Amplitude'  : raw.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'ICA', 'Channel':  channels[i], 'Amplitude'  : simple_ica.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'CWT_MSSA-ICA', 'Channel':  channels[i], 'Amplitude'  : cwt_mssa_ica.get_data()[i,::5]}),
            ])
        f = px.line(df, x='Time', y='Amplitude', color='Type', facet_row='Channel', height = 300*len(channel_idx), title=f'Sample CWT_MSSA-ICA results {save}')
        f.show()
        if save is not None:
            save = save + '.html'
            f.write_html(figures_path / 'ICA' / 'Rectified_CWT_MSSA' / save)




    else:
        cwt_mssa_ica  = apply_modified(ica, raw, corrected_signals, to_be_corrected, include_ica=False)

    return cwt_mssa_ica.get_data().T

def rectified_CWT_MSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels = None, visualize=True, save=None, save_ratio=True, **p):
    """
    Apply rectified Continuous Wavelet Transform (CWT) and Multistage Singular Spectrum Analysis (MSSA)
    corrections to ICA components, with optional PSD ratio analysis.

    This function performs the same high-level steps as `CWT_MSSA_ICA_correction`, but uses
    `rectified_CWT_MSSA` for artifact removal, which includes additional preprocessing such
    as rectification, smoothing, and artifact localization.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data object from MNE.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA object containing the decomposition of `raw`.
    to_be_corrected : list of int
        List of ICA component indices to be corrected.
    sampling_rate : float
        Sampling rate of the EEG data in Hz.
    channels : list of str, optional
        List of channel names, used for labeling plots. If None, channel labels are omitted.
    visualize : bool, default=True
        If True, generates interactive plots comparing raw, ICA, and corrected data.
    save : str or None, default=None
        Base filename (without extension) for saving the visualization as an HTML file.
        If None, no visualization file is saved.
    save_ratio : bool, default=True
        If True and `visualize=False`, computes and saves power spectral density (PSD) ratio plots.
    **p : dict
        Additional parameters passed to the `rectified_CWT_MSSA` function, including:
        - wavelets_frequency_spacing
        - wavelets_min_frequency
        - wavelets_max_frequency
        - filter_f_min
        - filter_f_max
        - metric_function
        - metric_factor
        - margins
        - widths
        - ssa_window_seconds
        - first_ssa_trajectories_indices
        - second_ssa_downsampling
        - second_ssa_window_seconds
        - second_ssa_trajectories_indices
        - return_unbias
        - return_corrected
        - return_localisation
        - artifact_margin
        - artifact_plateau
        - artifact_shift
        - smoothing_size
        - smoothing_first_pass_width
        - smoothing_first_pass_iteration
        - smoothing_second_pass_width
        - gradient_peaks_height
        - gradient_peaks_prominence
        - gradient_peaks_width
        - rectification_drift_highpass
        - ease_in_width

    Returns
    -------
    ndarray of shape (n_times, n_channels)
        The reconstructed EEG data after applying rectified CWT-MSSA corrections.

    Notes
    -----
    - If `save_ratio=True`, PSD ratios are computed for slow (0.05-0.5 Hz) and fast (0.5-200 Hz) bands.
    - This method is intended for artifact-heavy EEG where rectification improves saccade correction.

    See Also
    --------
    CWT_MSSA_ICA_correction : Similar method without rectification preprocessing.
    """

    # reference = 'Fz'
    rereferenced_raw = raw.copy()
    # rereferenced_raw.add_reference_channels(reference)
    # rereferenced_raw.set_eeg_reference(p['ICA_params']['set_reference'])
    source_signal = ica.get_sources(rereferenced_raw)

    print('Starting Rectified CWT MSSA corrections')
    def get_correction(k):
        single_channel_signal = source_signal.get_data()[k]

        if base_folder == Path('/mnt/data/RelaxCons/'):
            show = False
        else:
            show = True

        correction = rectified_CWT_MSSA( single_channel_signal, sampling_rate, 
                                        wavelets_frequency_spacing = p['wavelets_frequency_spacing'] , wavelets_min_frequency = p['wavelets_min_frequency'],
                                        wavelets_max_frequency = p['wavelets_max_frequency'], filter_f_min = p['filter_f_min'], 
                                        filter_f_max = p['filter_f_max'], metric_function=p['metric_function'], metric_factor=p['metric_factor'],
                                        margins = p['margins'], widths = p['widths'], ssa_window_seconds = p['ssa_window_seconds'],
                                        first_ssa_trajectories_indices = p['first_ssa_trajectories_indices'], second_ssa_downsampling = p['second_ssa_downsampling'], 
                                        second_ssa_window_seconds = p['second_ssa_window_seconds'], second_ssa_trajectories_indices =p['second_ssa_trajectories_indices'],
                                        return_unbias = p['return_unbias'], return_corrected = p['return_corrected'], show=show, save = None, details= True,
                                        return_localisation = p['return_localisation'], artifact_margin = p['artifact_margin'], artifact_plateau = p['artifact_plateau'],
                                        artifact_shift = p['artifact_shift'], smoothing_size=p['smoothing_size'], smoothing_first_pass_width = p['smoothing_first_pass_width'],
                                        smoothing_first_pass_iteration = p['smoothing_first_pass_iteration'], smoothing_second_pass_width = p['smoothing_second_pass_width'],
                                        gradient_peaks_height=p['gradient_peaks_height'], gradient_peaks_prominence=p['gradient_peaks_prominence'],
                                          gradient_peaks_width=p['gradient_peaks_width'], rectification_drift_highpass = p['rectification_drift_highpass'],
                                          ease_in_width = p['ease_in_width']
                                        )
        
        
        
        return correction


    # num_cores =  max(1, os.cpu_count() - 1)
    # corrected_signals = Parallel(n_jobs=num_cores)(delayed(get_correction)(k) for k in to_be_corrected)
    # corrected_signals = np.column_stack(corrected_signals).T.astype(float)
    corrected_signals = []
    print('Corrections computed')
    for k in to_be_corrected :
        corrected_signals.append(get_correction(k))
    corrected_signals = np.column_stack(corrected_signals).T.astype(float)

    if base_folder == Path('/mnt/data/RelaxCons/'):
        show = False
    else:
        show = True

    if visualize and show:

        simple_ica, rectified_cwt_mssa_ica  = apply_modified(ica, rereferenced_raw, corrected_signals, to_be_corrected, include_ica=True)
        # rectified_cwt_mssa_ica.set_eeg_reference([reference])

        channel_idx = [0, 22, 53, 46]
        t = raw.times[::5]
        df = pd.DataFrame()
        for i in channel_idx :
            df = pd.concat([df,
                pd.DataFrame({'Time' : t, 'Type' : 'Raw', 'Channel': channels[i], 'Amplitude'  : raw.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'ICA', 'Channel':  channels[i], 'Amplitude'  : simple_ica.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'rectified_CWT_MSSA-ICA', 'Channel':  channels[i], 'Amplitude'  : rectified_cwt_mssa_ica.get_data()[i,::5]}),
            ])
        f = px.line(df, x='Time', y='Amplitude', color='Type', facet_row='Channel', height = 300*len(channel_idx), title=f'Sample rectified_CWT_MSSA-ICA results {save}')
        f.show()
        if save is not None:
            save = save + '.html'
            f.write_html(figures_path / 'ICA' / 'Rectified_CWT_MSSA' / save)
    
    
    elif save_ratio :
        simple_ica, rectified_cwt_mssa_ica  = apply_modified(ica, rereferenced_raw, corrected_signals, to_be_corrected, include_ica=True)
        psd_slow = raw.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)
        slow_freqs = psd_slow.freqs
        psd_slow_2 = simple_ica.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)
        psd_slow_3 = rectified_cwt_mssa_ica.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)

        psd_fast = raw.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))
        fast_freqs = psd_fast.freqs
        psd_fast_2 = simple_ica.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))
        psd_fast_3 = rectified_cwt_mssa_ica.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))

        freqs = np.concatenate([slow_freqs, fast_freqs], axis = 0)
        psd = np.concatenate([psd_slow.get_data(), psd_fast.get_data()], axis = 1)
        psd_2 = np.concatenate([psd_slow_2.get_data(), psd_fast_2.get_data()], axis = 1)
        psd_3 = np.concatenate([psd_slow_3.get_data(), psd_fast_3.get_data()], axis = 1)

        fig_ratio = plot_psd_ratios(
            numerators=[psd_2, psd_3], denominators=[psd, psd], freqs=freqs,
            vline_x=13, hline_y=1, labels=['ICA/Detrended', 'R_CWT_MSSA/Detrended'],
            colors=['rgba(66, 135, 245, 1)', 'rgba(255, 100, 100, 1)'], show=False, return_fig=True, run_key=save
        )

        name = f'{save}_PSD_ratios_ICA_R_CWT_MSSA.html'
        if type(modular_eeg_preprocessing_job.get_filename('0')) == WindowsPath :
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('\\')[-2]
        else:
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('/')[-2]
        filepath = figures_path / 'ICA' / 'Rectified_CWT_MSSA' / job_hash_filename
        if not os.path.exists(filepath ):
            os.makedirs(filepath)
        fig_ratio.write_html(filepath  / name)





    else:
        rectified_cwt_mssa_ica  = apply_modified(ica, raw, corrected_signals, to_be_corrected, include_ica=False)
        # rectified_cwt_mssa_ica.set_eeg_reference([reference])
    
    

    return rectified_cwt_mssa_ica.get_data().T

def rectified_V2_CWT_MSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels = None, visualize=True, save=None, save_ratio=True, **p):
    """
    Apply Rectified V2 CWT-MSSA artifact correction to selected ICA components 
    and reconstruct the EEG signal.

    This function extracts ICA sources from an MNE Raw object, applies the 
    `rectified_V2_CWT_MSSA` algorithm to selected components, and reconstructs 
    the corrected signal. Optionally, it visualizes the corrections or computes 
    and saves PSD ratio plots.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG recording.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA decomposition of the EEG data.
    to_be_corrected : list of int
        Indices of ICA components to correct.
    sampling_rate : float
        Sampling rate in Hz.
    channels : list of str, optional
        Channel names for labeling visualizations. If None, names are omitted.
    visualize : bool, default=True
        Whether to display a time-domain comparison plot.
    save : str or None, default=None
        Base filename (without extension) for saving plots. If None, nothing is saved.
    save_ratio : bool, default=True
        If True and `visualize` is False, compute and save PSD ratio plots.
    **p : dict
        Parameters for `rectified_V2_CWT_MSSA`, including:
        
        - iterations
        - wavelets_frequency_spacing
        - wavelets_min_frequency
        - wavelets_max_frequency
        - filter_f_min
        - filter_f_max
        - metric_function
        - metric_factor
        - margins
        - widths
        - ssa_window_seconds
        - first_ssa_trajectories_indices
        - second_ssa_downsampling
        - second_ssa_window_seconds
        - second_ssa_trajectories_indices
        - artifact_margin
        - artifact_plateau
        - artifact_shift
        - wavelet_detection_iteration
        - bridging
        - bias_ssa_downsampling
        - bias_ssa_window_seconds
        - bias_ssa_trajectories_indices
        - cascade
        - rolling_average_duration

    Returns
    -------
    ndarray of shape (n_times, n_channels)
        Corrected EEG data.

    Notes
    -----
    - The PSD ratio plots compare ICA, corrected, and raw power spectra across
      slow (0.05-0.5 Hz) and fast (0.5-200 Hz) bands.

    """

    rereferenced_raw = raw.copy()
    source_signal = ica.get_sources(rereferenced_raw)

    print('Starting Rectified V2 CWT MSSA corrections')
    def get_correction(k):
        single_channel_signal = source_signal.get_data()[k]

        if base_folder == Path('/mnt/data/RelaxCons/'):
            show = False
        else:
            show = True

        correction = rectified_V2_CWT_MSSA( single_channel_signal, iterations = p['iterations'],  sampling_rate=sampling_rate, 
                                        wavelets_frequency_spacing = p['wavelets_frequency_spacing'] , wavelets_min_frequency = p['wavelets_min_frequency'],
                                        wavelets_max_frequency = p['wavelets_max_frequency'], filter_f_min = p['filter_f_min'], 
                                        filter_f_max = p['filter_f_max'], metric_function=p['metric_function'], metric_factor=p['metric_factor'],
                                        margins = p['margins'], widths = p['widths'], ssa_window_seconds = p['ssa_window_seconds'],
                                        first_ssa_trajectories_indices = p['first_ssa_trajectories_indices'], second_ssa_downsampling = p['second_ssa_downsampling'], 
                                        second_ssa_window_seconds = p['second_ssa_window_seconds'], second_ssa_trajectories_indices =p['second_ssa_trajectories_indices'], show=show, verbose=True,
                                         artifact_margin = p['artifact_margin'], artifact_plateau = p['artifact_plateau'],
                                        artifact_shift = p['artifact_shift'], wavelet_detection_iteration= p['wavelet_detection_iteration'],
                                        bridging = p['bridging'],  bias_ssa_downsampling = p['bias_ssa_downsampling'], bias_ssa_window_seconds = p['bias_ssa_window_seconds'],
                                        bias_ssa_trajectories_indices = p['bias_ssa_trajectories_indices'], cascade = p['cascade'], rolling_average_duration= p['rolling_average_duration'],
                                        return_iterative = False,
                                )
        
        return correction


    # num_cores =  max(1, os.cpu_count() - 1)
    # corrected_signals = Parallel(n_jobs=num_cores)(delayed(get_correction)(k) for k in to_be_corrected)
    # corrected_signals = np.column_stack(corrected_signals).T.astype(float)
    corrected_signals = []
    print('Corrections computed')
    for k in to_be_corrected :
        corrected_signals.append(get_correction(k))
    corrected_signals = np.column_stack(corrected_signals).T.astype(float)

    if base_folder == Path('/mnt/data/RelaxCons/'):
        show = False
    else:
        show = True

    if visualize and show:

        simple_ica, rectified_cwt_mssa_ica  = apply_modified(ica, rereferenced_raw, corrected_signals, to_be_corrected, include_ica=True)
        # rectified_cwt_mssa_ica.set_eeg_reference([reference])

        channel_idx = [0, 22, 53, 46]
        t = raw.times[::5]
        df = pd.DataFrame()
        for i in channel_idx :
            df = pd.concat([df,
                pd.DataFrame({'Time' : t, 'Type' : 'Raw', 'Channel': channels[i], 'Amplitude'  : raw.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'ICA', 'Channel':  channels[i], 'Amplitude'  : simple_ica.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'rectified_CWT_MSSA-ICA', 'Channel':  channels[i], 'Amplitude'  : rectified_cwt_mssa_ica.get_data()[i,::5]}),
            ])
        f = px.line(df, x='Time', y='Amplitude', color='Type', facet_row='Channel', height = 300*len(channel_idx), title=f'Sample rectified_CWT_MSSA-ICA results {save}')
        f.show()
        if save is not None:
            save = save + '.html'
            f.write_html(figures_path / 'ICA' / 'Rectified_V2_CWT_MSSA' / save)
    
    
    elif save_ratio :
        simple_ica, rectified_cwt_mssa_ica  = apply_modified(ica, rereferenced_raw, corrected_signals, to_be_corrected, include_ica=True)
        psd_slow = raw.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)
        slow_freqs = psd_slow.freqs
        psd_slow_2 = simple_ica.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)
        psd_slow_3 = rectified_cwt_mssa_ica.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)

        psd_fast = raw.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))
        fast_freqs = psd_fast.freqs
        psd_fast_2 = simple_ica.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))
        psd_fast_3 = rectified_cwt_mssa_ica.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))

        freqs = np.concatenate([slow_freqs, fast_freqs], axis = 0)
        psd = np.concatenate([psd_slow.get_data(), psd_fast.get_data()], axis = 1)
        psd_2 = np.concatenate([psd_slow_2.get_data(), psd_fast_2.get_data()], axis = 1)
        psd_3 = np.concatenate([psd_slow_3.get_data(), psd_fast_3.get_data()], axis = 1)

        fig_ratio = plot_psd_ratios(
            numerators=[psd_2, psd_3], denominators=[psd, psd], freqs=freqs,
            vline_x=13, hline_y=1, labels=['ICA/Detrended', 'R2_CWT_MSSA/Detrended'],
            colors=['rgba(66, 135, 245, 1)', 'rgba(255, 100, 100, 1)'], show=False, return_fig=True, run_key=save
        )

        name = f'{save}_PSD_ratios_ICA_R2_CWT_MSSA.html'
        if type(modular_eeg_preprocessing_job.get_filename('0')) == WindowsPath :
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('\\')[-2]
        else:
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('/')[-2]
        filepath = figures_path / 'ICA' / 'Rectified_V2_CWT_MSSA' / job_hash_filename
        if not os.path.exists(filepath ):
            os.makedirs(filepath)
        fig_ratio.write_html(filepath  / name)





    else:
        rectified_cwt_mssa_ica  = apply_modified(ica, raw, corrected_signals, to_be_corrected, include_ica=False)
        # rectified_cwt_mssa_ica.set_eeg_reference([reference])
    
    

    return rectified_cwt_mssa_ica.get_data().T

def rectified_V3_CWT_MSSA_ICA_correction(raw, ica, labels, sampling_rate, channels = None, visualize=True, save=None, save_ratio=True, **p):
    """
    Apply Rectified V2 CWT-MSSA artifact correction to ICA components selected 
    by artifact labels, and reconstruct the EEG signal.

    This version automatically selects components for correction based on 
    provided labels and applies label-specific parameters.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG recording.
    ica : mne.preprocessing.ICA
        Pre-fitted ICA decomposition of the EEG data.
    labels : list of str
        Artifact type labels for each ICA component. Components with labels in
        {'eye blink', 'other', 'channel noise', 'muscle artifact'} are corrected.
    sampling_rate : float
        Sampling rate in Hz.
    channels : list of str, optional
        Channel names for labeling visualizations. If None, names are omitted.
    visualize : bool, default=True
        Whether to display a time-domain comparison plot.
    save : str or None, default=None
        Base filename (without extension) for saving plots. If None, nothing is saved.
    save_ratio : bool, default=True
        If True and `visualize` is False, compute and save PSD ratio plots.
    **p : dict
        Parameters for `rectified_V2_CWT_MSSA`, including label-dependent
        `metric_factor[label]`.

    Returns
    -------
    ndarray of shape (n_times, n_channels)
        Corrected EEG data.

    Notes
    -----
    - Requires global variables/functions: `base_folder`, `apply_modified`,
      `rectified_V2_CWT_MSSA`, `plot_psd_ratios`, `figures_path`, and
      `modular_eeg_preprocessing_job`.
    - This function differs from `rectified_V2_CWT_MSSA_ICA_correction` in that 
      it selects components based on artifact labels and can vary parameters 
      accordingly.

    """

    rereferenced_raw = raw.copy()
    source_signal = ica.get_sources(rereferenced_raw)

    print('Starting Rectified V3 CWT MSSA corrections')
    def get_correction(k, label):
        single_channel_signal = source_signal.get_data()[k]

        if base_folder == Path('/mnt/data/RelaxCons/'):
            show = False
        else:
            show = True

        correction = rectified_V2_CWT_MSSA( single_channel_signal, iterations = p['iterations'],  sampling_rate=sampling_rate, unbiasing = p['unbiasing'],
                                        wavelets_frequency_spacing = p['wavelets_frequency_spacing'] , wavelets_min_frequency = p['wavelets_min_frequency'],
                                        wavelets_max_frequency = p['wavelets_max_frequency'], filter_f_min = p['filter_f_min'], 
                                        filter_f_max = p['filter_f_max'], metric_function=p['metric_function'], metric_factor=p['metric_factor'][label],
                                        margins = p['margins'], widths = p['widths'], ssa_window_seconds = p['ssa_window_seconds'],
                                        first_ssa_trajectories_indices = p['first_ssa_trajectories_indices'], second_ssa_downsampling = p['second_ssa_downsampling'], 
                                        second_ssa_window_seconds = p['second_ssa_window_seconds'], second_ssa_trajectories_indices =p['second_ssa_trajectories_indices'], show=False, verbose=True,
                                         artifact_margin = p['artifact_margin'], artifact_plateau = p['artifact_plateau'],
                                        artifact_shift = p['artifact_shift'], wavelet_detection_iteration= p['wavelet_detection_iteration'],
                                        bridging = p['bridging'],  bias_ssa_downsampling = p['bias_ssa_downsampling'], bias_ssa_window_seconds = p['bias_ssa_window_seconds'],
                                        bias_ssa_trajectories_indices = p['bias_ssa_trajectories_indices'], cascade = p['cascade'], rolling_average_duration= p['rolling_average_duration'],
                                        return_iterative = False,
                                )
        
        return correction


    corrected_signals = []
    to_be_corrected = []
    print('Corrections computed')
    for k, label in enumerate(labels) :
        if label in ['eye blink', 'other', 'channel noise', 'muscle artifact']:
            corrected_signals.append(get_correction(k, label))
            to_be_corrected.append(k)
    corrected_signals = np.column_stack(corrected_signals).T.astype(float)

    if base_folder == Path('/mnt/data/RelaxCons/'):
        show = False
    else:
        show = True

    if visualize and show:

        simple_ica, rectified_cwt_mssa_ica  = apply_modified(ica, rereferenced_raw, corrected_signals, to_be_corrected, include_ica=True)
        # rectified_cwt_mssa_ica.set_eeg_reference([reference])

        channel_idx = [0, 22, 53, 46]
        t = raw.times[::5]
        df = pd.DataFrame()
        for i in channel_idx :
            df = pd.concat([df,
                pd.DataFrame({'Time' : t, 'Type' : 'Raw', 'Channel': channels[i], 'Amplitude'  : raw.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'ICA', 'Channel':  channels[i], 'Amplitude'  : simple_ica.get_data()[i,::5]}),
                pd.DataFrame({'Time' : t, 'Type' : 'rectified_CWT_MSSA-ICA', 'Channel':  channels[i], 'Amplitude'  : rectified_cwt_mssa_ica.get_data()[i,::5]}),
            ])
        f = px.line(df, x='Time', y='Amplitude', color='Type', facet_row='Channel', height = 300*len(channel_idx), title=f'Sample rectified_CWT_MSSA-ICA results {save}')
        f.show()
        if save is not None:
            save = save + '.html'
            f.write_html(figures_path / 'ICA' / 'Rectified_V2_CWT_MSSA' / save)
    
    
    elif save_ratio :
        simple_ica, rectified_cwt_mssa_ica  = apply_modified(ica, rereferenced_raw, corrected_signals, to_be_corrected, include_ica=True)
        psd_slow = raw.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)
        slow_freqs = psd_slow.freqs
        psd_slow_2 = simple_ica.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)
        psd_slow_3 = rectified_cwt_mssa_ica.compute_psd(method='welch', fmin=0.05, fmax=0.5, n_fft = 110000, n_per_seg = 36000, n_overlap = 18000)

        psd_fast = raw.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))
        fast_freqs = psd_fast.freqs
        psd_fast_2 = simple_ica.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))
        psd_fast_3 = rectified_cwt_mssa_ica.compute_psd(method='welch', fmin=0.5, fmax=200, n_fft = 5*500, n_per_seg = 5*500, n_overlap = int(2.5*500))

        freqs = np.concatenate([slow_freqs, fast_freqs], axis = 0)
        psd = np.concatenate([psd_slow.get_data(), psd_fast.get_data()], axis = 1)
        psd_2 = np.concatenate([psd_slow_2.get_data(), psd_fast_2.get_data()], axis = 1)
        psd_3 = np.concatenate([psd_slow_3.get_data(), psd_fast_3.get_data()], axis = 1)

        fig_ratio = plot_psd_ratios(
            numerators=[psd_2, psd_3], denominators=[psd, psd], freqs=freqs,
            vline_x=13, hline_y=1, labels=['ICA/Detrended', 'R2_CWT_MSSA/Detrended'],
            colors=['rgba(66, 135, 245, 1)', 'rgba(255, 100, 100, 1)'], show=False, return_fig=True, run_key=save
        )

        name = f'{save}_PSD_ratios_ICA_R2_CWT_MSSA.html'
        if type(modular_eeg_preprocessing_job.get_filename('0')) == WindowsPath :
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('\\')[-2]
        else:
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('/')[-2]
        filepath = figures_path / 'ICA' / 'Rectified_V2_CWT_MSSA' / job_hash_filename
        if not os.path.exists(filepath ):
            os.makedirs(filepath)
        fig_ratio.write_html(filepath  / name)





    else:
        rectified_cwt_mssa_ica  = apply_modified(ica, raw, corrected_signals, to_be_corrected, include_ica=False)
        # rectified_cwt_mssa_ica.set_eeg_reference([reference])
    
    

    return rectified_cwt_mssa_ica.get_data().T



# ECG Artefact removal
###################

def remove_ecg_artifacts(data, ica, sampling_rate, run_key, channels= None, save=None, **p):
    """
    Remove ECG artifacts from EEG data using ICA-based methods.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        EEG data array.
    ica : mne.preprocessing.ICA
        Precomputed ICA object fitted to the EEG data.
    sampling_rate : float
        Sampling frequency of the EEG data in Hz.
    run_key : str
        Identifier for the current EEG run (used in logging and saving results).
    channels : list of str, optional
        List of channel names. If None, default names are inferred.
    save : str or pathlib.Path, optional
        Path to save figures or intermediate results, if applicable.
    **p : dict
        Additional parameters. Must contain:
        - ``name`` (str): artifact removal method name ('ICA' or 'CWT_MSSA_ICA').
        Other parameters are passed to the correction functions.

    Returns
    -------
    ecg_corrected : ndarray
        EEG data after ECG artifact removal, shape (n_samples, n_channels).

    Notes
    -----
    - The function uses `ica_component_selection` to identify ICA components
      related to ECG artifacts.
    - Depending on ``p['name']``, it applies either `remove_from_ICA` or
      `CWT_MSSA_ICA_correction`.
    """

    method = p['name']
    print(f'___________________________________Removing ECG artifacts using {method}___________________________________')

    raw = numpy_to_mne(data.T, channel_names=channels, sampling_rate=sampling_rate)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    to_be_corrected = ica_component_selection(raw, ica, run_key, **p)

    if len(to_be_corrected) > 0 :
        if method == 'ICA':
            ecg_corrected = remove_from_ICA(raw, ica, to_be_corrected, channels, **p)
        
        elif method == 'CWT_MSSA_ICA':
            ecg_corrected = CWT_MSSA_ICA_correction(raw, ica, to_be_corrected, sampling_rate, channels, save=save, **p)   

    else:
        ecg_corrected = data

    return ecg_corrected 



# EMG Artefact removal
###################

def remove_emg_artifacts(data, sampling_rate, channels = None, **p):
    """
    Remove EMG artifacts from EEG data.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        EEG data array.
    sampling_rate : float
        Sampling frequency in Hz.
    channels : list of str, optional
        Channel names.
    **p : dict
        Additional parameters. Must contain:
        - ``name`` (str): method name ('EMG_rms' supported).
        Other parameters are passed to the EMG correction function.

    Returns
    -------
    emg_corrected : ndarray
        EEG data after EMG artifact removal.
    mask : ndarray or None
        Boolean mask of detected artifact intervals. None if method not applied.

    Notes
    -----
    Currently only supports the ``EMG_rms`` method.
    """

    method = p['name']
    print(f'___________________________________Removing EMG artifacts using {method}___________________________________')

    if method == 'EMG_rms':
        emg_corrected, mask = emg_rms(data, sampling_rate, channels =channels , **p)

    else:
        emg_corrected = data
        mask = None

    return emg_corrected, mask

def emg_rms(data, sampling_rate, channels = None, verify_plot=0, save = 0, **p):
    """
    Correct EMG artifacts using RMS thresholding on each channel.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        EEG data array.
    sampling_rate : float
        Sampling frequency in Hz.
    channels : list of str, optional
        Channel names.
    verify_plot : bool, default=0
        If True, generate verification plots for each corrected channel.
    save : bool, default=0
        If True, save generated verification plots.
    **p : dict
        Additional parameters for `EMG_noise_patching`:
        - ``margin`` (int)
        - ``sliding_rms_window`` (float)
        - ``deviation_number`` (float)
        - ``fmin`` (float)

    Returns
    -------
    emg_corrected : ndarray
        EEG data after EMG artifact correction.
    masks : ndarray
        Boolean masks indicating corrected intervals for each channel.

    Notes
    -----
    Uses parallel processing to speed up per-channel corrections.
    """


    def emg_correction_one_channel(k):
        print(f'Correcting channel {k} : {channels[k]}')

        corrected, mask = EMG_noise_patching(data[:,k], sampling_rate=sampling_rate, margin=p['margin'],
                                        sliding_rms_window = p['sliding_rms_window'], n_deviation = p['deviation_number'],
                                        fmin = p['fmin'], seed = 42)
        
        if verify_plot:
            t = np.arange(data.shape[0])*1/sampling_rate

            df_correction = pd.concat([
                pd.DataFrame({'Time' :t[::4], 'Type' : 'Raw', 'Voltage' : data[::4,k] }),
                pd.DataFrame({'Time' :t[::4], 'Type' : 'Corrected', 'Voltage' : corrected[::4] }),
            ])
            name = f'{run_key}_{channels[k]}.png'

            fig = px.line(df_correction, x='Time', y='Voltage', color='Type', height = 800, width = 1600,
                         title= f'Corrected signal on artifact intervals {run_key}, {channels[k]}')
            if save :
                fig.write_image(figures_path / 'EMG' / 'EMG_rms_correction' / name)
        
        return corrected, mask
    
    with set_num_threads(1):
        num_cores =  max(1, os.cpu_count() - 1)
        results = Parallel(n_jobs=num_cores)(delayed(emg_correction_one_channel)(k) for k in range(data.shape[1]))
        emg_corrected, masks = zip(*results)

    return np.column_stack(emg_corrected).astype(float), np.column_stack(masks).astype(float)


# Big artifact removal
###################

def remove_big_artifacts(data, sampling_rate, **p):
    """
    Remove large-scale artifacts from EEG data.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        EEG data array.
    sampling_rate : float
        Sampling frequency in Hz.
    **p : dict
        Additional parameters. Must contain:
        - ``name`` (str): method name ('big_ssa' supported).

    Returns
    -------
    big_corrected : ndarray
        EEG data after large artifact removal.

    Notes
    -----
    Currently only supports the ``big_ssa`` method.
    """

    method = p['name']
    print(f'___________________________________Removing big artifacts using {method}___________________________________')

    if method == 'big_ssa':
        big_corrected = big_correction_ssa(data, sampling_rate, **p)

    else:
        big_corrected = data

    return big_corrected

def big_correction_ssa(data, sampling_rate, verify_plot=1, **p):
    """
    Apply Singular Spectrum Analysis (SSA) to remove large artifacts.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        EEG data array.
    sampling_rate : float
        Sampling frequency in Hz.
    verify_plot : bool, default=1
        If True, enable diagnostic plotting (currently commented out).
    **p : dict
        Additional parameters for `big_artefact_correction_ssa`:
        - ``mad_factor`` (float): threshold multiplier for artifact detection.

    Returns
    -------
    big_corrected : ndarray
        EEG data after artifact correction.

    Notes
    -----
    Uses parallel processing to handle multiple channels efficiently.
    """

    print("Parallel computation")

    def big_correction_ssa_one_channel(k):
        print(f'Correcting channel {k} ')

        corrected = big_artefact_correction_ssa(data[:,k], mad_factor=p['mad_factor'], sampling_rate=sampling_rate, )
        
        # if verify_plot:
        #     t = np.arange(data.shape[0])*1/sampling_rate

        #     df_correction = pd.concat([
        #         pd.DataFrame({'Time' :t[::4], 'Type' : 'Raw', 'Voltage' : data[::4,k] }),
        #         pd.DataFrame({'Time' :t[::4], 'Type' : 'Corrected', 'Voltage' : corrected[::4] }),
        #     ])

        #     fig = px.line(df_correction, x='Time', y='Voltage', color='Type', height = 800, width = 1600,
        #                  title= f'Corrected signal on artifact intervals {k}')
            
        #     fig.show()
        
        return corrected
    
    
    with set_num_threads(1):
        num_cores =  max(1, os.cpu_count() - 1)
        big_corrected = Parallel(n_jobs=num_cores)(delayed(big_correction_ssa_one_channel)(k) for k in range(data.shape[1]))

    return np.column_stack(big_corrected).astype(float)

# Rereferencing
###################

def rereference_eeg(data, channels, sampling_rate, **p):
    """
    Rereference EEG data using various reference schemes.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels) or (n_channels, n_samples)
        EEG data array.
    channels : list of str
        Channel names.
    sampling_rate : float
        Sampling frequency in Hz.
    **p : dict
        Additional parameters. Must contain:
        - ``method`` (str): reference method, one of:
          {'physical_channel', 'average', 'CSD', 'REST'}
        - ``reference_channels`` (list of str), required if method='physical_channel'.

    Returns
    -------
    rereferenced_data : ndarray of shape (n_samples, n_channels)
        Rereferenced EEG data.

    Notes
    -----
    - Uses MNE-Python's rereferencing utilities.
    - 'CSD' and 'REST' methods require electrode montage setup.
    - 'REST' method requires forward model computation.
    """

    method = p['method']
    print(f'___________________________________Rereferencing using {method}___________________________________')

    if data.shape[0] == len(channels) :
        rerefenced = numpy_to_mne(data, channels, sampling_rate)
    else:
        rerefenced = numpy_to_mne(data.T, channels, sampling_rate)

    if method == 'physical_channel':
        rerefenced, _ = mne.set_eeg_reference(rerefenced, ref_channels=p['reference_channels'], copy=True)
    
    elif method == 'average':
        rerefenced, _ = mne.set_eeg_reference(rerefenced, ref_channels='average', copy=True)

    elif method == 'CSD':
        montage = mne.channels.make_standard_montage('standard_1020')
        rerefenced.set_montage(montage)
        rerefenced = mne.preprocessing.compute_current_source_density(rerefenced, sphere = 'auto')
    
    elif method == 'REST':
        montage = mne.channels.make_standard_montage('standard_1020')
        rerefenced.set_montage(montage)
        subjects_dir = mne.datasets.sample.data_path() / 'subjects'  # Path to MNE's sample data subjects directory
        src = subjects_dir / 'sample' / "bem" / "sample-fsaverage-ico-5-src.fif" #mne.setup_source_space(subject='sample', spacing='oct6', subjects_dir=subjects_dir)
        bem = subjects_dir / 'sample' / 'bem' / 'sample-5120-5120-5120-bem-sol.fif'
        fwd = mne.make_forward_solution(rerefenced.info, trans="fsaverage", src=src, bem=bem, eeg=True, verbose=False)
        rerefenced,_ = mne.set_eeg_reference(rerefenced, ref_channels='REST', copy=True, forward = fwd, verbose=False)

    return rerefenced.get_data().T

############################################################################################
#                               JOBTOOLS implementation
############################################################################################





def compute_modular_eeg_preprocessing(run_key, testing=False, **p):
    """
    Apply a modular sequence of EEG preprocessing steps and convert the result 
    into an ``xarray.Dataset``.

    This function loads raw EEG data for a given run, applies a series of 
    preprocessing steps specified in ``p`` in the defined order, collects 
    intermediate results and metadata, and returns the processed data as an 
    ``xarray.Dataset`` suitable for further analysis.

    Parameters
    ----------
    run_key : str
        Identifier for the EEG recording (e.g., ``"sub-01_task-rest"``).
    testing : bool, default=False
        If True, only process the first 20 seconds (at 500 Hz) of data.
    **p : dict
        Dictionary mapping preprocessing step names to their parameters.
        Keys should correspond to one of the supported preprocessing steps:
        
        - ``rescaling``
        - ``trimming``
        - ``detrending``
        - ``centering``
        - ``notch_filtering``
        - ``filtering``
        - ``compute_ICA``
        - ``EOG_artifact_removal``
        - ``ECG_artifact_removal``
        - ``EMG_artifact_removal``
        - ``big_artifact_removal``
        - ``rereferencing``

        Values should be dictionaries containing method-specific parameters.
        Any required parameters for each method are documented in the 
        corresponding preprocessing function.
        If a step were to be applied several times, each application 
        should be precised by adding '__' followed by the iteration number
        to each of the key related to this step.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing:
        
        - ``eeg`` : (``Time``, ``Channel``) preprocessed EEG signal
        - Any additional variables (e.g., artifact masks) added by steps
        - Coordinate variables: ``time`` (seconds), ``channel`` (names)
        - Global attributes: run metadata and parameters used for each step

    Notes
    -----
    - The preprocessing pipeline is fully modular: the order and combination 
      of steps are determined by the keys in ``p``.
    - Each preprocessing function is responsible for its own logic and 
      validation.
    - If ``channels`` is specified in ``p``, only those channels are processed.
    - Attributes record the parameters used for reproducibility.
    """
    

    if 'Order' in p:
        p.pop('Order')

    # Get signal
    # try :
    sigs_eeg, sr, eeg_channels = read_eeg(run_key)
    sigs_eeg = sigs_eeg.astype(float)

    # Initialize dataset variables

    data_vars = dict()
    attributes = dict(run_key = run_key,
            # group = get_odor_group(run_key),
            sampling_rate = sr, 
            eeg_unit = 'µV', 
            time_unit = 's',
            original_reference='Fz')

    # Function correspondance
    functions_dict = {
        'correcting_channel_names' : {'corresponding_function' : correct_channel_names, 
                    'arguments' : ['run_key', 'channels'], 
                    'parameter' : {'variable_number': 1, 'name' : 'channels'}
                    },
        'subsetting' : {'corresponding_function' : subset, 
                    'arguments' : ['channels'], 
                    'parameter' : {'variable_number': 1, 'name' : 'channels'}
                    },
        'rescaling' : {'corresponding_function' : scale_eeg, 
                    'arguments' : [], 
                    'report' : ['rescaling_factor'],
                    },
        'trimming' : {'corresponding_function' : trim_eeg, 
                    'arguments' : ['sampling_rate', 'run_key'], 
                    'report' : ['method', 'session_adapted'],
                    },
        'detrending' : {'corresponding_function' : detrend_eeg, 
                    'arguments' : ['sampling_rate', 'run_key', 'channels'], 
                    'report' : 'name',
                    },
        'centering' : {'corresponding_function' : center_eeg, 
                    'arguments' : [], 
                    'report' : 'method',
                    },      
        'notch_filtering' : {'corresponding_function' : notch_filter_eeg, 
                    'arguments' : ['sampling_rate'], 
                    'report' : 'method',
                    },  
        'filtering' : {'corresponding_function' : filter_eeg, 
                    'arguments' : ['sampling_rate'], 
                    'report' : ['method', "low_cutoff", "high_cutoff"],
                    },     
        'compute_ICA' : {'corresponding_function' : compute_ICA, 
                    'arguments' : ['channels', 'sampling_rate'], 
                    'report' : 'name',
                    'parameter' : {'variable_number': 1, 'name' : 'ICA'},
                    }, 
        'EOG_artifact_removal' : {'corresponding_function' : remove_eog_artifacts, 
                    'arguments' : ['ICA', 'sampling_rate', 'channels', 'run_key'], 
                    'report' : 'name',
                    }, 
        'ECG_artifact_removal' : {'corresponding_function' : remove_ecg_artifacts, 
                    'arguments' : ['ICA', 'sampling_rate', 'run_key', 'channels'], #, 'run_key' ], 
                    'report' : 'name',
                    }, 
        'EMG_artifact_removal' : {'corresponding_function' : remove_emg_artifacts, 
                    'arguments' : ['sampling_rate', 'channels'], 
                    'report' : 'name',
                    'variable' : {'coordinates' : ["Time", "Channel"], 'variable_number': 1, 'name' : 'EMG_mask'}
                    }, 
        'big_artifact_removal' : {'corresponding_function' : remove_big_artifacts, 
                    'arguments' : ['sampling_rate'], 
                    'report' : 'name',
                    },
        'rereferencing' : {'corresponding_function' : rereference_eeg, 
                    'arguments' : ['channels', 'sampling_rate'], 
                    'report' : 'method',
                    }, 
    }

    parameters_dict = {
        'sampling_rate' : sr,
        'run_key' : run_key,
        'channels' : eeg_channels
    }


    for preprocess_name, preprocess_parameters in p.items():

        correspondence_dict = functions_dict[preprocess_name.split('__')[0]]

        preprocess_function = correspondence_dict['corresponding_function']
        arguments = [parameters_dict[key] for key in correspondence_dict['arguments']]

        preprocess_step_result = preprocess_function(sigs_eeg, *arguments, **preprocess_parameters)
        if type(preprocess_step_result) == tuple:
            sigs_eeg = preprocess_step_result[0]
        else:
            sigs_eeg = preprocess_step_result

        if 'report' in correspondence_dict :
            report = correspondence_dict['report']
            if type(report) == str:
                attributes[preprocess_name] = preprocess_parameters[report]
            else:
                attributes[preprocess_name] = str()
                for report_detail in report:
                    attributes[preprocess_name] += f'{preprocess_parameters[report_detail]}_'
                attributes[preprocess_name] = attributes[preprocess_name][:-1]

        if 'variable' in correspondence_dict :
            coordinates = correspondence_dict['variable']['coordinates']
            variable_number = correspondence_dict['variable']['variable_number']
            name = correspondence_dict['variable']['name']
            data_vars[name] = (coordinates, preprocess_step_result[variable_number])
        
        if 'parameter' in correspondence_dict :
            variable_number = correspondence_dict['parameter']['variable_number']
            name = correspondence_dict['parameter']['name']
            new_parameter = preprocess_step_result[variable_number]
            parameters_dict[name] = new_parameter
        
        # f = px.line(sigs_eeg[::20,0], title=preprocess_name)
        # f.show()


    print('_________________All steps completed_____________________')

    # Convert to dataset
    data_vars['eeg'] = (["Time", "Channel"], sigs_eeg)


    ds = xr.Dataset(
        data_vars= data_vars,
        coords=dict(
            time=("Time", np.arange(sigs_eeg.shape[0])*1/sr),
            channel=("Channel",parameters_dict['channels']),
        ),
        attrs=attributes,
    )
    
    print('_________________Dataset created_____________________')

    return ds
    
    # except Exception as e: 
    #         print(e)
    #         raise e

def plot_all_signals(ds, height = None, return_fig = False):
    """
    Plot all EEG signals from an xarray dataset as an interactive Plotly animation.

    This function converts the EEG data in the dataset to MNE format, downsamples it,
    and plots each channel's time series as an animation frame using Plotly Express.

    Parameters
    ----------
    ds : xarray.Dataset
        EEG dataset containing:
        - ``eeg`` : (``Time``, ``Channel``) EEG data in microvolts
        - ``channel`` : channel names
        - ``time`` : time points in seconds
        - ``run_key`` : dataset attribute specifying run identifier
    height : int, optional
        Height of the plot in pixels. If None, default Plotly height is used.
    return_fig : bool, default=False
        If True, return the generated Plotly figure object instead of just showing it.

    Returns
    -------
    fig : plotly.graph_objects.Figure, optional
        Plotly figure object if ``return_fig`` is True, otherwise None.

    Notes
    -----
    - The plot uses a fixed downsampling factor of 10 to speed up rendering.
    - Amplitude range is set to ± twice the maximum absolute signal value.
    - Each channel is shown as a separate animation frame.
    """

    df_init = pd.DataFrame()
    downsample = 10
    raw = dataset_to_mne(ds)
    r = 2*np.max(np.abs(raw.get_data()))
    for i, channel in enumerate(ds.channel.values):
        df_init = pd.concat([df_init,
                            pd.DataFrame({'Time' : raw.times[::downsample], 'Type' : 'EEG', 'Channel':channel, 'Signal' : raw.get_data()[i,::downsample]}),
                            ])
    if height is not None :
        f = px.line(df_init, x = 'Time', y = 'Signal', color='Type', animation_frame='Channel', title= ds.run_key, height= height, range_y=[-r,+r])
    else:
        f = px.line(df_init, x = 'Time', y = 'Signal', color='Type', animation_frame='Channel', title= ds.run_key, range_y=[-r,+r])
    f.show()

    if return_fig:
        return f

def modular_eeg_preprocessing_plot(run_key, show=False, **p):
    """
    Plot minimal vs. fully preprocessed EEG signals for a given run.

    This function:
    1. Runs a reduced preprocessing pipeline (only rescaling, trimming, and optional detrending)
       to create a "minimal" dataset.
    2. Retrieves the fully preprocessed dataset from the preprocessing job store.
    3. Generates an interactive Plotly animation comparing both signals channel by channel.
    4. Saves the HTML file to a structured figures directory.

    Parameters
    ----------
    run_key : str
        Identifier for the EEG recording.
    show : bool, default=False
        If True, display the interactive plot after saving.
    **p : dict
        Parameters for the preprocessing pipeline. Keys not in
        ``['rescaling', 'trimming', 'Order']`` are excluded from the minimal pipeline.

    Returns
    -------
    None
        Saves an HTML file to ``figures_path / 'Preprocess_eeg' / job_hash_filename``.

    Notes
    -----
    - If detrending is requested, a high-pass filter at 0.05 Hz is applied to the minimal dataset.
    - The plot is an animation with each channel as a frame and two traces:
      'Minimal' and 'Preprocessed'.
    - HTML output is named ``{run_key}_preprocess.html`` and stored in a folder
      corresponding to the preprocessing job hash.
    - The preprocessing job filenames are used to determine the storage directory
      (compatible with both Windows and POSIX paths).
    """
    try :
        p_minimal = p.copy()
        for key in p:
            if key.split('__')[0] not in ['rescaling', 'trimming', 'Order']:
                p_minimal.pop(key)

        ds_minimal = compute_modular_eeg_preprocessing(run_key, **p_minimal)
        
        if any([key.split('__')[0]=='detrending' for key in p]):
            ds_minimal['eeg'] = (['time', 'channel'], mne.filter.filter_data(ds_minimal['eeg'].values.T, 500, 0.05, None).T)

        # Get preprocessed eeg
        ds_preprocess = modular_eeg_preprocessing_job.get(run_key)

        df_init = pd.DataFrame()
        downsample = 10
        r = 2*np.max(np.abs(ds_minimal['eeg'].values))

        for i, channel in enumerate(ds_minimal.channel.values):
            df_init = pd.concat([df_init,
                                pd.DataFrame({'Time' : ds_minimal.time.values[::downsample], 'Type' : 'Minimal', 'Channel':channel,
                                            'Signal' : ds_minimal.eeg.values[::downsample, i]}),
                                pd.DataFrame({'Time' : ds_preprocess.time.values[::downsample], 'Type' : 'Preprocessed', 'Channel':channel,
                                            'Signal' : ds_preprocess.eeg.values[::downsample, i]}),
                                ])
        f = px.line(df_init, x = 'Time', y = 'Signal', color='Type', animation_frame='Channel', title= run_key, range_y=[-r,+r])

        name = f'{run_key}_preprocess.html'
        if type(modular_eeg_preprocessing_job.get_filename('0')) == WindowsPath :
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('\\')[-2]
        else:
            job_hash_filename = str(modular_eeg_preprocessing_job.get_filename('0')).split('/')[-2]
        filepath = figures_path / 'Preprocess_eeg' / job_hash_filename
        if not os.path.exists(filepath ):
            os.makedirs(filepath)

        f.write_html(filepath  / name)

        if show:
            f.show()
        
        return 

    except Exception as e:
        print(e)
        raise e
    

def test_modular_eeg_preprocessing():
    run_key = "sub-01_Repos"
    compute_modular_eeg_preprocessing(run_key, testing=True, **modular_eeg_preprocessing_params)



def compute_modular_eeg_proprocessing_all():
    
    run_keys = []
    for base in subject_keys:
        for sess in ['Repos', 'Distraction']:
            run_key = f"{base}_{sess}"
            run_keys.append(run_key)

    run_keys = tuple(run_keys)

    jobtools.compute_job_list(modular_eeg_preprocessing_job, run_keys, force_recompute=False, engine = 'loop')



modular_eeg_preprocessing_job = jobtools.Job(precomputedir, 'modular_eeg_preprocessing', modular_eeg_preprocessing_params, compute_modular_eeg_preprocessing)
jobtools.register_job(modular_eeg_preprocessing_job)

modular_eeg_preprocessing_plot_job = jobtools.Job(precomputedir, 'modular_eeg_preprocessing_plot', modular_eeg_preprocessing_params, modular_eeg_preprocessing_plot)
jobtools.register_job(modular_eeg_preprocessing_plot_job)


if __name__=='__main__':


    compute_modular_eeg_proprocessing_all()
    # test_modular_eeg_preprocessing()

