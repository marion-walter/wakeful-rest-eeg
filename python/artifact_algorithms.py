# Necessary imports
from configuration import *

# Base imports
import numpy as np
import pandas as pd
import plotly.express as px

# Specific imports used several times
from scipy.signal import resample
import scipy.signal as signal

# Intra package imports
from utils import *
from SSA import *


###############################################################################################################
#
#                                               SSA - Blinks
#
###############################################################################################################




def energy(multivariate_matrix):
    """
    Compute signal energy along first dimension of a multivariate signal.

    Parameters
    ----------
    multivariate_matrix : ndarray, shape (time, channels)
        Input multivariate time series data.

    Returns
    -------
    ndarray, shape (channels,)
        Energy (sum of squared values) computed along the time axis for each channel.

    Notes
    -----
    Energy is computed as the sum of squares of the signal amplitude over time.
    """
    return np.sum(multivariate_matrix**2, axis = 0)

def hjorth_mobility(multivariate_matrix):
    """
    Compute Hjorth mobility parameter along first dimension of a multivariate signal.

    Hjorth mobility is defined as the square root of the variance of the first derivative
    of the signal divided by the variance of the signal itself.

    Parameters
    ----------
    multivariate_matrix : ndarray, shape (time, channels)
        Input multivariate time series data.

    Returns
    -------
    ndarray, shape (channels,)
        Hjorth mobility values for each channel.
    """
    var_dt = np.var(np.gradient(multivariate_matrix, axis=0), axis=0)
    var = np.var(multivariate_matrix, axis=0)
    return np.sqrt(var_dt/var)

def kurtosis(multivariate_matrix):
    """
    Compute kurtosis along first dimension of a multivariate signal.

    Parameters
    ----------
    multivariate_matrix : ndarray, shape (time, channels)
        Input multivariate time series data.

    Returns
    -------
    ndarray, shape (channels,)
        Kurtosis values for each channel.

    Notes
    -----
    Uses scipy.stats.kurtosis with default parameters (Fisher's definition, bias corrected).
    """
    from scipy.stats import kurtosis
    return kurtosis(multivariate_matrix, axis = 0)

def peak_to_peak(multivariate_matrix):
    """
    Compute peak-to-peak amplitude along first dimension of a multivariate signal.

    Parameters
    ----------
    multivariate_matrix : ndarray, shape (time, channels)
        Input multivariate time series data.

    Returns
    -------
    ndarray, shape (channels,)
        Peak-to-peak amplitude (max - min) for each channel.

    Notes
    -----
    Peak-to-peak amplitude indicates the total amplitude range of the signal.
    """
    return np.max(multivariate_matrix, axis = 0) - np.min(multivariate_matrix, axis = 0)

def time_features(multivariate_matrix, standardize = False):
    """
    Compute time-domain features for each channel of a multivariate signal.

    Parameters
    ----------
    multivariate_matrix : ndarray, shape (time, channels)
        Input multivariate time series data.
    standardize : bool or str, optional
        Method to standardize features across channels. Options are:
        - False (default): no standardization
        - 'rz_score': apply robust z-score normalization
        - 'min_max': apply min-max scaling to [0, 1]

    Returns
    -------
    feature_matrix : ndarray, shape (4, channels)
        Computed features per channel. Each row corresponds to a feature:
        1. Energy
        2. Hjorth mobility
        3. Kurtosis
        4. Peak-to-peak amplitude

    Notes
    -----
    The features are computed along the first axis (time dimension).
    """

    feature_matrix = np.empty((4, multivariate_matrix.shape[1]))

    for i, f in enumerate([energy, hjorth_mobility, kurtosis, peak_to_peak]):
        feature_matrix[i,:] = f(multivariate_matrix)
        if standardize == 'rz_score' :
            feature_matrix[i,:] = rz_score(feature_matrix[i,:])
        elif standardize == 'min_max':
            feature_matrix[i,:] = min_max_scaling(feature_matrix[i,:])
    
    return feature_matrix

def k_multivariates(multivariate_matrix, kmeans_labels, kmeans_cluster_number ):
    """
    Separate a multivariate time series into cluster-specific components based on k-means labels.

    Parameters
    ----------
    multivariate_matrix : ndarray, shape (time, channels)
        Input multivariate time series data.
    kmeans_labels : ndarray, shape (time,)
        Cluster labels for each time point, typically from k-means clustering.
    kmeans_cluster_number : int
        Number of distinct clusters.

    Returns
    -------
    k_multivariate_matrix : ndarray, shape (time, channels, kmeans_cluster_number)
        Time series data separated by cluster. For each cluster, values outside
        that cluster's time points are zeroed out.
    """

    k_multivariate_matrix = np.zeros((multivariate_matrix.shape[0], multivariate_matrix.shape[1], kmeans_cluster_number ))

    for i in range(kmeans_cluster_number):
        k_multivariate_matrix[:,:,i] = multivariate_matrix*(kmeans_labels == i)
    
    return k_multivariate_matrix



def fractal_dimension(univariate_matrix, axis = 0):
    """
    Estimate the fractal dimension of a univariate signal or multiple signals.

    Parameters
    ----------
    univariate_matrix : ndarray
        Input signal(s). Shape typically (time, signals).
    axis : int, optional
        Axis along which to compute the fractal dimension (default is 0, time axis).

    Returns
    -------
    H : ndarray
        Estimated fractal dimension for each signal.
    Hm : ndarray
        Mean fractal dimension computed over several segments of the signal for stability.

    Notes
    -----
    The signal is normalized before computing the length of its curve.
    Fractal dimension is calculated using the length of the normalized curve
    and a multi-segment estimation to reduce noise.
    """

    n = univariate_matrix.shape[axis]

    x = np.arange(n) / (n-1)

    maxi = np.max(univariate_matrix, axis = 0)
    mini = np.min(univariate_matrix, axis = 0)
    y = (univariate_matrix-mini)/(maxi-mini)

    L = np.sqrt((y[1:]-y[:-1])**2 + (x[1:, np.newaxis]-x[:-1,np.newaxis])**2)
    L_total = np.sum(L, axis = 0)

    H = 1 + np.log(L_total)/np.log(2*(n-1))
    # Calculate fractal dimension estimates using overlapping 10% windows
    # with 50% overlap to gain precision in average fractal dimension.
    Hm = np.array([1 + np.log(np.sum(L[int(L.shape[axis]*i/10): int(L.shape[axis]*(i+1)/10)], axis = 0))/np.log(2*(n-1))  for i in np.linspace(0,9,19)])

    return H, np.mean(Hm, axis = 0)

def blink_draft(univariate_matrix, fractal_dimensions, threshold):
    """
    Identify potential blink artifacts by summing signal channels with fractal dimension below a threshold.

    Parameters
    ----------
    univariate_matrix : ndarray, shape (time, channels)
        Multivariate signal data.
    fractal_dimensions : ndarray, shape (channels,)
        Fractal dimension value for each channel.
    threshold : float
        Threshold below which a channel is considered part of a blink artifact.

    Returns
    -------
    draft : ndarray, shape (time,)
        Summed signal over channels suspected of blink artifacts.
    """
    return np.sum(univariate_matrix[:,fractal_dimensions<=threshold], axis = 1)

def artifact_template(draft):
    """
    Generate a binary artifact template from a draft signal by thresholding non-zero values.

    Parameters
    ----------
    draft : ndarray, shape (time,)
        Draft blink signal (summed from channels).

    Returns
    -------
    template : ndarray, shape (time,)
        Binary artifact template, 1 where artifact detected, else 0.
    """
    return (np.abs(draft) > 0).astype(int)

def craft_blink(raw, artifact_template, window_length, eigen_threshold):
    """
    Refine blink artifact signal by embedding, SSA decomposition, and subspace projection.

    Parameters
    ----------
    raw : ndarray, shape (time,)
        Original single-channel signal.
    artifact_template : ndarray, shape (time,)
        Binary artifact mask indicating blink occurrences.
    window_length : int
        Embedding window length in samples.
    eigen_threshold : float
        Threshold to select significant eigenvalues in SSA decomposition.

    Returns
    -------
    crafted_blink : ndarray, shape (time,)
        Reconstructed blink artifact signal after SSA-based refinement.

    Notes
    -----
    Uses embedding and diagonal averaging to reconstruct the signal from significant components.
    """

    recovered_blink = raw*artifact_template

    embedded_blink = embedding(recovered_blink, window_length=window_length)

    from scipy.linalg import svd
    u, s, v = svd(embedded_blink, full_matrices = False)

    subspace_projection_mask = np.diag(s*((s/np.sum(s)) > eigen_threshold).astype(int))

    projected = u @ subspace_projection_mask @ v

    crafted_blink = diagonal_averaging(projected)

    return crafted_blink

def kmeans_ssa(single_channel_signal, window_length = 0.5, sampling_rate=500, kmeans_cluster_count = 5, standardize =False, 
                      fractal_dimension_threshold = 1.4, ssa_threshold = 0.01, visualization = 0, random_state = 42):
    """
    Detect and refine blink artifacts in a single-channel signal using SSA and K-means clustering.

    Parameters
    ----------
    single_channel_signal : ndarray, shape (time,)
        Input raw signal.
    window_length : float, optional
        Length of embedding window in seconds (default 0.5).
    sampling_rate : int, optional
        Sampling rate in Hz (default 500).
    kmeans_cluster_count : int, optional
        Number of clusters for K-means (default 5).
    standardize : bool or str, optional
        Standardization method for features; False disables (default False).
    fractal_dimension_threshold : float, optional
        Threshold for fractal dimension to detect blink clusters (default 1.4).
    ssa_threshold : float, optional
        Threshold for eigenvalue significance in SSA (default 0.01).
    visualization : int, optional
        If 1, display clustering and artifact visualizations (default 0).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    final : ndarray, shape (time,)
        Final reconstructed blink artifact signal.
    template : ndarray, shape (time,)
        Binary artifact template indicating blink occurrences.

    Notes
    -----
    The method performs embedding, feature extraction, clustering, fractal dimension analysis,
    and SSA-based artifact reconstruction.
    Visualization uses Plotly if enabled.
    """

    # Calculate the window length in samples
    M = int(window_length * sampling_rate)

    # Embed the signal into a multivariate matrix
    X = embedding(single_channel_signal, window_length = M)
    
    # Extract time-domain features from the embedded matrix
    F = time_features(X, standardize=standardize)

    # Perform K-means clustering on the features
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=kmeans_cluster_count, random_state=random_state, n_init="auto")
    kmeans.fit(F.T)

    # Visualization of the clustering results
    if visualization:
        features = ["Energy", "Hjorth mobility", "Kurtosis", "Peak_to_peak", "Cluster"]
        df = pd.DataFrame(np.concatenate((F, kmeans.labels_[np.newaxis, : ]), axis = 0 ).T, columns=features)
        
        # 3D scatter plot of features colored by Peak-to-peak value
        fig_3d = px.scatter_3d(df[:], x='Kurtosis', y='Energy', z='Hjorth mobility',
              color="Peak_to_peak", height = 600, width = 700)
        fig_3d.show()

        # 3D scatter plot of features colored by cluster
        fig_cluster = px.scatter_3d(df[:], x='Kurtosis', y='Energy', z='Hjorth mobility',
              color='Cluster',
                height = 600, opacity = 0.3, width = 700)
        fig_cluster.show()

    # Create multivariate matrices for each cluster
    Xi = k_multivariates(X, kmeans.labels_, kmeans_cluster_count)
    
    # Perform diagonal averaging to reconstruct the signal
    Si = diagonal_averaging(Xi)

    # Calculate fractal dimensions of the reconstructed signal
    _, Hm = fractal_dimension(Si, axis=0)

    # Draft blink detection based on fractal dimension threshold
    draft = blink_draft(Si, Hm, fractal_dimension_threshold)
    
    # Create an artifact template from the draft
    template = artifact_template(draft)

    # Craft the final blink signal using SSA
    final = craft_blink(single_channel_signal, template, M, ssa_threshold)

    # Visualization of the final crafted blink signal
    if visualization:
        comparison = np.array([single_channel_signal*template, final*template, template*10])
        t0 = 0
        dt = 400000
        px.line(comparison[:, t0:t0+dt:10].T, title = f'Blink fit | Artifact proportion : {np.sum(template)/template.shape[0]*100} %')

    return final, template


###############################################################################################################
#
#                                              CWT Multi-SSA - Blinks- Saccade
#
###############################################################################################################


def get_univariate_wavelet_transform(univariate_signal, wavelets_frequencies, sampling_rate, n_cycles = 6):
    """
    Compute the Morlet wavelet time-frequency transform of a univariate signal using MNE.

    Parameters
    ----------
    univariate_signal : ndarray, shape (n_times,)
        The input 1D signal to analyze.
    wavelets_frequencies : ndarray, shape (n_frequencies,)
        Array of frequencies (in Hz) at which to compute the wavelet transform.
    sampling_rate : float
        Sampling frequency of the signal (in Hz).
    n_cycles : int, float or callable, optional
        Number of cycles in Morlet wavelet. If callable, it should accept
        wavelets_frequencies and return an array of cycles per frequency.
        Default is 6.

    Returns
    -------
    power : ndarray, shape (n_frequencies, n_times)
        Complex Morlet wavelet transform coefficients for each frequency and time point.
    """

    info = mne.create_info(ch_names=['univariate'], sfreq=sampling_rate, ch_types='eeg', verbose=False)

    raw = mne.io.RawArray(univariate_signal[np.newaxis,:], info, verbose=False)
    if type(n_cycles) != int  and type(n_cycles) != float:
        n_cycles = n_cycles(wavelets_frequencies)  
    
    power = raw.compute_tfr(method="morlet", freqs=wavelets_frequencies, n_cycles=n_cycles, use_fft=True, verbose=False)
    
    return power.get_data()[0,:,:]



def wavelet_scaling(wavelet_frequencies):
    """
    Compute a scaling factor for wavelet frequencies based on a specific formula.

    Parameters
    ----------
    wavelet_frequencies : ndarray
        Array of frequencies.

    Returns
    -------
    scaled_frequencies : ndarray
        Scaled frequency values.
    """
    return wavelet_frequencies*np.log(1+wavelet_frequencies**(0.5))/(1+wavelet_frequencies)

def get_pseudo_artifact(univariate_signal, sampling_rate = 500, filter_f_min = 0.1, filter_f_max = None, metric_function='MAD', metric_factor=5, already_filtered = 0):
    """
    Generate a pseudo-artifact signal segment based on the amplitude metric of an input signal.

    Parameters
    ----------
    univariate_signal : ndarray, shape (n_times,)
        Input 1D signal.
    sampling_rate : float, optional
        Sampling frequency of the input signal (Hz). Default is 500.
    filter_f_min : float, optional
        Minimum frequency for bandpass filtering. Default is 0.1 Hz.
    filter_f_max : float or None, optional
        Maximum frequency for bandpass filtering. Default is None (no upper limit).
    metric_function : {'MAD', 'std'}, optional
        Metric to compute amplitude baseline. 'MAD' uses Median Absolute Deviation,
        'std' uses standard deviation. Default is 'MAD'.
    metric_factor : float, optional
        Multiplicative factor applied to the amplitude metric to define artifact amplitude.
        Default is 5.
    already_filtered : bool or int, optional
        If True (or non-zero), signal is assumed pre-filtered and filtering is skipped.
        Default is False (0).

    Returns
    -------
    pseudo_artifact : ndarray, shape (3 * sampling_rate,)
        A synthetic artifact signal with a plateau at metric_factor times the amplitude
        metric, smoothly tapered at edges.
    """

    if not already_filtered :
        filtered = mne.filter.filter_data(univariate_signal, sampling_rate, filter_f_min, filter_f_max, verbose=False)
    else:
        filtered = univariate_signal
    pseudo_artifact = np.zeros((3*sampling_rate))
    if metric_function == 'MAD':
        func = MAD
    else:
        func = np.std
    metric = func(filtered)
    pseudo_artifact[sampling_rate:2*sampling_rate] = metric_factor*metric
    pseudo_artifact[sampling_rate:sampling_rate+10] *= np.linspace(0,1,10)
    pseudo_artifact[2*sampling_rate-10:2*sampling_rate] *= np.linspace(1,0,10)
    return pseudo_artifact

def gauss_plateau(plateau, width, n_points):
    """
    Create a 1D array with a central plateau and Gaussian-shaped edges.

    Parameters
    ----------
    plateau : int
        Length of the central plateau region with value 1.
    width : float
        Width parameter of the Gaussian edges (standard deviation).
        If 0 or less, edges are not smoothed.
    n_points : int
        Total length of the output array.

    Returns
    -------
    plat : ndarray, shape (n_points,)
        Array with plateau in the center and Gaussian tapering on edges.
    """
    from scipy.signal.windows import gaussian
    plat = np.zeros(n_points)
    start_plateau = int((n_points-plateau)/2)
    end_plateau = start_plateau+plateau
    plat[start_plateau:end_plateau] = 1
    if width > 0 :
        gauss = gaussian(n_points-plateau, width)
        plat[:start_plateau] = gauss[:start_plateau]
        plat[end_plateau:] = gauss[start_plateau:]
    return plat
    
def gaussianize_mask_edges(mask, margins, widths, windows = None):
    """
    Smooth binary mask edges by applying Gaussian tapered transitions.

    For each window of consecutive ones in `mask`, extend edges by `margins` and
    apply Gaussian smoothing with corresponding `widths`.

    Parameters
    ----------
    mask : ndarray, shape (n_times,)
        Binary input mask (1 for event, 0 for non-event).
    margins : list of int
        List of margin sizes (number of samples) to extend on each side of mask windows.
    widths : list of float
        List of Gaussian widths (std deviations) corresponding to each margin.
    windows : list of tuples or None, optional
        Precomputed list of (start, end) tuples of windows of ones in `mask`.
        If None, will be computed internally. Default is None.

    Returns
    -------
    smooth_edges_masks : list of ndarray
        List of smoothed masks for each margin-width pair, each same length as `mask`.
    """

    if windows is None:
        fw = find_windows(mask, 1, True)
    else:
        fw = windows

    smooth_edges_masks = [np.zeros(mask.shape[0]+2*margin).astype(float) for margin in margins]

    for win in fw:
        plateau = win[1]-win[0]
        for i, (margin, width) in enumerate(zip(margins, widths)):
            n_points = 2*margin + plateau
            smooth_edges_masks[i][win[0]: win[1]+2*margin] += gauss_plateau(plateau, width, n_points)

    for i, margin in enumerate(margins):
        smooth_edges_masks[i][smooth_edges_masks[i]>1] = 1
        smooth_edges_masks[i] = smooth_edges_masks[i][margin :-margin]
    
    return smooth_edges_masks




def CWT_MSSA(univariate_signal, sampling_rate,
                  wavelets_frequency_spacing = 0.1, wavelets_min_frequency = 0.1, wavelets_max_frequency = 12,
                  filter_f_min = 1, filter_f_max = None, metric_function='MAD', metric_factor=5,
                  margins = [500,5000], widths = [50, 1000], ssa_window_seconds = 0.3,
                  first_ssa_trajectories_indices = np.arange(6), second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = np.arange(30),
                  return_unbias = True, return_corrected = False, show=True, save = None, details= False,
                  return_localisation = False,
                  ):
    """
    Artifact correction in univariate EEG signals using Continuous Wavelet Transform (CWT)
    combined with Multivariate Singular Spectrum Analysis (MSSA).

    This function performs wavelet-based artifact localization and correction via
    a two-stage SSA decomposition. It reconstructs an estimate of the artifact signal,
    optionally returns localization masks, and supports visualization and saving results.

    Parameters
    ----------
    univariate_signal : np.ndarray
        1D array containing the raw EEG signal (single channel).

    sampling_rate : float
        Sampling rate of the signal in Hz.

    wavelets_frequency_spacing : float, optional
        Frequency step for wavelets (Hz). Default is 0.1.

    wavelets_min_frequency : float, optional
        Minimum frequency for wavelets (Hz). Default is 0.1.

    wavelets_max_frequency : float, optional
        Maximum frequency for wavelets (Hz). Default is 12.

    filter_f_min : float, optional
        Minimum frequency for pre-filtering during pseudo-artifact construction (Hz). Default is 1.

    filter_f_max : float or None, optional
        Maximum frequency for pre-filtering during pseudo-artifact construction (Hz). Default is None (no high cutoff).

    metric_function : {'MAD', 'std'}, optional
        Metric used for pseudo-artifact thresholding: Median Absolute Deviation (MAD) or standard deviation. Default 'MAD'.

    metric_factor : float, optional
        Factor multiplied by the metric to define artifact detection threshold. Default is 5.

    margins : list of int, optional
        Margins in samples to smooth artifact localization masks (for edge Gaussianization). Default is [500, 5000].

    widths : list of int, optional
        Widths of Gaussian edges for smoothing masks (in samples). Default is [50, 1000].

    ssa_window_seconds : float, optional
        Window length in seconds for first SSA embedding. Default is 0.3.

    first_ssa_trajectories_indices : array-like or None, optional
        Indices of SSA components used for the first decomposition. Default is np.arange(6).

    second_ssa_downsampling : int, optional
        Downsampling factor for second SSA decomposition. Default is 10.

    second_ssa_window_seconds : float, optional
        Window length in seconds for second SSA embedding after downsampling. Default is 5.

    second_ssa_trajectories_indices : array-like or None, optional
        Indices of SSA components used for the second decomposition. Default is np.arange(30).

    return_unbias : bool, optional
        If True, return the slightly unbiased (corrected) artifact signal estimate. Default is True.

    return_corrected : bool, optional
        If True, return the difference between original and corrected signal (artifact estimate). Default is False.

    show : bool, optional
        If True, display diagnostic plots of the correction results. Default is True.

    save : str or None, optional
        Filename prefix to save the diagnostic plot as HTML. If None, no saving is performed. Default is None.

    details : bool, optional
        If True, print intermediate processing steps for debugging. Default is False.

    return_localisation : bool, optional
        If True, return a tuple (corrected_signal, localization_mask). Default is False.

    Returns
    -------
    np.ndarray or tuple
        Corrected signal (or artifact estimate if `return_corrected` is True).
        If `return_localisation` is True, returns tuple (corrected_signal, localization_mask).

    Raises
    ------
    ValueError
        If `first_ssa_trajectories_indices` or `second_ssa_trajectories_indices` are invalid or undefined.
    """
    
    # print('______________Parameters_____________')
    # print("wavelets_frequency_spacing", wavelets_frequency_spacing)
    # print("wavelets_min_frequency", wavelets_min_frequency)
    # print("wavelets_max_frequency", wavelets_max_frequency)
    # print("filter_f_min", filter_f_min)
    # print("filter_f_max", filter_f_max)
    # print("metric_function", metric_function)
    # print("metric_factor", metric_factor)
    # print("margins", margins)
    # print("widths", widths)
    # print("ssa_window_seconds", ssa_window_seconds)
    # print("first_ssa_trajectories_indices", first_ssa_trajectories_indices)
    # print("second_ssa_downsampling", second_ssa_downsampling)
    # print("second_ssa_window_seconds", second_ssa_window_seconds)
    # print("second_ssa_trajectories_indices", second_ssa_trajectories_indices)
    
    # Get wavelet power
    wavelets_frequencies = np.linspace(wavelets_min_frequency,wavelets_max_frequency,int((wavelets_max_frequency-wavelets_min_frequency)/wavelets_frequency_spacing)+1)
    wavelet_power = get_univariate_wavelet_transform(univariate_signal, wavelets_frequencies, sampling_rate=sampling_rate, n_cycles=wavelet_scaling)
    if details:
        print('Wavelet power computed')

    # Get threshold to detect artifactual components
    pseudo_artifact = get_pseudo_artifact(univariate_signal, filter_f_min = filter_f_min, filter_f_max = filter_f_max, metric_function=metric_function, metric_factor=metric_factor)
    w_pseudo = get_univariate_wavelet_transform(pseudo_artifact, wavelets_frequencies, sampling_rate=sampling_rate, n_cycles=wavelet_scaling)
    wavelet_threshold = np.max(np.sum(w_pseudo, axis=0))
    if details:
        print('Wavelet threshold computed')
        print(wavelet_threshold)

    # Get localisation masks
    localisation = np.sum(wavelet_power, axis=0) > wavelet_threshold
    smooth_localisations = gaussianize_mask_edges(localisation, margins, widths)
    smooth_localisation = smooth_localisations[0]
    large_smooth_localisation = smooth_localisations[1]
    if details:
        # p1 = np.round(np.sum(smooth_localisation)/len(smooth_localisation)*100,2)
        # p2 = np.round(np.sum(large_smooth_localisation)/len(large_smooth_localisation)*100,2)
        # plot_comparison([univariate_signal, smooth_localisation, large_smooth_localisation], title = f"Local artefact : {p1} % | Global correction :{p2} % ")
        print('Artifact localisation masks computed')

    if np.sum(localisation) > 0 :

        # First SSA
        ssa_window_samples = int(ssa_window_seconds*sampling_rate)
        Y = embedding(univariate_signal, ssa_window_samples)
        first_ssa_trajectories = get_ssa_trajectories(Y, trajectory_indices=first_ssa_trajectories_indices)
        if details:
            print('First SSA computed')

        # Second SSA
        if len(second_ssa_trajectories_indices) > 0:
            ssa_window_samples_2 = int(second_ssa_window_seconds*sampling_rate/second_ssa_downsampling)
            second_ssa_univariate_signal = first_ssa_trajectories[0,::second_ssa_downsampling]
            Y_2 = embedding(second_ssa_univariate_signal, ssa_window_samples_2)
            second_ssa_trajectories = get_ssa_trajectories(Y_2, trajectory_indices=second_ssa_trajectories_indices)
            if details:
                print('Second SSA computed')

            # Unbias SSA
            unbias_ssa_univariate_signal = second_ssa_univariate_signal * (1-large_smooth_localisation[::second_ssa_downsampling])
            unbias_ssa_univariate_signal += second_ssa_trajectories[0] * smooth_localisation[::second_ssa_downsampling]
            Y_3 = embedding( unbias_ssa_univariate_signal, ssa_window_samples_2 )
            unbias_ssa_trajectories = get_ssa_trajectories(Y_3, trajectory_indices=[0])
            if details:
                print('Unbias SSA computed')

        # Reconstruction

        
        if len(second_ssa_trajectories_indices) > 0:
            ## Unbias reconstruction
            slow_bias =  resample(second_ssa_trajectories[0], univariate_signal.shape[0])
            slow_true =  resample(unbias_ssa_trajectories[0], univariate_signal.shape[0])
            unbias = 1*(slow_true-slow_bias) * large_smooth_localisation

            ## Artifact signal reconstruction
            partial_slow = second_ssa_trajectories[1:] * large_smooth_localisation[::second_ssa_downsampling]
            fast = resample(np.sum(partial_slow, axis=0), univariate_signal.shape[0])
            unbiased_fast = (fast-unbias)
            fast +=  np.sum(first_ssa_trajectories[1:] * smooth_localisation, axis=0)
            unbiased_fast += np.sum(first_ssa_trajectories[1:] * smooth_localisation, axis=0)
            
        else:
            ## Unbias reconstruction
            slow_bias =  0*univariate_signal
            slow_true =  0*univariate_signal
            unbias = 0*univariate_signal

            ## Artifact signal reconstruction
            partial_slow = second_ssa_trajectories[1:] * large_smooth_localisation[::second_ssa_downsampling]
            fast = 0*univariate_signal
            unbiased_fast = (fast-unbias)
            fast +=  np.sum(first_ssa_trajectories[1:] * smooth_localisation, axis=0)
            unbiased_fast += np.sum(first_ssa_trajectories[1:] * smooth_localisation, axis=0)
        
        if details:
                print('Correction computed')


        ## PLot
        if show or save :
            timings = (np.arange(len(univariate_signal))*1/sampling_rate)
            d = 5
            comparison_df = pd.concat([
            pd.DataFrame({'Time' : timings[::d], 'Type' : 'Original', 'Signal' : univariate_signal[::d]}),
            pd.DataFrame({'Time' : timings[::d], 'Type' : 'Slow bias', 'Signal' : slow_bias[::d]}),
            pd.DataFrame({'Time' : timings[::d], 'Type' : 'Slow unbias ', 'Signal' :  slow_true[::d]}),
            pd.DataFrame({'Time' : timings[::d], 'Type' : 'Biased correction', 'Signal' : (univariate_signal-fast)[::d]}),
            pd.DataFrame({'Time' : timings[::d], 'Type' : 'Unbiased correction', 'Signal' : (univariate_signal-unbiased_fast)[::d]}),
            pd.DataFrame({'Time' : timings[::d], 'Type' : 'Localisation', 'Signal' : np.max(np.abs(univariate_signal))/3*smooth_localisation[::d]}),
            pd.DataFrame({'Time' : timings[::d], 'Type' : 'Large localisation', 'Signal' : np.max(np.abs(univariate_signal))/3*large_smooth_localisation[::d]}),
            ])
            
            p1 = np.round(np.sum(smooth_localisation)/len(smooth_localisation)*100,2)
            p2 = np.round(np.sum(large_smooth_localisation)/len(large_smooth_localisation)*100,2)
            if not save is None:
                name = save
            else:
                name = ''
            f = px.line(comparison_df, x = 'Time', y= 'Signal', color= 'Type', 
                title = f'{name} : Local artefact : {p1} % | Global correction :{p2} % ')
            
            if show:
                f.show()
            
            if save:
                name_path = name + ".html"
                filepath = figures_path / 'ICA' / 'CWT_MSSA' / name_path
                f.write_html(filepath)
    else:
        unbiased_fast = np.zeros(len(localisation))
        fast = np.zeros(len(localisation))
        
    if return_unbias:
        output_signal = unbiased_fast
    else:
        output_signal = fast
    
    if return_corrected :
        output_signal = univariate_signal - output_signal
    
    if return_localisation :
        return output_signal, localisation
    else :
        return output_signal

    

def get_smooth_from_door(raw_signal, size=1000, first_pass_width = 10, first_pass_iteration = 4, second_pass_width = 200, scale_to_signal = True):
    """
    Perform iterative smoothing on a raw signal using rectangular "door" kernels and compute a smoothed gradient.

    Parameters
    ----------
    raw_signal : ndarray
        1D array of the raw input signal to be smoothed.
    size : int, optional
        Length of the rectangular smoothing kernel arrays (default is 1000).
    first_pass_width : int, optional
        Width of the rectangular kernel for the first smoothing pass (default is 10).
    first_pass_iteration : int, optional
        Number of iterations for the first smoothing convolution (default is 4).
    second_pass_width : int, optional
        Width of the rectangular kernel for smoothing the gradient in the second pass (default is 200).
    scale_to_signal : bool, optional
        If True, scale the smoothed results to match the amplitude of the raw signal using `change_scale` (default is True).

    Returns
    -------
    smoothed_signal : ndarray
        The iteratively smoothed version of the input raw_signal.
    smooth_gradient : ndarray
        The gradient of the smoothed signal, further smoothed by a second rectangular kernel.

    Notes
    -----
    The function applies two smoothing stages using rectangular kernels ("door" functions):
    1. Iterative smoothing of the raw signal.
    2. Smoothing of the gradient of the first smoothed signal.
    
    The kernels are centered rectangular windows with lengths controlled by the width parameters.
    The convolution is performed with mode='same' to preserve the original signal length.

    The `change_scale` function should be defined elsewhere to rescale signals appropriately.
    """

    width = int(min(size/2-1, first_pass_width))
    creneau = np.zeros(size)
    creneau[int(size/2-width/2):int(size/2+width/2)] = 1
    smoothed_signal = raw_signal.copy()
    for _ in range(first_pass_iteration):
        smoothed_signal = np.convolve(smoothed_signal, creneau, mode='same')
    
    if scale_to_signal :
        smoothed_signal = change_scale(smoothed_signal, raw_signal)

    width = int(min(size/2-1, second_pass_width))
    creneau = np.zeros(size)
    creneau[int(size/2-width/2):int(size/2+width/2)] = 1
    smooth_gradient = np.convolve(np.gradient(smoothed_signal), creneau, mode='same')
    if scale_to_signal:
        smooth_gradient = change_scale(smooth_gradient, raw_signal)

    return smoothed_signal, smooth_gradient


def get_eog_classification(univariate_signal, localisation, fixed_length_windows = 1, feature_type = 'sign_change', smoothing_artifacts = 0.25,
                           artifact_margin = 500, artifact_width = 0,  artifact_plateau = 250, artifact_shift = 50, 
                           smoothing_size=1000, smoothing_first_pass_width = 100, smoothing_first_pass_iteration = 4, smoothing_second_pass_width = 10,
                            gradient_peaks_height=3, gradient_peaks_prominence=3, gradient_peaks_width=10, 
                            bridging = 0.2, sampling_rate = 500):
    """
    Classify artifact windows in a univariate EOG signal based on features extracted from signal windows.

    This function segments the input signal into artifact and non-artifact windows using 
    provided localization data, applies smoothing and feature extraction methods, and
    labels windows based on detected artifact characteristics.

    Parameters
    ----------
    univariate_signal : ndarray
        1D array containing the raw EOG or related signal.
    localisation : ndarray
        1D binary array indicating artifact presence (1) or absence (0) at each sample.
    fixed_length_windows : int or bool, optional
        If True (default 1), use fixed length windows around artifacts; otherwise use localisation windows directly.
    feature_type : {'sign_change', 'gradient_peaks'}, optional
        Method for classifying artifacts. 'sign_change' counts sign changes, 'gradient_peaks' detects peaks in gradient.
    smoothing_artifacts : float, optional
        Window length in seconds for rolling mean smoothing of signal artifacts (default 0.25).
    artifact_margin : int, optional
        Number of samples added as margin around detected artifacts (default 500).
    artifact_width : int, optional
        Width parameter for the Gaussian plateau shaping artifact localization (default 0).
    artifact_plateau : int, optional
        Minimum plateau length in samples for artifact shaping (default 250).
    artifact_shift : int, optional
        Number of samples to shift the artifact localization window (default 50).
    smoothing_size : int, optional
        Size parameter for smoothing kernels in `get_smooth_from_door` (default 1000).
    smoothing_first_pass_width : int, optional
        First pass kernel width for smoothing in `get_smooth_from_door` (default 100).
    smoothing_first_pass_iteration : int, optional
        Number of iterations for first pass smoothing in `get_smooth_from_door` (default 4).
    smoothing_second_pass_width : int, optional
        Second pass kernel width for smoothing gradient in `get_smooth_from_door` (default 10).
    gradient_peaks_height : float, optional
        Minimum height of gradient peaks for peak detection (default 3).
    gradient_peaks_prominence : float, optional
        Minimum prominence of gradient peaks for peak detection (default 3).
    gradient_peaks_width : float, optional
        Minimum width of gradient peaks for peak detection (default 10).
    bridging : float, optional
        Maximum gap duration in seconds between windows to be bridged (default 0.2).
    sampling_rate : int, optional
        Sampling rate of the signal in Hz (default 500).

    Returns
    -------
    cluster_windows : list of tuples
        List of classified windows as tuples `(start_idx, end_idx, label)`, where
        `label` is a string indicating artifact class:
        - '0' : Non-artifact window
        - '1', '2', '3', ... : Artifact window with classification based on features detected

    Notes
    -----
    - For `feature_type='gradient_peaks'`, windows are classified by counting positive and negative gradient peaks.
    - For `feature_type='sign_change'`, windows are classified by counting zero crossings in the smoothed artifact signal.
    - Artifact windows closer than `bridging * sampling_rate` samples are optionally merged.


    """

    # Get windows around artifacts
    if fixed_length_windows :
        # Find windows of continuous artifact localization (fixed-length windows)
        fw = find_windows(localisation, 1, True)
        # Initialize an extended array for sharper localization with margin padding
        sharp_localisation = np.zeros(localisation.shape[0]+2*artifact_margin)

        # For each detected window, create a Gaussian plateau to model artifact shape
        for win in fw:

            # Ensure plateau length is at least artifact_plateau or the detected window length
            plateau = max(artifact_plateau,win[1]-win[0])
            # Total number of points in this shaped artifact window
            n_points = 2*artifact_margin + plateau

            # Define the segment indices accounting for shift and margins
            end_case = sharp_localisation[max(0,win[0]-artifact_shift): win[0]+plateau+2*artifact_margin-artifact_shift].shape[0]

            # Add a Gaussian plateau raised to the 3rd power to the sharp_localisation segment
            sharp_localisation[max(0,win[0]-artifact_shift): win[0]+plateau+2*artifact_margin-artifact_shift] += gauss_plateau(plateau, artifact_width, n_points)[:end_case]**3

        # Clip values > 1 to 1 to maintain binary-like mask
        sharp_localisation[sharp_localisation>1] = 1

        # Remove the padding margins to get final sharp localization mask aligned to original signal length
        sharp_localisation = sharp_localisation[artifact_margin :-artifact_margin]

        # Identify continuous artifact windows from the sharp localization mask  
        sharp_windows = find_windows(sharp_localisation,1,1)
    else:
        # If not fixed length, directly find windows from the input localization mask
        sharp_windows = find_windows(localisation,1,1)

    previous = 0
    completed_windows = []

    # Complete list of windows by adding non-artifactual windows and merge or bridge close windows based on bridging threshold (in samples)
    for w1, w2 in sharp_windows:
        # If gap between windows is larger than bridging threshold, add non-artifact window ('0')
        if w1 > previous and w1-previous > bridging*sampling_rate:
            completed_windows.append((previous, w1,'0'))
        # If gap is small, mark bridging window with label '1' (artifact)
        elif w1 > previous and w1-previous <= bridging*sampling_rate:
            completed_windows.append((previous, w1,'1'))

        # Add current artifact window with label '1'
        completed_windows.append((w1, w2,'1'))
        previous = w2

    # Add trailing non-artifact window if signal end is beyond last detected artifact window
    if sharp_windows[-1][1] < len(univariate_signal)-1:
        completed_windows.append((sharp_windows[-1][1], len(univariate_signal)-1,'0'))

    
    if feature_type == 'gradient_peaks' :
        # Classify artifact windows based on gradient variations (a step is caracterized by a single peak, while a blink by two peaks of opposite signs)
        _, smooth_gradient = get_smooth_from_door(univariate_signal, size=smoothing_size, first_pass_width = smoothing_first_pass_width,
                                                first_pass_iteration = smoothing_first_pass_iteration, second_pass_width = smoothing_second_pass_width, scale_to_signal=False)
        # Detect positive gradient peaks above thresholds
        peaks_plus, prop = signal.find_peaks(rz_score(smooth_gradient), height=gradient_peaks_height, prominence=gradient_peaks_prominence, width=gradient_peaks_width)
        # Detect negative gradient peaks (inverted signal) above thresholds
        peaks_minus, prop = signal.find_peaks(rz_score(-smooth_gradient), height=gradient_peaks_height, prominence=gradient_peaks_prominence, width=gradient_peaks_width)
        # Combine positive and negative peaks into a DataFrame
        df_peak = pd.concat([
                        pd.DataFrame({'Sign' : 1, 'Time' : peaks_plus}),
                        pd.DataFrame({'Sign' : -1, 'Time' : peaks_minus}),
                    ])

        feature = []
        # For each completed window labeled as artifact ('1'), count peaks of each sign
        for w1, w2, label in completed_windows:
            if int(label):
                counts = df_peak[(df_peak['Time'] > w1) & (df_peak['Time'] < w2)].groupby('Sign').count().values
                # Append number of unique peak signs detected (usually 1 or 2)
                feature.append(len(counts))
    else:
        # Use smoothed signal for sign-change based feature extraction
        feature = []
        n_points = int(smoothing_artifacts*sampling_rate)
        # Smooth signal with rolling mean, using reflection padding to reduce edge artifacts
        smoothed = pd.Series(np.pad(univariate_signal, n_points, mode='reflect')).rolling(n_points, center=True).mean()[n_points:-n_points].values

        # For each artifact window, count zero crossings in the smoothed segment
        for w1, w2, label in completed_windows:
            if int(label):
                artefact = smoothed[w1:w2]
                artefact = artefact - np.mean(artefact) # Demean
                # Count sign changes (mod 2) to discriminate artifact type
                feature.append(int(np.sum(np.abs(np.diff(0.5*np.sign(artefact))))%2))
    


    cluster_windows = []
    counter = 0
    # Assign final labels to windows based on computed features
    for w1, w2, label in completed_windows:
        if int(label):
            # Add 1 to feature to avoid label '0' for artifact windows
            cluster_windows.append((w1, w2,str(feature[counter]+1)))
            counter += 1
        else:
            cluster_windows.append((w1, w2,'0'))
    
    return cluster_windows


def rectify_saccades_old(univariate_signal, cluster_windows, correction, rectification_drift_highpass = 0.05, ease_in_width=100, sampling_rate=500, show=False):
    """
    Rectify saccades in a univariate signal by applying a correction to artifact windows and smoothing transitions.

    This function processes the input signal based on artifact windows (`cluster_windows`) and a provided
    correction signal. It applies rectification to artifact segments, enforces continuity between windows,
    filters the rectified signal to remove drift, and smooths the edges between artifact and non-artifact segments
    using an ease-in exponential weighting. Optionally, it displays a comparison plot of the original, rectified,
    and difference signals.

    Parameters
    ----------
    univariate_signal : np.ndarray
        The original univariate signal to be corrected.
    cluster_windows : list of tuples
        List of tuples `(start, end, label)` defining windows in the signal and their artifact label.
        Labels are interpreted as integers; non-zero means artifact.
    correction : np.ndarray
        Correction signal applied to artifact windows to rectify the original signal.
    rectification_drift_highpass : float, optional
        High-pass filter cutoff frequency in Hz applied to the rectification signal to remove drift.
        Default is 0.05 Hz.
    ease_in_width : int, optional
        Number of samples over which to apply exponential ease-in smoothing at the edges of artifact windows.
        Default is 100 samples.
    sampling_rate : int, optional
        Sampling rate of the signal in Hz. Used for filtering. Default is 500 Hz.
    show : bool, optional
        If True, plots a comparison of original and rectified signals using `plot_comparison`.
        Default is False.

    Returns
    -------
    np.ndarray
        The rectified and smoothed signal after applying artifact corrections.

    Notes
    -----
    - The function assumes `cluster_windows` cover the entire signal with artifact ('label' != 0) and
      non-artifact segments ('label' == 0).
    - Boundary checks are recommended if `cluster_windows` include windows at the signal edges.
    - Requires `mne` package for filtering and `plot_comparison` for visualization if `show=True`.
    """

    # Rectify saccades
    rectification = np.zeros_like(univariate_signal)
    # mask = np.zeros_like(univariate_signal)
    last_value = 0
    previous = 0

    for i, (w1, w2, label) in enumerate(cluster_windows):
        if int(label) :
            # mask[w1:w2] = int(label)
            if int(label)==3 or int(label)==1:
                rectification[w1:w2] = correction[w1:w2] - correction[w1] + last_value
            else:
                rectification[w1:w2] = correction[w1:w2] -correction[w1] + last_value
                rectification[w2:] = -previous
        else:
            if len(univariate_signal[w1:w2])> 0:
                previous = np.mean(univariate_signal[w1:w2])
        
        last_value = rectification[w2]

    for w1, w2, label in cluster_windows:
            if int(label) :
                rectification[w2:] = rectification[w2:]  - (rectification[w2]-rectification[w2-1]) 
            
    
    rectification_filtered = mne.filter.filter_data(rectification,500,0.05,None)
    rectification_flattened = rectification.copy()

    for w1, w2, label in cluster_windows:
            if int(label) :
                rectification_flattened[w1:w2] = rectification_filtered[w1:w2]
            else:
                rectification_flattened[w1-1:w2+1] = rectification_flattened[w1-1:w2+1]-np.mean(rectification_flattened[w1-1:w2+1]) + np.mean(rectification_filtered[w1-1:w2+1])
                
                ease_in = np.exp(-np.linspace(0,2.5,rectification_flattened[w1-1:w1+ease_in_width].shape[0])**2)
                rectification_flattened[w1-1:w1+ease_in_width] = (rectification_filtered[w1-1:w1+ease_in_width])*ease_in + (rectification_flattened[w1-1:w1+ease_in_width])*(1-ease_in)

                ease_in = np.exp(-np.linspace(0,2.5,rectification_flattened[w2-ease_in_width:w2+1].shape[0])**2)
                rectification_flattened[w2-ease_in_width:w2+1] = rectification_filtered[w2-ease_in_width:w2+1]*(1-ease_in) + rectification_flattened[w2-ease_in_width:w2+1]*ease_in

    rectification = mne.filter.filter_data(rectification, sampling_rate, rectification_drift_highpass, None)

    if show :
        plot_comparison([univariate_signal, rectification, univariate_signal-rectification, rectification_flattened, univariate_signal-rectification_flattened])

    return rectification_flattened


def rectified_CWT_MSSA(univariate_signal, sampling_rate, wavelets_frequency_spacing = 0.1, wavelets_min_frequency = 0.1, wavelets_max_frequency = 12,
                  filter_f_min = 1, filter_f_max = None, metric_function='MAD', metric_factor=5,
                  margins = [500,5000], widths = [50, 1000], ssa_window_seconds = 0.3,
                  first_ssa_trajectories_indices = np.arange(6), second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = np.arange(30),
                  return_unbias = True, return_corrected = False, show=True, save = None, details= False,
                  return_localisation = False,
                  artifact_margin = 500, artifact_width = 0, artifact_plateau = 250, artifact_shift = 50, smoothing_size=1000,
                  smoothing_first_pass_width = 100, smoothing_first_pass_iteration = 4, smoothing_second_pass_width = 10,
                  gradient_peaks_height=3, gradient_peaks_prominence=3, gradient_peaks_width=10,
                  rectification_drift_highpass = 0.05, ease_in_width=100, **kwargs):
    """
    Perform artifact rectification on a univariate signal using combined CWT-MSSA decomposition, artifact classification, and rectification.

    This function executes a multi-step process for detecting and correcting ocular artifacts (e.g., saccades)
    in a univariate time series signal. It uses continuous wavelet transform and multivariate singular spectrum
    analysis (CWT-MSSA) to generate a correction signal and localization of artifacts. Then, it classifies artifact
    windows and applies a rectification procedure to remove artifacts while preserving signal continuity.

    Parameters
    ----------
    univariate_signal : np.ndarray
        The raw input signal to be corrected (1D array).
    sampling_rate : float
        Sampling rate of the input signal in Hz.
    wavelets_frequency_spacing : float, optional
        Frequency spacing between wavelets used in the CWT step. Default is 0.1 Hz.
    wavelets_min_frequency : float, optional
        Minimum frequency for wavelet decomposition. Default is 0.1 Hz.
    wavelets_max_frequency : float, optional
        Maximum frequency for wavelet decomposition. Default is 12 Hz.
    filter_f_min : float, optional
        Minimum frequency cutoff for filtering in the CWT-MSSA pipeline. Default is 1 Hz.
    filter_f_max : float or None, optional
        Maximum frequency cutoff for filtering. None means no upper cutoff. Default is None.
    metric_function : str, optional
        Metric used for artifact detection. Default is 'MAD' (Median Absolute Deviation).
    metric_factor : float, optional
        Scaling factor for the artifact detection threshold. Default is 5.
    margins : list of int, optional
        Margins (in samples) around detected artifacts to include in analysis. Default is [500, 5000].
    widths : list of int, optional
        Width parameters (in samples) used in artifact detection. Default is [50, 1000].
    ssa_window_seconds : float, optional
        Window size in seconds for the first SSA step. Default is 0.3 seconds.
    first_ssa_trajectories_indices : np.ndarray, optional
        Indices of SSA trajectories used in the first SSA. Default is np.arange(6).
    second_ssa_downsampling : int, optional
        Downsampling factor for the second SSA step. Default is 10.
    second_ssa_window_seconds : float, optional
        Window size in seconds for the second SSA. Default is 5 seconds.
    second_ssa_trajectories_indices : np.ndarray, optional
        Indices of SSA trajectories used in the second SSA. Default is np.arange(30).
    return_unbias : bool, optional
        Whether to return the unbias signal from CWT-MSSA. Default is True.
    return_corrected : bool, optional
        Whether to return the corrected signal from CWT-MSSA. Default is False.
    show : bool, optional
        Whether to display a plot comparing original and corrected signals. Default is True.
    save : str or None, optional
        File path to save outputs, or None to disable saving. Default is None.
    details : bool, optional
        Whether to return detailed intermediate results from CWT-MSSA. Default is False.
    return_localisation : bool, optional
        Whether to return the artifact localisation mask from CWT-MSSA. Default is False.
    artifact_margin : int, optional
        Margin (in samples) around artifacts for classification. Default is 500.
    artifact_width : int, optional
        Width parameter for artifact classification. Default is 0.
    artifact_plateau : int, optional
        Plateau duration (in samples) used in artifact classification. Default is 250.
    artifact_shift : int, optional
        Shift applied in artifact classification windowing. Default is 50.
    smoothing_size : int, optional
        Smoothing window size (in samples) for artifact classification. Default is 1000.
    smoothing_first_pass_width : int, optional
        First pass smoothing width. Default is 100.
    smoothing_first_pass_iteration : int, optional
        Number of iterations in first pass smoothing. Default is 4.
    smoothing_second_pass_width : int, optional
        Second pass smoothing width. Default is 10.
    gradient_peaks_height : float, optional
        Height threshold for peak detection in gradient signal. Default is 3.
    gradient_peaks_prominence : float, optional
        Prominence threshold for peak detection. Default is 3.
    gradient_peaks_width : int, optional
        Width threshold for peak detection. Default is 10.
    rectification_drift_highpass : float, optional
        High-pass filter cutoff frequency for rectification drift removal (Hz). Default is 0.05.
    ease_in_width : int, optional
        Number of samples for ease-in smoothing at artifact edges. Default is 100.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    np.ndarray
        artifact signal of the same length as `univariate_signal`.

    Notes
    -----
    - If no artifacts are detected (`localisation` sum is zero), returns an array of zeros.
    """

    # Compute correction
    correction, localisation = CWT_MSSA(univariate_signal, sampling_rate,
                  wavelets_frequency_spacing = wavelets_frequency_spacing, wavelets_min_frequency = wavelets_min_frequency, wavelets_max_frequency = wavelets_max_frequency,
                  filter_f_min = filter_f_min, filter_f_max = filter_f_max, metric_function=metric_function, metric_factor=metric_factor,
                  margins = margins, widths = widths, ssa_window_seconds = ssa_window_seconds,
                  first_ssa_trajectories_indices = first_ssa_trajectories_indices, second_ssa_downsampling = second_ssa_downsampling, 
                  second_ssa_window_seconds = second_ssa_window_seconds, second_ssa_trajectories_indices = second_ssa_trajectories_indices,
                  return_unbias = return_unbias, return_corrected = return_corrected, show=False, save = save, details= details,
                  return_localisation = return_localisation,
                  )

    if np.sum(localisation) > 0:
        # Compute windows with classification of artifacts
        cluster_windows = get_eog_classification(univariate_signal, localisation, artifact_margin = artifact_margin, artifact_width = artifact_width,
                                                artifact_plateau = artifact_plateau, artifact_shift = artifact_shift, smoothing_size=smoothing_size,
                                smoothing_first_pass_width = smoothing_first_pass_width, smoothing_first_pass_iteration = smoothing_first_pass_iteration,
                                smoothing_second_pass_width = smoothing_second_pass_width, gradient_peaks_height=gradient_peaks_height, gradient_peaks_prominence=gradient_peaks_prominence,
                                gradient_peaks_width=gradient_peaks_width, sampling_rate=sampling_rate)
        
        # Compute rectification
        rectification = rectify_saccades_old(univariate_signal, cluster_windows, correction, rectification_drift_highpass = rectification_drift_highpass,
                                        ease_in_width=ease_in_width,  sampling_rate=sampling_rate, show=show)

    else:
        rectification = np.zeros(len(univariate_signal))
    

    return rectification




###############################################################################################################
#
#                                               SSA-ICA - Blinks
#
###############################################################################################################






def _pl(x, non_pl="", pl="s"):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, int | np.generic) else len(x)
    return non_pl if len_x == 1 else pl

# Slight modification of MNE code
def modify_sources(ica, data, modifications, modifications_index, include_ica=False,  include=None, exclude=None, n_pca_components=None, keep_all = False):
        """
    Modify selected ICA source components in the data and reconstruct the signal.

    Parameters
    ----------
    ica : object
        Fitted ICA object with attributes like `pca_components_`, `mixing_matrix_`, `unmixing_matrix_`, etc.
    data : ndarray, shape (n_channels, n_times)
        Input data to transform and modify in ICA space.
    modifications : ndarray, shape (len(modifications_index), n_times)
        Modifications (additive/subtractive) to apply to selected ICA components.
    modifications_index : array-like
        Indices of ICA components to be modified.
    include_ica : bool, optional
        If True, return both the signal reconstructed after ICA modification and the residual signal (default is False).
    include : array-like or None, optional
        Indices of ICA components to explicitly include (default is None).
    exclude : array-like or None, optional
        Indices of ICA components to explicitly exclude (default is None).
    n_pca_components : int or None, optional
        Number of PCA components to use in reconstruction (default is None, uses ica.n_pca_components).
    keep_all : bool, optional
        If True, keep all PCA components during reconstruction, otherwise only selected ones (default is False).

    Returns
    -------
    proj_back : ndarray, shape (n_channels, n_times)
        Reconstructed signal after modifying selected ICA components.

    or (if include_ica is True):

    data : ndarray, shape (n_channels, n_times)
        Data projected back without modified components.
    proj_back : ndarray, shape (n_channels, n_times)
        Data projected back with modified components.

    Raises
    ------
    ValueError
        If `n_pca_components` is outside valid range relative to ICA components.
    """
        import logging
        logger = logging.getLogger("mne")  # one selection here used across mne-python

        """Aux function."""
        if n_pca_components is None:
            n_pca_components = ica.n_pca_components
        data = ica._pre_whiten(data)
        exclude = ica._check_exclude(exclude)
        _n_pca_comp = ica._check_n_pca_components(n_pca_components)
        n_ch, _ = data.shape

        max_pca_components = ica.pca_components_.shape[0]
        if not ica.n_components_ <= _n_pca_comp <= max_pca_components:
            raise ValueError(
                f"n_pca_components ({_n_pca_comp}) must be >= "
                f"n_components_ ({ica.n_components_}) and <= "
                "the total number of PCA components "
                f"({max_pca_components})."
            )

        logger.info(
            f"    Transforming to ICA space ({ica.n_components_} "
            f"component{_pl(ica.n_components_)})"
        )

        # Apply first PCA
        if ica.pca_mean_ is not None:
            data -= ica.pca_mean_[:, None]

        sel_keep = np.arange(ica.n_components_)
        if include not in (None, []):
            sel_keep = np.unique(include)
        elif exclude not in (None, []):
            sel_keep = np.setdiff1d(np.arange(ica.n_components_), exclude)

        n_zero = ica.n_components_ - len(sel_keep)
        logger.info(f"    Zeroing out {n_zero} ICA component{_pl(n_zero)}")

        # Mixing and unmixing should both be shape (ica.n_components_, 2),
        # and we need to put these into the upper left part of larger mixing
        # and unmixing matrices of shape (n_ch, _n_pca_comp)
        pca_components = ica.pca_components_[:_n_pca_comp]
        assert pca_components.shape == (_n_pca_comp, n_ch)
        assert (
            ica.unmixing_matrix_.shape
            == ica.mixing_matrix_.shape
            == (ica.n_components_,) * 2
        )
        unmixing = np.eye(_n_pca_comp)
        unmixing[: ica.n_components_, : ica.n_components_] = ica.unmixing_matrix_
        unmixing = np.dot(unmixing, pca_components)

        logger.info(
            f"    Projecting back using {_n_pca_comp} "
            f"PCA component{_pl(_n_pca_comp)}"
        )
        mixing = np.eye(_n_pca_comp)
        mixing[: ica.n_components_, : ica.n_components_] = ica.mixing_matrix_
        mixing = pca_components.T @ mixing
        assert mixing.shape == unmixing.shape[::-1] == (n_ch, _n_pca_comp)

        # keep requested components plus residuals (if any)
        if keep_all:
            sel_keep = np.arange(ica.n_components_)
        sel_keep = np.concatenate(
            (sel_keep, np.arange(ica.n_components_, _n_pca_comp))
        )

        proj_to_ica = np.dot(unmixing[sel_keep, :], data)
        proj_to_ica[modifications_index, :] = proj_to_ica[modifications_index, :] - modifications

        proj_back =  np.dot(mixing[:, sel_keep], proj_to_ica)

        if ica.pca_mean_ is not None:
            proj_back += ica.pca_mean_[:, None]

        # restore scaling
        if ica.noise_cov is None:  # revert standardization
            proj_back *= ica.pre_whitener_
        else:
            proj_back = np.linalg.pinv(ica.pre_whitener_, rcond=1e-14) @ proj_back

        if include_ica :
            sel_keep = np.setdiff1d(np.arange(ica.n_components_), modifications_index)
            sel_keep = np.concatenate(
            (sel_keep, np.arange(ica.n_components_, _n_pca_comp))
            )
            proj_mat = np.dot(mixing[:, sel_keep], unmixing[sel_keep, :])
            data = np.dot(proj_mat, data)
            assert proj_mat.shape == (n_ch,) * 2

            if ica.pca_mean_ is not None:
                data += ica.pca_mean_[:, None]

            # restore scaling
            if ica.noise_cov is None:  # revert standardization
                data *= ica.pre_whitener_
            else:
                data = np.linalg.pinv(ica.pre_whitener_, rcond=1e-14) @ data
            return data, proj_back
        
        else:
            return proj_back



def apply_modified(ica, raw, modifications, modifications_index, include_ica=False, include=None, exclude=None, n_pca_components=None, keep_all = False):
        
        """
    Apply modifications to ICA source components on MNE Raw data and update the Raw object.

    Parameters
    ----------
    ica : object
        Fitted ICA object.
    raw : mne.io.Raw
        Raw MNE data object containing continuous recordings.
    modifications : ndarray, shape (len(modifications_index), n_times)
        Modifications to apply to selected ICA components.
    modifications_index : array-like
        Indices of ICA components to be modified.
    include_ica : bool, optional
        If True, returns tuple of (data with unmodified components, data with modified components) as Raw objects (default is False).
    include : array-like or None, optional
        ICA components to explicitly include (default None).
    exclude : array-like or None, optional
        ICA components to exclude (default None).
    n_pca_components : int or None, optional
        Number of PCA components to use (default None).
    keep_all : bool, optional
        If True, keep all PCA components during reconstruction (default False).

    Returns
    -------
    raw_modified : mne.io.Raw or tuple of (mne.io.Raw, mne.io.Raw)
        If `include_ica` is False, returns the Raw object with modified data.
        If `include_ica` is True, returns a tuple of two Raw objects:
            - Raw with unmodified components projected back
            - Raw with modified components projected back
    """
        """Aux method."""
        # _check_preload(raw, "ica.apply")

        # start, stop = _check_start_stop(raw, start, stop)
        from mne._fiff.pick import pick_types

        picks = pick_types(
            raw.info, meg=False, include=ica.ch_names, exclude="bads", ref_meg=False
        )

        data = raw[picks, :][0]
        
        if include_ica :
            data, proj_back = modify_sources(ica,data, modifications, modifications_index, include_ica, include, exclude, n_pca_components, keep_all=keep_all )
            
            simple = raw.copy()
            simple[picks, :] = data

            corrected = raw.copy()
            corrected[picks, :] = proj_back

            return simple, corrected
        else:
            proj_back = modify_sources(ica,data, modifications, modifications_index, include_ica, include, exclude, n_pca_components, keep_all=keep_all )

            raw[picks, :] = proj_back

            return raw


###############################################################################################################
#
#                                               EMG
#
###############################################################################################################

def get_EMG_mask(data, sampling_rate = 500, sliding_rms_window = 1, n_deviation = 5) :
    """
    Detect EMG artifact mask in the data using RMS thresholding on high-pass filtered signals.

    Parameters
    ----------
    data : ndarray, shape (n_samples,)
        Raw signal data.
    sampling_rate : int, optional
        Sampling frequency of the data in Hz (default is 500).
    sliding_rms_window : float, optional
        Window length in seconds for sliding RMS calculation (default is 1).
    n_deviation : float, optional
        Number of median absolute deviations above median to set threshold (default is 5).

    Returns
    -------
    mask : ndarray of bool, shape (n_samples,)
        Boolean mask where True indicates detected EMG artifact.
    """
    data_rms = rolling_rms(mne.filter.filter_data(data, sampling_rate, 30,None, verbose=False), window = sliding_rms_window)
    threshold = np.median(data_rms) + n_deviation*MAD(data_rms)
    mask = data_rms > threshold
    return mask

def spectrum_based_noise(data, length, sampling_rate = 500, fmin= 0.05, seed = 42):
    """
    Generate noise matching the amplitude spectrum of a signal for noise patching.

    Parameters
    ----------
    data : ndarray, shape (n_samples,)
        Signal to estimate the power spectrum from.
    length : int
        Length of the noise to generate (in samples).
    sampling_rate : int, optional
        Sampling frequency in Hz (default 500).
    fmin : float, optional
        High-pass filter cutoff frequency in Hz (default 0.05).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    generated_noise : ndarray, shape (length,)
        Noise signal shaped to match the input signal's spectrum, filtered above `fmin`.
    """

    # estimate psd sig
    _, spectrum = signal.welch(data, nperseg=length, nfft=length, noverlap=0, scaling='spectrum', window='box', return_onesided=False, average='median')

    spectrum = np.sqrt(spectrum)

    # pregenerate long noise piece
    rng = np.random.RandomState(seed=seed)

    long_noise = rng.randn(length)
    noise_F = np.fft.fft(long_noise)
    #long_noise = np.fft.ifft(np.abs(noise_F) * spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = np.fft.ifft(spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = long_noise.astype(data.dtype)
    sos = signal.iirfilter(2, fmin / (sampling_rate / 2), analog=False, btype='highpass', ftype='bessel', output='sos')
    long_noise = signal.sosfiltfilt(sos, long_noise, axis=0)

    filtered_sig = signal.sosfiltfilt(sos, data, axis=0)
    rms_sig = np.median(filtered_sig**2)
    rms_noise = np.median(long_noise**2)
    factor = np.sqrt(rms_sig) / np.sqrt(rms_noise)
    generated_noise = long_noise*factor

    return generated_noise

def EMG_noise_patching(data, sampling_rate=500, margin=0.2, sliding_rms_window = 1, n_deviation = 5, fmin = 0.05, seed = 42):
    """
    Detect EMG artifacts and replace artifact segments with noise patches shaped to the clean signal spectrum.

    Parameters
    ----------
    data : ndarray, shape (n_samples,)
        Raw signal data.
    sampling_rate : int, optional
        Sampling frequency in Hz (default 500).
    margin : float, optional
        Margin (in seconds) to widen the detected artifact mask for smoothing (default 0.2).
    sliding_rms_window : float, optional
        Window length in seconds for RMS calculation (default 1).
    n_deviation : float, optional
        Number of MAD deviations above median to detect artifacts (default 5).
    fmin : float, optional
        High-pass filter cutoff frequency in Hz used in noise generation (default 0.05).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    corrected_signal : ndarray, shape (n_samples,)
        Signal with EMG artifact segments replaced by noise patches.
    mask_widened : ndarray of bool, shape (n_samples,)
        Boolean mask indicating widened artifact regions after margin expansion.
    """
    
    mask = get_EMG_mask(data, sampling_rate = sampling_rate, sliding_rms_window = sliding_rms_window, n_deviation = n_deviation)

    if np.sum(mask)==0:
        print('No artifacts detected')
        return data, mask
    
    else:
    
        width = int(margin*sampling_rate)
        mask_widened = mask.values + (pd.Series(np.pad(mask, width, mode='reflect')).rolling(width, center=True).mean()[width:-width].astype(bool))

        mask_widened_2 = mask_widened.values.astype(float)
        signal_ease_in = np.ones_like(mask) - mask_widened_2

        n = int(width/2)+1

        from scipy.signal.windows import blackman
        ease_in = np.array(blackman(width-1)[:n])
        ease_out = np.flip(ease_in)

        windows = find_windows(mask_widened,1, integer=True)

        for w in windows:
            
            easing_length = len(signal_ease_in[w[0]:w[0]+n])
            if easing_length > 0 :
                signal_ease_in[w[0]:w[0]+n] = ease_out[:easing_length]
                mask_widened_2[w[0]:w[0]+n] = mask_widened_2[w[0]:w[0]+n] * ease_in[:easing_length]

            easing_length = len(signal_ease_in[w[1]-n+1:w[1]+1])
            if easing_length > 0 :
                signal_ease_in[w[1]-n+1:w[1]+1] = ease_in[:easing_length]     
                mask_widened_2[w[1]-n+1:w[1]+1] = mask_widened_2[w[1]-n+1:w[1]+1] * ease_out[:easing_length]
        
        noise_size = int(np.sum([w[1]-w[0] for w in windows]) )

        sig = data[~mask]

        generated = spectrum_based_noise(data=sig, sampling_rate=sampling_rate, length = noise_size, fmin= fmin, seed = seed)
        

        slicing_noise = np.cumsum(np.concatenate([[0],np.array([w[1]-w[0] for w in windows])]))

        masked_noise = np.zeros(mask_widened.shape[0])
        slicing_mask = windows
        for i in range(len(slicing_noise)-1):
            masked_noise[slicing_mask[i][0] : slicing_mask[i][1]] = generated[slicing_noise[i]:slicing_noise[i+1]]

        masked_noise = masked_noise * mask_widened_2

        corrected_signal = data*signal_ease_in  + masked_noise
        

        return corrected_signal, mask_widened
    


###############################################################################################################
#
#                                               SSA-ICA - Redresseur V2
#
###############################################################################################################


def iterative_localisation(data, wavelet_power, wavelets_frequencies, wavelet_detection_iteration = 6, filter_f_min=1, filter_f_max=None, 
                           metric_function='MAD', metric_factor=3, sampling_rate= 500, show = False):
    
    """
    Perform iterative localization of signal components based on wavelet power thresholds.

    The function filters the signal, then iteratively detects artifacts on signal portion, 
    compute a template pseudo-artifact, computes wavelet transforms of these artifacts, 
    and updates a localization mask  based on wavelet power thresholds computed from the pseudo-artifact

    Parameters
    ----------
    data : ndarray, shape (n_samples,)
        Raw input signal.
    wavelet_power : ndarray, shape (n_frequencies, n_samples)
        Precomputed wavelet power of the signal.
    wavelets_frequencies : ndarray, shape (n_frequencies,)
        Frequencies corresponding to the wavelet transform.
    wavelet_detection_iteration : int, optional
        Number of iterations for localization refinement (default 6).
    filter_f_min : float, optional
        Minimum frequency for bandpass filter (default 1 Hz).
    filter_f_max : float or None, optional
        Maximum frequency for bandpass filter (default None, no upper limit).
    metric_function : str, optional
        Metric used for pseudo-artifact detection, e.g., 'MAD' (default).
    metric_factor : float, optional
        Threshold factor for the metric (default 3).
    sampling_rate : int, optional
        Sampling frequency of the signal in Hz (default 500).
    show : bool, optional
        Whether to display histogram of wavelet power thresholds (default False).

    Returns
    -------
    localisation : ndarray of bool, shape (n_samples,)
        Boolean mask indicating localized signal components.
    """
    
    signal_to_analyse = mne.filter.filter_data(data, sampling_rate, filter_f_min, filter_f_max, verbose=False)
    localisation = ~ (np.sum(wavelet_power, axis=0) > -1)
    wavelet_thresholds = []
    for i in range(wavelet_detection_iteration):
        
        pseudo_artifact = get_pseudo_artifact(signal_to_analyse[~localisation], already_filtered=True, metric_function=metric_function, metric_factor=metric_factor)
        w_pseudo = get_univariate_wavelet_transform(pseudo_artifact, wavelets_frequencies, sampling_rate=sampling_rate, n_cycles=wavelet_scaling)
        wavelet_threshold = np.max(np.sum(w_pseudo, axis=0))
        # Get localisation masks
        localisation = np.sum(wavelet_power, axis=0) > wavelet_threshold
        

        wavelet_thresholds.append(wavelet_threshold)
    
    # if show :
        # f = px.histogram(np.sum(wavelet_power, axis=0), log_y=True)
        # colors = ['black', 'purple', 'blue', 'green', 'yellow', 'orange', 'red']
        # for i, thresh in enumerate(wavelet_thresholds):
        #     f.add_vline(x = thresh, line_color = colors[i%len(colors)])
        # f.show()

    return localisation




def upsample(data, reference, pad = True):
    """
    Upsample a signal to match the length of a reference signal, with optional padding.

    Parameters
    ----------
    data : ndarray, shape (n_samples,)
        Signal to upsample.
    reference : ndarray, shape (m_samples,)
        Reference signal whose length is the target upsampling length.
    pad : bool, optional
        Whether to pad both signals by reflection before resampling to reduce edge artifacts (default True).

    Returns
    -------
    upsampled : ndarray, shape (len(reference),)
        Upsampled signal matching the length of the reference.
    """
    if pad :
        padded = np.pad(data, len(data), mode='reflect')
        reference_padded = np.pad(reference, len(reference), mode='reflect')
        return resample(padded, len(reference_padded))[len(reference):-len(reference)]
    else:
        return resample(data, len(reference))


def rectify_saccades(univariate_signal, cluster_windows, correction, show=False, return_corrected = False):
    """
    Rectify saccade-related artifacts in a univariate signal based on cluster windows and correction signals.

    The function applies additive corrections over labeled windows, easing transitions between windows,
    and optionally plots the rectification and returns the corrected signal.

    Parameters
    ----------
    univariate_signal : ndarray, shape (n_samples,)
        Input signal to be corrected.
    cluster_windows : list of tuples
        List of tuples (start_idx, end_idx, label) defining segments for rectification.
        Labels indicate correction type.
    correction : ndarray or list of ndarray
        Correction signals applied to saccade segments.
        If a single array, applied for labels 1 and 3.
        If list, the first is for labels 1 and 3, second for others.
    show : bool, optional
        Whether to plot the original, rectified, and difference signals (default False).
    return_corrected : bool, optional
        If True, return the corrected signal (original - rectification).
        Otherwise, return only the rectification (default False).

    Returns
    -------
    result : ndarray, shape (n_samples,)
        Either the rectification signal or the corrected signal depending on `return_corrected`.
    """

    # Rectify saccades
    rectification = np.zeros_like(univariate_signal)
    # mask = np.zeros_like(univariate_signal)
    last_value = 0
    previous = 0
    if type(correction) != list:
        corrections = [correction, correction]
    else:
        corrections = correction

    for i, (w1, w2, label) in enumerate(cluster_windows):
        if int(label) :
            # mask[w1:w2] = int(label)
            if int(label)==3 or int(label)==1:
                rectification[w1:w2] = corrections[0][w1:w2] - corrections[0][w1] + last_value
            else:
                rectification[w1:w2] = corrections[1][w1:w2] -corrections[1][w1] + last_value
                rectification[w2:] = -previous
        else:
            if len(univariate_signal[w1:w2])> 0:
                previous = np.mean(univariate_signal[w1:w2])
        
        last_value = rectification[w2]

    for w1, w2, label in cluster_windows:
            if int(label) :
                rectification[w2:] = rectification[w2:]  - (rectification[w2]-rectification[w2-1]) 
            

    if show :
        plot_comparison([univariate_signal, rectification, univariate_signal-rectification])

    if return_corrected :
        return univariate_signal-rectification
    else:
        return rectification




def rectified_V2_CWT_MSSA(data, iterations = 2, unbiasing= True, sampling_rate =500,
                          wavelets_frequency_spacing = 0.1,
                            wavelets_min_frequency = 0.1,
                            wavelets_max_frequency = 12,
                            filter_f_min = 1,
                            filter_f_max = None,
                            metric_function='MAD',
                            metric_factor=[3,5],
                            margins = [500,5000],
                            widths = [50,1000],
                            ssa_window_seconds = 0.3,
                            first_ssa_trajectories_indices = np.arange(7),
                            second_ssa_downsampling = 10,
                            second_ssa_window_seconds = 5,
                            second_ssa_trajectories_indices = np.arange(80),
                            show=False,
                            artifact_margin = 500,
                            artifact_width = 0,
                            artifact_plateau = 350, 
                            artifact_shift = 75, 
                            wavelet_detection_iteration = 6,
                            bridging =0.1,
                            bias_ssa_downsampling = 20, #20
                            bias_ssa_window_seconds = 80 ,#15
                            bias_ssa_trajectories_indices = [4,5,6],
                            cascade = {0:{'indices':0}},
                            rolling_average_duration= 10,
                            verbose = True,
                            return_iterative = False
                          ):
    """
    Perform iterative rectification and artifact removal on univariate time series data 
    using a combination of Continuous Wavelet Transform (CWT), Multistage Singular Spectrum Analysis (MSSA),
    and saccade rectification procedures.

    The method identifies artifacts by iterative wavelet power thresholding and reconstructs clean signals
    by separating slow and fast components via MSSA. Optional unbiasing steps further refine artifact correction.

    Parameters
    ----------
    data : ndarray, shape (n_samples,)
        The input univariate time series signal to be processed.
    iterations : int, optional
        Number of iterative rectification cycles to perform. Default is 2.
    unbiasing : bool, optional
        Whether to perform unbiasing correction steps after initial artifact removal. Default is True.
    sampling_rate : float, optional
        Sampling frequency of the input data in Hz. Default is 500.
    wavelets_frequency_spacing : float, optional
        Frequency spacing for wavelet transform frequency vector in Hz. Default is 0.1.
    wavelets_min_frequency : float, optional
        Minimum frequency for wavelet transform in Hz. Default is 0.1.
    wavelets_max_frequency : float, optional
        Maximum frequency for wavelet transform in Hz. Default is 12.
    filter_f_min : float, optional
        Minimum frequency for optional bandpass filtering before artifact detection. Default is 1.
    filter_f_max : float or None, optional
        Maximum frequency for optional bandpass filtering. If None, no upper limit applied. Default is None.
    metric_function : str, optional
        Metric function name for artifact detection thresholding (e.g., 'MAD'). Default is 'MAD'.
    metric_factor : float or list of floats, optional
        Factor(s) to multiply the metric threshold for artifact detection. If list, uses per iteration. Default is [3, 5].
    margins : list of int, optional
        Margin sizes in samples for smoothing localization mask edges. Default is [500, 5000].
    widths : list of int, optional
        Widths in samples for smoothing localization mask edges. Default is [50, 1000].
    ssa_window_seconds : float, optional
        Window length in seconds for the first stage SSA. Default is 0.3.
    first_ssa_trajectories_indices : array-like, optional
        Indices of SSA trajectories used in the first SSA stage. Default is np.arange(7).
    second_ssa_downsampling : int, optional
        Downsampling factor for the second SSA stage. Default is 10.
    second_ssa_window_seconds : float, optional
        Window length in seconds for the second SSA stage. Default is 5.
    second_ssa_trajectories_indices : array-like, optional
        Indices of SSA trajectories used in the second SSA stage. Default is np.arange(80).
    show : bool, optional
        Whether to show intermediate plots during processing. Default is False.
    artifact_margin : int, optional
        Margin size in samples for artifact detection windows. Default is 500.
    artifact_width : int, optional
        Width in samples for artifact plateau. Default is 0.
    artifact_plateau : int, optional
        Plateau width in samples for artifact detection. Default is 350.
    artifact_shift : int, optional
        Shift in samples for artifact alignment. Default is 75.
    wavelet_detection_iteration : int, optional
        Number of iterations for wavelet-based artifact detection threshold refinement. Default is 6.
    bridging : float, optional
        Bridging parameter for artifact clustering. Default is 0.1.
    bias_ssa_downsampling : int, optional
        Downsampling factor for SSA bias correction stage. Default is 20.
    bias_ssa_window_seconds : float, optional
        Window length in seconds for bias SSA. Default is 80.
    bias_ssa_trajectories_indices : list, optional
        SSA trajectory indices for bias correction. Default is [4, 5, 6].
    cascade : dict, optional
        Configuration dictionary for multistage SSA processing. Default is {0: {'indices': 0}}.
    rolling_average_duration : float, optional
        Duration in seconds for rolling mean smoothing of final rectified signal. Default is 10.
    verbose : bool, optional
        If True, print progress and debugging information. Default is True.
    return_iterative : bool, optional
        If True, return list of intermediate corrected signals for each iteration, else return final corrected signal. Default is False.

    Returns
    -------
    corrected_signal : ndarray, shape (n_samples,)
        The artifact  signal if `return_iterative` is False.
    or
    iterative_corrections : list of ndarray
        List of rectified signals from each iteration if `return_iterative` is True.

    Notes
    -----

    The algorithm is designed for EEG or similar biosignals contaminated by artifacts (e.g., ocular movements).

    """
    
    if verbose :
        print('Starting computation')

    univariate_signal = data.copy()
    localisation_original = np.zeros_like(data)
    iterative_corrections = []

    wavelets_frequencies = np.linspace(wavelets_min_frequency,wavelets_max_frequency,int((wavelets_max_frequency-wavelets_min_frequency)/wavelets_frequency_spacing)+1)
    wavelet_power = get_univariate_wavelet_transform(univariate_signal, wavelets_frequencies, sampling_rate=sampling_rate, n_cycles=wavelet_scaling)

    if verbose :
        print('Wavelet power computed')

    if type(metric_factor)==list:
        factor = metric_factor[0]
    else:
        factor = metric_factor

    localisation = iterative_localisation(univariate_signal, wavelet_power, wavelets_frequencies,
                                              wavelet_detection_iteration = wavelet_detection_iteration, filter_f_min=filter_f_min, filter_f_max=filter_f_max, 
                           metric_function=metric_function, metric_factor=factor, sampling_rate= sampling_rate, show = show)
    
    if not np.sum(localisation) == 0 :



        trajectories = multistage_ssa(univariate_signal, cascade=cascade, windows = [ssa_window_seconds,second_ssa_window_seconds], 
                                    indices = [first_ssa_trajectories_indices, second_ssa_trajectories_indices], downsamplings = [1,second_ssa_downsampling])
        
        if verbose :
            print('Tracjectories computed')

        

        for k in range(iterations):
            print(f'Iteration {k}')

            if type(metric_factor)==list:
                if k>=len(metric_factor):
                    print('Not enough metric factors provided, last one used is chosen instead')
                    factor = metric_factor[-1]
                else:
                    factor = metric_factor[k]
            else:
                factor = metric_factor
            
            if k > 0 :
                localisation = iterative_localisation(univariate_signal, wavelet_power, wavelets_frequencies,
                                                    wavelet_detection_iteration = wavelet_detection_iteration, filter_f_min=filter_f_min, filter_f_max=filter_f_max, 
                                metric_function=metric_function, metric_factor=factor, sampling_rate= sampling_rate, show = show)
            
            if np.sum(localisation) == 0 :
                print('No new artifact detected')
                break
            
            localisation = (localisation.astype(bool) + localisation_original.astype(bool)).astype(int)
            
            smooth_localisations = gaussianize_mask_edges(localisation, margins, widths)
            smooth_localisation = smooth_localisations[0]
            large_smooth_localisation = smooth_localisations[1]

            if unbiasing :

                cluster_windows = get_eog_classification(data, localisation, fixed_length_windows=0,
                                                    bridging =bridging,  sampling_rate=sampling_rate)

                partial_slow = trajectories[1][1:] * large_smooth_localisation[::second_ssa_downsampling]
                fast = resample(np.sum(partial_slow, axis=0), univariate_signal.shape[0])
                fast +=  np.sum(trajectories[0][1:] * smooth_localisation, axis=0)

                # if show :
                #     plot_comparison([data, data-fast], title = 'Fast first SSA')

                bias = rectify_saccades(data, cluster_windows, [ fast, data], show=show, return_corrected=True)

                unbias = ssa_trajectories(bias, window = bias_ssa_window_seconds,
                                    indices = bias_ssa_trajectories_indices, 
                                    downsample = bias_ssa_downsampling, summation = 1)

                unbiased = upsample(unbias, data)

                # if show:
                #     plot_comparison([data, unbiased])

                if verbose :
                    print('Bias computed')

                ## Artifact signal reconstruction
                slow_bias =  resample(trajectories[1][0], univariate_signal.shape[0])
                unbias = unbiased-slow_bias

                partial_slow = trajectories[1][1:] * large_smooth_localisation[::second_ssa_downsampling]
                fast = resample(np.sum(partial_slow, axis=0), univariate_signal.shape[0])
                unbiased_fast = (fast-unbias)
                unbiased_fast += np.sum(trajectories[0][1:] * smooth_localisation, axis=0)
            
            else:
                partial_slow = trajectories[1][1:] * large_smooth_localisation[::second_ssa_downsampling]
                unbiased_fast = resample(np.sum(partial_slow, axis=0), univariate_signal.shape[0])
                unbiased_fast += np.sum(trajectories[0][1:] * smooth_localisation, axis=0)


            if show :
                timings = (np.arange(len(univariate_signal))*1/sampling_rate)
                d = 5
                if unbiasing :
                    comparison_df = pd.concat([
                    pd.DataFrame({'Time' : timings[::d], 'Type' : 'Original', 'Signal' : data[::d]}),
                    pd.DataFrame({'Time' : timings[::d], 'Type' : 'SSA correction', 'Signal' : (data-fast)[::d]}),
                    pd.DataFrame({'Time' : timings[::d], 'Type' : 'SSA correction unbias', 'Signal' : (data-unbiased_fast)[::d]}),
                    ])
                else:
                    comparison_df = pd.concat([
                    pd.DataFrame({'Time' : timings[::d], 'Type' : 'Original', 'Signal' : data[::d]}),
                    pd.DataFrame({'Time' : timings[::d], 'Type' : 'SSA correction unbias', 'Signal' : (data-unbiased_fast)[::d]}),
                    ])

                p1 = np.round(np.sum(smooth_localisation)/len(smooth_localisation)*100,2)
                p2 = np.round(np.sum(large_smooth_localisation)/len(large_smooth_localisation)*100,2)
                f = px.line(comparison_df, x = 'Time', y= 'Signal', color= 'Type', 
                    title = f'Local artefact : {p1} % | Global correction :{p2} % ')
                f.show()

            cluster_windows_sharp = get_eog_classification(data, localisation, fixed_length_windows=1,
                                                    artifact_margin = artifact_margin, artifact_width = artifact_width,
                                                        artifact_plateau = artifact_plateau, artifact_shift = artifact_shift,
                                                bridging =bridging,  sampling_rate=sampling_rate)
            
            rectification = rectify_saccades(data, cluster_windows_sharp, unbiased_fast, show=show, return_corrected=False)

            n_points = int(rolling_average_duration*500)
            rectified = data-rectification
            rolling_mean = pd.Series(np.pad(rectified, n_points, mode='reflect')).rolling(n_points, center=True).mean()[n_points:-n_points].values
            rectified = rectified - rolling_mean
            iterative_corrections.append(rectified)

            if k< iterations-1 :
                localisation_original = localisation.copy()
                univariate_signal = rectified
                wavelet_power = get_univariate_wavelet_transform(univariate_signal, wavelets_frequencies, sampling_rate=sampling_rate, n_cycles=wavelet_scaling)
            if verbose :
                print(f'Rectification {k} computed')

    if not iterative_corrections:
        result = np.zeros_like(data)
    else:
        result = data-rectified

    if return_iterative :
        return iterative_corrections
    else:
        return result


###############################################################################################################
#
#                                               SSA BIG
#
###############################################################################################################

def proportionnal_correction(artifact, trajectories, threshold, threshold_diviser = 5, rolling_window = 3):
    """
    Perform proportional correction of artifact components based on SSA trajectories.

    This function weights SSA trajectories by the magnitude of the artifact relative to a scaled threshold,
    optionally smoothed by a rolling mean, then sums the weighted trajectories to reconstruct the corrected signal.

    Parameters
    ----------
    artifact : ndarray
        1D array representing the artifact segment to correct.
    trajectories : ndarray
        2D array of SSA trajectories/components (shape: n_components x signal_length).
    threshold : float
        Threshold value used to scale the artifact amplitude for masking.
    threshold_diviser : float, optional
        Divisor for threshold to moderate sensitivity, by default 5.
    rolling_window : int or bool, optional
        Window size for rolling mean smoothing applied to scaled artifact amplitude.
        If 0 or False, no rolling smoothing is applied, by default 3.

    Returns
    -------
    ndarray
        1D array representing the sum of masked SSA trajectories weighted by artifact magnitude.

    Notes
    -----
    - Clipping is applied to ensure masking indices do not exceed the number of SSA components.
    - The masking is done by comparing the smoothed artifact magnitude to the SSA component index.
    """
    if rolling_window:
        sum_details = np.clip(rolling_mean(np.abs((artifact/(threshold/threshold_diviser))), rolling_window).astype(int), None, len(trajectories))
    else:
        sum_details = np.clip(np.abs((artifact/(threshold/threshold_diviser))).astype(int), None, len(trajectories))
        
    masked_trajectories = trajectories.copy()
    for i in range(len(masked_trajectories)):
        masked_trajectories[i] = masked_trajectories[i] * (sum_details >= i)
        
    return np.sum(masked_trajectories, axis=0)

def big_artefact_correction_ssa(data, mad_factor = 5, sampling_rate=500):
    """
    Correct large amplitude artifacts in a univariate EEG signal using SSA-based component subtraction.

    The method detects artifact windows based on rolling MAD thresholds, applies SSA to artifact segments,
    performs proportional correction, and then smooths the transitions using Gaussian masks.
    This two-stage process aims to reduce high-amplitude artifacts.

    Parameters
    ----------
    data : ndarray
        1D EEG signal array to be corrected.
    mad_factor : float, optional
        Multiplicative factor of the Median Absolute Deviation (MAD) to set artifact detection threshold, by default 5.
    sampling_rate : int, optional
        Sampling rate of the EEG signal in Hz, by default 500.

    Returns
    -------
    ndarray
        Corrected EEG signal with reduced large amplitude artifacts.

    Notes
    -----
    - Uses rolling peak-to-peak amplitude (np.ptp) over a 3-sample window to detect artifacts.
    - Applies SSA trajectory decomposition to artifact windows for signal reconstruction.
    - Two iterative correction passes with increasingly strict thresholds.
    - Blends corrected and original signals smoothly via Gaussian masks.
    - Requires external functions: MAD, rolling_function, find_windows, gaussianize_mask_edges, ssa_trajectories, proportionnal_correction.
    """

    T = mad_factor*MAD(data)
    criterium = rolling_function(np.abs(data), 3, np.ptp , sampling_rate=sampling_rate) > T
    w = find_windows(criterium, integer=True)
    smooth = gaussianize_mask_edges(data, [1000], [200], w)[0]
    w = find_windows(smooth > 1e-3, integer=True)

    if len(w) > 0 :
        data_corrected = data.copy()
        for window in w :
            artifact = data[slice(*window)]
            traj = ssa_trajectories(artifact, np.min([0.5, len(artifact)/sampling_rate]), indices=np.arange(20))
            data_corrected[slice(*window)] = artifact - proportionnal_correction(artifact, traj, T)
        
        data_corrected = data_corrected*smooth + data*(1-smooth)

        Tbis = 20*MAD(data_corrected)
        critbis = np.abs(data_corrected) > Tbis

        w_2 = find_windows(critbis, integer=True)
        smooth_2 = gaussianize_mask_edges(data, [50],[10],w_2)[0]
        w_2 = find_windows(smooth_2 > 1e-3, integer=True)

        if len(w_2) > 0:

            data_corrected_2 = data_corrected.copy()
            for window in w_2 :
                art = data_corrected[slice(*window)]
                traj = ssa_trajectories(art, np.min([0.5, len(art)/sampling_rate]), indices=[])
                data_corrected_2[slice(*window)] = art - proportionnal_correction(art, traj, Tbis, 2, 0)

            data_corrected_2 = data_corrected_2*smooth_2 + data_corrected*(1-smooth_2)

            return data_corrected_2
        
        else:
            return data_corrected

    else:
        return data


if __name__=='__main__':

    test_CWT_MSSA = 0
    test_CWT_MSSA_V2 = 0


    if test_CWT_MSSA:
        run_key = 'sub-45_Distraction'
        from params import *
        from modular_eeg_preprocess import compute_modular_eeg_preprocessing
        p = dict(
            rescaling = {'rescaling_factor' : 0.0488281},
            trimming = conservative_trimming, 
            detrending = highpass_detrending_slow,
            centering = {'method' : 'mean_centering'},
            notch_filtering = notch_filtering_params,
            filtering = filtering_params_slow, # dict(lowpass = None, highpass = None, bandpass = None, notch = None),
                )
        ds = compute_modular_eeg_preprocessing(run_key=run_key, **p)
        sampling_rate = int(ds.sampling_rate)
        channels = ds.channel.values

        raw = dataset_to_mne(ds)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        for i in range(len(raw.get_data())):
            print(i)
            data = raw.get_data()[i,:]

            _ = CWT_MSSA(data , sampling_rate,
                  wavelets_frequency_spacing = 0.1, wavelets_min_frequency = 0.1, wavelets_max_frequency = 12,
                  filter_f_min = 1, filter_f_max = None, metric_function=MAD, metric_factor=5,
                  margins = [500,5000], widths = [50, 1000], ssa_window_seconds = 0.3,
                  first_ssa_trajectories_indices = np.arange(6), second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = np.arange(50),
                  return_unbias = True, return_corrected = False, show=True, save = run_key+'_'+channels[i] , details= True,
                  )

    if test_CWT_MSSA_V2 == 1:
        from params import *
        from modular_eeg_preprocess import compute_modular_eeg_preprocessing
        run_key = 'sub-02_E2'
        p = dict(
        rescaling = {'rescaling_factor' : 0.0488281},
        trimming = conservative_trimming, 
        detrending = highpass_detrending_slow,
        centering = {'method' : 'mean_centering'},
        # notch_filtering = notch_filtering_params,
        # filtering = filtering_params, # dict(lowpass = None, highpass = None, bandpass = None, notch = None),

            )
        ds = compute_modular_eeg_preprocessing(run_key=run_key, **p)
        sampling_rate = int(ds.sampling_rate)
        channel = 0
        raw = dataset_to_mne(ds)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        # raw.pick('Fp1')

        # data0 = raw.get_data()[0,:]
        # data = raw.filter(1,None).get_data()[0,:]

        univariate_signal = raw.get_data()[channel,:]

        correction = rectified_V2_CWT_MSSA(univariate_signal, iterations = 1, verbose=True)

        plot_comparison([univariate_signal, correction])



