"""
This script provides a comprehensive implementation of Singular Spectrum Analysis (SSA)
and its multivariate extension (MSSA) for time series analysis and signal processing.
It includes core SSA operations, weighted correlation metrics, and specialized functions
for artifact detection and correction in EEG data.

KEY FEATURES:
-------------
1. CORE SSA FUNCTIONS:
   - Embedding: Convert time series to trajectory matrices
   - Diagonal averaging: Reconstruct time series from trajectory matrices
   - SVD decomposition: Extract SSA components
   - Multi-stage SSA: Sequential decomposition with cascading

2. WEIGHTED CORRELATION METRICS:
   - Weight vector computation for SSA diagonals
   - Weighted inner products, norms, and correlations
   - Correlation matrix computation (single and batch modes)

3. MULTIVARIATE SSA (MSSA):
   - Concatenated embedding for multiple channels
   - Channel-wise trajectory reconstruction
   - Channel similarity analysis

4. PERFORMANCE OPTIMIZATIONS:
   - Numba-accelerated diagonal averaging
   - Optional GPU acceleration for SVD computations
   - Batch processing for correlation matrices

5. VISUALIZATION:
   - Interactive correlation matrix heatmaps
   - Channel separation markers for multivariate data

"""


# Base imports
import numpy as np
import plotly.express as px

# Specific imports used several times
from numba import njit, prange
from scipy.linalg import svd

gpu_on= False

###############################################################################################################
#
#                                                      SSA 
#
###############################################################################################################


def embedding(x, window_length):
    """
    Construct a trajectory (Hankel) matrix from a 1D time series using windowed embedding.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input 1D time series signal.

    window_length : int
        The length of the embedding window. Determines the number of rows
        (i.e., time-delay dimensions) in the resulting Hankel matrix.

    Returns
    -------
    multivariate_matrix : ndarray of shape (window_length, n_samples - window_length + 1)
        The trajectory matrix, where each row is a lagged version of the original
        signal. This matrix is often used in Singular Spectrum Analysis (SSA) and
        other time series decomposition methods.

    Notes
    -----
    The resulting matrix is also referred to as a Hankel matrix, where each column 
    represents a lagged window of the original signal. This is a key step in SSA, 
    allowing the analysis of temporal structure in time series.
    
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> embedding(x, window_length=3)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    """

    W = int(window_length)
    N = x.shape[0]
    K = N - W + 1 # Number of columns in the Hankel matrix

    # Initialize the Hankel matrix with shape (window_length, K)
    multivariate_matrix = np.empty((W, K))

    # Populate each row with shifted views of the signal
    for i in range(W):
        multivariate_matrix[i,:] = x[i:i+K]

    return multivariate_matrix

def diagonal_averaging_old(k_multivariate_matrix):
    """
    Perform diagonal averaging (Hankelization) of a matrix or tensor.

    Parameters
    ----------
    k_multivariate_matrix : ndarray
        2D or 3D array. If 3D, averaging is applied channel-wise.

    Returns
    -------
    univariate_matrix : ndarray
        Reconstructed time series (1D or 2D).
    """

    if len(k_multivariate_matrix.shape)< 3:
        n = k_multivariate_matrix.shape[0] + k_multivariate_matrix.shape[1] -1
        univariate_matrix = np.empty((n))
        b = np.flip(k_multivariate_matrix[:,:], axis = 1)

        # Loop over diagonals
        for i in range(-k_multivariate_matrix.shape[0]+1, k_multivariate_matrix.shape[1]):
            univariate_matrix[-(i -(-k_multivariate_matrix.shape[0]+1))-1] = np.mean(np.diag(b, i))
    
    else:
        n = k_multivariate_matrix.shape[0] + k_multivariate_matrix.shape[1] -1
        l = k_multivariate_matrix.shape[2]
        univariate_matrix = np.empty((n, l))

        # Loop over channels and diagonals
        for j in range(l):
            b = np.flip(k_multivariate_matrix[:,:,j], axis = 1)
            for i in range(-k_multivariate_matrix.shape[0]+1, k_multivariate_matrix.shape[1]):
                univariate_matrix[-(i -(-k_multivariate_matrix.shape[0]+1))-1, j] = np.mean(np.diag(b, i))
    
    return univariate_matrix


@njit(parallel=True)
def hankelize_numba(X, out):
    """
    Perform diagonal averaging (Hankelization) of a trajectory matrix using Numba for fast reconstruction.

    This function reconstructs the original time series from its trajectory matrix
    by averaging along the anti-diagonals (constant-sum indices) of the matrix.

    Parameters
    ----------
    X : ndarray of shape (L, K)
        Trajectory (Hankel) matrix, typically obtained from windowed embedding
        of a time series.

    out : ndarray of shape (L + K - 1,)
        Pre-allocated output array to store the reconstructed 1D signal. 
        Must be of length equal to the sum of the matrix dimensions minus one.

    Notes
    -----
    This is often used as a final step in Singular Spectrum Analysis (SSA)
    to reconstruct the denoised or decomposed time series components.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> X = np.array([[1., 2., 3.],
    ...               [4., 5., 6.]])
    >>> out = np.zeros(4)
    >>> hankelize_numba(X, out)
    >>> out
    array([1. , 3. , 4. , 6. ])
    """

    L, K = X.shape[:2]
    N = L + K - 1 # Total length of reconstructed signal

    # Loop over each anti-diagonal index t
    for t in prange(N):
        # Compute valid index range for row (i)
        i_min = 0 if t < (K - 1) else t - (K - 1)
        i_max = t if t < L else (L - 1)


        s = 0.0 # Accumulator for summing elements on diagonal
        cnt = 0 # Counter for number of elements in the diagonal

        # Sum over all (i, j) such that i + j == t
        for i in range(i_min, i_max + 1):
            j = t - i
            s   += X[i, j]
            cnt += 1

         # Store the average in the output buffer
        out[t] = s / cnt

def diagonal_averaging(X):
    """
    Reconstruct time series from trajectory matrix using fast diagonal averaging. Wrapper for hankelize_numba.

    Parameters
    ----------
    X : ndarray of shape (L, K)
        Trajectory matrix.

    Returns
    -------
    out : ndarray of shape (L + K - 1,)
        Reconstructed time series.
    """

    L, K = X.shape[:2]
    N = L + K - 1
    out = np.empty(N)
    hankelize_numba(X,out)
    return out

def get_ssa_trajectories(embedded, trajectory_indices, precomputed_svd = None, return_svd = False, gpu=gpu_on, verbose=False ):
    """
    Extract SSA components (trajectories) from a trajectory matrix via SVD.

    This function performs Singular Spectrum Analysis (SSA) by computing or
    using a precomputed Singular Value Decomposition (SVD) of the trajectory matrix.
    It reconstructs specified components via rank-1 outer products and applies
    diagonal averaging to convert them back to time series form.

    Parameters
    ----------
    embedded : ndarray of shape (L, K)
        The trajectory matrix obtained from time series embedding.

    trajectory_indices : list of int
        Indices of SSA components (singular vectors) to extract and reconstruct.
        If an empty list is passed, all components will be used.

    precomputed_svd : tuple of (U, S, V), optional
        Optional precomputed SVD tuple:
        - U : ndarray of shape (L, L)
        - S : ndarray of shape (min(L, K),)
        - V : ndarray of shape (K, K)
        If provided, SVD computation is skipped.

    return_svd : bool, default=False
        Whether to return the computed SVD tuple along with trajectories.

    gpu : bool, default=gpu_on
        If True, use GPU acceleration with CuPy to compute the SVD.

    verbose : bool, default=False
        If True, print the index of each component being processed.

    Returns
    -------
    trajectories : ndarray of shape (len(trajectory_indices), L + K - 1)
        Array of reconstructed SSA components after diagonal averaging.

    computed_svd : tuple of (U, S, V), optional
        Returned only if `return_svd=True`. The computed or reused SVD components.

    Notes
    -----
    Each component is reconstructed as:

        s_i * np.outer(u[:, i], v[i, :])

    followed by diagonal averaging to return to the time domain.

    Examples
    --------
    >>> X = embedding(time_series, window_length=50)
    >>> components = get_ssa_trajectories(X, trajectory_indices=[0, 1])
    """

    # If SVD is not provided, compute it (optionally using GPU)
    if precomputed_svd is None: 
        if gpu:
            # Transfer data to GPU and perform SVD
            embedded_gpu = cp.asarray(embedded)
            u, s, v = cp.linalg.svd(embedded_gpu, full_matrices=False)
            # Transfer back to CPU
            u = u.get()
            s = s.get()
            v = v.get()
        else:
            # CPU-based SVD
            u, s, v = svd(embedded, full_matrices = False)
    else:
        # Unpack the precomputed SVD
        u,s,v = precomputed_svd

    
    # Handle case where no specific indices are provided
    if not len(trajectory_indices):
        trajectory_indices = np.arange(np.min(embedded.shape))

    trajectories = []

    # Reconstruct each requested component
    for i in trajectory_indices:
        if verbose:
            print(f"Processing component {i}")
        # Compute rank-1 approximation
        projected = s[i] * np.outer(u[:, i], v[i,:])
        # Apply diagonal averaging to convert back to 1D time series
        trajectories.append(diagonal_averaging(projected))

    trajectories = np.array(trajectories)
    
    # Return reconstructed components and optionally the SVD
    if return_svd:
        computed_svd = (u,s,v)
        return trajectories, computed_svd
    else:
        return trajectories



    

def ssa_trajectories(data, window, indices = [0], sampling_rate=500, downsample = 1, summation = 0, return_svd = False):
    """
    Perform Singular Spectrum Analysis (SSA) on a 1D signal and extract selected components.

    Parameters
    ----------
    data : ndarray of shape (n_samples,)
        Input 1D signal to decompose.
    window : float
        Window length in seconds used for the embedding step.
    indices : list of int, optional
        Indices of SSA components to extract after decomposition (default is [0]).
    sampling_rate : int, optional
        Sampling rate of the input signal in Hz (default is 500).
    downsample : int, optional
        Downsampling factor to apply before SSA (default is 1).
    summation : bool, optional
        If True, sum the selected components into one signal (default is False).
    return_svd : bool, optional
        If True, also return the singular vectors and values (U, S, V) from SSA (default is False).

    Returns
    -------
    trajectories : ndarray of shape (n_components, n_samples)
        The reconstructed SSA components. If `summation` is True, shape is (n_samples,).
    u : ndarray, optional
        Left singular vectors (only if `return_svd` is True).
    s : ndarray, optional
        Singular values (only if `return_svd` is True).
    v : ndarray, optional
        Right singular vectors (only if `return_svd` is True).
    """
    # Compute the number of samples in the embedding window
    samples_count = int(window*sampling_rate/downsample)

    # Embed the downsampled signal into a trajectory matrix
    Y = embedding(data[::downsample], samples_count )

    # Perform SSA decomposition and extract the desired components
    trajectories = get_ssa_trajectories(Y, trajectory_indices=indices, return_svd = return_svd )

    # If return_svd is True, unpack the singular vectors/values
    if return_svd :
        u,s,v = trajectories[1]
        trajectories = trajectories[0]

    # Return summed components or all separately
    if summation :
        if return_svd :
            return np.sum(trajectories, axis=0), u,s,v
        else:
            return np.sum(trajectories, axis=0)
    else:
        if return_svd :
            return trajectories, u,s,v
        else:
            return trajectories



def multistage_ssa(data, cascade, windows , indices, downsamplings = None, sampling_rate=500):
    """
    Perform multistage Singular Spectrum Analysis (SSA) with optional cascading.

    This function runs SSA sequentially across multiple stages, each with its own
    window length, reconstruction indices, and optional downsampling factor.
    The results of one stage can be fed as input to subsequent stages according
    to a user-defined cascade mapping.

    Parameters
    ----------
    data : array_like
        Input time series or multivariate data to be decomposed.
    cascade : dict
        Mapping of stage indices to cascading parameters.
        Each key is an integer stage index, and each value is a dictionary
        that can include:
            - ``'stage'`` : int, optional
                The stage index whose result will be cascaded forward.
                Defaults to the current stage if not provided.
            - ``'indices'`` : {'all', int, list of int}
                Specifies which SSA components to pass forward:
                    * ``'all'`` — sum all components.
                    * int — pass a single component.
                    * list of int — sum the specified components.
    windows : list of int
        List of SSA window lengths, one per stage.
    indices : list of int or list of list of int
        Reconstruction indices for SSA at each stage.
        Can be a single integer or a list of integers per stage.
    downsamplings : list of int, optional
        Downsampling factors for each stage. Defaults to 1 for all stages.
    sampling_rate : int, optional
        Sampling rate of the input data in Hz. Default is 500.

    Returns
    -------
    trajectories : dict of ndarray
        Dictionary where keys are stage indices (int) and values are
        the SSA trajectory matrices for that stage.

    Raises
    ------
    AssertionError
        If `windows` and `indices` lists are not of equal length,
        or if `downsamplings` is provided but not the same length as `windows`.

    Notes
    -----
    The function relies on an external function `ssa_trajectories` to perform
    the SSA decomposition. The cascading mechanism allows iterative processing
    of reconstructed components from previous stages.

    Examples
    --------
    >>> cascade = {1: {'stage': 0, 'indices': [0, 1]}}
    >>> windows = [100, 50]
    >>> indices = [[0, 1, 2], [0, 1]]
    >>> data = np.random.randn(1, 1000)
    >>> trajectories = multistage_ssa(data, cascade, windows, indices)
    >>> trajectories[0].shape
    (3, 1000)

    See Also
    --------
    ssa_trajectories : Performs SSA decomposition for a single stage.
    """

    # Validate that the number of windows matches the number of indices
    assert len(windows)==len(indices), "Arguments 'windows' and 'indices' must be of equal length"

    if downsamplings is not None :
        # Ensure downsampling list length matches windows length
        assert len(windows) == len(downsamplings), "Arguments 'windows' and 'downsamplings' must be of equal length"
    else:
        # If no downsampling factors are given, default to 1 for all stages
        downsamplings = [1]*len(windows)

    n = len(windows) # Number of SSA stages to perform
    trajectories = dict()  # Store SSA results per stage
    data_to_process = data # Data to feed into the current SSA stage

    for i in range(n):
        print(f'Computing stage {i}')

        # Perform SSA for the current stage
        trajectories[i] = ssa_trajectories(data_to_process, 
                                           windows[i], 
                                           indices = indices[i], 
                                           sampling_rate=sampling_rate, 
                                           downsample = downsamplings[i], 
                                           summation = 0)
        
        # Check if this stage has cascade parameters
        if i in cascade :
            params = cascade[i]

            # Determine which stage's result to use for cascading
            if 'stage' in params:
                index = params['stage']
            else:
                index = i
            
            # Determine which components to pass to the next stage
            if params['indices']== 'all':
                data_to_process = np.sum(trajectories[index], axis=0)
            elif type(params['indices']) == int :
                data_to_process = trajectories[index][params['indices']]
            else:
                 data_to_process = np.sum(trajectories[index][params['indices']])
    
    return trajectories

###############################################################################################################
#
#                                                      W correlation 
#
###############################################################################################################

def get_weight_separability(N, L):
    """
    Compute the weight vector used in weighted correlation or separability metrics
    for Singular Spectrum Analysis (SSA) based on data length and window size.

    The weight vector quantifies the number of overlapping elements in the trajectory
    matrix diagonals, which is useful for weighted correlations or reconstruction quality
    assessments.

    Parameters
    ----------
    N : int
        Length of the original time series data.
    L : int
        Window length used in SSA embedding (trajectory matrix size).

    Returns
    -------
    weights : ndarray of shape (N - L + 1,)
        Weight vector where each element corresponds to the number of elements
        overlapping along the respective diagonal of the trajectory matrix.

    Notes
    -----
    The output vector has a trapezoidal shape:
    - Increases linearly from 0 up to the minimum window size.
    - Remains constant for the middle elements.
    - Decreases linearly back down toward 1.

    This weighting is often used to adjust correlations between reconstructed components
    accounting for unequal overlap lengths in the trajectory matrix diagonals.

    Examples
    --------
    >>> get_weight_separability(100, 20)
    array([0, 1, 2, ..., 19, 19, ..., 1])
    """
    # Number of columns in the trajectory matrix
    K = N - L + 1

    # Effective length used for linear segments
    Lstar = min(L, K)

    # Increasing weights from 0 to Lstar-1
    a = np.arange(Lstar)

    # Constant weight segment (flat top of trapezoid)
    b = np.full(K - Lstar + 1, Lstar)

    # Decreasing weights from Lstar-1 down to 2
    c = np.arange(Lstar - 1, 0, -1)

    # Concatenate all segments to form the full weight vector
    return np.concatenate([a, b, c])

def w_inner_product(X1, X2, L, w=None):
    """
    Compute the weighted inner product between two signals using SSA weights.

    The weighted inner product accounts for the varying number of overlapping
    elements along the diagonals of the SSA trajectory matrix by applying
    a weight vector. This is commonly used in SSA to compute weighted correlations
    or distances between reconstructed components.

    Parameters
    ----------
    X1 : array_like, shape (N,)
        First input signal or reconstructed component.
    X2 : array_like, shape (N,)
        Second input signal or reconstructed component. Must be the same length as `X1`.
    L : int
        Window length used in SSA embedding (trajectory matrix size).
    w : array_like, optional, shape (N - L + 1,)
        Weight vector corresponding to the trajectory matrix diagonal overlaps.
        If None, weights are computed internally using `get_weight_separability`.

    Returns
    -------
    float
        Weighted inner product scalar value between the two input signals.

    Raises
    ------
    AssertionError
        If `X1` and `X2` do not have the same length.

    Examples
    --------
    >>> x1 = np.array([1, 2, 3, 4, 5])
    >>> x2 = np.array([5, 4, 3, 2, 1])
    >>> w_inner_product(x1, x2, L=3)
    60
    """
    N = len(X1)

    # Ensure both input signals have the same length
    assert N == len(X2), 'Input signals must have the same length'

    if w is None :
        w = get_weight_separability(N,L)

    return np.sum(w  *X1 * X2)

def w_norm(X1, L, w=None):
    """
    Compute the weighted norm of a signal using SSA weights.

    The weighted norm is defined as the square root of the weighted inner product
    of the signal with itself. Weights account for the overlapping elements
    in the SSA trajectory matrix diagonals, ensuring proper normalization.

    Parameters
    ----------
    X1 : array_like, shape (N,)
        Input signal or reconstructed component.
    L : int
        Window length used in SSA embedding (trajectory matrix size).
    w : array_like, optional, shape (N - L + 1,)
        Weight vector corresponding to the trajectory matrix diagonal overlaps.
        If None, weights are computed internally using `get_weight_separability`.

    Returns
    -------
    float
        Weighted norm of the input signal.
    """
    return np.sqrt(w_inner_product(X1,X1,L, w=w))

def w_correlation(X1,X2,L, w=None):
    """
    Compute the weighted correlation coefficient between two signals using SSA weights.

    The weighted correlation accounts for the varying overlap lengths of the
    SSA trajectory matrix diagonals by applying a weight vector. It is defined as
    the weighted inner product normalized by the weighted norms of each signal.

    Parameters
    ----------
    X1 : array_like, shape (N,)
        First input signal or reconstructed component.
    X2 : array_like, shape (N,)
        Second input signal or reconstructed component. Must be the same length as `X1`.
    L : int
        Window length used in SSA embedding (trajectory matrix size).
    w : array_like, optional, shape (N - L + 1,)
        Weight vector corresponding to the trajectory matrix diagonal overlaps.
        If None, weights are computed internally using `get_weight_separability`.

    Returns
    -------
    float
        Weighted correlation coefficient between the two input signals,
        ranging from -1 to 1.
    """
    r = w_inner_product(X1, X2, L, w)/(w_norm(X1, L, w) *w_norm(X2, L, w))
    return r


def w_correlation_matrix(trajectories, L, multivariate= False, absolute=True,  show=True, **kwargs):
    """
    Compute and optionally visualize the weighted correlation matrix between SSA trajectory components.

    The function calculates the pairwise weighted correlations between trajectory vectors,
    accounting for overlapping window weights in SSA. Supports multivariate input by
    concatenating trajectories from multiple channels. The correlation matrix can
    be displayed as an interactive heatmap with optional absolute values.

    Parameters
    ----------
    trajectories : dict or ndarray
        If `multivariate` is True, a dict of channel trajectories (each is ndarray).
        Otherwise, a single ndarray of trajectories with shape (n_components, length).
    L : int
        Window length used in SSA embedding (trajectory matrix size).
    multivariate : bool, optional, default False
        Whether to treat `trajectories` as multivariate (dict) or univariate (ndarray).
    absolute : bool, optional, default True
        Whether to show the absolute value of the weighted correlation matrix in the visualization.
    show : bool, optional, default True
        Whether to display the interactive heatmap plot.
    **kwargs : keyword arguments
        Additional keyword arguments passed to the plotting function `px.imshow`.

    Returns
    -------
    correlation_matrix : ndarray, shape (n, n)
        Weighted correlation matrix of all trajectory components.
    """
    
    # Concatenate trajectories across channels if multivariate input is provided
    if multivariate:
        trajectory = np.concatenate([np.array(channel_trajectory) for channel_trajectory in trajectories.values()], axis=0)
    else:
        trajectory = trajectories

    n = len(trajectory)
    # Compute weight vector for SSA diagonal overlaps
    w = get_weight_separability(trajectory.shape[1], L)

    correlation_matrix = np.ndarray((n,n))
    # Compute weighted correlations pairwise for each component pair
    for i in range(n):
        # print(i/n)
        for j in range(n):
            correlation_matrix[i,j] = w_correlation(trajectory[i,:], trajectory[j,:], L, w)
    
    if show:
        data = [correlation_matrix, np.abs(correlation_matrix)][absolute]
        color = [px.colors.sequential.RdBu_r, px.colors.sequential.Plasma][absolute]
        zmin = [-1,0][absolute]

        f = px.imshow(data, title='W correlation matrix', aspect='equal', zmin=zmin, zmax=1, color_continuous_scale=color, **kwargs)

        # If multivariate, add dividing lines between channels
        if multivariate:
            c = len(trajectories[list(trajectories.keys())[0]])
            for i in range(1, len(trajectories)):
                f.add_hline(i*c-0.5, line_width=1, line_color='aquamarine', line_dash='dot')
                f.add_vline(i*c-0.5, line_width=1, line_color='aquamarine', line_dash='dot')
        f.show()
    return correlation_matrix

def w_correlation_matrix_batch(trajectories, L, multivariate= False, absolute=True,  show=True, **kwargs):
    """
    Compute and optionally visualize the weighted correlation matrix between SSA trajectory components using batch matrix operations.

    This function efficiently computes the weighted correlation matrix by leveraging
    matrix multiplication and broadcasting, avoiding explicit double loops.
    Supports multivariate input by concatenating trajectories from multiple channels.
    The resulting matrix can be visualized as an interactive heatmap with optional
    absolute values.

    Parameters
    ----------
    trajectories : dict or ndarray
        If `multivariate` is True, a dict of channel trajectories (each an ndarray).
        Otherwise, a single ndarray of trajectories with shape (n_components, length).
    L : int
        Window length used in SSA embedding (trajectory matrix size).
    multivariate : bool, optional, default False
        Whether to treat `trajectories` as multivariate (dict) or univariate (ndarray).
    absolute : bool, optional, default True
        Whether to show the absolute value of the weighted correlation matrix in the visualization.
    show : bool, optional, default True
        Whether to display the interactive heatmap plot.
    **kwargs : keyword arguments
        Additional keyword arguments passed to the plotting function `px.imshow`.

    Returns
    -------
    correlation_matrix : ndarray, shape (n, n)
        Weighted correlation matrix of all trajectory components.
    """

    # Concatenate trajectories across channels if multivariate input is provided
    if multivariate:
        trajectory = np.concatenate([np.array(channel_trajectory) for channel_trajectory in trajectories.values()], axis=0)
    else:
        trajectory = trajectories

    # Number of samples in each trajectory vector
    N = trajectory.shape[1]

    # Compute weight vector for SSA diagonal overlaps
    w = get_weight_separability(N, L)

    # Apply weights element-wise to each trajectory component
    WX = trajectory * w  

    # Compute weighted Gram matrix (weighted inner products between trajectories)
    G  = WX @ trajectory.T 

    # Compute weighted norms from diagonal entries of Gram matrix
    norms = np.sqrt(np.diag(G))

    # Compute denominator matrix for correlation normalization via outer product of norms
    denom = np.outer(norms, norms)

    # Normalize weighted Gram matrix to get weighted correlation matrix
    correlation_matrix = G / denom

    if show:
        data = [correlation_matrix, np.abs(correlation_matrix)][absolute]
        color = [px.colors.sequential.RdBu_r, px.colors.sequential.Plasma][absolute]
        zmin = [-1,0][absolute]

        f = px.imshow(data, title='W correlation matrix', aspect='equal', zmin=zmin, zmax=1, color_continuous_scale=color, **kwargs)

        # If multivariate, add dividing lines between channels
        if multivariate:
            c = len(trajectories[list(trajectories.keys())[0]])
            for i in range(1, len(trajectories)):
                f.add_hline(i*c-0.5, line_width=1, line_color='aquamarine', line_dash='dot')
                f.add_vline(i*c-0.5, line_width=1, line_color='aquamarine', line_dash='dot')
        f.show()

    return correlation_matrix


###############################################################################################################
#
#                                                      MSSA 
#
###############################################################################################################    

def get_mssa_trajectories(concatenated_embedding, channel_number, channel_names=None, trajectory_indices=None, precomputed_svd = None, return_svd = False, gpu=gpu_on, verbose=False):
    """
    Extract MSSA trajectory components from a concatenated embedding matrix using SVD.

    The function computes the singular value decomposition (SVD) of the concatenated embedding
    matrix either on CPU or GPU, then projects onto selected SVD components to obtain
    reconstructed trajectories for each channel. The trajectory is reconstructed by
    applying diagonal averaging to the projected rank-1 approximation for each channel.

    Parameters
    ----------
    concatenated_embedding : ndarray, shape (M, total_length)
        The concatenated embedding matrix from multiple channels (M rows, total length columns).
    channel_number : int
        Number of individual channels concatenated in the embedding matrix.
    channel_names : list or array-like, optional
        Names or labels of each channel. If None, channels are labeled by their integer indices.
    trajectory_indices : list or array-like, optional
        Indices of SVD components to project and reconstruct trajectories from.
        If None, all components are used.
    precomputed_svd : tuple (u, s, v), optional
        Precomputed SVD components to reuse, avoiding recomputation.
        If None, SVD is computed inside the function.
    return_svd : bool, optional, default False
        Whether to return the computed SVD components along with trajectories.
    gpu : bool, optional, default depends on global gpu_on variable
        Whether to perform SVD on GPU using CuPy. If False, CPU NumPy/SciPy SVD is used.
    verbose : bool, optional, default False
        Whether to print progress information for each trajectory index.

    Returns
    -------
    trajectories : dict
        Dictionary mapping channel names to lists of reconstructed trajectory arrays,
        each corresponding to a selected SVD component.
    computed_svd : tuple (u, s, v), optional
        Returned only if `return_svd` is True. Contains the SVD components computed or passed in.

    Notes
    -----
    - Diagonal averaging is applied to each projected rank-1 component to obtain time series.
    - The length of each trajectory component equals the length of the original time series.

    """

    # Compute or reuse SVD of concatenated embedding
    if precomputed_svd is None: 
        if gpu:
            # Transfer data to GPU and compute SVD with CuPy
            embedded_gpu = cp.asarray(concatenated_embedding)
            u, s, v = cp.linalg.svd(embedded_gpu, full_matrices=False)
            # Transfer results back to CPU memory (NumPy arrays)
            u = u.get()
            s = s.get()
            v = v.get()
        else:
            # Compute SVD on CPU using SciPy / NumPy
            u, s, v = svd(concatenated_embedding, full_matrices = False)
    else:
        # Use provided precomputed SVD components
        u,s,v = precomputed_svd
    
    # Use all components if indices not provided
    if trajectory_indices is None:
        trajectory_indices = np.arange(len(u))
    
    # Use numeric channel names if not provided
    if channel_names is None:
        channel_names = np.arange(channel_number)
    
    # Initialize dictionary of trajectory lists per channel
    trajectories = {name : [] for name in channel_names}

    # Length of trajectory segments per channel after concatenation
    K = int(v.shape[1]/channel_number)

    # Reconstruct trajectories by projecting onto selected SVD components
    for i in trajectory_indices:
        if verbose:
            print(i)
        for j, name in enumerate(channel_names) :
            # Compute rank-1 projected matrix for component i and channel j
            projected = s[i] * np.outer(u[:, i], v[i,j*K:(j+1)*K])
            # Apply diagonal averaging to reconstruct the trajectory time series
            trajectories[name].append(diagonal_averaging(projected))
    
    # Optionally return computed SVD along with trajectories
    if return_svd:
        computed_svd = (u,s,v)
        return trajectories, computed_svd
    else:
        return trajectories


###############################################################################################################
#
#                                                      Analysis 
#
###############################################################################################################


def reshape_v(v, channel_number):
    """
    Reshape the right singular vectors matrix `v` from MSSA SVD into a 3D array separating channels.

    Given the matrix `v` of shape (n_components, total_length), where `total_length` is
    the concatenation of trajectory lengths from multiple channels, this function splits
    the second dimension into separate channel blocks and returns a 3D array with shape
    (n_components, length_per_channel, channel_number).

    Parameters
    ----------
    v : ndarray, shape (n_components, total_length)
        Right singular vectors from the SVD of the concatenated embedding matrix.
    channel_number : int
        Number of channels concatenated in the input `v`.

    Returns
    -------
    reshaped_v : ndarray, shape (n_components, length_per_channel, channel_number)
        Reshaped array where the second dimension corresponds to the trajectory length per channel,
        and the third dimension indexes channels.

    Examples
    --------
    >>> reshaped_v = reshape_v(v, channel_number=3)
    >>> reshaped_v.shape
    (n_components, length_per_channel, 3)
    """
    # Calculate trajectory length per channel by dividing total length by number of channels
    k = v.shape[1]//channel_number

    # Initialize output array with shape (components, length_per_channel, channels)
    reshaped_v = np.zeros((v.shape[0], k, channel_number))

    # For each channel, slice the corresponding block along the second dimension and assign it
    for i in range(channel_number):
        reshaped_v[:,:,i] = v[:,i*k:(i+1)*k]

    return reshaped_v




def v_similarity(v, channel_number):
    """
    Compute cosine similarity matrices for reshaped right singular vectors across channels.

    The function reshapes the right singular vectors `v` into separate channel blocks,
    then computes the cosine similarity between the vectors of each component across channels.
    The output is an array of similarity matrices, one per singular vector component.

    Parameters
    ----------
    v : ndarray, shape (n_components, total_length)
        Right singular vectors from the SVD of the concatenated embedding matrix.
    channel_number : int
        Number of channels concatenated in `v`.

    Returns
    -------
    similarities : ndarray, shape (n_components, channel_number, channel_number)
        Array of cosine similarity matrices for each singular vector component.
        Each matrix shows similarities between channels for that component.

    Examples
    --------
    >>> sims = v_similarity(v, channel_number=3)
    >>> sims.shape
    (n_components, 3, 3)
    """
    # Reshape v to separate channel blocks: shape (components, length_per_channel, channel_number)
    vr = reshape_v(v, channel_number)

    similarities = []
    # Compute cosine similarity between channel vectors for each component
    from sklearn.metrics.pairwise import cosine_similarity
    for vi in vr:
        # vi.T shape is (channel_number, length_per_channel), so cosine_similarity compares channels
        similarities.append(cosine_similarity(vi.T))
    
    return np.array(similarities)