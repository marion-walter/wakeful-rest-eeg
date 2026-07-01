# Necessary imports
from configuration import *
from dataio import *

# Base imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import WindowsPath
import os
import pandas as pd


# Specific imports used several times
import mne
import plotly.graph_objects as go

from contextlib import contextmanager
from keying import *
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns



def get_one_metadata_info(info):
    """
    Load and format the age metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the first 60 rows of `subject` and a chosen meta data columns
        from the meta-data CSV file.

    """
    meta_data = pd.read_csv(data_path / 'meta-data.csv', sep = ";", encoding='latin')
    df = pd.DataFrame(meta_data)
    # df = df[:60]
    df = df[['subject', info]]
    return df


def get_metadata():
    """
    Load metadata
    
    ----------
    Parameters
    ----------

    -------
    Returns
    -------
    - pd.DataFrame
    """   
    meta_data = pd.read_csv(Path(data_path / 'meta-data.csv'), sep = ";", encoding='latin')
    meta_data_df = pd.DataFrame(meta_data)
    return meta_data_df



def get_mental_activities():
    df = pd.read_csv(data_path / 'data_mental_activities.csv', sep = ";", encoding='latin')
    df.drop('Other type', axis=1, inplace=True)
    return df


def concatenate_psycho():
    df_mental_activities = get_mental_activities()
    df_distraction_task_difficulty = get_distraction_task_difficulty_data()

    behavior_df = pd.merge(df_mental_activities, df_distraction_task_difficulty, on=['subject'], how='left')

    metadata_infos = ['age','gender', 'DFS_total']

    for info in metadata_infos: 
        df_info = get_one_metadata_info(info)
        behavior_df = behavior_df.merge(df_info, on='subject', how='left')

    return behavior_df



def plot_correlation_matrix(df, cols_to_correlate, with_correction=True, method='fdr_bh', alpha=0.05, annot_kws=14, rotation=30,custom_label=None, save=False, figname=None):
    """
    Compute and plot a correlation matrix for the specified columns in the DataFrame.
    Optionally applies multiple comparison correction to p-values.
 
    Parameters:
    - cols_to_correlate: List of column names to correlate.
    - df: DataFrame containing the data.
    - with_correction: If True, applies multiple comparison correction.
    - method: Correction method (by default = fdr_bh).
        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)
    - alpha: Significance level for correction.
    """
    df_to_correlate = df[cols_to_correlate].dropna()
    n = len(cols_to_correlate)

    # Initialize matrices
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))

    # Compute correlations and p-values
    for i in range(n):
        for j in range(n):
            rho, pval = spearmanr(df_to_correlate.iloc[:, i], df_to_correlate.iloc[:, j])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval
            

    # Apply correction if requested
    if with_correction:
        _, pval_matrix_adjusted, _, _ = multipletests(pval_matrix.flatten(), alpha=alpha, method=method)
        pval_matrix_adjusted = pval_matrix_adjusted.reshape(n, n)
    else:
        pval_matrix_adjusted = pval_matrix

    labels = cols_to_correlate.copy()
    if custom_label:
        for old, new in custom_label.items():
            if old in labels:
                idx = labels.index(old)
                labels[idx] = new

    # annot = np.empty((n, n), dtype=object)
    # for i in range(n):
    #     for j in range(n):
    #         rho = corr_matrix[i, j]
    #         pval = pval_matrix_adjusted[i, j]
    #         pval_rounded = round(pval, 2)
    #         annot[i, j] = f"**r = {rho:.2f}**\n*p = {pval_rounded:.2f}*"

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        corr_matrix,
        cmap="RdYlBu",
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        fmt="",
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=rotation, fontsize=16)
    ax.collections[0].colorbar.ax.tick_params(labelsize=16)

    # Add title
    correction_str = f" ({method}-adjusted)" if with_correction else " without correction"
    ax.set_title(f"Correlation matrix{correction_str}", fontsize=16, fontweight='bold')

    for i in range(n):
        for j in range(n):
            rho = corr_matrix[i, j]
            pval = pval_matrix_adjusted[i, j]
            # pval_rounded = np.round(pval, 2)
            # Bold r, italic p
            ax.text(
                j + 0.5, i + 0.5,
                f"{rho:.2f}\np = {pval:.2f}",
                ha='center', va='center',
                fontsize=annot_kws,
                weight='bold' if pval < alpha and i != j else 'normal',  
                style='italic' if i == j else 'normal'  
            )

    # Add rectangles for significant correlations
    for i in range(n):
        for j in range(n):
            if i != j and pval_matrix_adjusted[i, j] < alpha:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='darkred', lw=4))

    plt.show()

    if save == True:
        fig.savefig(os.path.join(figures_path, f'{figname}.png'), dpi=300, format='png', bbox_inches='tight')
        fig.savefig(os.path.join(figures_path, f'{figname}.pdf'), format='pdf', bbox_inches='tight') 
    return corr_matrix, pval_matrix_adjusted


def radar_factory(num_vars, frame='circle'):
    """
    # from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html


    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections import register_projection
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D



    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



###############################################################################################################
#
#                                               USEFUL FUNCTIONS
#
###############################################################################################################


def get_session_list(session, remove_peculiar=False, even_if_different_sessions = False):
    """
    Generate a list of subject run keys for a given session, optionally excluding peculiar subjects.

    Parameters
    ----------
    session : str
        Session identifier string (e.g., 'baseline').
    remove_peculiar : bool, optional
        Whether to exclude peculiar subjects from the list (default is False).
    even_if_different_sessions : bool, optional
        If True, remove subjects whose IDs match peculiar subjects regardless of session (default is False).

    Returns
    -------
    list of str
        List of subject run keys in the format 'sub-XX_session'.

    Notes
    -----
    Relies on a global `peculiar` list of subject run keys to exclude peculiar subjects.

    Examples
    --------
    >>> get_session_list('baseline', remove_peculiar=True)
    ['sub-01_baseline', 'sub-02_baseline', ..., 'sub-61_baseline']
    """
    if remove_peculiar :
        subjects = [f'sub-{add_0_to_str(i)}_{session}' for i in range(1,62) if i != 11]
        
        if even_if_different_sessions :
            subjects = [sub for sub in subjects if not sub.split('_')[0] in [peculiar_sub.split('_')[0] for peculiar_sub in peculiar]]
        else:
            subjects = [sub for sub in subjects if not sub in peculiar]
        return subjects
    else:
        return [f'sub-{add_0_to_str(i)}_{session}' for i in range(1,62) if i != 11]


def session_job_done(job, sessions, start_i = 1, end_i = 62, show = False):
    """
    Check which subject-session jobs are completed.

    Parameters
    ----------
    job : object
        Job manager object with method `is_job_done(run_key)` returning bool.
    sessions : list of str
        List of session identifiers to check.
    start_i : int, optional
        Start subject index (default 1).
    end_i : int, optional
        End subject index (default 62, exclusive).
    show : bool, optional
        Whether to print unfinished jobs and completion percentage (default False).

    Returns
    -------
    list of str
        List of run keys (subject-session) for which the job is done.

    Examples
    --------
    >>> session_job_done(job, ['baseline', 'E1'], show=True)
    """

    subject_list = [f'sub-{add_0_to_str(i)}_{session}' for i in range(start_i,end_i) for session in sessions  if i != 11]
    job_not_done_list = [run_key for run_key in subject_list if not job.is_job_done(run_key)]
    if show:
        print(job_not_done_list)
        print(len(job_not_done_list), (1-len(job_not_done_list)/len(subject_list))*100)

    job_done_list = [run_key for run_key in subject_list if job.is_job_done(run_key)]
    return job_done_list


def numpy_to_mne(data_numpy, channel_names, sampling_rate, channel_type='eeg', add_montage='standard_1020', equalize_electrode_number = False):
    """
    Convert a NumPy array into an MNE Raw object.

    Converts EEG/ECG/other channel data from a NumPy array into an MNE 
    RawArray object with specified channel names, types, and sampling rate.

    Parameters
    ----------
    data_numpy : np.ndarray
        Input data array of shape (channels, time) or (time,) for single channel.
    channel_names : list of str
        Names of each channel in the array.
    sampling_rate : float
        Sampling frequency in Hz.
    channel_type : str, optional
        Type of the channels (default is 'eeg'). Can be 'ecg', 'eog', etc.

    Returns
    -------
    raw : mne.io.RawArray
        MNE Raw object containing the data, channel info, and sampling rate.

    Notes
    -----
    - If `data_numpy` is 1D, it is assumed to be a single channel and reshaped accordingly.
    - Input data should be in shape (channels, time). Transpose if your data is (time, channels).
    """


    if len(data_numpy.shape)<2:
        data = data_numpy[np.newaxis,:]
    else:
        data = data_numpy

    # Define channel information
    ch_types = [channel_type] * len(channel_names) 
    info = mne.create_info(ch_names=list(channel_names), sfreq=sampling_rate, ch_types=ch_types)


    # Create MNE RawArray object
    raw = mne.io.RawArray(data, info, verbose=False)
    if add_montage is not None :
        montage = mne.channels.make_standard_montage(add_montage)
        raw.set_montage(montage)
    
    if equalize_electrode_number:
        raw.drop_channels({'FCz', 'Iz'}, on_missing='ignore')

    return raw

def dataset_to_mne(ds, add_montage='standard_1020', equalize_electrode_number = False):
    """
    Convert an xarray Dataset into an MNE Raw object with optional montage.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing EEG data. Must have:
        - ds.eeg.values: array of shape (time, channels)
        - ds.channel.values: list of channel names
        - ds.sampling_rate: float, sampling frequency in Hz
    add_montage : str or None, optional
        Name of the standard montage to add (default is 'standard_1020').
        If None, no montage is added.

    equalize_electrode_number : bool, optional
        If ``True``, drop the channels ``'FCz'`` and ``'Iz'`` if present.
        Defaults to ``False``.

    Returns
    -------
    raw : mne.io.RawArray
        An MNE Raw object containing the EEG data, channel information, and
        optionally a standard montage.

    Notes
    -----
    The EEG data is transposed internally from ``(n_times, n_channels)`` to
    ``(n_channels, n_times)`` to match the MNE expected input format.

    See Also
    --------
    mne.channels.make_standard_montage : Create standard EEG montages.
    numpy_to_mne : Helper function to create an MNE RawArray from NumPy arrays.
    """
    raw =  numpy_to_mne(ds.eeg.values.T, list(ds.channel.values), float(ds.sampling_rate))
    if add_montage is not None :
        montage = mne.channels.make_standard_montage(add_montage)
        raw.set_montage(montage)
    
    if equalize_electrode_number:
        raw.drop_channels({'FCz', 'Iz'}, on_missing='ignore')
    return raw

def get_info(equalize_electrode_number= 'both', montage = 'standard_1020'):

    from dataio import read_eeg
    
    idx = 10
    if equalize_electrode_number == 'Fz':
        idx = 40

    run_key = f'sub-{add_0_to_str(idx)}_baseline'
    sigs_eeg, sr, eeg_channels = read_eeg(run_key)

    if equalize_electrode_number == 'both':
        eq = 1
    else:
        eq = 0

    raw = numpy_to_mne(sigs_eeg.T, eeg_channels, sr, equalize_electrode_number=eq, add_montage=montage)
    info = raw.info
    
    return info


def add_0_to_str(x):
    """
    Add a leading zero to a number or numeric string if less than 10.

    Parameters
    ----------
    x : int or str
        Input number or numeric string.

    Returns
    -------
    str
        String representation of the number with a leading zero if `x < 10`.
    """

    if int(x) < 10:
        return '0' + str(x)
    else:
        return str(x)



###############################################################################################################
#
#                                               PLOTS
#
###############################################################################################################


def plot_psd_ratios(numerators, denominators, freqs, 
                    vline_x=None, hline_y=1.0,
                    labels=None, colors=None, variability='IQ',
                    show= True, return_fig = False, run_key=None):
    """
    Plot mean PSD ratios for multiple conditions with variability bands.

    For each condition, computes the ratio of numerator to denominator PSD arrays,
    averages across channels, and plots the mean along with variability shading.

    Parameters
    ----------
    numerators : list of array-like
        List of PSD arrays (n_channels, n_freqs) for the numerator condition(s).
    denominators : list of array-like
        List of PSD arrays (n_channels, n_freqs) for the denominator condition(s).
    freqs : array-like
        1D array of frequency values in Hz.
    vline_x : float or None, optional
        Frequency at which to draw a vertical reference line. Default is None.
    hline_y : float or None, optional
        Y-value at which to draw a horizontal reference line. Default is 1.0.
    labels : list of str, optional
        Legend labels for each condition. Default generates generic labels.
    colors : list of str, optional
        List of base colors (e.g., 'rgba(r,g,b,1)') for each condition.
        Transparency for shading is auto-generated.
    variability : {'IQ', 'std'}, optional
        Type of variability band:
        - 'IQ': 5th-95th percentile range
        - 'std': ±1 standard deviation
        Default is 'IQ'.
    show : bool, optional
        If True (default), displays the figure interactively.
    return_fig : bool, optional
        If True, returns the `plotly.graph_objects.Figure` object.
    run_key : str or None, optional
        Optional run identifier appended to the plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The plotly figure if `return_fig=True`, otherwise None.

    Notes
    -----
    - All numerator and denominator arrays must have the same shape and frequency axis.
    - Frequency axis is plotted on a logarithmic scale.
    - Colors with alpha channel exactly `'1)'` are modified for transparency shading.
    """
    fig = go.Figure()
    n_conditions = len(numerators)

    if labels is None:
        labels = [f'PSD Ratio {i+1}' for i in range(n_conditions)]
    if colors is None:
        colors = ['rgba(66, 135, 245, 1)', 'rgba(255, 100, 100, 1)', 
                  'rgba(100, 200, 100, 1)', 'rgba(200, 100, 255, 1)'][:n_conditions]

    for i in range(n_conditions):
        num = np.array(numerators[i])
        denom = np.array(denominators[i])
        ratio = num / denom

        mean_ratio = ratio.mean(axis=0)
        if variability == 'IQ':
            bottom_variability = np.quantile(ratio, 0.05, axis=0)
            top_variability = np.quantile(ratio, 0.95, axis=0)
        else:
            bottom_variability = mean_ratio - ratio.std(axis=0)
            top_variability = mean_ratio + ratio.std(axis=0)

        color = colors[i]
        fillcolor = color.replace('1)', '0.2)')  # transparent version

        # Upper bound
        fig.add_trace(go.Scatter(
            x=freqs, y=bottom_variability,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Lower bound and fill
        fig.add_trace(go.Scatter(
            x=freqs, y=top_variability,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=fillcolor,
            name=f'Std Dev {labels[i]}',
            hoverinfo='skip'
        ))
        # Mean line
        fig.add_trace(go.Scatter(
            x=freqs, y=mean_ratio,
            mode='lines',
            line=dict(color=color, width=2),
            name=labels[i]
        ))

    # Reference lines
    if hline_y is not None:
        fig.add_hline(y=hline_y, line_dash="dot", line_color="black", line_width=1)
    if vline_x is not None:
        fig.add_vline(x=vline_x, line_dash="dot", line_color="black", line_width=1)

    # Layout
    if variability == 'IQ':
        deviation = '5-95% Percentile Bands'
    else:
        deviation = 'Standard Deviation'
    
    if run_key is not None :
        name = ' '+ run_key
    else :
        name = ' '
    fig.update_layout(
        title=f'Mean PSD Ratios{name} with {deviation}',
        xaxis_title='Frequency (Hz)',
        yaxis_title='PSD Ratio',
        xaxis_type='log',
        template='plotly_white'
    )

    if show:
        fig.show()
    
    if return_fig:
        return fig


def plot_comparison(signals, downsample = 5, start=None, end=None, show= True, scatter= False, **kwargs):
    """
    Plot a comparison of multiple signals with optional downsampling.

    Parameters
    ----------
    signals : list of array-like
        List of signals to compare. Each signal should be 1D and of equal length.
    downsample : int, default=5
        Downsampling factor. Every `downsample`-th point will be used.
    show : bool, default=True
        If True, displays the figure. If False, returns the figure object.
    scatter : bool, default=False
        If True, use a scatter plot instead of a line plot.
    **kwargs
        Additional keyword arguments passed to :func:`plotly.express.line` or
        :func:`plotly.express.scatter`.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The created Plotly figure, or None if `show=True`.
    """
    comparison = np.array([a[start:end:downsample] for a in signals]).T
    if scatter:
        f = px.scatter(comparison, **kwargs)
    else:
        f = px.line(comparison, **kwargs)
    if show :
        f.show()
    else:
        return f


def save_figure(fig, name, job, folder, plotly=True, with_job=True):
    """
    Save a figure to a specific job-related folder.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure to save.
    name : str
        File name including extension (`.html` or `.png`).
    job : object
        Object with method `get_filename('0')` returning a Path-like to determine job hash.
    folder : pathlib.Path
        Base folder in which the figure will be saved.
    plotly : bool, default=True
        If True, use Plotly saving methods; otherwise use Matplotlib's `savefig`.

    Raises
    ------
    ValueError
        If unsupported file extension is provided for the chosen backend.
    """

    if with_job == True:
        if type(job.get_filename('0')) == WindowsPath :
            job_hash_filename = str(job.get_filename('0')).split('\\')[-2]
        else:
            job_hash_filename = str(job.get_filename('0')).split('/')[-2]
        filepath = folder / job_hash_filename
        if not os.path.exists(filepath ):
            os.makedirs(filepath)
    else: 
        filepath = folder 
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    extension = name.split('.')[-1]

    if plotly :        
        if extension == 'html':
            fig.write_html(filepath  / name)
        elif extension == 'pdf':
            fig.write_image(str(filepath / name))
        else:
            raise ValueError
        
    else:
        if extension == 'pdf' or extension == 'png':
            # print(filepath)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            fig.savefig(filepath  / name, dpi=300)
        else:
            raise ValueError



        

###############################################################################################################
#
#                                               Simple maths
#
###############################################################################################################

def find_windows(arr, sr=1, integer = False, values = None):
    """
    Identify contiguous windows of '1's in a binary array.

    Parameters
    ----------
    arr : array_like
        One-dimensional binary array (0s and 1s).
    sr : float, optional
        Sampling rate. Default is 1.
    integer : bool, optional
        If True, returned indices are integers. Default is False.
    values : array_like, optional
        If provided, returns the corresponding values instead of indices.

    Returns
    -------
    list of tuple
        List of (start, end) tuples for each contiguous window. 
        Indices or values depending on `integer` and `values`.
    """

    if values is not None and len(values) != len(arr):
        raise ValueError

    arr = np.asarray(arr)  
    
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional")
    
    # Find start indices: where 0 → 1 transition occurs
    starts = (1/sr*np.where(np.diff(np.concatenate(([0], arr))) == 1)[0]).astype(float)

    # Find end indices: where 1 → 0 transition occurs
    ends = (1/sr*np.where(np.diff(np.concatenate((arr, [0]))) == -1)[0]).astype(float)

    if values is not None :
        return list(zip(values[starts.astype(int)], values[ends.astype(int)]))
    elif not integer:
        return list(zip(starts, ends))
    else:
        return list(zip(starts.astype(int), ends.astype(int)))



def MAD(data, axis=0, standardize_with_normal = True):
    """
    Compute the Median Absolute Deviation (MAD) of the data.

    Parameters
    ----------
    data : array_like
        Input data array.
    axis : int, optional
        Axis along which to compute the MAD. Default is 0.
    standardize_with_normal : bool, optional
        If True, scale MAD to match standard deviation for normal distribution (factor 1.4826).

    Returns
    -------
    np.ndarray
        Median Absolute Deviation of the data along the specified axis.
    """
    if standardize_with_normal :
        factor = 1.4826
    else:
        factor = 1
    return factor*np.median(np.absolute(data - np.median(data, axis=axis)), axis=axis)


def rms(x):
    """
    Compute the root mean square (RMS) of a 1D array.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    float
        Root mean square of the input array.
    """
    return np.sqrt(np.mean(x**2))


def rolling_rms(x, window, sampling_rate = 500, engine='numba'):
    """
    Compute rolling root mean square (RMS) over a sliding window.

    Parameters
    ----------
    x : array_like
        Input one-dimensional array.
    window : float
        Window length in seconds.
    sampling_rate : float, optional
        Sampling rate in Hz. Default is 500.

    Returns
    -------
    pd.Series
        Rolling RMS of the input signal.

    Notes
    -----
    Uses reflective padding to minimize edge effects. Window is centered.
    Requires 'numba' if engine='numba' is used.
    """
    n_points = int(window*sampling_rate)
    return pd.Series(np.pad(x, n_points, mode='reflect')).rolling(n_points, center=True).apply(rms, raw=True, engine=engine)[n_points:-n_points]

def rolling_mean(x, window, sampling_rate = 500):
    """
    Compute rolling mean over a sliding window.

    Parameters
    ----------
    x : array_like
        Input one-dimensional array.
    window : float
        Window length in seconds.
    sampling_rate : float, optional
        Sampling rate in Hz. Default is 500.

    Returns
    -------
    pd.Series
        Rolling mean of the input signal.

    Notes
    -----
    Uses reflective padding to minimize edge effects. Window is centered.
    """
    n_points = int(window*sampling_rate)
    return pd.Series(np.pad(x, n_points, mode='reflect')).rolling(n_points, center=True).mean()[n_points:-n_points]


###############################################################################################################
#
#                                               Optimization
#
###############################################################################################################




@contextmanager
def set_num_threads(n):
    """
    Context manager to temporarily set the number of threads for various libraries.

    Parameters
    ----------
    n : int
        Number of threads to set.

    Yields
    ------
    None
        Context manager yields control to the block of code inside the `with` statement.

    Notes
    -----
    Temporarily sets environment variables for:
    - OMP_NUM_THREADS
    - MKL_NUM_THREADS
    - OPENBLAS_NUM_THREADS
    - NUMEXPR_NUM_THREADS

    Original values are restored after exiting the context.
    """
    original_env = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
    }

    # Set to 1
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

    try:
        yield
    finally:
        # Restore original values
        for key, value in original_env.items():
            if value:
                os.environ[key] = value
            else:
                del os.environ[key]


def rolling_function(x, window, function, window_step = None, sampling_rate=500, numba_on = True ):
    """
    Apply a rolling function over a 1D array.

    Parameters
    ----------
    x : array_like
        Input signal array.
    window : float
        Window size in seconds.
    function : callable
        Function to apply to each rolling window.
    window_step : int, optional
        Step size in number of points between windows. Default is None (overlapping).
    sampling_rate : float, optional
        Sampling rate in Hz. Default is 500.
    numba_on : bool, optional
        If True, uses numba engine for acceleration. Default is True.

    Returns
    -------
    pd.Series
        Result of the rolling function applied to the signal.

    Notes
    -----
    Uses reflective padding to reduce edge effects. Window is centered.
    """
    n_points = int(window*sampling_rate)
    if numba_on:
        return pd.Series(np.pad(x, n_points, mode='reflect')).rolling(n_points, step=window_step, center=True).apply(function, raw=True, engine='numba')[n_points:-n_points]
    else:
        return pd.Series(np.pad(x, n_points, mode='reflect')).rolling(n_points, step=window_step, center=True).apply(function)[n_points:-n_points]
    


###############################################################################################################
#
#                                               Constants
#
###############################################################################################################

peculiar = ['sub-56_Distraction']

