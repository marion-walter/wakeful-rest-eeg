# Necessary imports
from configuration import *

# Base imports
import numpy as np
import pandas as pd
import xarray as xr
import jobtools

# Specific imports used several times
import mne
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

# Intra package imports
from modular_eeg_preprocess import modular_eeg_preprocessing_job
from analyses_eeg_params import spectrum_params_reversed_ref, spectrum_params_replication_reversed_ref
from utils import dataset_to_mne, get_metadata, save_figure
from keying import key_adapter
from analyse_memory import get_all_trials_data_with_trial_numbers, compute_global_memory_change, compute_relative_memory_change
from params import run_keys, subject_keys

############################################################################################
#                               Full session spectrum 
############################################################################################

def get_frequency_resolution(first_duration, second_duration, sampling_rate=500):
    """
    Calculate the FFT length needed to resolve two target periods in a PSD.

    This function determines the number of FFT points (`n_fft`) required
    so that the Power Spectral Density (PSD) computation can distinguish
    between two signals with given periods (`first_duration` and 
    `second_duration`). It is useful for setting `n_fft` in PSD calculations,
    such as `raw.compute_psd()` in MNE.

    Parameters
    ----------
    first_duration : float
        The first target period (in seconds) to resolve.
    second_duration : float
        The second target period (in seconds) to resolve.
    sampling_rate : int or float, optional
        Sampling rate in Hz (samples per second). Default is 500.

    Returns
    -------
    int
        The required FFT length (`n_fft`) in samples.

    """
    diff = np.abs(2/(1/first_duration-1/second_duration))
    return int(diff*sampling_rate)

def compute_spectrum(run_key, **p):
    """
    Compute band-averaged power spectral density (PSD) features for an EEG
    recording using both high-resolution and default-resolution PSD estimates.

    Parameters
    ----------
    run_key : str
        Identifier used to load the preprocessed EEG dataset.
    **p : dict
        Parameter dictionary containing:
        
        equalize_electrode_number : bool
            Whether to force consistent channel count across runs.
        reference : str or list or None
            Reference channel(s). If ``None``, the original Fz reference is assumed.
        frequency_to_resolve : tuple
            (f_low, f_high) pair used to determine desired PSD frequency
            resolution. Passed to ``get_frequency_resolution``.
        frequency_bands : list of tuple
            Each tuple defines a (low, high) frequency interval in Hz.
        frequency_names : list of str
            Human-readable names corresponding to ``frequency_bands``.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        
        PSD : (Channel, Frequency_band)
            Normalized band-averaged PSD based on high-resolution PSD.
        PSD_short : (Channel, Frequency_band)
            Normalized band-averaged PSD based on default-resolution PSD.
        Coordinates for channels and frequency-band labels.
        Metadata including sampling rate, condition, odor group, and reference.

    Notes
    -----
    - PSDs are normalized per channel (sum over frequencies equals 1).
    - Band edges are treated with inclusive bounds.
    - Two PSD estimates are computed: one with explicit ``n_fft`` and one
      using MNE's default Welch parameters.
    """
    
    ds = modular_eeg_preprocessing_job.get(run_key)
    eeg = dataset_to_mne(ds, equalize_electrode_number=p['equalize_electrode_number'])
    channels = eeg.info.ch_names

    if p['reference'] is not None :
        eeg , _ = mne.set_eeg_reference(eeg, ref_channels=p['reference'], copy=True)
        reference = f"{p['reference']}"
    else:
        reference = 'FZ'

    
    if p == spectrum_params_reversed_ref:
        print("Computing spectrum with replication-inspired parameters on long window with 50% overlap.")

        n_fft = get_frequency_resolution(*p['frequency_to_resolve'])
        n_per_seg = n_fft
        n_overlap = n_per_seg // 2
        spectrum = eeg.compute_psd(n_fft = n_fft, n_per_seg = n_per_seg, n_overlap = n_overlap, fmax= p['spectrum_fmax'])
        psd = spectrum.get_data()
        psd_norm = psd / np.sum(psd, axis = 1)[:,np.newaxis] #norm par puissance totale jusqu'à fmax
        frequency = spectrum.freqs
        print('PSD computed with 20.0 s window and 50% overlap.')

    elif p == spectrum_params_replication_reversed_ref:
        print("Computing spectrum with replication-inspired parameters on 4-second window.")
    
        sfreq = eeg.info['sfreq']
        n_per_seg = int(4 * sfreq)     # 4-second window
        n_overlap = n_per_seg // 2     # 50% overlap
        spectrum = eeg.compute_psd(n_per_seg=n_per_seg,n_overlap=n_overlap,n_fft=n_per_seg,fmax=p['spectrum_fmax'])
        psd = spectrum.get_data()
        psd_norm = psd / np.sum(psd, axis=1)[:, np.newaxis]
        frequency = spectrum.freqs
        print(f"PSD computed with 4.0 s window and 50% overlap")

    frequencies = p['frequency_bands']
    names = p['frequency_names']
    frequency_bands = [f"{name} ({f[0]} - {f[1]} Hz)" for name, f in zip(names, frequencies)]

    psd_bands_mean  = np.zeros([len(channels), len(names)])  # density (raw)
    psd_bands_norm_mean  = np.zeros([len(channels), len(names)])  # density (normalized)

    psd_bands_sum  = np.zeros([len(channels), len(names)])  # absolute power
    psd_bands_norm_sum  = np.zeros([len(channels), len(names)])  # relative power

    for i, band in enumerate(frequencies):

        mask = (frequency >= band[0]) & (frequency <= band[1])

        psd_bands_mean[:, i] = np.mean(psd[:, mask], axis=1)
        psd_bands_norm_mean[:, i] = np.mean(psd_norm[:, mask], axis=1)
        psd_bands_sum[:, i] = np.sum(psd[:, mask], axis=1)
        psd_bands_norm_sum[:, i] = np.sum(psd_norm[:, mask], axis=1)

    data_vars = dict()

    # Absolute power
    data_vars['PSD_abs'] = (["Channel", "Frequency_band"], psd_bands_sum)

    # Relative power
    data_vars['PSD_rel'] = (["Channel", "Frequency_band"], psd_bands_norm_sum) # to use 

    # Density
    data_vars['PSD_density'] = (["Channel", "Frequency_band"], psd_bands_mean)
    data_vars['PSD_density_norm'] = (["Channel", "Frequency_band"], psd_bands_norm_mean)

    meta_data = get_metadata()
    subject_id = run_key.split("_")[0] 
    order_val = meta_data.set_index("subject").loc[subject_id, "order"]

    attributes = dict(run_key = run_key,
                order = order_val,
                sampling_rate = ds.sampling_rate, 
                eeg_unit = 'µV', 
                time_unit = 's',
                reference = reference,
                )

    ds = xr.Dataset(
            data_vars= data_vars,
            coords=dict(
                Channel=("Channel", channels),
                Frequency_band=("Frequency_band", frequency_bands),
            ),
            attrs=attributes,
        )
    return ds



def update_figure_axes(title, band, axes, x, y, slope, intercept, r_value, p_value,  r_values, p_values, info, rank_data, replication_inspired):
    """
    Update a row of subplots with scatter-regression, r-value topomap,
    and thresholded p-value topomap.

    Parameters
    ----------
    title : str
        Title prefix for the scatter panel (e.g., ``"PSD 6 channels"``).
    band : str
        Frequency band label (e.g., ``"Alpha (8–12 Hz)"``).
    axes : numpy.ndarray of matplotlib.axes.Axes
        A 2D array of axes with shape ``(n_rows, 3)`` where each row contains:
        left = scatter plot, middle = r-value topomap, right = p-value topomap.
    x : array_like
        Predictor variable for regression (e.g., PSD or PSD_short values).
    y : array_like
        Outcome variable (e.g., change in memory recall).
    slope : float
        Regression slope for the line overlay.
    intercept : float
        Regression intercept for the line overlay.
    r_value : float
        Pearson correlation for the (x, y) regression.
    p_value : float
        p-value associated with ``r_value``.
    r_values : array_like
        Channel-wise correlation values for the topomap.
    p_values : array_like
        Channel-wise thresholded p-values for the topomap. Expected to contain:
        ``1`` for uncorrected sig, ``-1`` for corrected sig, ``0`` otherwise.
    info : mne.Info
        MNE Info object corresponding to the EEG montage of ``r_values`` and
        ``p_values``.

    Notes
    -----
    - Significant scatter points are colored red; nonsignificant are cyan.
    - The function draws:
        * scatter + regression line,
        * r-value topomap (``RdBu_r`` colormap),
        * corrected/uncorrected p-value topomap.
    - No figure is returned; the function modifies ``axes`` in place.

    Returns
    -------
    None
    """

    title_font = {'fontsize': 20, 'color': 'black', 'fontweight':'bold'}
    axes_font = {'fontsize': 18, 'color': 'black'}
    pad=20

    # Scatter plot (left panel)
    if p_value <0.05 :
        color = 'red'
    else:
        color = 'lightseagreen'
    axes[0].scatter(x, y, c='grey', edgecolor='k', s=40)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
    axes[0].plot(np.linspace(np.min(x)*0.95, np.max(x)*1.05, 2), slope*np.linspace(np.min(x)*0.95, np.max(x)*1.05, 2)+intercept,  '-', c =color,)
    axes[0].tick_params(labelsize=16)

    if replication_inspired == True and rank_data == True:
        axes[0].set_xlabel(f"{band} Relative Power \n Ranked values", **axes_font)
        axes[0].set_ylabel("GMC - Ranked values", **axes_font)
    elif replication_inspired == True and rank_data == False:
        axes[0].set_xlabel(f"{band} Relative Power (a.u.)", **axes_font)
        axes[0].set_ylabel("GMC (%)", **axes_font)
    elif replication_inspired == False and rank_data == True:
        axes[0].set_xlabel(f"{band} Relative Power \n Ranked values", **axes_font)
        axes[0].set_ylabel("RMC - Ranked values", **axes_font)
    elif replication_inspired == False and rank_data == False:
        axes[0].set_xlabel(f"{band} Relative Power (a.u.)", **axes_font)
        axes[0].set_ylabel("RMC (%)", **axes_font)
    axes[0].set_title(f"r={r_value:.2f}, p={p_value:.3f}", **title_font, pad=pad)

    

    # Topomap (right panel)
    im, _ = mne.viz.plot_topomap(
        r_values, 
        info, 
        axes=axes[1], 
        show=False, 
        cmap="RdBu_r", 
        vlim=[-1,1]
    )
    cbar = plt.colorbar(im, ax=axes[1], shrink=1)
    cbar.set_label("r-values",fontsize=18, fontweight= 'bold')
    cbar.ax.tick_params(labelsize=16)
    
    import matplotlib.colors as mcolors

    colors = ['#29AF7FFF', '#481567FF', '#FDE725FF']
    cmap = mcolors.ListedColormap(colors)

    # Define boundaries for -1, 0, and 1
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im2, _ = mne.viz.plot_topomap(
        p_values, 
        info, 
        axes=axes[2], 
        show=False, 
        # cmap="viridis",
        cmap=cmap,
        cnorm=norm,
        # vlim=[-1,1],
        image_interp='nearest',
        contours=False,
    )

    axes[2].legend
    axes[2].set_label("p-value thresholded")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#29AF7FFF', label='* corrected'),
        Patch(facecolor='#FDE725FF', label='* uncorrected'),
        Patch(facecolor= '#481567FF', label='ns'),
    ]
    legend = axes[2].legend(handles=legend_elements, bbox_to_anchor=(0.9, -0.6), loc='lower right', fontsize=16, title = 'p-values')
    legend.get_title().set_fontsize(16)  
    legend.get_title().set_fontweight('bold')

def compute_spectrum_figure(run_key, replication_inspired, **p):
    """
    Compute PSD/PSD-short correlations with memory performance across
    subjects, groups, sessions and conditions, and generate Brokaw-style
    EEG topographic regression figures for all frequency bands.

    Parameters
    ----------
    key : str, optional
        Identifier for subject grouping or selection. Default is ``'all'``.
    **p : dict
        A parameter dictionary expected to contain:

        - ``frequency_bands`` : list of tuple
            Each tuple is ``(fmin, fmax)`` for a frequency band.
        - ``frequency_names`` : list of str
            Human-readable names of frequency bands.

        Additional parameters are passed through to file system utilities,
        preprocessing jobs, or plotting locations depending on the workflow.

    Workflow Summary
    ----------------
    1. Loads frequency bands and labels.
    2. Retrieves preprocessing output (PSD and PSD_short) for each subject,
       condition, and order from the Brokaw spectrum job.
    3. Computes memory recall scores before/after RF trials.
    4. For each combination of:
         * order: ``['D-R', 'R-D', 'both']``
         * band: all frequency bands
       it performs:
         - subject-level averaging of spectral power,
         - linear regressions between spectral measures and recall change,
         - channel-wise correlation maps,
         - FDR-like thresholding of p-values,
         - topographic plotting (r-values and corrected p-values),
         - scatter/regression plots (6 channels and all channels),
         - automated saving of figures.

    Notes
    -----
    - Uses functions such as ``brokaw_spectrum_job.get``,
      ``modular_eeg_preprocessing_job.get``, ``dataset_to_mne`` and
      ``save_figure`` which must exist in the project environment.
    - Heavily dependent on specific dataframe schema from the scoring job.
    - No value is returned; figures are stored on disk via ``save_figure``.

    Returns
    -------
    None
    """
    from scipy.stats import rankdata, linregress, spearmanr
    from statsmodels.stats.multitest import multipletests

    frequencies = p['frequency_bands']
    names = p['frequency_names']
    frequency_bands = [f"{name} ({f[0]} - {f[1]} Hz)" for name, f in zip(names, frequencies)]
    
    rank_data = p['rank_data']
    print(f'rank_data: {rank_data}')

    sub_chan = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

    ds = modular_eeg_preprocessing_job.get(run_key)
    eeg = dataset_to_mne(ds, equalize_electrode_number=1)

    
    if replication_inspired == True:
        print("Using replication-inspired GMC memory performance computation.")
        df = get_all_trials_data_with_trial_numbers()
        df_memory = compute_global_memory_change(df)

    else: 
        print("Using RMC memory performance computation.")
        df_memory = compute_relative_memory_change()
    
    df_memory['condition'] = df_memory['condition'].replace('Rest', 'Repos')

    results=[]
    for condition in ['Repos', 'Distraction']:
        subject_list = []
        for base in subject_keys: 
            key = f"{base}_{condition}"
            if replication_inspired == True:
                # print(f"Checking if job done for replication-inspired params: {key}")
                if compute_spectrum_replication_inspired_params_rev_ref_job.is_job_done(key):
                    subject_list.append(key)
            else:
                # print(f"Checking if job done for data-adapted params: {key}")
                if compute_spectrum_rev_ref_job.is_job_done(key):
                    subject_list.append(key)

        df_all = []
        for subject in subject_list:
            if replication_inspired == True:
                # print(f"Loading replication-inspired spectrum for subject: {subject}")
                ds = compute_spectrum_replication_inspired_params_rev_ref_job.get(subject)
            else:
                # print(f"Loading data-adapted spectrum for subject: {subject}")
                ds = compute_spectrum_rev_ref_job.get(subject)
            df_ds = ds.to_dataframe()
            df_ds['order'] = ds.order
            df_ds['subject'] = key_adapter(subject, 'key')
            df_ds = df_ds.reset_index(level=['Channel', 'Frequency_band'])
            df_all.append(df_ds)
        df_all = pd.concat(df_all)
        # print(df_all)

        # for order in ['D-R', 'R-D', 'both']:
        for order in ['both']:        
            print(f"Processing order: {order}")
            for band in frequency_bands:
                    print(f"Processing band: {band}")
                    if order == 'both':
                        order_mask = (df_all['order']=='R-D') | (df_all['order']=='D-R')
                    else:
                        order_mask = (df_all['order']==order)

                    if replication_inspired == True:
                        print("Using replication-inspired params for correlation analysis.")
                        df_o_d = df_all.loc[order_mask & (df_all['Frequency_band']== band)]
                        df_o_d = df_o_d.loc[([a in sub_chan for a in df_o_d['Channel']])]
                        df_memory_sub = df_memory[(df_memory['condition'] == condition)]
                        if order != 'both':
                            df_o_d_order = df_o_d[df_o_d['order'] == order]
                            df_memory_order = df_memory_sub[df_memory_sub['order'] == order]
                        else:
                            df_o_d_order = df_o_d
                            df_memory_order = df_memory_sub
                        
                        df_memory_order = df_memory_order[df_memory_order['condition'] == condition]
                        if condition == 'Distraction':
                            df_memory_order = df_memory_order[df_memory_order['subject'] != 'sub-56']




                        x = df_o_d_order.groupby('subject')['PSD_rel'].mean().values
                        y = df_memory_order['GMC'].values
                        # print(f'GMC: {y}')
                        

                        version = 'PSD_rel'
                        r_channels = []
                        p_channels_uncorrected = []

                        reorder = []
                        for channel in df_o_d['Channel'].unique():
                            if channel in sub_chan :
                                reorder.append(channel)
                                x = df_o_d.loc[df_o_d['Channel'] == channel, version].values

                                if rank_data == True:
                            
                                    r_value, p_value = spearmanr(x, y)
                                    r_channels.append(r_value)
                                    p_channels_uncorrected.append(p_value)
                                    slope, intercept, _, _, _ = linregress(rankdata(x), rankdata(y))

                                elif rank_data == False:
                                    slope, intercept, r_value, p_value, _ = linregress(x, y)
                                    r_channels.append(r_value)
                                    p_channels_uncorrected.append(p_value)

                                results.append({
                                    'condition': condition,
                                    'order': order,
                                    'frequency_band': band,
                                    'channel': channel,
                                    'version': 'short_window',

                                    # ALL PSD METRICS (median per channel) 
                                    'PSD_abs': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_abs'].mean(),
                                    'PSD_rel': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_rel'].mean(),
                                    'PSD_density': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_density'].mean(),
                                    'PSD_density_norm': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_density_norm'].mean(),
                                    
                                    # CURRENT METRIC USED FOR CORRELATION 
                                    'metric_used': 'PSD_rel',

                                    # STATS 
                                    'r': r_value,
                                    'p_uncorrected': p_value
                                })

                    else: 
                        print("Using data-adapted params for correlation analysis.")
                        df_o_d = df_all.loc[order_mask & (df_all['Frequency_band']== band)]
                        df_memory_sub = df_memory[(df_memory['condition'] == condition)]
                        if order != 'both':
                            df_o_d_order = df_o_d[df_o_d['order'] == order]
                            df_memory_order = df_memory_sub[df_memory_sub['order'] == order]
                        else:
                            df_o_d_order = df_o_d
                            df_memory_order = df_memory_sub
                        df_memory_order = df_memory_order[df_memory_order['condition'] == condition]
                        if condition == 'Distraction':
                            df_memory_order = df_memory_order[df_memory_order['subject'] != 'sub-56']
                        

                        x = df_o_d_order.groupby('subject')['PSD_rel'].mean().values
                        y = df_memory_order['RMC'].values
                        

                        r_channels = []
                        p_channels_uncorrected = []

                        version = 'PSD_rel'
                        for channel in df_o_d['Channel'].unique():
                            
                            x = df_o_d.loc[df_o_d['Channel'] == channel, version].values

                            if rank_data == True:
                                r_value, p_value = spearmanr(x, y)
                                slope, intercept, _, _, _ = linregress(rankdata(x), rankdata(y))
                                r_channels.append(r_value)
                                p_channels_uncorrected.append(p_value)

                            elif rank_data == False:
                                slope, intercept, r_value, p_value, _ = linregress(x, y)
                                r_channels.append(r_value)
                                p_channels_uncorrected.append(p_value)

                            results.append({
                                'condition': condition,
                                'order': order,
                                'frequency_band': band,
                                'channel': channel,
                                'version': 'long_window',

                                # ALL PSD METRICS (median per channel) 
                                'PSD_abs': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_abs'].mean(),
                                'PSD_rel': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_rel'].mean(),
                                'PSD_density': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_density'].mean(),
                                'PSD_density_norm': df_o_d.loc[df_o_d['Channel'] == channel, 'PSD_density_norm'].mean(),
                                
                                # CURRENT METRIC USED FOR CORRELATION 
                                'metric_used': 'PSD_rel',

                                # STATS 
                                'r': r_value,
                                'p_uncorrected': p_value
                            })

                        _, p_channels_corrected = fdrcorrection(p_channels_uncorrected, alpha=0.05)
                
                        p_channels = np.sign(
                            (np.array(p_channels_uncorrected) < 0.05).astype(float)  # 1 if uncorrected p < 0.05
                            - 2 * (np.array(p_channels_corrected) < 0.05).astype(float)  # 1 if FDR-corrected p < 0.05
                        )

                    df_results = pd.DataFrame(results)
                    df_results = df_results.sort_values(by=['condition', 'order', 'frequency_band', 'channel'])

                    df_results = df_results.assign(
                        p_bonferroni=df_results.groupby(['condition', 'order', 'frequency_band', 'version'])['p_uncorrected'].transform(lambda x: multipletests(x, method='bonferroni')[1]),
                        p_fdr=df_results.groupby(['condition', 'order', 'frequency_band', 'version'])['p_uncorrected'].transform(lambda x: multipletests(x, method='fdr_bh')[1])
                    )
                    df_results = df_results.reset_index()
                    # print(df_results.columns)
                    if replication_inspired == True:
                        df_results.to_csv(figures_path / 'eeg_correlations_results_replication_inspired_reversed_reference.csv', index=False)
                    if replication_inspired == False:
                        df_results.to_csv(figures_path / 'eeg_correlations_results_reversed_reference.csv', index=False)

                    # Apply FDR correction to p_channels_uncorrected
                    _, p_channels_corrected = fdrcorrection(p_channels_uncorrected, alpha=0.05)
            
                    p_channels = np.sign(
                        (np.array(p_channels_uncorrected) < 0.05).astype(float)  # 1 if uncorrected p < 0.05
                        - 2 * (np.array(p_channels_corrected) < 0.05).astype(float)  # 1 if FDR-corrected p < 0.05
                    )

                    if replication_inspired == True:
                        info = eeg.copy().pick(reorder).info
                    else:
                        info = eeg.info

                    if condition == 'Repos':
                        condition = 'Rest'

                    x_ranked = rankdata(x)
                    y_ranked = rankdata(y)
                    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), constrained_layout=True)

                    if replication_inspired == True:
                        
                        if rank_data == True:
                            update_figure_axes('PSD (window duration = 4 sec), 6 channels', band, axes, x_ranked, y_ranked, slope, intercept, r_value, p_value, r_channels, p_channels, info, rank_data = True, replication_inspired = True)
                            name = f"Order-{order}_{band}_Cond-{condition}_ranked.png"
                            folder = figures_path / 'unused_figures' / 'reversed_reference' / 'average_replication_inspired_analysis' / 'ranked'
                        else:
                            update_figure_axes('PSD (window duration = 4 sec), 6 channels', band, axes, x, y, slope, intercept, r_value, p_value, r_channels, p_channels, info, rank_data = False, replication_inspired = True)
                            name = f"Order-{order}_{band}_Cond-{condition}.png"
                            folder = figures_path / 'unused_figures' / 'reversed_reference' / 'average_replication_inspired_analysis' / 'absolute'
                            
                    else:
                        if rank_data == True:
                            update_figure_axes('PSD (window duration = 20 sec), 62 channels', band, axes, x_ranked, y_ranked, slope, intercept, r_value, p_value, r_channels, p_channels, info, rank_data = True, replication_inspired = False)
                            name = f"Order-{order}_{band}_Cond-{condition}_ranked.png"
                            folder = figures_path / 'unused_figures' / 'reversed_reference' / 'mastoids_data_adapted_analysis' /'ranked'
                        else:
                            update_figure_axes('PSD (window duration = 20 sec), 62 channels', band, axes, x, y, slope, intercept, r_value, p_value, r_channels, p_channels, info, rank_data = False, replication_inspired = False)
                            if band == 'Slow (0.3 - 1 Hz)' and condition == 'Rest':
                                name = f"Supplementary_Figure_10B.png"
                                folder = figures_path 
                            else:
                                name = f"Order-{order}_{band}_Cond-{condition}.png"
                                folder = figures_path / 'unused_figures' / 'reversed_reference' / 'mastoids_data_adapted_analysis' /'absolute'

########################################################################



                    
                    if replication_inspired:
                        print(
                            f"\n--- Generating figure ---\n"
                            f"Condition: {condition}\n"
                            f"Frequency Band: {band}\n"
                            f"Replication Inspired: {replication_inspired}\n"
                            f"Reference: Average\n"
                            f"Rank Data: {rank_data}\n"
                            f"Figure Name: {name}\n"
                            f"Folder: {folder}\n"
                            f"------------------------\n"
                        )

                    else:
                        print(
                            f"\n--- Generating figure ---\n"
                            f"Condition: {condition}\n"
                            f"Frequency Band: {band}\n"
                            f"Replication Inspired: {replication_inspired}\n"
                            f"Reference: Mastoïds-like\n"
                            f"Rank Data: {rank_data}\n"
                            f"Figure Name: {name}\n"
                            f"Folder: {folder}\n"
                            f"------------------------\n"
                        )
                

                    if replication_inspired:
                        save_figure(fig, name, compute_spectrum_replication_inspired_params_rev_ref_job, folder, plotly=False, with_job=True)
                        save_figure(fig, name, compute_spectrum_replication_inspired_params_rev_ref_job, folder, plotly=False, with_job=False)
                    else:
                        save_figure(fig, name, compute_spectrum_figure_rev_ref_job, folder, plotly=False, with_job=True)
                        save_figure(fig, name, compute_spectrum_figure_rev_ref_job, folder, plotly=False, with_job=False)

                    
                    plt.show()
                    plt.close(fig)
                    
                    if condition == 'Rest':
                        condition = 'Repos'

    return

def compute_spectrum_all_reversed_ref():

    run_keys = []
    for base in subject_keys:
        for sess in ['Repos', 'Distraction']:
            run_key = f"{base}_{sess}"
            run_keys.append(run_key)

    run_keys = tuple(run_keys)

    jobtools.compute_job_list(compute_spectrum_replication_inspired_params_rev_ref_job, run_keys, force_recompute=False, engine = 'loop')
    if spectrum_params_replication_reversed_ref['rank_data'] == True:
        print("spectrum for replication-inspired analysis with reversed reference (average) on ranked data computed")
    else:
        print("spectrum for replication-inspired analysis with reversed reference on real data computed")

    jobtools.compute_job_list(compute_spectrum_rev_ref_job, run_keys, force_recompute=False, engine = 'loop')
    if spectrum_params_reversed_ref['rank_data'] == True:
        print("spectrum for our analysis with reversed reference (mastoïd-like) on ranked data computed")
    else:
        print("spectrum for our analysis with reversed reference (mastoïd-like) on real data computed")

def plot_correlation_EEG_memory_reversed_reference(ranked_data, replication_inspired_analysis):

    temp_spectrum_params_replication = spectrum_params_replication_reversed_ref.copy()
    temp_spectrum_params = spectrum_params_reversed_ref.copy()

    temp_spectrum_params_replication['rank_data'] = ranked_data
    temp_spectrum_params['rank_data'] = ranked_data

    if replication_inspired_analysis:
        print("\n--- Starting Replication-Inspired Analysis with Average Reference ---")
        compute_spectrum_figure(
            run_key='sub-01_Repos',
            replication_inspired=True,
            **temp_spectrum_params_replication
        )
        rank_type = "ranked" if ranked_data else "real"
        print(f"Figures for replication-inspired analysis ({rank_type} data) computed")
    else:
        print("\n--- Starting Data-Adapted Analysis with Mastoïd-like Reference ---")
        compute_spectrum_figure(
            run_key='sub-01_Repos',
            replication_inspired=False,
            **temp_spectrum_params
        )
        rank_type = "ranked" if ranked_data else "real"
        print(f"Figures for data-adapted analysis ({rank_type} data) computed")


compute_spectrum_replication_inspired_params_rev_ref_job = jobtools.Job(precomputedir, 'compute_spectrum_replication_inspired_rev_ref', spectrum_params_replication_reversed_ref, compute_spectrum)
jobtools.register_job(compute_spectrum_replication_inspired_params_rev_ref_job)

compute_spectrum_rev_ref_job = jobtools.Job(precomputedir, 'compute_spectrum_rev_ref', spectrum_params_reversed_ref, compute_spectrum)
jobtools.register_job(compute_spectrum_rev_ref_job)

compute_spectrum_figure_replication_inspired_rev_ref_params_job = jobtools.Job(precomputedir, 'compute_spectrum_figure_replication_inspired_rev_ref', spectrum_params_replication_reversed_ref, compute_spectrum_figure)
jobtools.register_job(compute_spectrum_figure_replication_inspired_rev_ref_params_job)

compute_spectrum_figure_rev_ref_job = jobtools.Job(precomputedir, 'compute_spectrum_figure_rev_ref', spectrum_params_reversed_ref, compute_spectrum_figure)
jobtools.register_job(compute_spectrum_figure_rev_ref_job)

if __name__ == "__main__":


    plot_correlation_EEG_memory_reversed_reference(ranked_data=False, replication_inspired_analysis=True)
    plot_correlation_EEG_memory_reversed_reference(ranked_data=False, replication_inspired_analysis=False)



