"""
EEG PREPROCESSING PARAMETER CONFIGURATION SCRIPT

This script defines all parameters and configurations for EEG data preprocessing and analysis
in the RelaxCons project. It serves as a centralized configuration hub for:

1. SUBJECT AND SESSION MANAGEMENT:
   - Lists of subject identifiers (all subjects, DR group, RD group)
   - Session types (Repos, Distraction)
   - Run keys combining subjects and sessions

2. EEG PREPROCESSING PARAMETERS:
   - Trimming: Methods to remove artifacts from signal edges (centered, both, delay, conservative)
   - Detrending: Methods to remove low-frequency drifts (linear, robust linear, trend filter, highpass)
   - Filtering: Notch and bandpass filter configurations
   - Artifact Removal: Parameters for EOG, ECG, and EMG artifact correction
   - Rereferencing: Options for different referencing schemes (physical, average, REST, CSD)

3. PREDEFINED PIPELINE:
   - `modular_eeg_preprocessing_params`: Advanced modular pipeline with step ordering

IMPORTANT: This is a critical configuration script used throughout the analysis pipeline.

USAGE:
------
Import this script in your analysis scripts to access all predefined parameters.
Example:
    from configuration import modular_eeg_preprocessing_params, subject_keys
    # Then use these parameters in your processing functions
"""


subject_keys = ['sub-01', 'sub-10', 'sub-20', 'sub-22', 'sub-24', 'sub-30', 'sub-38', 
                 'sub-40', 'sub-41', 'sub-45', 'sub-46', 'sub-50', 'sub-52', 'sub-53', 
                 'sub-58', 'sub-07', 'sub-12', 'sub-14', 'sub-16', 'sub-19', 'sub-23', 
                 'sub-25', 'sub-26', 'sub-31', 'sub-34', 'sub-43', 'sub-44', 'sub-47', 
                 'sub-49', 'sub-59']


subject_keys_DR = ['sub-01','sub-10','sub-20','sub-22','sub-24','sub-30','sub-38','sub-40','sub-41','sub-45','sub-46','sub-50','sub-52','sub-53','sub-58']
subject_keys_RD = ['sub-07','sub-12','sub-14','sub-16','sub-19','sub-23','sub-25','sub-26','sub-31','sub-34','sub-43','sub-44','sub-47','sub-49','sub-59']

session_keys = ["Repos", "Distraction"]

run_keys = [f'{sub_key}_{ses_key}' for sub_key in subject_keys for ses_key in session_keys]

distraction_keys = [f'{sub_key}_Distraction' for sub_key in subject_keys]
rest_keys = [f'{sub_key}_Repos' for sub_key in subject_keys]


    
################################## Parameters for EEG preprocessing ##################################



# Trimming parameters
'''
Parameters used to trim both ends of EEG signals to avoid artifacts caused by the beginning and end of the experiment



- centered : find the duration center, then keep a symmetrical window around it
             * window_duration : total window duration in seconds, i.e there will be half the window before and half after the center

- both : trim both ends based on specified amounts
         * start : trim 'start' seconds from the beginning
         * end : trim 'end' seconds from the end

- delay : trim a fixed amount at the beginning, a fixed amount at the end and then keep only a certain duration of the signal after the start
          * start : trim 'start' seconds from the beginning
          * end : trim 'end' seconds from the end, in case the window is too long
          * window_duration : keep a window of duration 'window' after the starting point

- conservative : insures that the duration is at least either the minimum between the window duration and the session duration and
                 while doing so trims up to the specficied start value at the beginning

- task_trimming : only keeps the signal obtained during the execution of the task
          

'''

session_timmings = {
    'E' : {'fixation_cross' : -1, 'image_display_time': 5, 'maximum_answer_time' : 5},
    'R' : {'fixation_cross' : -1, 'image_display_time': 5, 'maximum_answer_time' : 15}
}


session_duration = {
    'baseline' : 5*60, 
    'Repos' : 10*60,
    'Distraction' : 10*60,
    'E1' : 25*15,
    'E2' : 25*15,
    'R1' : 25*15,
    'R2' : 25*15,
    'RF' : 50*15,
}


centered_trimming = dict(method = 'centered', window_duration = session_duration, session_adapted = True, session_parameters = session_timmings, interpolate_triggers = True  )

both_trimming = dict(method = 'both', start = 10, end = 10, session_adapted = True, session_parameters = session_timmings, interpolate_triggers = True)

delay_trimming = dict(method = 'delay', start = 10, end = 10, window_duration = session_duration, session_adapted = True, session_parameters = session_timmings, interpolate_triggers = True)

conservative_trimming = dict(method  = 'conservative', window_duration = session_duration, start = 10, session_adapted = True, session_parameters = session_timmings, interpolate_triggers = True)

trimming_parameters = conservative_trimming



# Detrending parameters
'''
The following dicts should be modified according to user preferences and
their variable name should be associated to the detrending key in the 
eeg_preprocessing_params dict

- Linear_detrending : substract a linear fit of the signal using scipy.detrend

- robust_linear_detrending : using meegkit.detrend.detrend, substract a polynomial fit computed on an iteratively weighted
                             to reduce influence of large deviations
                             * order : degree of the polynomial function used of the fit, defaults to 1 (linear fit)
                             * threshold : number of standard deviation for outlier detection and weight update, defaults to 3
                             * n_iter : number of iterations, defaults to 6

- trend_filter_detrending : based on L1 norm trend filtering, see Tibshirani 2014 for details. 
                            * vlambda : controls the regularization, the higher the value, the closer to a linear fit, defaults to 0.001
                            * downsample : reduce the size of the fitted data, avoiding prohibitingly long computations and 
                                           fit-induced deletion of frequency of interest, defaults to 5000

- highpass_detrending : using a FIR zero-phase filter to high-pass the signal and remove ultra low frequency components
                        resulting in detrending. Creates large amplitude ringing artifacts at every signal discontinuity
                        * cutoff_frequency : defaults to 0.05 Hz
                        * step_removal : scan signals for steps and other big variations to partially prevent filter's artifacts
                        * time_window : window length for the embedding
                        * energy_threshold : for the step-removal method, ditactes which epochs are artifactual based on their energy
                        * peak_to_peak_threshold : for the step-removal method, ditactes which epochs are artifactual based on their 
                                                    peak-to-peak amplitude.
'''

linear_detrending = dict(name='linear',)
robust_linear_detrending= dict(name='robust_linear', order=1, threshold=3, n_iter=6)
trend_filter_detrending = dict(name='trend_filter', vlambda = 0.001, downsample = 5000)
highpass_detrending_slow = dict(name='highpass', cutoff_frequency = 0.05, step_removal=True, time_window=500, energy_threshold=10, peak_to_peak_threshold=5)
highpass_detrending_fast = dict(name='highpass', cutoff_frequency = 1, step_removal=True, time_window=500, energy_threshold=10, peak_to_peak_threshold=5)

# Notch filtering parameters
'''
Parameters configuring a notch filter implemented using mne.filter.notch_filter:
    * notch_frequency : frequencies around which the signal will be notched
    * method : finite impulse response (fir) or infinite impulse response (iir)
    * iir_params : parameters for the iir method (ex dict(order=6, ftype="butter") ), should be None for fir method
    * phase : if zero, will compensate for the introduced delay by making the filter non-causal
'''
notch_filtering_params = dict(notch_frequency = [50, 100, 150, 200], method='fir', iir_params=None, phase='zero')

# Filtering parameters
'''
Parameters configuring a filter implemented using mne.filter.filter_data:
     
    * method : finite impulse response (fir) or infinite impulse response (iir)
    * iir_params : parameters for the iir method (ex dict(order=6, ftype="butter") ), should be None for fir method
    * phase : if zero, will compensate for the introduced delay by making the filter non-causal
'''
filtering_params_slow = dict(low_cutoff = None, high_cutoff = 200, method='fir', iir_params=None, phase='zero')
filtering_params_fast = dict(low_cutoff = 1, high_cutoff = 200, method='fir', iir_params=None, phase='zero')

# ICA parameters
'''
Parameters to compute ICA
            * n_components : number of components kept during the PCA step preceding ICA
            * filter_signal : filters a copy of the raw array before feeding it to ICA, defaults to [1,100] (Hz)
            * set_reference : sets the reference for the copy of the raw array
            ** For other parameters, see https://mne.tools/1.8/generated/mne.preprocessing.ICA.html
'''

ICA_computation_params = dict(name = 'ICA', n_components = 30, filter_signal = [1,100], set_reference = None,
                              random_state=42, method='infomax', fit_params=dict(extended=True), max_iter=100, use_torch=True, )

# EOG artifact removal parameters
'''
Parameters configuring different methods to correct EOG artifacts:

    - ICA_EOG : allows user to select EOG ICA components and remove them from the signal
            * ICA_params : indicates the parameters used to prepare the raw array for the ICA fit
            * selection_method : indicates how components should be choosen, defaults to 'iclabel' (automatic detection, otherwise 'manual' to allow manual selection)
            * automatic_label : which automatic label should be used for the selection

    - KSSA_ICA : uses ICA transformation, then K-means and SSA to locally correct the selected components and finally ICA to revert back to sensor signals
                 * window_length_seconds : time window considered for the embedding. While dealing with EOG artifacts, 500ms is appropriate
                 * kmeans_cluster_count : number of clusters to use for kmeans, 5 seems to work quite well
                 * standardize : whether to use robust z score to standardize the features before clusters (if not priority is given to energy since Kmeans is used)
                 * fractal_dimension_threshold : EOG artifacts tend to have a lower fractal dimension than EEG signal, which allow discrimination
                 * ssa_threshold : EOG artifacts tend to have high energy, hence a threshold on the SVD singular values to discriminate again
                 * ICA_params : indicates the parameters used to prepare the raw array for the ICA fit
                 * selection_method : indicates how components should be choosen, defaults to 'iclabel' (automatic detection, otherwise 'manual' to allow manual selection)
                 * automatic_label : which automatic label should be used for the selection

    - CWT_MSSA_ICA : uses ICA transformation, then wavelet transform and SSA to locally correct the selected components and finally ICA to revert back to sensor signals
                 * wavelets_frequency_spacing : frequency spacing for the wavelet transform, defaults to 0.1 Hz
                 * wavelets_min_frequency : minimal frequency for the wavelet transform, defaults to 0.1 Hz (blinks and saccades are considered here)
                 * wavelets_max_frequency : maximal frequency for the wavelet transform, defaults to 12 Hz (blinks and saccades are considered here)
                 * filter_f_min : frequency to use to highpass filtering the data to compute metric function for simulated artifact, defaults to 1 Hz (to remove large variations)
                 * filter_f_max : frequency to use to lowpass filtering the data to compute metric function for simulated artifact, defaults to None
                 * metric_function : function to use for a measure of variation to build the simulated artifact, defaults to MAD
                 * metric_factor : factor to multiply the obtained metric to obtain the amplitude of the simulated artifact, defaults to 5
                 * margins : list of sample point numbers to determine the number of points used to build the smoothing of the artifact mask, defaults to [500,5000]
                 * widths : list of sample point numbers to determine the width of the edge smoothing to build the artifact mask, defautls to [50,1000]
                 * ssa_window_seconds: time window considered for the first ssa embedding, defaults to 0.3 s
                 * first_ssa_trajectories_indices : first ssa trajectories to compute based on the list of indices of the ordered eigenvalues of the SVD decomposition, defaults to [0,...,6]
                 * second_ssa_downsampling : downsampling used to allow for larger windows for the second ssa, defaults to 10
                 * second_ssa_window_seconds : time window considered for the second ssa embedding, defaults to 5 s
                 * second_ssa_trajectories_indices : second ssa trajectories to compute based on the list of indices of the ordered eigenvalues of the SVD decomposition, defaults to [0,...,30]
                 * return_unbias : whether to return the unbiased version or the biased version of the fit
                 * return_corrected : whether to return the signal with the subtraction of the artifacts, or the artifactual signal
                 * ICA_params : indicates the parameters used to prepare the raw array for the ICA fit
                 * selection_method : indicates how components should be choosen, defaults to 'iclabel' (automatic detection, otherwise 'manual' to allow manual selection)
                 * automatic_label : which automatic label should be used for the selection

    - Rectified_CWT_MSSA_ICA : rectifies the signal to avoid false positive oscillations provoked by saccades
                 * ... see above for CWt_MSSA parameters
                 * artifact_margin :  sample point numbers to determine the number of points used to build the smoothing of the artifact mask, defaults to 500, 
                 * artifact_plateau : width in sample points of the plateau around the artifact, defaults to 250 
                 * artifact_shift : number of points to shift the plateau to the past to take the beginning of the artifact into account, defaults to 50
                 * smoothing_size : convolution mask size in number of points for smoothing, defaults to 1000
                 * smoothing_first_pass_width : size of the door in number of points for the convolution smoothing, defaults to 100
                 * smoothing_first_pass_iteration: number times to apply the convolution smoothing, defaults to 4
                 * smoothing_second_pass_width : size of the door in number of points for the convolution smoothing of the gradient, defaults to 10
                 * gradient_peaks_height : heigth used by scipy find peaks to find gradient exrtema, defaults to 3 (zscore)
                 * gradient_peaks_prominence :promionence used by scipy find peaks to find gradient exrtema, defaults to 3 (zscore)
                 * gradient_peaks_width : width used by scipy find peaks to find gradient exrtema, defaults to 10
                 * rectification_drift_highpass : highpass frequency to filter the final correction and remove its drift
                 

'''


ICA_EOG_params = dict(name = 'ICA', ICA_params = ICA_computation_params, selection_method = 'iclabel', automatic_label='eye blink' )

KSSA_ICA_params = dict(name = 'KSSA_ICA', window_length_seconds = 0.5, kmeans_cluster_count = 5, standardize = False,
                        fractal_dimension_threshold = 1.4, ssa_threshold = 0.01, ICA_params = ICA_computation_params, selection_method = 'iclabel', automatic_label='eye blink')


CWT_MSSA_ICA_EOG_params = dict(name = 'CWT_MSSA_ICA', wavelets_frequency_spacing = 0.1, wavelets_min_frequency = 0.1, wavelets_max_frequency = 12,
                  filter_f_min = 1, filter_f_max = None, metric_function='MAD', metric_factor=3,
                  margins = [500,5000], widths = [50, 1000], ssa_window_seconds = 0.3,
                  first_ssa_trajectories_indices = [i for i in range(7)], second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = [i for i in range(80)],
                  return_unbias = True, return_corrected = False,
                  ICA_params = ICA_computation_params, selection_method = 'iclabel', automatic_label='eye blink')


Rectified_CWT_MSSA_ICA_EOG_params = dict(name = 'Rectified_CWT_MSSA_ICA', wavelets_frequency_spacing = 0.1, wavelets_min_frequency = 0.1, wavelets_max_frequency = 12,
                  filter_f_min = 1, filter_f_max = None, metric_function='MAD', metric_factor=1.5,
                  margins = [500,5000], widths = [50, 1000], ssa_window_seconds = 0.3,
                  first_ssa_trajectories_indices = [i for i in range(7)], second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = [i for i in range(80)],
                  return_unbias = False, return_corrected = False, show=True, return_localisation = True,
                  artifact_margin = 500, artifact_plateau = 250, artifact_shift = 50, smoothing_size=1000,
                  smoothing_first_pass_width = 100, smoothing_first_pass_iteration = 4, smoothing_second_pass_width = 10,
                  gradient_peaks_height=3, gradient_peaks_prominence=3, gradient_peaks_width=10,
                  rectification_drift_highpass = 0.05, ease_in_width=100,
                  ICA_params = ICA_computation_params, selection_method = 'iclabel', automatic_label='eye blink')

Rectified_V2_CWT_MSSA_ICA_EOG_params_slow = dict(name = 'Rectified_V2_CWT_MSSA_ICA', iterations = 3,
                    wavelets_frequency_spacing = 0.1, wavelets_min_frequency = 0.1, wavelets_max_frequency = 12,
                  filter_f_min = 1, filter_f_max = None, metric_function='MAD', metric_factor=[3,5],
                  margins = [500,5000], widths = [100, 1000], ssa_window_seconds = 0.3,
                  first_ssa_trajectories_indices = [i for i in range(8)], second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = [i for i in range(80)], show=False, 
                  artifact_margin = 500, artifact_plateau = 350, artifact_shift = 75, wavelet_detection_iteration = 6,
                  bridging =0.1, bias_ssa_downsampling = 20, bias_ssa_window_seconds = 80 , bias_ssa_trajectories_indices = [5,6,7],
                    cascade = {0:{'indices':0}}, rolling_average_duration= 10,
                  ICA_params = ICA_computation_params, selection_method = 'iclabel', automatic_label='eye blink')

Rectified_V3_CWT_MSSA_ICA_EOG_params_slow = dict(name = 'Rectified_V3_CWT_MSSA_ICA', iterations = 3, unbiasing=False,
                    wavelets_frequency_spacing = 0.1, wavelets_min_frequency = 0.1, wavelets_max_frequency = 12,
                  filter_f_min = 1, filter_f_max = None, metric_function='MAD', 
                  metric_factor={'eye blink' : [5, 3, 2], 'other' : [5, 5, 5], 'channel noise': [5, 5, 5], 'muscle artifact' : [5, 5, 5]},
                  margins = [500,5000], widths = [100, 1000], ssa_window_seconds = 0.3,
                  first_ssa_trajectories_indices = [i for i in range(8)], second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = [i for i in range(80)], show=False, 
                  artifact_margin = 500, artifact_plateau = 350, artifact_shift = 75, wavelet_detection_iteration = 6,
                  bridging =0.1, bias_ssa_downsampling = 20, bias_ssa_window_seconds = 80 , bias_ssa_trajectories_indices = [5,6,7],
                    cascade = {0:{'indices':0}}, rolling_average_duration= 20,
                  ICA_params = ICA_computation_params, selection_method = 'iclabel', automatic_label='eye blink')




# ECG artifact removal parameters
'''
Parameters configuring different methods to correct ECG artifacts:

    - ICA_EOG : allows user to select ECG ICA components and remove them from the signal
            * ICA_params : indicates the parameters used to prepare the raw array for the ICA fit
            * selection_method : indicates how components should be choosen, defaults to 'iclabel' (automatic detection, otherwise 'manual' to allow manual selection)
            * automatic_label : which automatic label should be used for the selection

    - CWT_MSSA_ICA : uses ICA transformation, then wavelet transform and SSA to locally correct the selected components and finally ICA to revert back to sensor signals
                 * wavelets_frequency_spacing : frequency spacing for the wavelet transform, defaults to 0.1 Hz
                 * wavelets_min_frequency : minimal frequency for the wavelet transform, defaults to 0.1 Hz (blinks and saccades are considered here)
                 * wavelets_max_frequency : maximal frequency for the wavelet transform, defaults to 12 Hz (blinks and saccades are considered here)
                 * filter_f_min : frequency to use to highpass filtering the data to compute metric function for simulated artifact, defaults to 1 Hz (to remove large variations)
                 * filter_f_max : frequency to use to lowpass filtering the data to compute metric function for simulated artifact, defaults to None
                 * metric_function : function to use for a measure of variation to build the simulated artifact, defaults to MAD
                 * metric_factor : factor to multiply the obtained metric to obtain the amplitude of the simulated artifact, defaults to 5
                 * margins : list of sample point numbers to determine the number of points used to build the smoothing of the artifact mask, defaults to [500,5000]
                 * widths : list of sample point numbers to determine the width of the edge smoothing to build the artifact mask, defautls to [50,1000]
                 * ssa_window_seconds: time window considered for the first ssa embedding, defaults to 0.3 s
                 * first_ssa_trajectories_indices : first ssa trajectories to compute based on the list of indices of the ordered eigenvalues of the SVD decomposition, defaults to [0,...,6]
                 * second_ssa_downsampling : downsampling used to allow for larger windows for the second ssa, defaults to 10
                 * second_ssa_window_seconds : time window considered for the second ssa embedding, defaults to 5 s
                 * second_ssa_trajectories_indices : second ssa trajectories to compute based on the list of indices of the ordered eigenvalues of the SVD decomposition, defaults to [0,...,30]
                 * return_unbias : whether to return the unbiased version or the biased version of the fit
                 * return_corrected : whether to return the signal with the subtraction of the artifacts, or the artifactual signal
                 * ICA_params : indicates the parameters used to prepare the raw array for the ICA fit
                 * selection_method : indicates how components should be choosen, defaults to 'iclabel' (automatic detection, otherwise 'manual' to allow manual selection)
                 * ecg_threshold : threshold above which the correlation will be considered as indicative of heartbeat artifacts
                 * trimming parameters : trimming parameters to trim the ecg signal

'''


ICA_ECG_params = dict(name = 'ICA', ICA_params = ICA_computation_params, selection_method = 'iclabel', automatic_label='eye blink' )

CWT_MSSA_ICA_ECG_params = dict(name = 'CWT_MSSA_ICA', wavelets_frequency_spacing = 0.5, wavelets_min_frequency = 8, wavelets_max_frequency = 16,
                  filter_f_min = 8, filter_f_max = 16, metric_function='std', metric_factor=3,
                  margins = [50,1000], widths = [5, 500], ssa_window_seconds = 0.05,
                  first_ssa_trajectories_indices = [i for i in range(10)], second_ssa_downsampling = 10, 
                  second_ssa_window_seconds = 5, second_ssa_trajectories_indices = [i for i in range(80)],
                  return_unbias = False, return_corrected = False,
                  ICA_params = ICA_computation_params, selection_method = 'mne_bad_ecg', ecg_threshold=0.7, trimming_parameters = trimming_parameters)


# EMG artifact removal parameters
'''
Parameters configuring different methods to detect and potnetially correct EMG artifacts:

    - EMG_rms : based on Valentin Ghibaudo's work, finds EMG artifacts using sliding rms values and a median/MAD deviation threshold
            * window_duration : duration for the sliding rms window
            * deviation_number : number of MAD relatively to the median the rms value has to be to overcome the threshold
            * sliding_rms_window : duration of the sliding window used to compute the rms
            * margin : padding duration to smothly insert the correction
            * fmin : high-pass cutoff used to filter the noise patch
            * correction : define the methods to use to correct artifact patches
                           ** average_noise : replace the artifact by noise having the same psd as the clean segments


'''


EMG_rms_params = dict(name = 'EMG_rms', window_duration=1, deviation_number = 5, sliding_rms_window = 1, margin = 0.2, fmin = 0.05,  correction = 'average_noise' )


# Big artifact removal parameters
'''
Parameters configuring different methods to detect and potnetially correct EMG artifacts:

    - big_ssa : based on SSA, finds big artifacts using sliding peak to peak and a MAD deviation threshold
            * mad_factor : factor used to build threshold from MAD multiplication
'''

big_artifact_removal_params_slow = dict(name='big_ssa', mad_factor = 10)
big_artifact_removal_params_fast = dict(name='big_ssa', mad_factor = 10)

# Rereferencing parameters
'''
The following dicts should be modified according to user preferences and
their variable name should be associated to the rereferencing key in the 
eeg_preprocessing_params dict

- physical_channel_rereferencing : rereference with provided physical channels using mne.set_eeg_reference
                                   * reference_channels : list of physical channels to use as reference,
                                     if several, the average of the channels will be used

- average_rereferencing : rereference with average reference using mne.set_eeg_reference

- rest_rereferencing : rereference with the REST paradigm using mne.set_eeg_reference

- csd_rereferencing : rereference with current source density using mne.preprocessing.compute_current_source_density
'''
physical_channel_rereferencing = dict(method = 'physical_channel', reference_channels=['TP9', 'TP10'])
average_rereferencing = dict(method = 'average', )
rest_rereferencing = dict(method = 'REST', )
csd_rereferencing = dict(method = 'CSD', )


# Main preprocessing dictionnary
'''
Possible keys in the dictionnary :
- rescaling : rescale the signal using a given rescaling_factor value
- trimming : trim the signals to the desired duration
- detrending : remove trends such as drifts from the signal 
- centering : center the signal by removing the mean value
- notch_filtering : removes line noise by notch filtering the signal
- filtering : removes unwanted frequency components from the signal
- compute_ICA : computes an ICA transformation
- EOG_artifact_removal : removs EOG artifacts from the signal
- ECG_artifact_removal : removes ECG artifacts from the signal
- EMG_artifact_removal :remove EMG artifacts from the signal
- rereferencing : rereference the signal

'''

modular_eeg_preprocessing_params = dict(
    correcting_channel_names = dict(),
    rescaling = {'rescaling_factor' : 0.0488281},
    trimming = trimming_parameters,
    detrending = highpass_detrending_slow, # Linear, Robust_linear, Trend_filter, Highpass
    centering__0 = {'method' : 'mean_centering'}, # True/False
    compute_ICA = ICA_computation_params,
    EOG_artifact_removal = Rectified_V3_CWT_MSSA_ICA_EOG_params_slow, # None, ICA_params, KSSA_ICA_params,
    ECG_artifact_removal = CWT_MSSA_ICA_ECG_params,
    EMG_artifact_removal = EMG_rms_params,
    notch_filtering = notch_filtering_params,
    filtering = filtering_params_slow, # filtering_params_fast  dict(lowpass = None, highpass = None, bandpass = None, notch = None),
    big_artifact_removal = big_artifact_removal_params_slow,
    centering__1 = {'method' : 'mean_centering'},
    # rereferencing = None,
)
modular_eeg_preprocessing_params['Order'] = list(modular_eeg_preprocessing_params.keys())


trimming_params = trimming_parameters

ecg_subject_sign = {
    'sub-01': -1.,
    'sub-07': -1.,
    'sub-10': -1.,
    'sub-19': -1.,
    'sub-20': -1.,
    'sub-22': -1.,
    'sub-24': -1.,
}
