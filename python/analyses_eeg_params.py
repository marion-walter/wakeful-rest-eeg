"""
SPECTRAL ANALYSIS PARAMETER CONFIGURATIONS

This script defines parameter sets for spectral analysis of EEG data, with different configurations
for reference schemes and analysis approaches. Each parameter dictionary specifies how to process
EEG signals for frequency domain analysis, including:

Key Features:
- Reference schemes: TP9/TP10 (mastoid) or average reference
- Frequency bands: Slow (0.3-1 Hz) and Alpha (8-12 Hz)
- Spectral resolution: Focused on 2.5 Hz and 3.33 Hz (1/0.4 and 1/0.3)
- Maximum frequency: Limited to 20 Hz
- Data ranking: Option to use raw values or ranks for correlation analysis

Parameter Descriptions:
- equalize_electrode_number: Balance electrode counts across subjects (True/False)
- reference: Reference scheme ('average' or ['TP9', 'TP10'])
- frequency_to_resolve: Frequencies for spectral resolution (in Hz)
- frequency_bands: List of [low, high] frequency ranges for bandpower analysis
- frequency_names: Labels for each frequency band
- spectrum_fmax: Maximum frequency for the spectrum (Hz)
- rank_data: Whether to use ranked data (True) or raw values (False) for analysis
"""
###################################################################################
"""
Parameters for replication-inspired analysis using TP9/TP10 (mastoid) reference
"""
spectrum_params_replication = dict(
    equalize_electrode_number=True,  # Balance the number of electrodes across subjects
    reference=['TP9', 'TP10'],       # Use TP9 and TP10 (mastoids) as reference electrodes
    frequency_to_resolve=[1 / 0.4, 1 / 0.3],  
    frequency_bands=[
        [0.3, 1],   # Slow band: 0.3-1 Hz
        [8, 12],    # Alpha band: 8-12 Hz
    ],
    frequency_names=['Slow', 'Alpha'],  # Names for the frequency bands
    spectrum_fmax=20,  # Maximum frequency for the spectrum: 20 Hz
    rank_data=False,   # By default, rank the data for the correlation analysis. Set to False to use raw values instead of ranks 
)

"""
Parameters for data-specific analysis using average reference
"""
spectrum_params = dict(
    equalize_electrode_number=True,  # Balance the number of electrodes across subjects
    reference='average',              # Use average reference (common average re-referencing)
    frequency_to_resolve=[1 / 0.4, 1 / 0.3],  
    frequency_bands=[
        [0.3, 1],   # Slow band: 0.3-1 Hz
        [8, 12],    # Alpha band: 8-12 Hz
    ],
    frequency_names=['Slow', 'Alpha'],  # Names for the frequency bands
    spectrum_fmax=20,  # Maximum frequency for the spectrum: 20 Hz
    rank_data=False,   
)

"""
Parameters for replication-inspired analysis using average reference
"""
spectrum_params_replication_reversed_ref = dict(
    equalize_electrode_number=True,  # Balance the number of electrodes across subjects
    reference='average',              # Use average reference
    frequency_to_resolve=[1 / 0.4, 1 / 0.3], 
    frequency_bands=[
        [0.3, 1],   # Slow band: 0.3-1 Hz
        [8, 12],    # Alpha band: 8-12 Hz
    ],
    frequency_names=['Slow', 'Alpha'],  # Names for the frequency bands
    spectrum_fmax=20,  # Maximum frequency for the spectrum: 20 Hz
    rank_data=False,    
)

"""
Parameters for data-specific analysis using TP9/TP10 (mastoid) reference)
"""
spectrum_params_reversed_ref = dict(
    equalize_electrode_number=True,  # Balance the number of electrodes across subjects
    reference=['TP9', 'TP10'],       # Use TP9 and TP10 (mastoids) as reference electrodes
    frequency_to_resolve=[1 / 0.4, 1 / 0.3],  
    frequency_bands=[
        [0.3, 1],   # Slow band: 0.3-1 Hz
        [8, 12],    # Alpha band: 8-12 Hz
    ],
    frequency_names=['Slow', 'Alpha'],  # Names for the frequency bands
    spectrum_fmax=20,  # Maximum frequency for the spectrum: 20 Hz
    rank_data=False,   
)