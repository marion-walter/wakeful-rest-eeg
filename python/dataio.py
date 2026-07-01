"""
Data Loading Utilities for EEG and Behavioral Data

This script provides functions to load and process EEG and behavioral data for the RelaxCons project.
It uses a centralized index (main_index) to locate files and extract metadata.

Key Features:
- Load EEG data from BrainVision files using Neo
- Read behavioral data from CSV files
- Link encoding and recall phases for score analysis
- Load metadata and distraction task data
- Handle special cases like missing files or scores
"""

import numpy as np
import neo
import xarray as xr
import pandas as pd
import physio

from configuration import *

def get_main_index():
    """
    Load the main index DataFrame from NetCDF file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata for all runs, including file paths and annotations.
    """
    ds = xr.open_dataset(main_index_filename)
    main_index = ds.to_dataframe().copy()
    ds.close()
    return main_index

def get_neo_reader(run_key):
    """
    Initialize a Neo RawIO reader for a BrainVision file.

    Parameters
    ----------
    run_key : str
        Unique identifier for the run (e.g., 'sub-01_Repos').

    Returns
    -------
    neo.rawio.BrainVisionRawIO
        Neo RawIO object with parsed header for the specified BrainVision file.
    """
    main_index = get_main_index()
    vhdr_file = main_index.at[run_key, "vhdr_file"]
    vhdr_file = base_folder / vhdr_file

    reader = neo.rawio.BrainVisionRawIO(str(vhdr_file))
    reader.parse_header()

    return reader

def _read_on_channel(run_key, channel_name):
    main_index = get_main_index()
    vhdr_file = main_index.at[run_key, "vhdr_file"]
    vhdr_file = base_folder / vhdr_file

    trace, sr = physio.read_one_channel(
        filename=vhdr_file,
        format='brainvision',
        channel_name=channel_name,
        scaled=True)

    return trace, sr

def read_ecg(run_key):
    from params import ecg_subject_sign
    subject = run_key.split('_')[0]
    
    sign = ecg_subject_sign.get(subject, 1)

    ecg, sr = _read_on_channel(run_key, 'ECG')
    if sign < 0:
        ecg *= sign
    return ecg, sr

def read_eeg(run_key):
    """
    Read EEG signals for a specific run.

    Parameters
    ----------
    run_key : str
        Unique identifier for the run (e.g., 'sub-01_Repos').

    Returns
    -------
    tuple
        - sigs : ndarray
            2D array of EEG signals (timepoints × channels)
        - sr : float
            Sampling rate in Hz
        - chan_names : list
            List of channel names (excluding last 3 channels)
    """
    reader = get_neo_reader(run_key)
    chan_names = reader.header["signal_channels"]["name"]
    sr = reader.header["signal_channels"]["sampling_rate"][0]
    chan_names = chan_names[:-3]  # Exclude last 3 channels (respiration and heart channels)

    raw_signal = reader.get_analogsignal_chunk(
        stream_index=0,
        channel_names=chan_names)
    sigs = reader.rescale_signal_raw_to_float(raw_signal)

    return sigs, sr, chan_names

def read_behavior(run_key, with_related_score=False, with_final_score=False):
    """
    Read behavioral data for a run, optionally with related recall scores.

    Parameters
    ----------
    run_key : str
        Unique identifier for the run (e.g., 'sub-01_E1').
    with_related_score : bool, optional
        If True, adds related_score_initial by matching with recall phase data.
    with_final_score : bool, optional
        If True, adds related_score_final by matching with final recall phase data.

    Returns
    -------
    pd.DataFrame or None
        Behavioral data as DataFrame. Returns None if no behavior file exists for the run_key.
        May include additional columns for recall scores if requested.
    """
    main_index = get_main_index()

    behavior_file = main_index.at[run_key, "behavior_file"]
    if behavior_file is None or behavior_file == '':
        return None

    behavior_file = base_folder / behavior_file

    # Read CSV with flexible separator and Latin-1 encoding
    behavior = pd.read_csv(behavior_file, sep=None, engine="python", encoding='latin-1')
    behavior = behavior.rename(columns={"subject_ID": "subject"})

    sub, phase = run_key.split("_")

    if with_related_score and phase in ('E1', 'E2'):
        # Get recall phase data (R1 or R2)
        run_key_related = sub + "_R" + run_key[-1]
        recall_behavior = read_behavior(run_key_related, with_related_score=False)

        # Add initial recall scores
        behavior['related_score_initial'] = pd.Series(dtype='int64')
        for _, row in recall_behavior.iterrows():
            # Find matching image_object_id in encoding data
            inds = np.flatnonzero(behavior['image_object_id'].values == row["image_object_encodage_id"])
            assert inds.size == 1, f"Expected 1 match, found {inds.size}"
            ind = inds[0]
            behavior.at[ind, 'related_score_initial'] = row["score"]

        if with_final_score:
            # Get final recall phase data (RF)
            run_key_related_final = sub + "_RF"
            recall_behavior_final = read_behavior(run_key_related_final, with_related_score=False)

            # Add final recall scores
            behavior['related_score_final'] = pd.Series(dtype='int64')
            for _, row in recall_behavior_final.iterrows():
                print(row)
                print(behavior['image_object_id'].values)
                inds = np.flatnonzero(behavior['image_object_id'].values == row["image_object_encodage_id"])
                if inds.size == 1:
                    ind = inds[0]
                    behavior.at[ind, 'related_score_final'] = row["score"]
                elif inds.size == 0:
                    behavior.at[ind, 'related_score_final'] = None

    return behavior

def get_one_metadata_info(info):
    """
    Load a specific metadata column from the meta-data CSV file.

    Parameters
    ----------
    info : str
        Name of the metadata column to extract (e.g., 'age').

    Returns
    -------
    pd.DataFrame
        DataFrame with 'subject' and the requested metadata column for the first 60 subjects.
    """
    meta_data = pd.read_csv(data_path / 'meta-data.csv', sep=";", encoding='latin')
    df = pd.DataFrame(meta_data)
    df = df[:60]  # Limit to first 60 rows
    df = df[['subject', info]]
    return df

def get_distraction_task_data():
    """
    Load and format distraction task behavioral data.

    Returns
    -------
    pd.DataFrame
        DataFrame with distraction task data containing:
        - subject: Subject identifier
        - trial_number: Trial number
        - score: Task score
        - rt: Reaction time
    """
    main_index = get_main_index()

    # Query for distraction phase runs
    query = f"phase == '{'dist'}'"
    print(query)
    run_keys = list(main_index.query(query).index)

    df = []
    for run_key in run_keys:
        df.append(read_behavior(run_key))

    df = pd.concat(df, axis=0)

    # Select and clean columns
    df.drop(['type', 'sequence_number_1', 'sequence_number_2',
             'correct_response', 'subject_response'], axis=1, inplace=True)
    df = df[['subject', 'trial_number', 'score', 'rt']]

    return df

def get_distraction_task_difficulty_data():
    """
    Load and format distraction task difficulty ratings from final questionnaire.

    Returns
    -------
    pd.DataFrame
        DataFrame with distraction task difficulty ratings, containing:
        - Distraction_task_difficulty: Renamed from original 'Difficulté_Dist' column
    """
    main_index = get_main_index()

    # Query for final questionnaire runs
    query = f"phase == '{'qfinal'}'"
    print(query)
    run_keys = list(main_index.query(query).index)

    df = []
    for run_key in run_keys:
        df.append(read_behavior(run_key))

    df = pd.concat(df, axis=0)

    # Select and clean columns
    df.drop(['date', 'Relaxation',
       'IntensitÃ©_Emotions', 'IntensitÃ©_Souvenirs', 'Attention_respi'], axis=1, inplace=True)
    df.rename(columns={"DifficultÃ©_Dist": "Distraction_task_difficulty"}, inplace=True)

    return df

if __name__ == '__main__':
    # Example usage: Load and print distraction task data
    df = get_distraction_task_data()
    print(df)