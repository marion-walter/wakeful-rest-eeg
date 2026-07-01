# This script contains functions to analyze memory performance based on encoding and recall data.
# It includes functions to retrieve and concatenate encoding and recall data, merge them, assign trial numbers, 
# and compute memory change metrics such as Global Memory Change (GMC) and Relative Memory Change (RMC).

"""
Import necessary libraries and functions from other modules.
- pandas for data manipulation.
- get_main_index, read_behavior, get_one_metadata_info from dataio for data retrieval.
- get_recall_data, get_encoding_data from behavior_score for specific data retrieval functions.
"""
import pandas as pd
from dataio import get_main_index, read_behavior, get_one_metadata_info
# from behavior_score import get_recall_data, get_encoding_data
from configuration import *

def get_encoding_data(phase):
    """
    Retrieve and concatenate encoding data for a specified encoding pahse (E1 or E2).

    Parameters
    ----------
    phase : str
        The experimental phase to filter data by ('E1', 'E2').

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing all encoding data for the specified phase.
        Columns = ['dates', 'subject', 'image_object_id', 'image_landscape_id', 'Rating'].

    Notes
    -----
    - Assumes `get_main_index()` returns a queryable object (e.g., pandas DataFrame).
    - Assumes `read_behavior(run_key)` returns a pandas DataFrame for each run.
    - The query string is printed for debugging purposes.
    """

    main_index = get_main_index()

    query = f"phase == '{phase}'"
    print(query)
    run_keys = list(main_index.query(query).index)
    
    df = []
    for run_key in run_keys:
        df.append(read_behavior(run_key))
    
    df = pd.concat(df, axis=0)

    return df


def concat_encoding_data():
    """
    Concatenate encoding data from phases E1 and E2, and enrich with metadata.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing concatenated encoding data for E1 and E2,
        with added columns for condition, age, group, and order.

    Notes
    -----
    - Uses `get_encoding_data` to fetch data for phases E1 and E2.
    - Adds an 'E_type' column to distinguish between E1 and E2.
    - Selects specific columns: ['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type'].
    - Further enriches the DataFrame with additional metadata (age, group, order) using `get_one_metadata_info`.
    - The final DataFrame is a concatenation of E1 and E2 data with all metadata.
    """

    df_E1 = get_encoding_data(phase='E1')
    df_E2 = get_encoding_data(phase='E2')

    df_E1['E_type'] = 'E1'
    df_E2['E_type'] = 'E2'

    cols = ['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type']
    df_E1 = df_E1[cols]
    df_E2 = df_E2[cols]

    df_encodings = pd.concat([df_E1, df_E2], axis=0, ignore_index=True)

    meta_data = pd.read_csv(data_path / 'meta-data.csv', sep = ";", encoding='latin')

    df_E1_cond = pd.merge(df_E1, meta_data[['subject', 'R1']], on='subject', how='left').rename(columns={'R1': 'condition'})
    df_E2_cond = pd.merge(df_E2, meta_data[['subject', 'R2']], on='subject', how='left').rename(columns={'R2': 'condition'})

    cols = ['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type', 'condition']
    df_E1_cond = df_E1_cond[cols]
    df_E2_cond = df_E2_cond[cols]

    metadata_infos = ['age','order']

    for info in metadata_infos: 
        df_info = get_one_metadata_info(info)
        df_E1_cond = df_E1_cond.merge(df_info, on='subject', how='left')
        df_E2_cond = df_E2_cond.merge(df_info, on='subject', how='left')

    df_encodings = pd.concat([df_E1_cond, df_E2_cond], axis=0, ignore_index=True)

    return df_encodings



def get_recall_data(phase):
    """
    Retrieve and concatenate recall data for a specified recall phase (R1, R2, or RF).

    Parameters
    ----------
    phase : str
        The experimental phase to filter data by ('R1', 'R2', 'RF').

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing all recall data for the specified phase.
        Columns = ['dates', 'subject', 'image_object_encodage_id', 'image_landscape_id', 'Name', 'rt', 'score'].

    Notes
    -----
    - Assumes `get_main_index()` returns a queryable object (e.g., pandas DataFrame).
    - Assumes `read_behavior(run_key)` returns a pandas DataFrame for each run.
    - The query string is printed for debugging purposes.
    """

    main_index = get_main_index()

    query = f"phase == '{phase}'"
    print(query)
    run_keys = list(main_index.query(query).index)
    
    df = []
    for run_key in run_keys:
        df.append(read_behavior(run_key))
    
    df = pd.concat(df, axis=0)
    # print(df)

    df['score'] = df['score'].astype('float')

    return df

def concat_recall_data():
    """
    Concatenate recall data from phases R1, R2, and RF, and enrich with metadata.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing concatenated recall data for R1, R2, and RF,
        with added columns for condition, age, group, order, and recall phase.

    Notes
    -----
    - Uses `get_recall_data` to fetch data for phases R1, R2, and RF.
    - Adds a 'recall' column to distinguish between R1, R2, and RF.
    - Merges with metadata to add condition information for R1 and R2.
    - Selects specific columns for the final output.
    - Merges R1 and R2 data with RF data, adding suffixes to distinguish initial and final recall.
    - Renames and drops columns to clean up the DataFrame.
    - Enriches the DataFrame with additional metadata (age, group, order...) using `get_one_metadata_info`.
    """
    # Fetch recall data for R1, R2, and RF
    df_R1 = get_recall_data(phase='R1')
    df_R2 = get_recall_data(phase='R2')
    df_RF = get_recall_data(phase='RF')

    # Add 'recall' column to distinguish phases
    df_R1['recall'] = 'R1'
    df_R2['recall'] = 'R2'
    df_RF['recall'] = 'RF'

    # Get condition metadata for R1 and R2
    cond_R1 = get_one_metadata_info('R1')
    cond_R2 = get_one_metadata_info('R2')

    # Merge condition information
    df_R1_cond = df_R1.merge(cond_R1, on='subject', how='left').rename(columns={'R1': 'condition'})
    df_R2_cond = df_R2.merge(cond_R2, on='subject', how='left').rename(columns={'R2': 'condition'})

    # Select relevant columns
    cols = ['subject', 'image_object_encodage_id', 'image_landscape_id', 'rt', 'score', 'Name', 'condition', 'recall']
    df_R1_cond = df_R1_cond[cols]
    df_R2_cond = df_R2_cond[cols]

    # Concatenate R1 and R2 data
    df_recalls = pd.concat([df_R1_cond, df_R2_cond], axis=0, ignore_index=True)

    # Merge with RF data, adding suffixes to distinguish initial and final recall
    df_recalls_all = pd.merge(
        df_recalls, df_RF,
        left_on=['subject', 'image_object_encodage_id'],
        right_on=['subject', 'image_object_encodage_id'],
        suffixes=('_initial', '_final')
    )

    # Select and rename columns for the final output
    new_cols = [
        'subject', 'condition', 'image_object_encodage_id',
        'image_landscape_id_initial', 'Name_initial', 'Name_final',
        'score_initial', 'score_final', 'rt_initial', 'rt_final',
        'recall_initial', 'recall_final'
    ]
    df_recalls_all = df_recalls_all[new_cols]

    # Add and clean up columns
    df_recalls_all['image_object_id'] = df_recalls_all['image_object_encodage_id']
    df_recalls_all['image_landscape_id'] = df_recalls_all['image_landscape_id_initial']
    df_recalls_all.drop(
        columns=['image_object_encodage_id', 'image_landscape_id_initial', 'Name_initial', 'Name_final'],
        inplace=True
    )

    # Enrich with additional metadata
    metadata_infos = ['age', 'order', 'R1', 'R2']
    for info in metadata_infos:
        df_info = get_one_metadata_info(info)
        df_recalls_all = df_recalls_all.merge(df_info, on='subject', how='left')

    return df_recalls_all


def merge_encoding_recall_data():
    """
    Merge encoding and recall data into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing both encoding and recall data,
        with redundant columns removed and suffixes cleaned up.

    Columns = ['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type',
       'condition', 'age', 'order', 'score_initial', 'score_final',
       'rt_initial', 'rt_final', 'recall_initial', 'recall_final', 'R1', 'R2']

    Notes
    -----
    - Uses `concat_encoding_data` and `concat_recall_data` to fetch the respective datasets.
    - Merges the encoding and recall DataFrames on 'subject' and 'image_object_id'.
    - Drops redundant columns (e.g., duplicate metadata columns from the recall DataFrame).
    - Removes '_x' suffixes from column names for clarity.
    """
    # Fetch encoding and recall data
    df_encodings = concat_encoding_data()
    df_recalls = concat_recall_data()

    # Merge encoding and recall data on 'subject' and 'image_object_id'
    df_merged = df_encodings.merge(df_recalls, on=['subject', 'image_object_id'], how='left')

    # Drop redundant columns from the recall DataFrame
    cols_to_drop = ['image_landscape_id_y','age_y','order_y','condition_y']
    df_cleaned = df_merged.drop(columns=cols_to_drop)

    # Remove '_x' suffixes from column names
    df = df_cleaned.rename(columns=lambda x: x.replace('_x', ''))

    return df

def assign_trial_numbers(df_long):
    """
    Assign trial numbers to initial and final phase data for each subject.

    Parameters
    ----------
    ouptut of merge_encoding_recall_data : pd.DataFrame, put in long format
        Containing data for both 'initial' and 'final' phases with columns: 'subject', 'phase', 'E_type', and 'image_object_id'.
        Used in the 'get_all_trials_data_with_trial_numbers' function to assign trial numbers based on the phase and encoding type.

    Returns
    -------
    pd.DataFrame
        A DataFrame with trial numbers assigned as follows:
        - Initial phase: E1 trials numbered 1–25, E2 trials numbered 26–50.
        - Final phase: Trial numbers match the initial phase but with a 'bis' suffix
          (e.g., '1bis', '2bis', etc.) based on matching 'image_object_id'.

    Notes
    -----
    - The function first separates the data into 'initial' and 'final' phases.
    - For the initial phase, trial numbers are assigned sequentially for E1 (1–25) and E2 (26–50).
    - For the final phase, trial numbers are assigned by matching 'image_object_id' with the initial phase
      and appending 'bis' to the corresponding initial trial number.
    - The result is a concatenated DataFrame sorted by 'subject' and 'trial_number'.
    """
    # Separate initial and final phase data
    initial_df = df_long[df_long['phase'] == 'initial'].copy()
    final_df = df_long[df_long['phase'] == 'final'].copy()

    # Initialize trial_number columns
    initial_df['trial_number'] = -1
    final_df['trial_number'] = ''

    # Assign trial numbers for each subject
    for subject in df_long['subject'].unique():
        subj_initial = initial_df[initial_df['subject'] == subject]

        # Assign trial numbers for E1 (1-25)
        trials_E1 = subj_initial[subj_initial['E_type'] == 'E1']
        initial_df.loc[trials_E1.index, 'trial_number'] = list(range(1, len(trials_E1) + 1))

        # Assign trial numbers for E2 (26-50)
        trials_E2 = subj_initial[subj_initial['E_type'] == 'E2']
        initial_df.loc[trials_E2.index, 'trial_number'] = list(range(26, 26 + len(trials_E2)))

        # Assign trial numbers for final phase based on matching image_object_id
        for _, row in initial_df[initial_df['subject'] == subject].iterrows():
            match = (
                (final_df['subject'] == subject) &
                (final_df['image_object_id'] == row['image_object_id'])
            )
            final_df.loc[match, 'trial_number'] = f"{row['trial_number']}bis"

    # Combine and return the DataFrames, sorted by subject and trial_number
    return pd.concat([initial_df, final_df], ignore_index=True).sort_values(['subject', 'trial_number'])

def get_all_trials_data_with_trial_numbers():
    """
    Compile all trial data (score, reaction time) with assigned trial numbers.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all trial data with the following columns:
        - 'subject', 'age', 'trial_number', 'condition', 'order', 'image_object_id',
          'image_landscape_id', 'learning_order', 'rating', 'recall_phase', 'phase',
          'score', 'rt'.
        Data is sorted by 'subject' and 'trial_number'.

    Notes
    -----
    - Starts with merged encoding and recall data from `merge_encoding_recall_data`.
    - Uses `pd.melt` to transform score, reaction time (rt), and recall data from wide to long format.
    - Assigns trial numbers using `assign_trial_numbers` for the score data.
    - Merges reaction time and recall data with the trial numbers from the score data.
    - Renames columns for clarity: 'Rating' -> 'rating', 'E_type' -> 'learning_order', 'recall' -> 'recall_phase'.
    - Sorts the final DataFrame by 'subject' and 'trial_number'.
    """
    # Start with merged encoding and recall data
    df = merge_encoding_recall_data()

    # Melt score columns to long format
    df_long_score = pd.melt(df,id_vars=['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type','condition', 'age', 'order'],
        value_vars=['score_initial', 'score_final'],
        var_name='phase',
        value_name='score')
    df_long_score['phase'] = df_long_score['phase'].str.replace('score_', '')

    # Assign trial numbers to the score data
    df_trials = assign_trial_numbers(df_long_score)

    # Melt reaction time (rt) columns to long format
    df_long_rt = pd.melt(df, id_vars=['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type','condition', 'age', 'order'],
        value_vars=['rt_initial', 'rt_final'],
        var_name='phase',
        value_name='rt')
    df_long_rt['phase'] = df_long_rt['phase'].str.replace('rt_', '')

    # Merge reaction time data with trial numbers
    df_long_rt = df_long_rt.merge(df_trials[['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type','condition', 'age', 'order', 'phase', 'trial_number']],
        on=['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type',
            'condition', 'age', 'order', 'phase'], how='left')

    # Melt recall columns to long format
    df_long_recall = pd.melt(df,
                id_vars=['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type',
                'condition', 'age', 'order'],
                value_vars=['recall_initial', 'recall_final'],
                var_name='phase',
                value_name='recall')
    df_long_recall['phase'] = df_long_recall['phase'].str.replace('recall_', '')

    # Merge recall data with trial numbers
    df_long_recall = df_long_recall.merge(df_trials[['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type','condition', 'age', 'order', 'phase', 'trial_number']],
        on=['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type',
            'condition', 'age', 'order', 'phase'], how='left')

    # Merge all DataFrames (score, rt, recall) into one
    df_long = df_trials.merge(df_long_rt,on=['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type',
            'condition', 'age', 'order', 'phase', 'trial_number'],how='outer')

    df_long = df_long.merge(
        df_long_recall,on=['subject', 'image_object_id', 'image_landscape_id', 'Rating', 'E_type',
            'condition', 'age', 'order', 'phase', 'trial_number'],how='outer')

    # Rename columns for clarity
    df_long = df_long.rename(columns={'Rating': 'rating',
                                      'E_type': 'learning_order',
                                      'recall': 'recall_phase'})

    # Sort and select final columns
    df_long.sort_values(['subject', 'trial_number'], inplace=True)
    cols_to_sort = ['subject', 'age', 'trial_number', 'condition', 'order', 'image_object_id',
                    'image_landscape_id', 'learning_order', 'rating', 'recall_phase', 'phase', 'score', 'rt']
    df = df_long[cols_to_sort]

    return df

def compute_global_memory_change(df):
    """
    Compute the Global Memory Change (GMC) score for each subject and condition.

    Parameters
    ----------
    df : pd.DataFrame
        Output from 'get_all_trials_data_with_trial_numbers' function containing recall data with columns:'subject', 'condition', 'phase', 'score', and 'order'.
        

    Returns
    -------
    pd.DataFrame
        A DataFrame with the computed GMC scores for each subject and condition,
        along with the corresponding 'order' information.

    Notes
    -----
    - GMC is calculated as the percentage change between the final and initial scores:
      `((final_score - initial_score) / initial_score) * 100`.
    - The function iterates over each subject and condition to compute the GMC.
    - The result includes the subject, condition, GMC score, and order.
    """
    GMC_scores = []

    # Iterate over each subject
    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]

        # Iterate over each condition for the subject
        for cond in subject_data['condition'].unique():
            cond_data = subject_data[subject_data['condition'] == cond]

            # Compute mean scores for final and initial phases
            final_score = cond_data[cond_data['phase'] == 'final']['score'].mean()
            initial_score = cond_data[cond_data['phase'] == 'initial']['score'].mean()

            # Calculate GMC score
            GMC_cond = ((final_score - initial_score) / initial_score) * 100

            # Get the order for the condition
            order = cond_data['order'].iloc[0]

            # Append results
            GMC_scores.append({'subject': subject,'condition': cond,'GMC': GMC_cond,'order': order})

    # Create a DataFrame from the computed scores
    df_GMC = pd.DataFrame(GMC_scores)

    return df_GMC

def compute_relative_memory_change():
    """
    Compute relative memory change metrics, optionally aggregated by subject, condition, and order.
    Based on the data from 'data_memory_learned_trials.csv', this function calculates performance, memorised, learned, forgotten, and never_learned metrics.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated memory metrics, includes:
        - 'performance', 'memorised', 'forgotten' (summed),
        - 'trial_number' (count, converted to int),
        - 'memorised_pct': Percentage of memorised trials,
        - 'RMC': Relative Memory Change, calculated as:
          `((memorised - trial_number) / trial_number) * 100`.

    """
    # Read the input data
    df = pd.read_csv(data_path / 'data_memory_learned_trials.csv', sep=';')


    # Group by subject, condition, and order, and aggregate all metrics
    df_grouped = (
        df.groupby(['subject', 'condition', 'order'])
        .agg({
            'memorised': 'sum',
            'forgotten': 'sum',
            'trial_number': 'count'
        })
        .reset_index()
    )
    # Convert trial_number from count (float) to int
    df_grouped['trial_number'] = df_grouped['trial_number'].astype(int)
    df = df_grouped

    # Convert trial_number to int and compute additional metrics
    df['trial_number'] = df['trial_number'].astype(int)
    df['memorised_pct'] = (df['memorised'] / df['trial_number'] * 100)
    df['RMC'] = ((df['memorised'] - df['trial_number']) / df['trial_number'] * 100)

    return df


if __name__ == "__main__":

    # df = concat_recall_data()

    df = get_all_trials_data_with_trial_numbers()
    df = compute_global_memory_change(df)
    # print(df.columns)
    print(df)