
"""
Script for Scanning and Indexing BrainVision and Behavior Data Files

This script scans directories containing BrainVision (.vhdr) and behavior (.csv) files,
extracts metadata (e.g., subject, phase, session), and builds a centralized index (`main_index`)
to track all files and their associated metadata. The index is stored in both NetCDF and Excel formats
for easy access and further processing.

Key Features:
- Scans `individual_data_path` for BrainVision files (in `sub-*/Physio/`) and behavior files (in `sub-*/Psycho/`).
- Extracts metadata from filenames (e.g., subject ID, phase, session).
- Generates unique `run_key` identifiers for each file based on metadata.
- Handles duplicates and errors (e.g., mismatched subject folders, redundant keys).
- Updates and saves the `main_index` (a pandas DataFrame) to `main_index_filename` (NetCDF) and an Excel file.

Dependencies:
- Uses `configuration.py` for paths and settings (e.g., `precomputedir`, `main_index_filename`, `unique_keys`).
- Requires `pandas`, `xarray`, and `pathlib` for file handling and data storage.

Usage:
- Run the script directly to scan and index all files in `individual_data_path`.
- The script prints summaries of new insertions, already indexed files, and errors (e.g., redundant keys).
"""

from configuration import *
import pandas as pd
import xarray as xr

def get_annotations(vhdr_file):
    """
    Extract subject and phase annotations from a BrainVision (.vhdr) file path.

    Parameters
    ----------
    vhdr_file : Path
        Path to the .vhdr file (e.g., 'sub-01_Repos.vhdr').

    Returns
    -------
    dict
        Dictionary with keys:
        - 'subject': Subject identifier (e.g., 'sub-01').
        - 'phase': Experimental phase (e.g., 'Repos').
        - 'session': Hardcoded as '02' (default session).
    """
    res = vhdr_file.stem.split("_")
    subject, phase = res

    annotations = dict(
        subject=subject,
        phase=phase,
        session='02',
    )

    return annotations

def get_key(annotations):
    """
    Generate a unique run key by joining values from the annotations dictionary.

    Parameters
    ----------
    annotations : dict
        Dictionary containing keys defined in `unique_keys` (e.g., subject, phase, session).

    Returns
    -------
    str
        A string key formed by joining annotation values with underscores (e.g., 'sub-01_Repos').
    """
    return '_'.join(annotations[k] for k in unique_keys )

def scan_brainvision_files():
    """
    Scan BrainVision (.vhdr) files in `individual_data_path`, extract metadata,
    and update the main index file with new entries.

    Notes
    -----
    - Creates `precomputedir` if it doesn't exist.
    - Loads or initializes `main_index` (a DataFrame stored as a NetCDF file).
    - Skips files with mismatched subject folders.
    - Checks for redundant or already inserted run keys.
    - Saves the updated index to both NetCDF and Excel formats.
    - Prints summaries of inserted, already present, and redundant keys.
    """

    if not precomputedir.is_dir():
        precomputedir.mkdir()

    if not main_index_filename.exists():
        main_index = pd.DataFrame()
    else:
        ds = xr.open_dataset(main_index_filename)
        main_index = ds.to_dataframe().copy()
        ds.close()
    
    list_already =[]
    list_inserted = []
    list_error_redundant_key = []
    
    print(individual_data_path)
    if '47' in str(individual_data_path) :
        print(individual_data_path)
    for vhdr_file in individual_data_path.glob('sub-*/Physio/*.vhdr'):

        print(vhdr_file)
        subject = vhdr_file.stem.split("_")[0]
        if subject != vhdr_file.parents[1].stem:
            print("  !!! Error file , ", vhdr_file.stem, "not in correct folder")
            continue
    
        try:
            annotations = get_annotations(vhdr_file)
            print(annotations)
            run_key = get_key(annotations)
            print(run_key)
        except Exception as e:
            print()
            print(' ####Erreur de nomenclature')
            print(vhdr_file.stem)
            continue

        vhdr_file_str = str(vhdr_file).replace(str(base_folder), '').replace('\\', '/')
        if vhdr_file_str.startswith('/'):
            vhdr_file_str = vhdr_file_str[1:]

        
        if run_key in main_index.index:
            if main_index.loc[run_key, 'vhdr_file'] == vhdr_file_str:
                print(run_key, ': Already inserted')
                list_already.append(run_key)
                
                continue
            else:
                print(run_key, ': !!!ERROR!!! redundant run_key')
                list_error_redundant_key.append(run_key)
                
                continue
        
        
        print(run_key, ': new insertion')
        print(vhdr_file)
        
        for k in unique_keys:
            main_index.loc[run_key, k] = annotations[k]
        
        for k in annotations.keys():
            if k not in unique_keys:
                main_index.loc[run_key, k] = annotations[k]


        main_index.loc[run_key, 'rec_datetime'] = annotations.get('rec_datetime', None)
        main_index.loc[run_key, 'vhdr_file'] = vhdr_file_str
        
        list_inserted.append(run_key)
    print(main_index)
    main_index = main_index.sort_values(by=['subject', 'session', 'phase'])

    xr.Dataset(main_index).to_netcdf(main_index_filename)
    main_index.to_excel(precomputedir / 'main_index.xlsx', sheet_name='main_index')
    
    print()
    print()
    print('### Already inserted ', len(list_already))
    print('### New insertion ', len(list_inserted))
    for run_key in list_inserted:
        print(run_key)

    print('### ERROR REDUNDANT ###', len(list_error_redundant_key))
    for run_key in list_error_redundant_key:
        print(run_key)


def scan_behavior_files():
    """
    Scan behavior (.csv) files in `individual_data_path`, extract metadata,
    and update the main index file with behavior file paths.

    Notes
    -----
    - Skips files containing 'Training' in the filename.
    - Extracts subject, session, and phase from the filename.
    - Updates `main_index` with behavior file paths.
    - Handles cases where run_key already exists (appends or skips).
    - Saves the updated index to NetCDF and Excel.
    - Prints summaries of processed files.
    """
    if not precomputedir.is_dir():
        precomputedir.mkdir()

    if not main_index_filename.exists():
        main_index = pd.DataFrame()
    else:
        ds = xr.open_dataset(main_index_filename)
        main_index = ds.to_dataframe().copy()
        ds.close()
    
    list_already =[]
    list_inserted = []
    list_error_redundant_key = []
    
    print(individual_data_path)

    for behavior_file in individual_data_path.glob('sub-*/Psycho/*.csv'):

        if 'Training' in behavior_file.stem:
            continue
        print()
        print(behavior_file.stem)

        subject = behavior_file.stem.split('_')[0]
        # print(subject)
        session = behavior_file.stem.split('_')[1].replace('sess-', '')
        # print(session)
        phase = behavior_file.stem.split('_')[2].replace('task-', '')
        # print(phase)

        run_key = f"{subject}_{phase}"
        # print(run_key)

        behavior_file_txt = str(behavior_file).replace(str(base_folder), '').replace('\\', '/')
        if behavior_file_txt.startswith('/'):
            behavior_file_txt = behavior_file_txt[1:]
        print(behavior_file_txt)

        if run_key in main_index:
            if main_index.at[run_key, "behavior_file"] == "" or main_index.at[run_key, "behavior_file"] is None:
                main_index.at[run_key, "behavior_file"] = behavior_file_txt
                list_already.append(run_key)
            else:
                list_inserted.append(run_key)
        else:
            main_index.at[run_key, "behavior_file"] = behavior_file_txt
            list_already.append(run_key)

            main_index.at[run_key, "subject"] = subject
            main_index.at[run_key, "session"] = session
            main_index.at[run_key, "phase"] = phase

    # print(main_index)


    main_index = main_index.sort_values(by=['subject', 'session', 'phase'])
    

    xr.Dataset(main_index).to_netcdf(main_index_filename)
    main_index.to_excel(precomputedir / 'main_index.xlsx', sheet_name='main_index')
    
    print()
    print()
    print('### Already inserted ', len(list_already))
    print('### New insertion ', len(list_inserted))
    for run_key in list_inserted:
        print(run_key)

    print('### ERROR REDUNDANT ###', len(list_error_redundant_key))
    for run_key in list_error_redundant_key:
        print(run_key)


if __name__ == '__main__':
    # test_get_annotations()

    scan_brainvision_files()
    scan_behavior_files()

