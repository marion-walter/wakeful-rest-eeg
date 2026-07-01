"""
Configuration Script for Project Paths

This script defines the file system paths for the project.
Users must customize the `base` path to match their local environment.

Key Features:
- `base_folder`: Base directory to the folder containing the scripts, data, and output folder.
  **Users must set this path according to their system.**
- Derived paths for data, scripts, figures, and precomputed results.
- Global variables like `unique_keys` for consistent data indexing.
"""

from pathlib import Path

# --- User-Specific Base Path ---
# SET THIS PATH ACCORDING TO YOUR LOCAL ENVIRONMENT.

# base_folder = Path('YOUR_BASE_PATH_HERE')  # <-- REPLACE THIS WITH YOUR ACTUAL PATH

base_folder = Path('')

# --- Project-Specific Paths ---
# All other paths are derived from `base_folder` and the project structure.

# Data directories
data_path = base_folder / 'data'                     # Data folder
r_data_path = base_folder / 'scripts/R/data'         # R-specific data files
individual_data_path = data_path / 'individual_data' # Individual participant data

# Scripts and output directories
path_scripts = base_folder / 'scripts/python'        # Python scripts
figures_path = base_folder / 'figures'               # Output figures
precomputedir = base_folder / 'precompute'           # Precomputed eeg results (e.g., preprocessing, spectrum)

# Analysis scripts directory
path_scripts_analysis = path_scripts / 'analyses'

# --- Global Variables ---
# File for storing the main index (NetCDF format)
main_index_filename = precomputedir / 'index.nc'

# Keys used to generate unique run identifiers (e.g., 'sub-01_Repos')
unique_keys = ['subject', 'phase']