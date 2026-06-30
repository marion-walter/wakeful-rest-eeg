# wakeful-rest-eeg
Analyses scripts (Python and R) for the article "No Evidence for a Wakeful Rest Benefit on Associative Memory: A Within-Participant EEG Study"

There is two types of scripts : Python and R scripts, both findable here.

## Python scripts

Python scripts contain the majority of analyses and data exploration, except Generalized Linear Mixed Models analyses (see section 4.2/ R scripts).

There is two types of Python scripts (see image below): 
- Python files (.py files): contain all homemade functions necessary for data analyses and statistics.
- Jupyter Notebooks (.ipynb files, in dark orange): structured files containing behavioural and EEG analyses.

![python scripts workflow](https://osf.io/download/z9rjh/?direct)


The key scripts are the following three notebooks:
- analyses.ipynb: contains all the analyses that do not have a figure associated with them.
- figures.ipynb: contains all the analyses that have a figure associated with them in the article.
- supplementary_figures.ipynb: contains all the analyses and figures used in the supplementary file. 

Brief explanations of the other Python scripts: 
- configuration: defines the path to the project folder and all its subfolders (data, figures, etc.).
- construct_dataset_index: scan all the individual data files and organises them into a file located in the ‘precompute’ folder. 
- dataio (for ‘in/out’): functions for reading different types of data from the file created by `construct_dataset_index`. 
- All other scripts in dark blue or light blue are functions used in other scripts or in a notebook.

USAGE: 
The scripts to be used are highlighted in orange (light and dark) in the figure 
1. Set paths in configuration.py
2. Run construct_dataset_index.py
3. Explore notebooks : analyses, figures and supplementaty figures. Notebooks organisation is as follow : IMPORTS section with all necessary imports; CODE section : contains subsection for all analyses/figures computed in the notebook.

## R scripts

R scripts contain the Generalised Linear Mixed Model analyses. Analyses are divised in 3 scripts : 
- 00_import_data : load, prepare and save data for further analyses
- 01_modelling : perform the GLMM analyses
- 02_plot_glmm_results : plot the GLMM figure used in the article (Figure 6)
They should be used in this order.

![R scripts workflow](https://osf.io/download/pcz5j/?direct) 

All scripts are commented and include a short introduction briefly describing their content.
