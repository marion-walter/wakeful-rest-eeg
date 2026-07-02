# wakeful-rest-eeg
Analyses scripts (Python and R) for the article "No Evidence for a Wakeful Rest Benefit on Associative Memory: A Within-Participant EEG Study"

There is two types of scripts : Python and R scripts, both findable here.

## Python scripts

Python scripts contain the majority of analyses and data exploration, except Generalized Linear Mixed Models analyses (see section 4.2/ R scripts).

There is two types of Python scripts (see image below): 
- Python files (.py files): contain all homemade functions necessary for data analyses and statistics.
- Jupyter Notebooks (.ipynb files, in dark orange): structured files containing behavioural and EEG analyses.

![python_scripts_workflow.png](https://github.com/marion-walter/wakeful-rest-eeg/blob/e15d7574d17e1df0a7a6c80cc7f6d9d340a7cd63/python_scripts_workflow.png)

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

### Setup environment

You can use the requirements.txt file to download useful packages and libraries.
For statistical analyses, I used a homemade toolbox made by Valentin Ghibaudo called 'ghibtools', available here : https://github.com/ValentinGhibaudo/ghibtools and slightly customed by me for vizualisation purposes. This toolbox allows to perform statistical tests and display the results automatically using the pingouin library.
Download ghibtools here : https://github.com/ValentinGhibaudo/ghibtools and replace the 'stats.py' file by the one findable in this git repository, then use the setup.py file to install ghibtools.

## R scripts

R scripts contain the Generalised Linear Mixed Model analyses. Analyses are divised in 3 scripts : 
- 00_import_data : load, prepare and save data for further analyses
- 01_modelling : perform the GLMM analyses
- 02_plot_glmm_results : plot the GLMM figure used in the article (Figure 6)
They should be used in this order.

![R scripts workflow](https://github.com/marion-walter/wakeful-rest-eeg/blob/e15d7574d17e1df0a7a6c80cc7f6d9d340a7cd63/R_scripts_workflow.png) 

All scripts are commented and include a short introduction briefly describing their content.
