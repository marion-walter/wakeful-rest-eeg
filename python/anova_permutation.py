# Base imports
import numpy as np

# Specific imports used several times
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter
import locale
locale.setlocale(locale.LC_ALL, 'C')
from rpy2.robjects.packages import importr


def anova_permutation(data, dependent_variable, between_factors=[], within_factors = [], random_factor = None, n_permutation=5000, method="freedman_lane", 
                        verbose=True, show_summary=True):
    """
    Runs a permutation-based ANOVA using the `permuco` R package.

    Supports between-subjects, within-subjects, and mixed designs using permutation inference
    with methods such as Freedman-Lane or Rd_kheradPajouh_renaud.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format dataframe containing all variables, including the dependent variable and factors.
    dependent_variable : str
        Column name of the dependent variable.
    between_factors : list of str, optional
        Names of between-subjects factors in `data`.
    within_factors : list of str, optional
        Names of within-subjects (repeated measures) factors in `data`.
    random_factor : str or None, optional
        Column name of the random factor (e.g., 'subject') required for within- or mixed-designs.
    n_permutation : int, optional
        Number of permutations to perform. Default is 5000.
    method : str, optional
        Permutation method. Default is "freedman_lane". Ignored in mixed designs (overridden).
    verbose : bool, optional
        If True, print model info and unused variables. Default is True.
    show_summary : bool, optional
        If True, print the ANOVA summary from R. Default is True.

    Returns
    -------
    result : dict
        Nested dictionary of ANOVA results converted from R to Python.

    Notes
    -----
    - Mixed or within-subject designs require both `within_factors` and `random_factor`.
    - The permutation test is run using R's `aovperm()` from the `permuco` package.
    - The formula is automatically constructed and converted for R.
    - The presence of within-subject factors override the permutation method to `'Rd_kheradPajouh_renaud'`

    Raises
    ------
    AssertionError
        If input assumptions are not met.
    """

    # =============================
    # Input validation
    # =============================
    assert dependent_variable in data.columns, f"Dependent variable is not provided in data"
    assert set(between_factors).issubset(data.columns), f" Some between factors are not provided in the data : {[feature for feature in between_factors if not feature in data.columns ]}"
    assert set(within_factors).issubset(data.columns), f" Some within factors are not provided in the data : {[feature for feature in within_factors if not feature in data.columns ]}"
    assert bool(within_factors)==bool(random_factor), "Both within factors and random_factor must be provided for mixed design"
    if random_factor is not None :
        assert random_factor in data.columns, f" Random factor is not provided in the data"
    
    # =============================
    # Verbose metadata diagnostics
    # =============================
    if verbose :
        unused = [feature for feature in data.columns if (feature not in between_factors and feature not in within_factors and feature !=random_factor and feature !=dependent_variable)]
        if len(unused) > 1 :
            print(f"Metadata features {unused} are not used in the model")
        elif len(unused) == 1 :
            print(f"Metadata feature {unused} is not used in the model")

    # =============================
    # Determine appropriate method
    # =============================
    if within_factors :
        print(f"Mixed design detected: 'method' has been overridden and set to 'Rd_kheradPajouh_renaud'.")
        valid_method = "Rd_kheradPajouh_renaud"
    else:
        valid_method = method


    # =============================
    # Convert to R dataframe
    # =============================
    dataframe_to_r(data, "data")
    for feature in data.columns:
        if not feature == dependent_variable:
            r(f'data${feature} <- as.factor(data${feature})')
    r('Sys.setlocale("LC_ALL", "English_United States.UTF-8")')

    # =============================
    # Build formula for R's aovperm
    # =============================
    formula = f"{dependent_variable} ~ "
    formula += (' * ').join(between_factors + within_factors) 
    if random_factor is not None:
        formula += f" + Error({random_factor}/({(' * ').join(within_factors)}))"
    if verbose :
        print(f"Model formula : {formula}")

    # =============================
    # Run permutation-based ANOVA
    # =============================
    permuco = importr("permuco")
    results =  r(f'''
        aovperm({formula},
                    data = data,
                    np = {n_permutation},
                    method = "{valid_method}",)
                ''')

    if show_summary:
        print(r('summary')(results))
        
    return recursive_converter(results)


def dataframe_to_r(df, name, return_r=False):
    """
    Converts a pandas DataFrame to an R data frame and assigns it to the R environment.

    This function uses `rpy2`'s `pandas2ri` and `localconverter` to safely convert 
    Python DataFrame objects for use in R code via the global `r` environment.

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas DataFrame to convert and pass to the R environment.
    name : str
        The variable name to assign the R data frame to in the R global environment.
    return_r : bool
        Whether to return the r dataframe

    Returns
    -------
    None

    Notes
    -----
    This function assumes `rpy2` is installed and that an `rpy2.robjects` interface
    is already initialized in the Python session (e.g., via `from rpy2.robjects import r`).

    Examples
    --------
    >>> dataframe_to_r(my_dataframe, "data")
    >>> r("summary(data)")
    """
    # Convert pandas to R dataframe
    with localconverter(pandas2ri.converter):
        r_version = pandas2ri.py2rpy(df)
    
    # Assign to R global environment
    r.assign(name, r_version)

    if return_r:
        return r_version


def recursive_converter(r_dict):
    """
    Recursively converts R-like nested data structures to Python-native structures.

    This function is primarily used to convert outputs from `rpy2` (such as R's
    `ListVector` or `DataFrame`) into Python `dict`, `list`, or `numpy.ndarray`
    formats. It ensures that deeply nested or wrapped R objects are usable in
    Python workflows.

    Parameters
    ----------
    r_dict : object
        An object returned by R through `rpy2`, typically a `ListVector`,
        `DataFrame`, or similar nested structure.

    Returns
    -------
    python_dict : object
        A Python-native representation of the input:
        - `dict` if the R object has named elements,
        - `list` or `numpy.ndarray` if it's unnamed or simplified,
        - Base types (int, float, str, etc.) otherwise.

    Notes
    -----
    - If the top-level keys of the dictionary are all `None`, the function assumes
      the R object is a vector and tries to convert it to a `numpy.ndarray`.
    - Singleton arrays are automatically unwrapped.

    Examples
    --------
    >>> r_output = r('list(a=1, b=2)')
    >>> py_output = recursive_converter(r_output)
    >>> print(py_output)
    {'a': 1, 'b': 2}
    """

    # Check for dict-like (named R list or ListVector)
    if hasattr(r_dict, 'items'):
        python_dict = {a:recursive_converter(b) if a is not None else b for (a,b) in r_dict.items() }

        # Special case: unnamed R list -> numpy array
        if list(python_dict.keys()) == [None]:
            python_dict = np.array(r_dict)

            # Unwrap singleton array
            if python_dict.size == 1:
                python_dict = python_dict[0]
    else:
        # Base case: already Python-native
        python_dict = r_dict

    return python_dict

