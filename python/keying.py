
"""
UTILITY FUNCTIONS FOR KEY AND STRING MANIPULATION

This script provides essential utility functions for consistent handling of subject/run identifiers
and string formatting throughout the analysis pipeline. It includes:

USAGE:
These functions are designed to be imported and used throughout the analysis pipeline.

IMPORTANT:
This is a core utility script. Modifications may affect the entire analysis pipeline.
"""

# Specific imports 
from functools import wraps
import numpy as np


def add_0_to_str(x):
    """
    Add a leading zero to a number or numeric string if less than 10.

    Parameters
    ----------
    x : int or str
        Input number or numeric string.

    Returns
    -------
    str
        String representation of the number with a leading zero if `x < 10`.
    """

    if int(x) < 10:
        return '0' + str(x)
    else:
        return str(x)

def remove_0_from_str(x):
    """
    Remove a leading zero from a string, if present.

    Parameters
    ----------
    x : str
        Input string.

    Returns
    -------
    str
        String without leading zero if it was present.

    Notes
    -----
    If `x` does not start with '0', it is returned unchanged.
    """

    if x[0]=='0':
        return x[1:]
    else:
        return x


def key_adapter(input, output_type = 'key', session = None):
    """
    Convert between different representations of subject/run identifiers.

    Parameters
    ----------
    input : str or int
        The input subject or run identifier. Can be:
        - int: subject number (e.g. 1)
        - str: one of the following formats:
          - '1' (id)
          - '01' (full_id)
          - 'sub-01' (key)
          - 'sub-01_session' (run_key), where session is a session identifier string
    output_type : {'number', 'id', 'full_id', 'key', 'run_key'}, optional
        The desired output format (default is 'key'):
        - 'number': returns subject number as int (e.g. 1 for 'sub-01')
        - 'id': returns subject number as str without leading zero (e.g. '1' for 'sub-01')
        - 'full_id': returns subject number as str with leading zero (e.g. '01' for 'sub-01')
        - 'key': returns subject identifier string like 'sub-01'
        - 'run_key': returns string like 'sub-01_session'; requires `session` parameter if input does not already contain it
    session : str, optional
        Session identifier string to append when `output_type` is 'run_key' and input is not already a run_key.
        Required if `output_type` is 'run_key' and `input` is not a run_key.

    Returns
    -------
    str or int
        Converted identifier in the requested format.

    Raises
    ------
    AssertionError
        If `input` is not a str or int.
        If `output_type` is 'run_key' and `session` is None while input is not a run_key.

    Examples
    --------
    >>> key_adapter('sub-01', output_type='number')
    1
    >>> key_adapter(1, output_type='key')
    'sub-01'
    >>> key_adapter('1', output_type='full_id')
    '01'
    """
    # Determine input type
    assert type(input) in [str, int, np.str_], f"Bad input type, expected str or int, received {type(input)}"

    if type(input)==int :
        input_type = 'number'
    
    else:
        if len(input) == 1:
            input_type = 'id'
        else:
            input_type = 'full_id'

        if '-' in input:
            input_type = 'key'
        
        if '_' in input:
            input_type = 'run_key'
        
    
    
    if output_type == 'run_key' and input_type !='run_key':
        assert session is not None, "'session' parameter is required for run_key type output"

    # Transfer functions
    transfert = {
        'number' : {
        'number' : lambda x:x,
        'id' : lambda x:int(x) ,
        'full_id' : lambda x:int(remove_0_from_str(x)) ,
        'key' : lambda x:int(remove_0_from_str(x.split('-')[1])) ,
        'run_key' : lambda x: int(remove_0_from_str(x.split('_')[0].split('-')[1])),
    },
        'id' : {
        'number' : lambda x: str(x) ,
        'id' : lambda x:x,
        'full_id' : lambda x: remove_0_from_str(x) ,
        'key' : lambda x: remove_0_from_str(x.split('-')[1]),
        'run_key' : lambda x: remove_0_from_str(x.split('_')[0].split('-')[1]),
    },
        'full_id' : {
        'number' : lambda x: add_0_to_str(str(x)) ,
        'id' : lambda x: add_0_to_str(x),
        'full_id' : lambda x:x,
        'key' : lambda x: x.split('-')[1],
        'run_key' : lambda x: x.split('_')[0].split('-')[1],
    },
        'key' : {
        'number' : lambda x: f'sub-{add_0_to_str(str(x))}' ,
        'id' : lambda x: f'sub-{add_0_to_str(x)}',
        'full_id' : lambda x: f'sub-{x}',
        'key' : lambda x:x,
        'run_key' : lambda x: x.split('_')[0],
    },
        'run_key' : {
        'number' : lambda x: f'sub-{add_0_to_str(str(x))}_{session}',
        'id' : lambda x: f'sub-{add_0_to_str(x)}_{session}',
        'full_id' : lambda x: f'sub-{x}_{session}',
        'key' : lambda x: f'{x}_{session}',
        'run_key' : lambda x:x,
    },

    }

    return transfert[output_type][input_type](input)



def keying(output_type, session=None):
    """
    Decorator to convert the first argument of a function to a normalized key format.

    Parameters
    ----------
    output_type : str
        Desired output key format passed to `key_adapter`.
    session : str, optional
        Session identifier required if `output_type` is 'run_key' and input lacks session info.

    Returns
    -------
    function
        Decorated function with the first argument normalized to the specified key format.

    Examples
    --------
    >>> @keying('key')
    ... def func(subject_key):
    ...     return subject_key
    >>> func('1')
    'sub-01'
    """
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            good_key = key_adapter(args[0], output_type, session=session )
            retval = function(good_key, *args[1:], **kwargs)
            return retval
        return wrapper
    return decorator

