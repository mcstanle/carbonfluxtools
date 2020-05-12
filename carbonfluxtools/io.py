"""
A collection of IO related functions to support
1. reading scale factors
2. reading optimal scale factors
3. transforming scale factors

Author   : Mike Stanley
Created  : May 12, 2020
Modified : May 12, 2020

================================================================================
"""
from glob import glob
import numpy as np
import PseudoNetCDF as pnc


def read_sf_objs(base_df_dir, sf_prefix):
    """
    Reads in all files in directory with provided scale factor prefix.

    E.g. ./scale_factors/sf_*

    where base_df_dir == 'scale_factors' and sf_prefix == 'sf_'

    Parameters:
        base_df_dir (str) : base directory where all scale factors can be found
        sf_prefix   (str) : prefix for each scale factor file

    Returns:
        list of sf objects

    NOTE:
    - tracerinfo and diaginfo files must be present in the given directory
    - all scale factor files are assumed to have the same prefix form
    """
    # obtain the scale factor file names (NOTE: file order doesn't matter)
    file_names = glob(base_df_dir + '/' + sf_prefix)

    return [pnc.pncopen(fn, format='bpch') for fn in file_names]


def create_sf_arr(list_of_sf_objs, var_oi='IJ-EMS-$_CO2bal'):
    """
    Creates a 4D stacked array all scale factors across all OSSEs
    and months.

    Parameters:
        list_of_sf_objs (list) : list of pnc objects -- inputting the output
                                 from read_sf_objs will work well
        var_oi          (str)  : the variable of interest in each of the above
                                 elements

    Returns:
        - numpy array (# iterations, lon, lat)
        - longitude array
        - latitude array
    """
    # extract the scale factors from each object
    extr_arrs = [sf_i.variables[var_oi].array()[0, :, :, :]
                 for sf_i in list_of_sf_objs]

    # stack the above
    stacked_arrs = np.stack(extr_arrs, axis=0)

    # obtain longitude and latitude
    lon = list_of_sf_objs[0].variables['longitude'].array()
    lat = list_of_sf_objs[0].variables['latitude'].array()

    return stacked_arrs, lon, lat


def read_opt_sfs(file_path):
    """
    Read in a numpy file containing optimal scale factors

    Parameters:
        file_path (str) : loction of optimal scale factors (numpy .npy file)

    Returns:
        numpy array containing optimal scale factors (M x 72 x 46)
    """
    assert file_path[-3:] == 'npy'
    opt_sf = np.load(
        file=file_path
    )

    return opt_sf
