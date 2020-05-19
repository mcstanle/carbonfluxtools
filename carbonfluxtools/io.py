"""
A collection of IO related functions to support
1. reading scale factors
2. reading optimal scale factors
3. transforming scale factors

Author   : Mike Stanley
Created  : May 12, 2020
Modified : May 19, 2020

================================================================================
"""
from glob import glob
import json
import netCDF4 as nc4
import numpy as np
import PseudoNetCDF as pnc
import xbpch


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
    file_names = glob(base_df_dir + '/' + sf_prefix + '*')

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

    # make sure that longitude comes before latitude
    extr_arrs_ll = []
    for sf_i in extr_arrs:
        if sf_i.shape[1] != 72:
            extr_arrs_ll.append(np.swapaxes(sf_i, axis1=1, axis2=2))
        else:
            extr_arrs_ll.append(sf_i)

    # stack the above
    stacked_arrs = np.stack(extr_arrs_ll, axis=0)

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


def read_geo_info(file_path):
    """
    Reads geographic point and region infomation from a json containing
    individual locations and regions.

    Parameters:
        file_path (str) : path to the json file

    Return:
        dictionary with geographic info
    """
    with open(file_path) as json_file:
        geo_dict = json.load(json_file)

    # pop out the "information" key \\ useless
    geo_dict.pop('information')

    return geo_dict


def get_single_loc_info(geo_dict, location_nm):
    """
    Retrieves point and extent information for a dictionary of the form
    returned by read_geo_info

    Parameters:
        geo_dict    (dict) :
        location_nm (str)  :

    Returns:
        tuple - (lon, lat) (tuple), extent (list)
    """
    # get lon/lat point
    lon_lat = tuple(geo_dict['locations'][location_nm]['point'])

    # get the extent
    extent_lst = geo_dict['locations'][location_nm]['extent']

    return lon_lat, extent_lst


def get_regional_info(geo_dict, region_nm):
    """
    Retrieves point and extent information for a dictionary of the form
    returned by read_geo_info

    Parameters:
        geo_dict  (dict) :
        region_nm (str)  :

    Returns:
        tuple - lon_pts, lat_pts
    """
    # get the reference location name
    ref_loc = geo_dict['regions'][region_nm]['reference_point']

    # get the information for the above location
    pt, ext = get_single_loc_info(geo_dict=geo_dict, location_nm=ref_loc)

    # get lat/lon perturb values
    lon_perturb = geo_dict['regions'][region_nm]['lon_perturb']
    lat_perturb = geo_dict['regions'][region_nm]['lat_perturb']

    # create the nump arrays for the grid
    lon_pts = np.arange(lon_perturb[0], lon_perturb[1]) + pt[0]
    lat_pts = np.arange(lat_perturb[0], lat_perturb[1]) + pt[1]

    return lon_pts, lat_pts


def create_netcdf_flux_file(
    write_loc,
    lon, lat, lev, co2_vals,
    co2_field_nm='CO2_SRCE_CO2bf'
):
    """
    Create a netcdf file for a single time instance of flux

    The array of interest contained within is lon x lat (72 x 46)

    Parameters:
        write_loc    (str)    : file path destination
        lon          (np arr) :
        lat          (np arr) :
        lev          (np arr) : atmosphere levels
        co2_vals     (np arr) :
        co2_field_nm (str)    : name of co2 field in the netcdf file

    Returns:
        None - writes netcdf file to path specified in write_loc
    """
    # create and save netcdf file
    f = nc4.Dataset(write_loc, 'w', format='NETCDF4')

    # create dimensions
    f.createDimension('lon', 72)
    f.createDimension('lat', 46)
    f.createDimension('lev', 72)

    # build variables
    longitude = f.createVariable('Longitude', 'f4', 'lon')
    latitude = f.createVariable('Latitude', 'f4', 'lat')
    levels = f.createVariable('Levels', 'f4', 'lev')
    co2_srce = f.createVariable(co2_field_nm, 'f4', ('lon', 'lat'))

    # passing data into variables
    longitude[:] = lon
    latitude[:] = lat
    levels[:] = lev
    co2_srce[:, :] = co2_vals

    # close the dataset
    f.close()


def generate_nc_files(
    bpch_files, output_dir, tracer_path, diag_path,
    co2_var_nm='CO2_SRCE_CO2bf'
):
    """
    Creates eight netcdf files for every bpch file (since there are eight
    time steps in each one). Builds the time indices from the sequence of
    provided bpch files. New file names are simply extentions of input
    bpch files.

    e.g.
     input  - nep.geos.4x5.001
     output - nep.geos.4x5.001.00001

    Parameters:
        bpch_files  (str) : an ordered sequential collection of daily
                            bpch files
        output_dir  (str) : output directory for netcdf files
        tracer_path (str) : path to tracer file
        diag_path   (str) : path to diag file
        co2_var_nm  (str) : name of co2 variable of interest

    Returns:
        None - write netcdf file to path in output_file

    NOTE:
    - use the convention of 5 digit time suffix
    """
    # read in the binary punch files
    bpch_data = xbpch.open_mfbpchdataset(
        bpch_files,
        dask=True,
        tracerinfo_file=tracer_path,
        diaginfo_file=diag_path
    )

    # create file suffixes
    file_suffs = [str(i).zfill(5) for i in np.arange(0, len(bpch_files)*8)]

    # create new file names
    output_file_nms = []
    count = 0
    for file_nm in bpch_files:
        for i in range(8):
            output_file_nms.append(file_nm + '.' + file_suffs[count])
            count += 1

    # create output paths
    output_file_nms = [
        output_dir + '/' + i.split('/')[-1] for i in output_file_nms
    ]

    # extract non-time dependent info from first bpch file
    lon = bpch_data.variables['lon'].values
    lat = bpch_data.variables['lat'].values
    lev = bpch_data.variables['lev'].values
    co2_arr = bpch_data.variables[co2_var_nm].values

    # create netcdf files
    for time_count, file_nm in enumerate(output_file_nms):

        # create netcdf file with time_count index co2 values
        create_netcdf_flux_file(
            write_loc=file_nm,
            lon=lon,
            lat=lat,
            lev=lev,
            co2_vals=co2_arr[time_count, :, :],
            co2_field_nm=co2_var_nm
        )


def open_netcdf_flux(file_path, co2_field_nm='CO2_SRCE_CO2bf'):
    """
    Open a single netcdf file as generated by generate_nc_files and return
    - co2 values
    - lon
    - lat

    in tuple

    Parameters:
        file_path    (str) :
        co2_field_nm (str) : name of co2 field

    Returns:
        co2 values, lon, lat (all numpy arrays)
    """
    # read in the file
    f_in = nc4.Dataset(file_path, 'r')

    # extract arrays
    co2_arr = np.array(f_in.variables['CO2_SRCE_CO2bf'][:, :])
    lon = np.array(f_in.variables['Longitude'][:])
    lat = np.array(f_in.variables['Latitude'][:])

    # close the file
    f_in.close()

    return co2_arr, lon, lat


def read_flux_files(
    file_dir,
    file_pre,
    tracer_fp=None,
    diag_fp=None
):
    """
    Since scale factors and results are examined on a monthly time-scale, raw
    3hr flux files need to be processed to produce a monthly flux for each grid
    point.

    Assumptions -
    1. flux files are bpch files

    Parameters:
        file_dir  (str) : directory where files are stored
        file_pre  (str) : prefix for flux files, e.g. nep.geos.4x5.2010
        tracer_fp (str) : path to relevant tracer file
                          (if none, will look in file_dir)
        diag_fp   (str) : path to relevant diag file
                          (if none, will look in file_dir)

    Returns:
        xbpch object which will contain a flux of interest in additiona to
        dimension parameters (e.g. lon/lat/lev)
    """
    if tracer_fp:
        tracer_fp_1 = tracer_fp
    else:
        tracer_fp_1 = file_dir + '/tracerinfo.dat'

    if diag_fp:
        diag_fp_1 = diag_fp
    else:
        diag_fp_1 = file_dir + '/diaginfo.dat'

    # find the flux file names
    file_names = sorted(
        [file_nm for file_nm in glob(file_dir + '/%s*' % file_pre)]
    )

    assert len(file_names) > 0

    # read in all the prior fluxes
    fluxes = xbpch.open_mfbpchdataset(
        file_names,
        dask=True,
        tracerinfo_file=tracer_fp_1,
        diaginfo_file=diag_fp_1
    )

    return fluxes


def find_time_idxs(start, end, fluxes):
    """
    Find the numpy arr indices between two month indexes (counting from 0)

    Parameters:
        start  (int)       : start month index
        end    (int)       : end month index
        fluxes (xbpch obj) : i.e. output of read_flux_files

    Returns:
        numpy array with time indices
    """
    assert start < end

    if end > 12:
        less = np.where(
            fluxes.time.values >= np.datetime64('1985-%i-01' % start)
        )[0]
        geq = np.where(
            fluxes.time.values >= np.datetime64('1985-%i-01' % start)
        )[0]
    elif start > 9:
        less = np.where(
            fluxes.time.values < np.datetime64('1985-%i-01' % end)
        )[0]
        geq = np.where(
            fluxes.time.values >= np.datetime64('1985-%i-01' % start)
        )[0]
    elif end > 9:
        less = np.where(
            fluxes.time.values < np.datetime64('1985-%i-01' % end)
        )[0]
        geq = np.where(
            fluxes.time.values >= np.datetime64('1985-0%i-01' % start)
        )[0]
    else:
        less = np.where(
            fluxes.time.values < np.datetime64('1985-0%i-01' % end)
        )[0]
        geq = np.where(
            fluxes.time.values >= np.datetime64('1985-0%i-01' % start)
        )[0]

    # find the intersection between the above
    time_idxs = np.intersect1d(geq, less)

    return time_idxs


def find_month_idxs(fluxes):
    """
    Find the indices for each month for a given xbpch obj.

    Parameters:
        fluxes (xbpch object) : i.e. output of read_flux_files

    Returns:
        dictionary of month abbreviations with numpy array values

    NOTE:
    - this function can only handle a single year.
    """
    # find the month indices
    month_idxs = {
        'jan': None, 'feb': None, 'mar': None,
        'apr': None, 'may': None, 'june': None,
        'jul': None, 'aug': None, 'sep': None,
        'oct': None, 'nov': None, 'dec': None
    }

    for month_idx, month in enumerate(month_idxs.keys()):
        # find the start and end values
        start_val = month_idx + 1
        end_val = month_idx + 2

        # find the time indices
        month_idxs[month] = find_time_idxs(
            start=start_val,
            end=end_val,
            fluxes=fluxes
        )

    return {
        key: value for key, value in month_idxs.items() if value is not None
    }


def read_cfn_files(file_dir):
    """
    Read the cost function output files from GEOS-Chem Adjoint Runs.

    These files should be organized in one directory to work with this function

    Parameters:
        file_dir (str) :

    Returns:
        float - the cost function evaluation
    """
    # get file names for cost functions
    cfn_fp = glob(file_dir + '/cfn*')

    cfn = []
    for i, fp in enumerate(cfn_fp):
        with open(fp, 'r') as f:
            cfn.append(float(f.readlines()[0].replace(
                ' %i ' % (i + 1), ''
            ).replace(' ', '').replace('\n', '')))

    return cfn
