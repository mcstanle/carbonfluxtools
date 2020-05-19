"""
A collection of computation related functions to support
1. finding bias
2. computing regional areas

Author   : Mike Stanley
Created  : May 12, 2020
Modified : May 18, 2020

================================================================================
"""
from area import area
from carbonfluxtools import io
import numpy as np


def find_bias(sf_stack, opt_sf):
    """
    Finds difference between avg(sf) and optimal sf for one 46x72 grid

    Parameters:
        sf_stack (numpy array) : nxMx46x72 array with inverted scale factors
                                 (n=number OSSEs and M=number of months)
        opt_sf   (numpy array) : Mx46x72 optimal scale factor array

    Returns:
        Mx46x76 numpy array of E(sf) - opt_sf

    NOTE:
     - we assume that the 0th index of sf_stack is the OSSE iterations
    """
    assert sf_stack.shape[1] == opt_sf.shape[0]

    # make sure the dimensions are correct
    if opt_sf.shape[1] != 46:

        # the only problem equiped to handle is lat/lon switch
        opt_sf_proc = np.swapaxes(opt_sf, axis1=1, axis2=2)

    else:
        opt_sf_proc = opt_sf.copy()

    # find mean of the given stack of sf draws
    sf_stack_avg = sf_stack.mean(axis=0)

    return sf_stack_avg - opt_sf_proc


def subgrid_rect_obj(lon_llc, lat_llc):
    """
    Create a polygon object for a single area rectangle that
    can be used with the area function

    Parameters:
        lon_llc (int) : lower left hand corner of the longitude
        lat_llc (int) : lower left hand corner of the latitude

    Returns:
        dictionary with 'type' and 'coordinates' keys

    Note:
    - assumes that we are working with a 4x5 (lat, lon) grid
    - I am following the counter-clockwise coordinate ordering of the example
    """
    # create coordinates
    lrc = [lon_llc + 5, lat_llc]
    urc = [lon_llc + 5, lat_llc + 4]
    ulc = [lon_llc, lat_llc + 4]

    # create the object
    return {
        'type': 'Polygon',
        'coordinates': [
            [[lon_llc, lat_llc], lrc, urc, ulc, [lon_llc, lat_llc]]
        ]
    }


def w_avg_sf(sfs_arr, lon, lat, lon_bounds, lat_bounds, month,
             var_oi='IJ-EMS-$_CO2bal'
             ):
    """
    Find the weighted avg scale factor taking the curvature of the earth
    into account.

    Parameters:
        sfs_arr    (numpy array) : contains sfs over lon/lat
        lon        (numpy array) : longitudes
        lat        (numpy array) : latitudes
        lon_bounds (tuple)       : eastern oriented
        lat_bounds (tuple)       : northern oriented
        month      (int)         : the month of interest
        var_oi     (str)         : variable of interest in sfs

    Returns:
        weighted avg of scale factors (float)

    NOTES:
    - the variable of interest in sfs is assumed to have shape (1, *, 46, 72)
      where * is the number of months
    - the bounds for lon/lat are inclusive
    - orients each grid box so that the scale factor is in the center
    """
    # test array dimensions
    assert len(sfs_arr.shape) == 4
    assert sfs_arr.shape[2] == 46
    assert sfs_arr.shape[3] == 72

    # find the lon/lat indices that correspond to the the bounds of interest
    lon_lb = np.where(lon >= lon_bounds[0])
    lon_ub = np.where(lon <= lon_bounds[1])
    lon_idxs = np.intersect1d(lon_lb, lon_ub)

    lat_lb = np.where(lat >= lat_bounds[0])
    lat_ub = np.where(lat <= lat_bounds[1])
    lat_idxs = np.intersect1d(lat_lb, lat_ub)

    # compute areas
    areas = []
    raw_weighted_sfs = []
    for lat_idx in lat_idxs:
        for lon_idx in lon_idxs:

            # get the lower right corner endpoints of the box
            lon_lrc = lon[lon_idx] - 2.5
            lat_lrc = lat[lat_idx] - 2

            # find area
            grid_area = area(subgrid_rect_obj(lon_lrc, lat_lrc))

            # get the scale factor
            grid_sf = sfs_arr[0, month, lat_idx, lon_idx]

            # compute the weighted scale factor
            w_sf = grid_area * grid_sf

            # store data
            areas.append(grid_area)
            raw_weighted_sfs.append(w_sf)

    # find the total area
    tot_area = np.array(areas).sum()

    # divide all weighted scale factors by total area
    weighted_sfs = np.array(raw_weighted_sfs) / tot_area

    return weighted_sfs.sum()


def region_sf_ts(lon_idx, lat_idx, sf_arr, lon, lat):
    """
    Given lat/lon bounds, find average scale factors over some time interval

    Parameters:
        lon_idx (numpy arr) : array of longitude indices for region oi
        lat_idx (numpy arr) : array of latitude indices for region oi
        sf_arr  (numpy arr) : global scale factors dim - T x 46 x 72
        lon     (numpy arr) : array of longitudes
        lat     (numpy arr) : array of latitudes

    Returns:
        1d numpy array, one value per time in first dimension of sf_arr
    """
    assert sf_arr.shape[1] == 46
    assert sf_arr.shape[2] == 72

    # determine bounds of region
    lon_bds = (lon[lon_idx[0]], lon[lon_idx[-1]])
    lat_bds = (lat[lat_idx[0]], lat[lat_idx[-1]])

    # find the optimal NA scale factors
    avg_sfs = np.zeros(sf_arr.shape[0])

    for idx in range(sf_arr.shape[0]):
        avg_sfs[idx] = w_avg_sf(
            sfs_arr=sf_arr[np.newaxis, :, :, :],
            lon=lon,
            lat=lat,
            lon_bounds=lon_bds,
            lat_bounds=lat_bds,
            month=idx
        )

    return avg_sfs


def region_sf_iters(lon_idx, lat_idx, sf_arr, lon, lat):
    """
    Computes 2d array of average scale factors over region where rows
    represent different OSSE runs and columns represent months

    Parameters:
        lon_idx (numpy arr) : array of longitude indices for region oi
        lat_idx (numpy arr) : array of latitude indices for region oi
        sf_arr  (numpy arr) : global scale factors dim - M x T x 46 x 72
                              M number of OSSEs, T number of months
        lon     (numpy arr) : array of longitudes
        lat     (numpy arr) : array of latitudes

    Returns:
        2d numpy array, as described above
    """
    osse_avg_sf = np.zeros(shape=(sf_arr.shape[0], sf_arr.shape[1]))

    for it in range(sf_arr.shape[0]):

        # compute monthly avg scale factors for OSSE it
        osse_avg_sf[it, :] = region_sf_ts(
            lon_idx, lat_idx, sf_arr[it, :, :, :], lon, lat
        )

    return osse_avg_sf


def create_monthly_flux(
    flux_xbpch,
    flux_var_nm,
    agg_type='mean'
):
    """
    Given an xbpch object of fluxes, create an aggregated monthly form of the
    fluxes.

    Parameters:
        flux_xbpch  (xbpch) : output from io.read_flux_files
        flux_var_nm (str)   : name of flux variable in the bpch file
        agg_type    (str)   : specification as to the aggregatation type

    Returns:
        numpy array with monthly aggregated values and lon/lat

    NOTE:
    - currently supports mean and sum
    - fluxes are expected to have form {time}x{lon}x{lat}
    """
    # create monthly indices
    month_idxs = io.find_month_idxs(fluxes=flux_xbpch)

    fluxes_agg_raw = []

    for month, idxs in month_idxs.items():

        if agg_type == 'mean':
            fluxes_agg_raw.append(
                flux_xbpch[flux_var_nm].values[idxs, :, :].mean(axis=0)
            )

        elif agg_type == 'sum':
            fluxes_agg_raw.append(
                flux_xbpch[flux_var_nm].values[idxs, :, :].sum(axis=0)
            )
        else:
            raise ValueError

    # concatenate the above together
    fluxes_mean = np.stack(fluxes_agg_raw)

    return fluxes_mean, flux_xbpch['lon'].values, flux_xbpch['lat'].values


def rmse_total_global():
    """
    Finds the RMSE over all months and all grid points. Uses actual average
    flux values.

    Parameters:
        prior_flux (numpy arr) : processed monthly prior fluxes
        true_flux  (numpy arr) : processed monthly true fluxes
        sfs        (numpy arr) : obtained scale factors

    Returns:
        float
    """
    pass
