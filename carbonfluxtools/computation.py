"""
A collection of computation related functions to support
1. finding bias
2. computing regional areas

Author   : Mike Stanley
Created  : May 12, 2020
Modified : May 22, 2020

================================================================================
"""
from area import area
from carbonfluxtools import io
import numpy as np


def find_bias(sf_stack, opt_sf):
    """
    Finds difference between avg(sf) and optimal sf for one 72x46 grid

    Parameters:
        sf_stack (numpy array) : nxMx72x46 array with inverted scale factors
                                 (n=number OSSEs and M=number of months)
        opt_sf   (numpy array) : Mx72x46 optimal scale factor array

    Returns:
        Mx46x76 numpy array of E(sf) - opt_sf

    NOTE:
     - we assume that the 0th index of sf_stack is the OSSE iterations
    """
    assert sf_stack.shape[1] == opt_sf.shape[0]  # same number of months
    assert sf_stack.shape[2] == 72
    assert opt_sf.shape[1] == 72

    # find mean of the given stack of sf draws
    sf_stack_avg = sf_stack.mean(axis=0)

    return sf_stack_avg - opt_sf


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


def w_avg_sf(sfs_arr, lon, lat, lon_bounds, lat_bounds, month):
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

    Returns:
        weighted avg of scale factors (float)

    NOTES:
    - the variable of interest in sfs is assumed to have shape (1, *, 72, 46)
      where * is the number of months
    - the bounds for lon/lat are inclusive
    - orients each grid box so that the scale factor is in the center
    """
    # test array dimensions
    assert len(sfs_arr.shape) == 4
    assert sfs_arr.shape[2] == 72
    assert sfs_arr.shape[3] == 46

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
            grid_sf = sfs_arr[0, month, lon_idx, lat_idx]

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


def w_avg_flux(
    flux_arr, ocean_idxs,
    lon, lat,
    lon_bounds, lat_bounds,
    month
):
    """
    Find the weighted avg flux for one time slice. The weighted average only
    takes land grid cells into account, i.e. it skips the ocean indices

    Parameters:
        flux_arr   (numpy array) : contains sfs over lon/lat
        ocean_idxs (numpy array) : indices of ocean grid cells when
                                   flattening a 72x46 array
        lon        (numpy array) : longitudes
        lat        (numpy array) : latitudes
        lon_bounds (tuple)       : eastern oriented
        lat_bounds (tuple)       : northern oriented
        month      (int)         : the month of interest

    Returns:
        weighted avg of fluxes (float)

    NOTES:
    - the variable of interest in sfs is assumed to have shape (*, 72, 46)
      where * is the number of months
    - the bounds for lon/lat are inclusive
    - orients each grid box so that the scale factor is in the center
    """
    # test array dimensions
    assert len(flux_arr.shape) == 3
    assert flux_arr.shape[1] == 72
    assert flux_arr.shape[2] == 46

    # find the lon/lat indices that correspond to the the bounds of interest
    lon_lb = np.where(lon >= lon_bounds[0])
    lon_ub = np.where(lon <= lon_bounds[1])
    lon_idxs = np.intersect1d(lon_lb, lon_ub)

    lat_lb = np.where(lat >= lat_bounds[0])
    lat_ub = np.where(lat <= lat_bounds[1])
    lat_idxs = np.intersect1d(lat_lb, lat_ub)

    # make reference array for lat/lon indices
    ref_arr = np.arange(46*72).reshape((72, 46))

    # compute areas
    areas = []
    raw_weighted_flux = []
    for lat_idx in lat_idxs:
        for lon_idx in lon_idxs:

            # check if reference array index is in the ocean mask
            if ref_arr[lon_idx, lat_idx] in ocean_idxs:
                continue

            # get the lower right corner endpoints of the box
            lon_lrc = lon[lon_idx] - 2.5
            lat_lrc = lat[lat_idx] - 2

            # find area
            grid_area = area(subgrid_rect_obj(lon_lrc, lat_lrc))

            # get the fluxes
            grid_flux = flux_arr[month, lon_idx, lat_idx]

            # compute the weighted fluxes
            w_flux = grid_area * grid_flux

            # store data
            areas.append(grid_area)
            raw_weighted_flux.append(w_flux)

    # find the total area
    tot_area = np.array(areas).sum()

    # divide all weighted scale factors by total area
    weighted_flux = np.array(raw_weighted_flux) / tot_area

    return weighted_flux.sum()


def w_avg_flux_month(
    flux_arr, ocean_idxs,
    lon, lat
):
    """
    Find the weighted avg flux for one time slice. This function differs from
    w_avg_flux_month in that the full scale factor and flux arrays do not have
    to be provided. This avoids a lot of having to pass indices around.

    Parameters:
        flux_arr   (numpy array) : contains sfs over lon/lat
        ocean_idxs (numpy array) : indices of ocean grid cells when
                                   flattening a 72x46 array
        lon        (numpy array) : longitudes
        lat        (numpy array) : latitudes

    Returns:
        weighted avg of fluxes (float)

    NOTES:
    - the variable of interest in flux_arr is assumed to have shape (72, 46)
    - orients each grid box so that the scale factor is in the center
    """
    # test array dimensions
    assert flux_arr.shape[0] == 72
    assert flux_arr.shape[1] == 46

    # make reference array for lat/lon indices
    ref_arr = np.arange(46*72).reshape((72, 46))

    # compute areas
    areas = []
    raw_weighted_flux = []
    for lat_idx in np.arange(len(lat)):
        for lon_idx in np.arange(len(lon)):

            # check if reference array index is in the ocean mask
            if ref_arr[lon_idx, lat_idx] in ocean_idxs:
                continue

            # get the lower right corner endpoints of the box
            lon_lrc = lon[lon_idx] - 2.5
            lat_lrc = lat[lat_idx] - 2

            # find area
            grid_area = area(subgrid_rect_obj(lon_lrc, lat_lrc))

            # get the fluxes
            grid_flux = flux_arr[lon_idx, lat_idx]

            # compute the weighted fluxes
            w_flux = grid_area * grid_flux

            # store data
            areas.append(grid_area)
            raw_weighted_flux.append(w_flux)

    # find the total area
    tot_area = np.array(areas).sum()

    # divide all weighted scale factors by total area
    weighted_flux = np.array(raw_weighted_flux) / tot_area

    return weighted_flux.sum()


def region_sf_ts(lon_idx, lat_idx, sf_arr, lon, lat):
    """
    Given lat/lon bounds, find average scale factors over some time interval

    Parameters:
        lon_idx (numpy arr) : array of longitude indices for region oi
        lat_idx (numpy arr) : array of latitude indices for region oi
        sf_arr  (numpy arr) : global scale factors dim - T x 72 x 46
        lon     (numpy arr) : array of longitudes
        lat     (numpy arr) : array of latitudes

    Returns:
        1d numpy array, one value per time in first dimension of sf_arr
    """
    assert sf_arr.shape[1] == 72
    assert sf_arr.shape[2] == 46

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
        sf_arr  (numpy arr) : global scale factors dim - M x T x 72 x 46
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


def posterior_sf_compute(prior_flux, sfs):
    """
    Given prior and optimized scale factors, find the posterior scale factors.

    Parameters:
        prior_flux (numpy arr) : processed monthly prior fluxes Mxlonxlat
        sfs        (numpy arr) : obtained scale factors Mxlonxlat

    Returns:
        numpy arr (Mxlonxlat)

    NOTE:
    - keep the dimensions in mind because they can matter when flattening to
      find aggregated errors and things of that sort
    """
    assert prior_flux.shape[0] == sfs.shape[0]
    assert prior_flux.shape[1] == 72
    assert sfs.shape[1] == 72

    return prior_flux * sfs


def rmse_total_global(prior_flux, true_flux, sfs, ocean_mask, lon, lat):
    """
    Finds the RMSE over all months and land grid points. Uses actual average
    flux values.

    Parameters:
        prior_flux (numpy arr) : processed monthly prior fluxes
        true_flux  (numpy arr) : processed monthly true fluxes
        sfs        (numpy arr) : obtained scale factors
        ocean_mask (numpy arr) : indices of ocean grid cells
        lon        (numpy arr) : longitude array (72)
        lat        (numpy arr) : longitude array (46)

    Returns:
        float

    NOTE:
    - prior_flux/true_flux/sfs have dim (M, 72, 46)
    - ocean_mask has dim (N,), where N is number of ocean grid points
    """
    assert prior_flux.shape[0] == true_flux.shape[0]

    # find the posterior flux arr
    post = posterior_sf_compute(
        prior_flux=prior_flux,
        sfs=sfs
    )

    # find the gridwise squared error
    gw_sq_err = np.square(post - true_flux)

    # sum over months
    gw_sq_err = gw_sq_err.sum(axis=0)

    # find the weighted avg error
    weighted_error = w_avg_flux(
        flux_arr=gw_sq_err[np.newaxis, :, :],
        ocean_idxs=ocean_mask,
        lon=lon,
        lat=lat,
        lon_bounds=(-180, 180),
        lat_bounds=(-90, 90),
        month=0
    )

    return np.sqrt(weighted_error)


def rmse_month_global(prior_flux, true_flux, sfs, ocean_mask, lon, lat, month):
    """
    Finds the RMSE over input month and land grid points. Uses actual average
    flux values.

    Parameters:
        prior_flux (numpy arr) : processed monthly prior fluxes
        true_flux  (numpy arr) : processed monthly true fluxes
        sfs        (numpy arr) : obtained scale factors
        ocean_mask (numpy arr) : indices of ocean grid cells
        lon        (numpy arr) : longitude array (72)
        lat        (numpy arr) : longitude array (46)
        month      (int)       : month of interest

    Returns:
        float

    NOTE:
    - prior_flux/true_flux/sfs have dim (M, 72, 46)
    - ocean_mask has dim (N,), where N is number of ocean grid points
    """
    assert prior_flux.shape[0] == true_flux.shape[0]

    # find the posterior flux arr
    post = posterior_sf_compute(
        prior_flux=prior_flux,
        sfs=sfs
    )

    # find the gridwise squared error
    gw_sq_err = np.square(post - true_flux)

    # find the weighted avg error
    weighted_error = w_avg_flux(
        flux_arr=gw_sq_err,
        ocean_idxs=ocean_mask,
        lon=lon,
        lat=lat,
        lon_bounds=(-180, 180),
        lat_bounds=(-90, 90),
        month=month
    )

    return np.sqrt(weighted_error)


def rmse_all_months(prior_flux, true_flux, sfs, ocean_mask, lon, lat):
    """
    Find global RMSE for each individual month of a given scale factor set.

    Parameters:
        prior_flux (numpy arr) : processed monthly prior fluxes
        true_flux  (numpy arr) : processed monthly true fluxes
        sfs        (numpy arr) : obtained scale factors
        ocean_mask (numpy arr) : indices of ocean grid cells
        lon        (numpy arr) : longitude array (72)
        lat        (numpy arr) : longitude array (46)

    Returns:
        list of rmse's, one float for each month
    """
    monthly_rmse = [None] * sfs.shape[0]

    for month_idx in range(sfs.shape[0]):
        monthly_rmse[month_idx] = rmse_month_global(
            prior_flux=prior_flux,
            true_flux=true_flux,
            sfs=sfs,
            ocean_mask=ocean_mask,
            lon=lon,
            lat=lat,
            month=month_idx
        )

    return monthly_rmse


"""
Computation with GOSAT observations
"""


def lon_lat_to_IJ(lon, lat, lon_size=5, lat_size=4):
    """
    Transform (lon, lat) coordinates to 4x5 grid

    Parameters:
        lon      (float) : longitude
        lat      (float) : latitude
        lon_size (int)   : number of degrees in lon grid
                           default is 5
        lat_size (int)   : number of degress in lat grid
                           default is 4

    Returns:
        (I, J) longitude/latitude coordinates

    NOTES:
    - this code is copied from the ./code/modified/grid_mod.f
      - the primary difference is that python arrays are indexed from 0
    """
    LON_idx = int((lon + 180) / lon_size + .5)
    LAT_idx = int((lat + 90) / lat_size + .5)

    if LON_idx >= 72:
        LON_idx = LON_idx - 72

    return LON_idx, LAT_idx


def num_region_obs(count_arr, lon_idxs, lat_idxs):
    """
    Given lon/lat indices and array of GOSAT counts, determine number of
    observations in the region as defined by the indices.

    Parameters:
        count_arr (np arr) : array of counts (Mx72x46), M num months
        lon_idxs  (np arr) : array of longitude indices
        lat_idxs  (np arr) : array of latitude indices

    Returns:
        numpy array of counts per month
    """
    return count_arr[
        :,
        lon_idxs[0]:lon_idxs[-1],
        lat_idxs[0]:lat_idxs[-1]
    ].sum(axis=1).sum(axis=1)
