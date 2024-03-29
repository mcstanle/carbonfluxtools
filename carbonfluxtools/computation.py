"""
A collection of computation related functions to support
1. finding bias
2. computing regional areas

Author   : Mike Stanley
Created  : May 12, 2020
Modified : Oct 2, 2020

================================================================================
"""
from area import area
from carbonfluxtools import io
import numpy as np
from scipy import constants


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


def compute_gridded_areas(lons, lats):
    """
    Creates numpy array of grid areas

    Parameters:
        lons (np arr) : array of longitudes
        lats (np arr) : array of latitudes

    Returns:
        np array of dim len(lon) x len(lat) with region areas in m^3
    """
    reg_areas = np.zeros(shape=(len(lons), len(lats)))

    for lon_idx, lon in enumerate(lons):
        for lat_idx, lat in enumerate(lats):

            # create geo object
            area_obj = subgrid_rect_obj(lon_llc=lon, lat_llc=lat)

            # find region area
            reg_areas[lon_idx, lat_idx] = area(area_obj)

    return reg_areas


def compute_global_flux(flux_arr, lons, lats):
    """
    Finds total flux over some time period of interest which is defined by the
    array inputted.

    Computes global flux in grams.

    Uses Avogadro's number for particle count per mole and
    12.0107 for grams of Carbon per mole

    Parameters:
        flux_arr (np arr) : dim Tx72x46
        lons     (np arr) : dim 72
        lats     (np arr) : dim 46

    Returns:
        float
    """
    # find the grid cell areas
    reg_areas = compute_gridded_areas(lons=lons, lats=lats)

    # array with total time counts per grid cell (in cubic meters)
    grid_count = 1e6 * reg_areas * flux_arr.sum(axis=0)

    # translate to grams for each grid cell
    grid_grams = grid_count * 12.0107 / constants.Avogadro

    return grid_grams.sum()


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


def w_avg_flux_variance(
    flux_var_arr, ocean_idxs,
    lon, lat,
    lon_bounds, lat_bounds,
    month
):
    """
    Find the weighted avg flux variance for one time slice. The weighted
    average only takes land grid cells into account, i.e. it skips the
    ocean indices

    Parameters:
        flux_var_arr (numpy array) : contains sfs over lon/lat
        ocean_idxs   (numpy array) : indices of ocean grid cells when
                                     flattening a 72x46 array
        lon          (numpy array) : longitudes
        lat          (numpy array) : latitudes
        lon_bounds   (tuple)       : eastern oriented
        lat_bounds   (tuple)       : northern oriented
        month        (int)         : the month of interest

    Returns:
        weighted avg of fluxes (float)

    NOTES:
    - the variable of interest in sfs is assumed to have shape (*, 72, 46)
      where * is the number of months
    - the bounds for lon/lat are inclusive
    - orients each grid box so that the scale factor is in the center
    """
    # test array dimensions
    assert len(flux_var_arr.shape) == 3
    assert flux_var_arr.shape[1] == 72
    assert flux_var_arr.shape[2] == 46

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
    raw_weighted_flux_var = []
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
            grid_flux = flux_var_arr[month, lon_idx, lat_idx]

            # compute the weighted fluxes
            w_flux_var = np.square(grid_area) * grid_flux

            # store data
            areas.append(grid_area)
            raw_weighted_flux_var.append(w_flux_var)

    # find the total area
    tot_area = np.array(areas).sum()

    # divide all weighted scale factors by total area
    weighted_flux = np.array(raw_weighted_flux_var) / np.square(tot_area)

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
    agg_type='mean',
    month_list=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep']
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
    month_idxs = io.find_month_idxs(fluxes=flux_xbpch, month_list=month_list)

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
        elif agg_type is None:
            fluxes_agg_raw.append(
                flux_xbpch[flux_var_nm].values[idxs, :, :]
            )

    # concatenate the above together
    if agg_type is not None:
        return np.stack(fluxes_agg_raw), flux_xbpch['lon'].values, flux_xbpch['lat'].values

    else:
        return fluxes_agg_raw, flux_xbpch['lon'].values, flux_xbpch['lat'].values


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


def rmse_total_monthly_pre(
    prior_flux, true_flux, sfs, month_idxs,
    ocean_mask, lon, lat
):
    """
    Finds the RMSE over each months for land grid points. Uses actual average
    flux values.

    Multiply every element of the month by scale factor instead of
    multiplying monthly average.

    pre refers to finding the errors before taking average over time

    Parameters:
        prior_flux (nump arry) : array of full prior fluxes (T x 72 x 46)
        true_flux  (nump arry) : array of full true fluxes (T x 72 x 46)
        sfs        (numpy arr) : obtained scale factors (M x 72 x 46)
        month_idxs (dict)      : key: month name, value array indices
        ocean_mask (numpy arr) : indices of ocean grid cells
        lon        (numpy arr) : longitude array (72)
        lat        (numpy arr) : longitude array (46)

    Returns:
        list of monthly errors

    NOTE:
    - T is the number of 3hr steps
    - prior_flux/true_flux/sfs have dim (M, 72, 46)
    - ocean_mask has dim (N,), where N is number of ocean grid points
    """
    # defin the sequential list of months
    month_list = [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]

    # for each month, find posterior
    monthly_errors = []
    month_idx = 0
    for idx, month_nm in enumerate(month_list):

        if month_nm not in month_idxs:
            break

        # get the month indices
        month_idxs_idx = month_idxs[month_nm]

        # find posterior flux (T_i x 72 x 46) X (72 x 46)
        post_month = prior_flux[month_idxs_idx, :, :] * sfs[month_idx, :, :]

        # find sq error and find average over time
        sq_err = np.square(
            true_flux[month_idxs_idx, :, :] - post_month
        ).mean(axis=0)

        # find the weighted avg error
        weighted_error = w_avg_flux(
            flux_arr=sq_err[np.newaxis, :, :],
            ocean_idxs=ocean_mask,
            lon=lon,
            lat=lat,
            lon_bounds=(-180, 180),
            lat_bounds=(-90, 90),
            month=0
        )

        # append the above weighted error to to monthly errors
        monthly_errors.append(np.sqrt(weighted_error))

        month_idx += 1

    return monthly_errors


def rmse_total_global(
    prior_flux, true_flux, sfs,
    month_idx, ocean_mask, lon, lat
):
    """
    Finds the RMSE over all months and land grid points. Uses actual average
    flux values.

    Parameters:
        # prior_flux (numpy arr) : processed monthly prior fluxes
        # true_flux  (numpy arr) : processed monthly true fluxes
        prior_flux (numpy arr) : full prior fluxes (T x 72 x 46)
        true_flux  (numpy arr) : full true fluxes (T x 72 x 46)
        sfs        (numpy arr) : obtained scale factors
        month_idx  (dict)      : for each month, provides flux indices
                                 output from io.find_month_idxs
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
    # post = posterior_sf_compute(
    #     prior_flux=prior_flux,
    #     sfs=sfs
    # )

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


def rmse_w(flux1, flux2, ocean_mask, lon, lat):
    """
    RMSE but weighted by grid areas.

    Designed to take whole 3hr flux arrays!

    First take the square different at each grid cell over time,
    then average over time and multiply each grid cell by a weight
    which is proportional to its weight with respect to all grid
    areas. Then add and take square root.

    T := # 3hr time steps
    N := number of longitude cuts
    M := number of latitude cuts
    P := number of ocean grid cells

    Parameters:
        flux1      (np arr) : (T, N, M)
        flux2      (np arr) : (T, N, M)
        ocean_mask (np arr) : (P,)
        lon        (np arr) : (N,)
        lat        (np arr) : (M,)

    Returns:
        sqrt(weighted error) (float)

    """
    # find square difference across space and time
    sq_diff = np.square(flux1 - flux2)

    # average over time
    sq_diff_ta = sq_diff.mean(axis=0)

    # find the error weighted by area
    weighted_error = w_avg_flux(
        flux_arr=sq_diff_ta[np.newaxis, :, :],
        ocean_idxs=ocean_mask,
        lon=lon,
        lat=lat,
        lon_bounds=(-180, 180),
        lat_bounds=(-90, 90),
        month=0
    )

    return np.sqrt(weighted_error)


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


"""
Computation of analytic optimal scale factors

There are two varieties referred to as follows...
1. Inner (take square differences over all time/grid points and then average)
2. Outer (take average difference and then square over all grid points)
"""


def optimal_sfs_1m(prior_flux, true_flux, land_idx, outer=False):
    """
    For a single month, finds the optimal scale factor for land grid points

    Inner Cost function refers to the one where every 3hr time step square
    error is considered

    Outer Cost function refers to the one where all errors are summed and
    then squared

    NOTE -
    - to take care of infinite values, whenever there is a 0 in the
      denominator, we simply return a 0

    Parameters:
        prior_flux (np arr) : Tx72x46 array
        true_flux  (np arr) : Tx72x46 array
        land_idx   (np arr) : 1D array with indices of 72*46 array that
                              are land
        outer      (bool)   : switch to optimize the outer vs. inner cost
                              function

    Returns:
        np array 72x46 with optimized scale factors in the land indices
    """
    # get time step constant
    T = prior_flux.shape[0]
    N = 72 * 46

    # reshape arrays to be Tx(72*46)
    prior_flux_rs = prior_flux.reshape(T, N)
    true_flux_rs = true_flux.reshape(T, N)

    # instantiate default one scale factors
    opt_sfs = np.ones(N)

    for idx in np.arange(0, N):

        # skip non land areas
        if idx not in land_idx:
            continue

        # find optimal scale factor
        if outer:

            # find the denominator
            den_val = np.sum(prior_flux_rs[:, idx])

            if den_val == 0:
                opt_sfs[idx] = 1
            else:
                opt_sfs[idx] = np.sum(true_flux_rs[:, idx]) / den_val

        else:

            # find the denominator
            den_val = np.sum(np.square(prior_flux_rs[:, idx]))

            if den_val == 0:
                opt_sfs[idx] = 1
            else:
                opt_sfs[idx] = np.sum(
                    prior_flux_rs[:, idx] * true_flux_rs[:, idx]
                ) / den_val

    # reshape back to spatial grid for output
    return opt_sfs.reshape(72, 46)


def optimal_sfs_allm(prior_flux, true_flux, land_idx, month_idx, outer=False):
    """
    Finds optimal scale factors for all months by extending optimal_sfs_1m.

    M := 3hr time steps
    N := number of months

    Parameters:
        Parameters:
        prior_flux (np arr) : Mx72x46 array
        true_flux  (np arr) : Mx72x46 array
        land_idx   (np arr) : 1D array with indices of 72*46 array that
                              are land
        month_idx  (dict)   : keys are month names values are time indices
        outer      (bool)   : switch to optimize the outer vs. inner cost
                              function

    Returns:
        np array Nx72x46 with optimized scale factors in the land indices
    """
    # initialize array to hold optimized values
    opt_sfs = np.ones((len(month_idx), 72, 46))

    month_count = 0
    for month_nm, month_idxs in month_idx.items():

        # find the monthly fluxes
        prior_month = prior_flux[month_idxs, :, :]
        true_month = true_flux[month_idxs, :, :]

        # find the optimal sfs
        opt_sfs[month_count, :, :] = optimal_sfs_1m(
            prior_flux=prior_month,
            true_flux=true_month,
            land_idx=land_idx,
            outer=outer
        )

        month_count += 1

    return opt_sfs
