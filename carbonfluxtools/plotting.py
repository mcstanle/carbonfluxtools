"""
A collection of basic plotting tools including
1. plot single location dot
2. plot regional dots

Author   : Mike Stanley
Created  : May 12, 2020
Modified : May 12, 2020

================================================================================
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
plt.style.use('ggplot')


def plot_single_loc(lon_pt, lat_pt, extent_lst,
                    lon, lat, save_loc=None, title=None,
                    ):
    """
    Plot a single point on a map

    Parameters:
        lon_pt     (int)    : longitude idx
        lat_pt     (int)    : latitude idx
        extent_lst (list)   : plotting region
        lon        (np arr) : array of longitudes (len == 72)
        lat        (np arr) : array of latitudes (len == 46)
        save_loc   (str)    : save location of plot (default None)
        title      (str)    : title for plot if given

    Returns:
        matplotlib plot save if location give
    """
    assert lon.shape[0] == 72
    assert lat.shape[0] == 46

    fig = plt.figure(figsize=(12.5, 8))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')

    ax.scatter(lon[lon_pt], lat[lat_pt], transform=ccrs.PlateCarree(),
               marker='s', s=50)

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.OCEAN)

    ax.set_extent(extent_lst)

    if title:
        ax.set_title(title)

    if save_loc:
        plt.savefig(save_loc)

    plt.show()


def plot_region_dots(lon_pts, lat_pts, extent_lst,
                     lon, lat, save_loc=None, title=None
                     ):
    """
    Plot an entire region

    Parameters:
        lon_pts     (int)    : longitude idxs
        lat_pts     (int)    : latitude idxs
        extent_lst  (list)   : plotting region
        lon        (np arr) : array of longitudes
        lat        (np arr) : array of latitudes
        save_loc   (str)    : save location of plot (default None)
        title      (str)    : title for plot if given

    Returns:
        matplotlib plot save if location give
    """

    fig = plt.figure(figsize=(12.5, 8))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')

    for lat_ in lat_pts:
        ax.scatter([lon[i] for i in lon_pts], [lat[lat_]] * len(lon_pts),
                   transform=ccrs.PlateCarree(), color='red')

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.OCEAN)

    ax.set_extent(extent_lst)

    if title:
        ax.set_title(title)

    if save_loc:
        plt.savefig(save_loc)

    plt.show()


def bw_plot(
    sf_arr, opt_sf_arr, lon_idx, lat_idx,
    num_obs=None,
    num_months=8,
    title=None,
    save_loc=None,
    text_h_offset=0.5,
    text_v_offset=0.3
):
    """
    Make a box-whisker plot for a single location

    Parameters
        sf_arr        (np arr) : {num OSSEs} x {num Months}
        opt_sf_arr    (np arr) : {months} x {lon} x {lat}
        lon_idx       (int)    : index of longitude
        lat_idx       (int)    : index of latitude
        num_obs       (np arr) : array of number of observations per month
        num_months    (int)    : number of months to plot
        title         (str)    : title for plot
        save_loc      (str)    : location where to save image
        text_h_offset (float)  : count labels - horizonal offset
        text_v_offset (float)  : count labels - vertical offset

    Note:
    - expects to only be given Jan through Aug
    """
    MONTH_LIST = [
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec'
    ]

    plt.figure(figsize=(12.5, 7))

    plt.boxplot(sf_arr[:, :num_months])

    # find the mean scale factors for each month
    sf_mean = sf_arr[:, :num_months].mean(axis=0)

    plt.scatter(
        np.arange(1, num_months + 1),
        opt_sf_arr[:num_months, lon_idx, lat_idx],
        color='red', label='Optimal Scale Factors'
    )

    # labels
    if title:
        plt.title(title)
    plt.xlabel('Months')
    plt.ylabel('Scale Factors')

    plt.xticks(np.arange(1, 10), MONTH_LIST[:num_months])

    if num_obs is not None:

        # add the text with the observation count
        for month_idx in range(num_months):
            plt.text(
                x=month_idx+1 + text_h_offset,
                y=sf_mean[month_idx] + text_v_offset,
                s=str(num_obs[month_idx])
            )

    plt.legend(loc='best')
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc)
    plt.show()


def bw_region_plot(
    sf_arr, opt_sf_arr,
    num_obs=None,
    num_months=8,
    title=None, save_loc=None,
    text_h_offset=0.5,
    text_v_offset=0.3
):
    """
    Make a box-whisker plot for a region

    Parameters
        sf_arr     (np arr)   : {num OSSEs} x {num Months}
        opt_sf_arr (np arr)   : {months}
        num_obs    (np arr)   : array of number of observations per month
        num_months (int)      : number of months to plot
        title      (str)      : title for plot
        save_loc   (str)      : location where to save image
        text_h_offset (float) : count labels - horizonal offset
        text_v_offset (float) : count labels - vertical offset

    Note:
    - expects to only be given Jan through Aug
    """
    MONTH_LIST = [
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec'
    ]

    plt.figure(figsize=(12.5, 7))
    plt.boxplot(sf_arr[:, :num_months])

    # find the mean scale factors for each month
    sf_mean = sf_arr[:, :num_months].mean(axis=0)

    plt.scatter(np.arange(1, num_months+1), opt_sf_arr[:num_months],
                color='red', label='Optimal Scale Factors')

    # labels
    if title:
        plt.title(title)
    plt.xlabel('Months')
    plt.ylabel('Scale Factors')

    plt.xticks(np.arange(1, num_months+2), MONTH_LIST[:num_months])

    if num_obs is not None:

        # add the text with the observation count
        for month_idx in range(num_months):
            plt.text(
                x=month_idx+1 + text_h_offset,
                y=sf_mean[month_idx] + text_v_offset,
                s=str(num_obs[month_idx])
            )

    plt.legend(loc='best')
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc)
    plt.show()


def plot_month_sfs(sf_arr, lon, lat, write_loc=None):
    """
    Plot a global heatmap of scale factors for a single month

    Parameters:
        sf_arr    (np arr) : {lon} x {lat} array
        lon       (np arr) :
        lat       (np arr) :
        write_loc (str)    : file path to which plot should be written

    Returns:
        None - but write plot to file
    """
    assert sf_arr.shape[0] == 72
    assert sf_arr.shape[1] == 46

    fig = plt.figure(figsize=(12.5, 6))
    norm = colors.DivergingNorm(vcenter=1)

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
    contour = ax.contourf(
        lon, lat, sf_arr.T, levels=100,
        transform=ccrs.PlateCarree(), cmap='bwr', norm=norm
    )
    fig.colorbar(contour, ax=ax, orientation='vertical', extend='both')
    ax.add_feature(cfeature.COASTLINE)

    if write_loc:
        plt.savefig(write_loc)
    else:
        plt.show()
