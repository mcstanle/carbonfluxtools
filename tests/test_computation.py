"""
Unit tests for computation functions in the carbonfluxtools package.

===============================================================================
Author        : Mike Stanley
Created       : May 20, 2020
Last Modified : May 20, 2020

===============================================================================
NOTE:
"""

from area import area
from carbonfluxtools import computation
import numpy as np
from os.path import expanduser

# testing constants
BASE = expanduser('~')
BASE_DIR = BASE + '/Research/Carbon_Flux'
LON_LAT_DIR = BASE_DIR + '/data/lon_lat_arrs'


def test_subgrid_rect_obj():
    """
    Function that creates polygon object of 4x5 lat/lon squares.
    """
    LAT_TEST_VAL = 20
    LON_TEST_VAL = 20

    # get the polygon object
    poly_obj = computation.subgrid_rect_obj(
        lon_llc=LON_TEST_VAL,
        lat_llc=LAT_TEST_VAL
    )

    assert poly_obj['coordinates'][0][0] == [20, 20]
    assert poly_obj['coordinates'][0][1] == [25, 20]
    assert poly_obj['coordinates'][0][2] == [25, 24]
    assert poly_obj['coordinates'][0][3] == [20, 24]
    assert poly_obj['coordinates'][0][4] == [20, 20]


def test_w_avg_sf():
    """
    Function that finds average scale factor over a region, weighting each
    scale factor in proportion to its land's area.

    This test averages four grid boxes with lon bounds (-175, 170) and lat
    bounds (-86, -82). These correspond to the boxes at [1, 1], [2, 1],
    [2, 2], and [1, 2] (counterclock-wise)
    """
    # get lon/lat arrays
    lon = np.load(LON_LAT_DIR + '/lon.npy')
    lat = np.load(LON_LAT_DIR + '/lat.npy')

    # define some test lon and lat bounds
    lon_bounds = (-175., -170.)
    lat_bounds = (-86., -82.)

    # make a faux scale factor array
    sfs_arr = np.zeros((1, 1, 72, 46))

    # fill in values
    sfs_arr[0, 0, 1, 1] = 1
    sfs_arr[0, 0, 2, 1] = 2
    sfs_arr[0, 0, 2, 2] = 3
    sfs_arr[0, 0, 1, 2] = 4

    # manually compute the weighted area average
    box_ar_1 = area(computation.subgrid_rect_obj(-177.5, -88))
    box_ar_2 = area(computation.subgrid_rect_obj(-172.5, -88))
    box_ar_3 = area(computation.subgrid_rect_obj(-172.5, -84))
    box_ar_4 = area(computation.subgrid_rect_obj(-177.5, -84))
    areas = [box_ar_1, box_ar_2, box_ar_3, box_ar_4]
    box_vals = [1, 2, 3, 4]

    # find the weighted average
    weighted_avg = sum([areas[i] * box_vals[i] / sum(areas) for i in range(4)])

    # compute using the function
    weighted_avg_comp = computation.w_avg_sf(
        sfs_arr=sfs_arr,
        lon=lon,
        lat=lat,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        month=0
    )

    assert weighted_avg == weighted_avg_comp


def test_w_avg_flux():
    """
    Essentially the same test as test_w_avg_sf. Some inputs are different.
    """
    # get lon/lat arrays
    lon = np.load(LON_LAT_DIR + '/lon.npy')
    lat = np.load(LON_LAT_DIR + '/lat.npy')

    # define some test lon and lat bounds
    lon_bounds = (-175., -170.)
    lat_bounds = (-86., -82.)

    # make a faux scale factor array
    sfs_arr = np.zeros((1, 72, 46))

    # fill in values
    sfs_arr[0, 1, 1] = 1
    sfs_arr[0, 2, 1] = 2
    sfs_arr[0, 2, 2] = 3
    sfs_arr[0, 1, 2] = 4

    # manually compute the weighted area average
    box_ar_1 = area(computation.subgrid_rect_obj(-177.5, -88))
    box_ar_2 = area(computation.subgrid_rect_obj(-172.5, -88))
    box_ar_3 = area(computation.subgrid_rect_obj(-172.5, -84))
    box_ar_4 = area(computation.subgrid_rect_obj(-177.5, -84))
    areas = [box_ar_1, box_ar_2, box_ar_3, box_ar_4]
    box_vals = [1, 2, 3, 4]

    # find the weighted average
    weighted_avg = sum([areas[i] * box_vals[i] / sum(areas) for i in range(4)])

    # compute using the function
    weighted_avg_comp = computation.w_avg_flux(
        flux_arr=sfs_arr,
        ocean_idxs=np.array([]),
        lon=lon,
        lat=lat,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        month=0
    )

    assert weighted_avg == weighted_avg_comp