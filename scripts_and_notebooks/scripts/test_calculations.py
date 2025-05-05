import sys
import os
import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from calculations import get_total_deformation, get_thetae, get_pv

test_data_dir = "data/test/"

def test_get_total_deformation():
    """
    Function to test the calculation of total deformation.
    
    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts if the total deformation values are calculated correctly and have the expected dimensions.
    
    """
    mock_data = xr.Dataset(
        {"U": (("latitude", "longitude"), np.random.rand(5, 5)),
         "V": (("latitude", "longitude"), np.random.rand(5, 5))},
        coords={
            "latitude": np.linspace(-90, 90, 5),
            "longitude": np.linspace(0, 360, 5)})

    deformation = get_total_deformation(mock_data)
    assert deformation is not None
    assert deformation.shape == (5, 5)

def test_get_thetae():
    """
    Function to test the calculation of equivalent potential temperature (thetae).
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Asserts if the thetae values are calculated correctly and have the expected dimensions.
    
    """
    mock_data = xr.Dataset(
        {"Q": (("level", "latitude", "longitude"), np.random.rand(37, 5, 5) * 1e-3),
         "T": (("level", "latitude", "longitude"), np.random.rand(37, 5, 5) * 300)},
        coords={
            "level": np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70.,
                               100., 125., 150., 175., 200., 225., 250., 300., 350., 400.,
                               450., 500., 550., 600., 650., 700., 750., 775., 800., 825.,
                               850., 875., 900., 925., 950., 975., 1000.]),
            "latitude": np.linspace(-90, 90, 5),
            "longitude": np.linspace(0, 360, 5)})

    level = 925 # units: hPa
    thetae = get_thetae(mock_data, level)
    assert thetae is not None
    assert thetae.shape == (5, 5)

def test_get_pv():
    """
    Function to test the calculation of potential vorticity (PV) at a specific pressure level.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts if the PV values are calculated correctly and have the expected dimensions.

    """
    mock_data = xr.Dataset(
        {"PV": (("level", "latitude", "longitude"), np.random.rand(37, 5, 5) * 1e-6)},
        coords={
            "level": np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70.,
                               100., 125., 150., 175., 200., 225., 250., 300., 350., 400.,
                               450., 500., 550., 600., 650., 700., 750., 775., 800., 825.,
                               850., 875., 900., 925., 950., 975., 1000.]),
            "latitude": np.linspace(-90, 90, 5),
            "longitude": np.linspace(0, 360, 5)})

    level = 500  # units: hPa
    pv = get_pv(mock_data, level)
    assert pv is not None
    assert pv.shape == (5, 5)
    assert_almost_equal(pv.values, mock_data["PV"].sel(level=level).values)