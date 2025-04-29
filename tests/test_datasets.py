import sys
import os
import xarray as xr
import numpy as np
from numpy.testing import assert_equal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pre_processing')))
from data_processing import load_local_datasets

def test_load_local_datasets():
    """
    Function to test the loading of local datasets.
    
    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts if the datasets are loaded correctly and have the expected dimensions and variables.

    """

    year, month, day, hour = 2025, 4, 14, 0
    data_dir = "C:\\Users\\Tony\\Documents\\GitHub\\EAE-598-Project\\data\\era5"

    ds_pl, ds_sfc = load_local_datasets(year, month, day, hour, data_dir=data_dir)

    expected_dims_pl = {'time': 1, 'level': 37, 'latitude': 141, 'longitude': 201}
    for dim, expected_size in expected_dims_pl.items():
        result = ds_pl.dims[dim]
        assert result == expected_size, f"Pressure level dataset: Expected {dim} size {expected_size}, got {result}"

    expected_levels = np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70.,
                                 100., 125., 150., 175., 200., 225., 250., 300., 350., 400.,
                                 450., 500., 550., 600., 650., 700., 750., 775., 800., 825.,
                                 850., 875., 900., 925., 950., 975., 1000.])
    assert 'level' in ds_pl.coords, "Pressure level dataset is missing 'level' coordinate"
    assert len(ds_pl.coords['level']) == 37, f"Expected 37 levels, got {len(ds_pl.coords['level'])}"
    assert np.allclose(ds_pl.coords['level'].values, expected_levels), \
        f"Pressure level dataset levels do not match expected values"

    expected_dims_sfc = {'time': 1, 'latitude': 141, 'longitude': 201}
    for dim, expected_size in expected_dims_sfc.items():
        result = ds_sfc.dims[dim]
        assert result == expected_size, f"Surface dataset: Expected {dim} size {expected_size}, got {result}"

    expected_vars_pl = ['Z', 'T', 'Q', 'V', 'U', 'W', 'PV']
    for var in expected_vars_pl:
        assert var in ds_pl.variables, f"Variable {var} is missing in pressure level dataset"

    expected_vars_sfc = ['mslp', 'u10', 'v10', 't2m', 'd2m']
    for var in expected_vars_sfc:
        assert var in ds_sfc.variables, f"Variable {var} is missing in surface dataset"

