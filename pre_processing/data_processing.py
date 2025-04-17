import os
import xarray as xr
import pandas as pd
import numpy as np
from siphon.catalog import TDSCatalog

# Code from Tony's EAE 595 Project (https://github.com/anthony-illenden/EAE-593-Project)
def load_datasets(year, month, start_day, start_hour=0, end_day=None, end_hour=23):
    """
    Load hourly ERA5 data from the THREDDS server.

    Parameters
    ----------
    year : int
        Year of the data to load.
    month : int
        Month of the data to load.
    start_day : int
        Start day of the data to load.
    start_hour : int, optional
        Start hour of the data to load (default is 0).
    end_day : int, optional
        End day of the data to load (default is None, which means the same as start_day).
    end_hour : int, optional
        End hour of the data to load (default is 23).

    Returns
    -------
    ds_pl : xarray.Dataset
        Dataset containing pressure level data.
    ds_sfc : xarray.Dataset
        Dataset containing surface data.
    
    """
    # Set end_day to start_day if not provided
    if end_day is None:
        end_day = start_day
    
    # Get the last day of the month
    last_day_of_month = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    last_day_str = f"{last_day_of_month.day:02d}"  # format last day as two digits

    # Format date and time strings
    year_month = f'{year}{month:02d}'
    start_time = f'{year}{month:02d}{start_day:02d}{start_hour:02d}'  # yyyymmddhh (start)
    end_time = f'{year}{month:02d}{end_day:02d}{end_hour:02d}'  # yyyymmddhh (end)

    # Define URLs for pressure level datasets with specific time ranges
    urls = {
        'temperature_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_130_t.ll025sc.{start_time}_{end_time}.nc',
        'geopotential_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_129_z.ll025sc.{start_time}_{end_time}.nc',
        'humidity_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_133_q.ll025sc.{start_time}_{end_time}.nc',
        'v_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_132_v.ll025uv.{start_time}_{end_time}.nc',
        'u_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_131_u.ll025uv.{start_time}_{end_time}.nc',
        'w_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_135_w.ll025sc.{start_time}_{end_time}.nc',
        'pv_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_060_pv.ll025sc.{start_time}_{end_time}.nc',
        
        # Define URLs for surface datasets to cover the full month using last_day_of_month
        'mslp_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.128_151_msl.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc',
        'u_wind_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.228_131_u10n.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc',
        'v_wind_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.228_132_v10n.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc',
        'temperature_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.128_167_2t.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc',
        'dew_point_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.128_168_2d.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc'
    }

    # Initialize empty dictionaries for datasets
    datasets = {}

    # Try to load datasets from the URLs
    for var, url in urls.items():
        try:
            tds_catalog = TDSCatalog(url)
            ds_url = tds_catalog.datasets[0].access_urls['OPENDAP']
            ds = xr.open_dataset(ds_url)
            datasets[var] = ds
            print(f"Successfully loaded {var}")

        except Exception as e:
            print(f"Error loading {var}: {e}")

    # Merge pressure level datasets if available
    ds_pl, ds_sfc = None, None

    try:
        ds_pl = xr.merge([datasets['temperature_pl'], datasets['geopotential_pl'], datasets['humidity_pl'], 
                        datasets['v_wind_pl'], datasets['u_wind_pl'], datasets['w_wind_pl'], datasets['pv_pl']])
        print("Successfully merged pressure level datasets")
    except KeyError as e:
        print(f"Error merging pressure level datasets: {e}")

    # Merge surface datasets if available
    try:
        ds_sfc = xr.merge([datasets['mslp_sfc'], datasets['v_wind_sfc'], datasets['u_wind_sfc'],
                        datasets['temperature_sfc'], datasets['dew_point_sfc']])
        print("Successfully merged surface datasets")
    except KeyError as e:
        print(f"Error merging surface datasets: {e}")

    # Synchronize time dimensions
    try:
        if ds_pl is not None and ds_sfc is not None:
            first_time_pl, last_time_pl = ds_pl['time'].min().values, ds_pl['time'].max().values
            ds_sfc = ds_sfc.sel(time=slice(first_time_pl, last_time_pl))
    except KeyError as e:
        print(f"Error accessing 'time' in the datasets: {e}")
    except Exception as e:
        print(f"An error occurred during slicing: {e}")
        
    return ds_pl, ds_sfc

def load_local_datasets(year, month, day, hour, data_dir="C:/Users/Tony/Documents/GitHub/EAE-598-Project/data/era5"):
    """
    Load local datasets from the specified directory.
    
    Parameters
    ----------
    year : int
        Year of the event data to load.
    month : int
        Month of the event data to load.
    day : int
        Day of the event data to load.
    hour : int
        Hour of the event data to load.
    data_dir : str, optional
        Directory where the data files are stored (default is "C:/Users/Tony/Documents/GitHub/EAE-598-Project/data/era5").

    Returns
    -------
    ds_pl : xarray.Dataset
        Dataset containing pressure level data.
    ds_sfc : xarray.Dataset
        Dataset containing surface data.
    
    """
    pl_filename = f"pl_{year:04d}_{month:02d}_{day:02d}_{hour:02d}.nc"
    pl_path = os.path.join(data_dir, pl_filename)

    sfc_filename = f"sfc_{year:04d}_{month:02d}_{day:02d}_{hour:02d}.nc"
    sfc_path = os.path.join(data_dir, sfc_filename)

    try:
        ds_pl = xr.open_dataset(pl_path)
    except FileNotFoundError:
        print(f"Missing file: {pl_path}")
        ds_pl = None

    try:
        ds_sfc = xr.open_dataset(sfc_path)
    except FileNotFoundError:
        print(f"Missing file: {sfc_path}")
        ds_sfc = None

    return ds_pl, ds_sfc

def slice_dataset_to_domain(ds_pl, ds_sfc, directions):
     """
     Slice the datasets to the specified geographical domain.

     Parameters
     ----------
     ds_pl : xarray.Dataset
        Dataset containing pressure level data.
     ds_sfc : xarray.Dataset
        Dataset containing surface data.
     directions : dict
        Dictionary containing the geographical boundaries for slicing.

     Returns
     -------
     ds_pl_sliced : xarray.Dataset
        Sliced pressure level dataset.
     ds_sfc_sliced : xarray.Dataset
        Sliced surface dataset.

     """
     ds_pl_sliced = ds_pl.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
     ds_sfc_sliced = ds_sfc.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
     return ds_pl_sliced, ds_sfc_sliced

def get_point_data(final_ds, lat, lon, buffer):
    """
    Extract data for a specific latitude and longitude with a buffer and calculate the mean.

    Parameters
    ----------
    final_ds : xarray.Dataset
        Dataset containing the event data to extract.
    lat : float
        Latitude of the point.
    lon : float
        Longitude of the point.
    buffer : float
        Buffer size in degrees.
    
    Returns
    -------
    ds_area_point_mean : xarray.Dataset
        Dataset containing the mean values for the specified point and buffer.
    
    """
    # Convert longitude from west to east
    lon_e = 360 - lon 
    ds_area_point = final_ds.sel(latitude=slice(lat + buffer, lat - buffer), longitude=slice(lon_e - buffer, lon_e + buffer))
    ds_area_point_mean = ds_area_point.mean(dim=['latitude', 'longitude'])
    return ds_area_point_mean

def add_time_dimension(final_ds, year, month, day, start_hour):
    """
    Add a time dimension to the dataset.

    Parameters
    ----------
    final_ds : xarray.Dataset
        Dataset to which the time dimension will be added.
    year : int
        Year of the event data.
    month : int
        Month of the event data.
    day : int
        Day of the event data.
    start_hour : int
        Hour of the event data.
    
    Returns 
    -------
    ds_final : xarray.Dataset
        Dataset with the added time dimension.

    """
    formatted_time = np.datetime64(pd.to_datetime(f"{year}-{month:02d}-{day:02d} {start_hour:02d}:00"))
    ds_final = final_ds.expand_dims(time=[formatted_time])
    
    return ds_final

def save_to_csv(ds_point, year, month, day, label, output_file="condensed_final_all_events.csv"):
    """
    Save the dataset to a CSV file.
    
    Parameters
    ----------
    ds_point : xarray.Dataset
        Dataset containing the event data to save.
    year : int
        Year of the event data.
    month : int
        Month of the event data.
    day : int
        Day of the event data.
    label : str
        Label for the event data.
    output_file : str, optional
        Path to the output CSV file (default is "condensed_final_all_events.csv").
    
    Returns
    -------
    None

    """
    df = ds_point.to_dataframe().reset_index()
    df['label'] = label
    df['year'] = year
    df['month'] = month
    df['day'] = day

    # Check if the file exists to determine if headers should be written
    file_exists = os.path.isfile(output_file)

    # Append data to a single CSV
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)