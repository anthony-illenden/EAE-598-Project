import os
import pandas as pd
import xarray as xr
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
    start_hour = f'{year}{month:02d}{start_day:02d}{start_hour:02d}'  # yyyymmddhh (start)
    end_hour = f'{year}{month:02d}{end_day:02d}{end_hour:02d}'  # yyyymmddhh (end)

    # Define URLs for pressure level datasets with specific time ranges
    urls = {
        'temperature_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_130_t.ll025sc.{start_hour}_{end_hour}.nc',
        'geopotential_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_129_z.ll025sc.{start_hour}_{end_hour}.nc',
        'humidity_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_133_q.ll025sc.{start_hour}_{end_hour}.nc',
        'v_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_132_v.ll025uv.{start_hour}_{end_hour}.nc',
        'u_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_131_u.ll025uv.{start_hour}_{end_hour}.nc',
        'w_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_135_w.ll025sc.{start_hour}_{end_hour}.nc',
        'pv_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_060_pv.ll025sc.{start_hour}_{end_hour}.nc',
        
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

def slice_dataset_to_domain(ds_pl, ds_sfc, directions):
     """
     Slices the datasets to the specified geographical domain.

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

def save_dataset(ds, filename, directory="data\era5"):
    """
    Save the dataset to a NetCDF file.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to save.
    filename : str
        Name of the output file.
    directory : str, optional
        Directory to save the file (default is "data/era5").
    
    Returns 
    -------
    None
    
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    ds.to_netcdf(filepath)
    print(f"Dataset saved to {filepath}")

def main():
    """
    Main function to process the datasets for specific events.
    
    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    events = [
        {"year": 2017, "month": 2, "start_day": 20, "start_hour": 3},
        {"year": 2019, "month": 2, "start_day": 13, "start_hour": 17},
        {"year": 2015, "month": 11, "start_day": 19, "start_hour": 2},
        {"year": 2016, "month": 1, "start_day": 28, "start_hour": 17},
        {"year": 2014, "month": 2, "start_day": 7, "start_hour": 13},
        {"year": 2012, "month": 10, "start_day": 18, "start_hour": 9},
        {"year": 2017, "month": 11, "start_day": 15, "start_hour": 6},
        {"year": 2017, "month": 3, "start_day": 17, "start_hour": 6},
        {"year": 2011, "month": 11, "start_day": 22, "start_hour": 14},
        {"year": 2023, "month": 1, "start_day": 9, "start_hour": 12},
        {"year": 2010, "month": 3, "start_day": 12, "start_hour": 0},
        {"year": 2014, "month": 11, "start_day": 3, "start_hour": 10},
        {"year": 2018, "month": 1, "start_day": 23, "start_hour": 9},
        {"year": 2018, "month": 11, "start_day": 22, "start_hour": 11},
        {"year": 2010, "month": 2, "start_day": 26, "start_hour": 3},
        {"year": 2012, "month": 3, "start_day": 9, "start_hour": 4},
        {"year": 2013, "month": 11, "start_day": 11, "start_hour": 17},
        {"year": 2016, "month": 1, "start_day": 17, "start_hour": 9},
        {"year": 2016, "month": 11, "start_day": 7, "start_hour": 11},
        {"year": 2017, "month": 3, "start_day": 28, "start_hour": 18},
        {"year": 2020, "month": 11, "start_day": 14, "start_hour": 21},
        {"year": 2021, "month": 10, "start_day": 24, "start_hour": 2},
        {"year": 2023, "month": 7, "start_day": 23, "start_hour": 6},
        {"year": 2006, "month": 11, "start_day": 4, "start_hour": 7},
        {"year": 2023, "month": 1, "start_day": 14, "start_hour": 19},
        {"year": 2014, "month": 12, "start_day": 10, "start_hour": 21},
        {"year": 2021, "month": 12, "start_day": 18, "start_hour": 13},
        {"year": 2015, "month": 8, "start_day": 27, "start_hour": 23},
        {"year": 2014, "month": 11, "start_day": 21, "start_hour": 8},
        {"year": 2012, "month": 11, "start_day": 19, "start_hour": 19},
        {"year": 2005, "month": 1, "start_day": 18, "start_hour": 11},
        {"year": 2010, "month": 1, "start_day": 20, "start_hour": 6},
        {"year": 2018, "month": 2, "start_day": 13, "start_hour": 2},
        {"year": 2017, "month": 12, "start_day": 28, "start_hour": 0},
        {"year": 2017, "month": 11, "start_day": 19, "start_hour": 3},
        {"year": 2016, "month": 11, "start_day": 14, "start_hour": 6},
        {"year": 2015, "month": 2, "start_day": 6, "start_hour": 0},
        {"year": 2015, "month": 3, "start_day": 25, "start_hour": 6},
        {"year": 2014, "month": 11, "start_day": 5, "start_hour": 21},
        {"year": 2014, "month": 4, "start_day": 16, "start_hour": 14},
        {"year": 2010, "month": 2, "start_day": 4, "start_hour": 18},
        {"year": 2008, "month": 1, "start_day": 3, "start_hour": 10},
        {"year": 2008, "month": 2, "start_day": 23, "start_hour": 0},
        {"year": 2019, "month": 2, "start_day": 2, "start_hour": 1},
        {"year": 2019, "month": 1, "start_day": 17, "start_hour": 3},
        {"year": 2022, "month": 12, "start_day": 30, "start_hour": 20},
        {"year": 2024, "month": 12, "start_day": 28, "start_hour": 18},
        {"year": 2022, "month": 6, "start_day": 10, "start_hour": 21},
        {"year": 2017, "month": 2, "start_day": 6, "start_hour": 2},
        {"year": 2005, "month": 3, "start_day": 26, "start_hour": 9}]
    

    directions = {'North': 55, 'East': 250, 'South': 20, 'West': 200}

    for event in events:
        year, month, day, start_hour = event["year"], event["month"], event["start_day"], event["start_hour"]
        end_hour = start_hour + 1 

        print(f"Processing data for {year}-{month:02d}-{day:02d} from {start_hour}:00 to {end_hour}:00...")
        
        ds_pl, ds_sfc = load_datasets(year, month, day, start_hour=0, end_hour=23)
        if ds_pl is not None and ds_sfc is not None:
            ds_pl_sliced, ds_sfc_sliced = slice_dataset_to_domain(ds_pl, ds_sfc, directions)

            for hour in range(start_hour, end_hour):  
                print(f"Processing hour {hour}:00...")

                ds_pl_time_sliced = ds_pl_sliced.isel(time=hour)
                ds_sfc_time_sliced = ds_sfc_sliced.isel(time=hour)

                date_str = f"{year}_{month:02d}_{day:02d}_{hour:02d}"
                save_dataset(ds_pl_time_sliced, f"pl_{date_str}.nc")
                save_dataset(ds_sfc_time_sliced, f"sfc_{date_str}.nc")

if __name__ == "__main__":
    main()