import os
import pandas as pd
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from siphon.catalog import TDSCatalog

# Code from Tony's EAE 595 Project (https://github.com/anthony-illenden/EAE-593-Project)
def load_datasets(year, month, start_day, start_hour=0, end_day=None, end_hour=23):
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
     ds_pl_sliced = ds_pl.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
     ds_sfc_sliced = ds_sfc.sel(latitude=slice(directions['North'], directions['South']), longitude=slice(directions['West'], directions['East']))
     return ds_pl_sliced, ds_sfc_sliced

def get_pv(ds_pl, level): 
    pv = ds_pl['PV'].sel(level=level) * 1e-6  # Convert PVU to PV (1 PVU = 1e-6 K m^2/kg/s)
    return pv

def get_thickness(ds_pl, level1, level2):
    z1 = ds_pl['Z'].sel(level=level1)
    z2 = ds_pl['Z'].sel(level=level2)
    thickness = z2 - z1
    return thickness

def get_wnd(ds_pl, level):
    u = ds_pl['U'].sel(level=level)
    v = ds_pl['V'].sel(level=level)
    wnd_speed = np.sqrt(u**2 + v**2)
    return wnd_speed

def get_ivt(ds_pl, g): 
    u_sliced = ds_pl['U'].sel(level=slice(500, 1000)) # units: m/s
    v_sliced = ds_pl['V'].sel(level=slice(500, 1000)) # units: m/s
    q_sliced = ds_pl['Q'].sel(level=slice(500, 1000)) # units: kg/kg

    # Flip the order of the pressure levels and convert them to Pa from hPa
    pressure_levels = u_sliced['level'][::-1] * 100 * units.Pa # units: Pa

    lats, lons = ds_pl['latitude'].metpy.unit_array, ds_pl['longitude'].metpy.unit_array

    # Calculate the integrated vapor transport (IVT) using the u- and v-wind components and the specific humidity
    u_ivt = -1 / g * np.trapz(u_sliced * q_sliced, pressure_levels, axis=0)
    v_ivt = -1 / g * np.trapz(v_sliced * q_sliced, pressure_levels, axis=0)

    # Calculate the IVT magnitude
    ivt = np.sqrt(u_ivt**2 + v_ivt**2)

    # Create an xarray DataArray for the IVT
    ivt_da = xr.DataArray(ivt, dims=['latitude', 'longitude'], coords={'latitude': u_sliced['latitude'], 'longitude': u_sliced['longitude']})

    return ivt_da

def get_qvec(ds_pl, g):
    t_sliced = ds_pl['T'].sel(level=slice(500, 700)) # units: K
    u_sliced = ds_pl['U'].sel(level=slice(500, 700)) # units: m/s
    v_sliced = ds_pl['V'].sel(level=slice(500, 700)) # units: m/s
    z_sliced = ds_pl['Z'].sel(level=slice(500, 700)) / g # units: m
    plevels_sliced = t_sliced['level'] # units: hPa

    # Smoothing to focus on the synoptic scale
    n_reps = 80
    u_sliced_s = mpcalc.smooth_n_point(u_sliced, 9, n_reps)
    v_sliced_s = mpcalc.smooth_n_point(v_sliced, 9, n_reps)
    z_sliced_s = mpcalc.smooth_n_point(z_sliced, 9, n_reps)
    t_sliced_s = mpcalc.smooth_n_point(t_sliced, 9, n_reps)

    lons, lats = t_sliced['longitude'], t_sliced['latitude']

    # Grid spacing
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    # Calculate some things
    u_qvec, v_qvec = mpcalc.q_vector(u=u_sliced_s, v=v_sliced_s, temperature=t_sliced_s, pressure=plevels_sliced) # units: m^2/kg*s

    # Calculate the average q-vector components between 700-500 hPa
    u_qvec_layer_avg = u_qvec.mean(dim='level') # units: m^2/kg*s
    v_qvec_layer_avg = v_qvec.mean(dim='level') # units: m^2/kg*s

    # Calculate the q-vec div
    qvec_div_layer_avg = -2 * mpcalc.divergence(u_qvec_layer_avg, v_qvec_layer_avg) * 1e18 # units: m/kg*s

    # Compute the q-vec magnitude
    qvec_magnitude = np.sqrt(u_qvec_layer_avg**2 + v_qvec_layer_avg**2) * 1e13

    return qvec_div_layer_avg, qvec_magnitude

def get_absolute_vorticity(ds_pl, level, g):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    z_sliced = ds_pl['Z'].sel(level=level) / g # units: m

    # Get lats and lons and grid spacing
    lat, lon = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

    # Calculate absolute vorticity
    absolute_vorticity = mpcalc.absolute_vorticity(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s
    return absolute_vorticity

def get_geopotential_height(ds_pl, level):
    z = ds_pl['Z'].sel(level=level)  # Geopotential height in meters
    return z

def get_temperature(ds_pl, level):
    t = ds_pl['T'].sel(level=level)  # Temperature in Kelvin
    return t

def get_specific_humidity(ds_pl, level):
    q = ds_pl['Q'].sel(level=level)  # Specific humidity in kg/kg
    return q

def get_thetae(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    q_sliced = ds_pl['Q'].sel(level=level) # units: kg/kg
    t_sliced = ds_pl['T'].sel(level=level) # units: K
    pv_sliced = ds_pl['PV'].sel(level=level) * 1e6 # units: PVU

    # Convert them to Pa from hPa
    pressure_levels = u_sliced.level.metpy.convert_units('hPa')  # units: hPa

    # Calculate dewpoint and theta-e
    td = mpcalc.dewpoint_from_specific_humidity(pressure_levels, t_sliced, q_sliced)
    theta_e = mpcalc.equivalent_potential_temperature(pressure_levels, t_sliced, td) # units: K

    return theta_e 

def get_temp_grad(var, var_name):
    dT_dx, dT_dy = mpcalc.geospatial_gradient(var)  # units: K/m
    temp_grad = np.sqrt(dT_dx**2 + dT_dy**2) * 1000 * 100  * units.meters / units.kilometers # units: K/100 km
    temp_grad_da = xr.DataArray(temp_grad, dims=['latitude', 'longitude'], coords={'latitude': var['latitude'], 'longitude': var['longitude']})

    return temp_grad_da

def get_fgen(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    t_sliced = ds_pl['T'].sel(level=level) # units: K
    theta = mpcalc.potential_temperature(level*units.hPa, t_sliced) # units: K
    fgen = mpcalc.frontogenesis(theta, u_sliced, v_sliced) * 1000*100*3600*3 # units: K per 100 km 3h

    return fgen

def get_rel_vort(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    rel_vort = mpcalc.vorticity(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return rel_vort

def get_tadv(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    t_sliced = ds_pl['T'].sel(level=level) # units: K
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m

    tadv = mpcalc.advection(t_sliced, u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 3600 * units.seconds / units.hour # units: k/hr

    return tadv

def get_total_deformation(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    total_deformation = mpcalc.total_deformation(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return total_deformation

def get_shearing_deformation(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    shearing_deformation = mpcalc.shearing_deformation(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return shearing_deformation

def get_stretching_deformation(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    stretching_deformation = mpcalc.stretching_deformation(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return stretching_deformation

def get_point_data(final_ds, lat, lon, buffer):
    # Convert longitude from west to east
    lon_e = 360 - lon 
    ds_area_point = final_ds.sel(latitude=slice(lat + buffer, lat - buffer), longitude=slice(lon_e - buffer, lon_e + buffer))
    ds_area_point_mean = ds_area_point.mean(dim=['latitude', 'longitude'])
    return ds_area_point_mean

def add_time_dimension(final_ds, year, month, day, start_hour):
    formatted_time = np.datetime64(pd.to_datetime(f"{year}-{month:02d}-{day:02d} {start_hour:02d}:00"))
    ds_final = final_ds.expand_dims(time=[formatted_time])
    
    return ds_final

def save_to_csv(ds_point, year, month, day, label, output_file="all_events.csv"):
    df = ds_point.to_dataframe().reset_index()
    df['label'] = label
    df['year'] = year
    df['month'] = month
    df['day'] = day

    # Check if the file exists to determine if headers should be written
    file_exists = os.path.isfile(output_file)

    # Append data to a single CSV
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)

def main():
    data_mode = "local" # "local" or "download"
    events = [
        {"year": 2017, "month": 2, "start_day": 20, "start_hour": 3, "lat": 39.5, "lon": 130, "label": "MFW"},
        {"year": 2019, "month": 2, "start_day": 13, "start_hour": 17, "lat": 34, "lon": 131, "label": "MFW"},
        {"year": 2015, "month": 11, "start_day": 19, "start_hour": 2, "lat": 45.5, "lon": 131, "label": "MFW"},
        {"year": 2016, "month": 1, "start_day": 28, "start_hour": 17, "lat": 41, "lon": 141, "label": "MFW"},
        {"year": 2014, "month": 2, "start_day": 7, "start_hour": 13, "lat": 40, "lon": 134, "label": "MFW"},
        {"year": 2012, "month": 10, "start_day": 18, "start_hour": 9, "lat": 45, "lon": 141, "label": "MFW"},
        {"year": 2017, "month": 11, "start_day": 15, "start_hour": 6, "lat": 35, "lon": 135, "label": "MFW"},
        {"year": 2017, "month": 3, "start_day": 17, "start_hour": 6, "lat": 39, "lon": 138, "label": "MFW"},
        {"year": 2011, "month": 11, "start_day": 22, "start_hour": 14, "lat": 40, "lon": 135, "label": "MFW"},
        {"year": 2023, "month": 1, "start_day": 9, "start_hour": 12, "lat": 31, "lon": 130, "label": "MFW"},
        {"year": 2010, "month": 3, "start_day": 12, "start_hour": 0, "lat": 40, "lon": 132, "label": "noMFW"},
        {"year": 2014, "month": 11, "start_day": 3, "start_hour": 10, "lat": 39, "lon": 145, "label": "noMFW"},
        {"year": 2018, "month": 1, "start_day": 23, "start_hour": 9, "lat": 40, "lon": 135, "label": "noMFW"},
        {"year": 2018, "month": 11, "start_day": 22, "start_hour": 11, "lat": 40, "lon": 132, "label": "noMFW"},
        {"year": 2010, "month": 2, "start_day": 26, "start_hour": 3, "lat": 35, "lon": 131, "label": "noMFW"},
        {"year": 2012, "month": 3, "start_day": 9, "start_hour": 4, "lat": 44, "lon": 131, "label": "noMFW"},
        {"year": 2013, "month": 11, "start_day": 11, "start_hour": 17, "lat": 36, "lon": 132, "label": "noMFW"},
        {"year": 2016, "month": 1, "start_day": 17, "start_hour": 9, "lat": 35, "lon": 130, "label": "noMFW"},
        {"year": 2016, "month": 11, "start_day": 7, "start_hour": 11, "lat": 36, "lon": 138, "label": "noMFW"},
        {"year": 2017, "month": 3, "start_day": 28, "start_hour": 18, "lat": 40, "lon": 140, "label": "noMFW"}
    ]

    directions = {'North': 55, 'East': 250, 'South': 20, 'West': 200}  # units: degrees North, degrees East
    g = 9.81  # units: m/s^2
    buffer = 0.25  # units: degrees

    for event in events:
        year, month, day, start_hour = event["year"], event["month"], event["start_day"], event["start_hour"]
        lat, lon, label = event["lat"], event["lon"], event["label"]
        end_hour = start_hour + 1

        print(f"Processing data for {year}-{month:02d}-{day:02d} from {start_hour}:00 to {end_hour}:00...")

        if data_mode == "local":
            ds_pl, ds_sfc = load_local_datasets(year=year, month=month, day=day, hour=start_hour)
        else:
            ds_pl, ds_sfc = load_datasets(year=year, month=month, start_day=day, end_day=day, start_hour=0, end_hour=23)

        if ds_pl is None or ds_sfc is None:
            print(f"Skipping {year}-{month:02d}-{day:02d} due to missing datasets.")
            continue

        ds_pl_sliced, ds_sfc_sliced = slice_dataset_to_domain(ds_pl=ds_pl, ds_sfc=ds_sfc, directions=directions)

        for hour in range(start_hour, end_hour):
            print(f"Processing hour {hour}:00...")
            ds_pl_time_sliced = ds_pl_sliced.isel(time=hour)
            ds_sfc_time_sliced = ds_sfc_sliced.isel(time=hour)

            # Process variables
            pv_300, pv_700, pv_850, pv_925, pv_1000 = get_pv(ds_pl_time_sliced, level=300), get_pv(ds_pl_time_sliced, level=700), get_pv(ds_pl_time_sliced, level=850), get_pv(ds_pl_time_sliced, level=925), get_pv(ds_pl_time_sliced, level=1000)
            wnd_300, wnd_500, wnd_850 = get_wnd(ds_pl_time_sliced, level=300), get_wnd(ds_pl_time_sliced, level=500), get_wnd(ds_pl_time_sliced, level=850)
            z_250, z_500, z_850, z_925, z_1000 = get_geopotential_height(ds_pl_time_sliced, level=250), get_geopotential_height(ds_pl_time_sliced, level=500), get_geopotential_height(ds_pl_time_sliced, level=850), get_geopotential_height(ds_pl_time_sliced, level=925), get_geopotential_height(ds_pl_time_sliced, level=1000)
            t_250, t_500, t_850, t_925, t_1000 = get_temperature(ds_pl_time_sliced, level=250), get_temperature(ds_pl_time_sliced, level=500), get_temperature(ds_pl_time_sliced, level=850), get_temperature(ds_pl_time_sliced, level=925), get_temperature(ds_pl_time_sliced, level=1000)
            q_850, q_925, q_1000 = get_specific_humidity(ds_pl_time_sliced, level=850), get_specific_humidity(ds_pl_time_sliced, level=925), get_specific_humidity(ds_pl_time_sliced, level=1000)
            ivt = get_ivt(ds_pl_time_sliced, g=g)
            thickness_1000_500 = get_thickness(ds_pl_time_sliced, level1=1000, level2=500)
            qvec_div, qvec_magn = get_qvec(ds_pl_time_sliced, g=g)
            abs_vort = get_absolute_vorticity(ds_pl_time_sliced, level=500, g=g)
            thetae_850, thetae_925, thetae_1000 = get_thetae(ds_pl_time_sliced, level=850), get_thetae(ds_pl_time_sliced, level=925), get_thetae(ds_pl_time_sliced, level=1000)
            fgen_700, fgen_850, fgen_925, fgen_1000 = get_fgen(ds_pl_time_sliced, level=700), get_fgen(ds_pl_time_sliced, level=850), get_fgen(ds_pl_time_sliced, level=925), get_fgen(ds_pl_time_sliced, level=1000)
            tadv_500, tadv_850, tadv_925, tadv_1000 = get_tadv(ds_pl_time_sliced, level=500), get_tadv(ds_pl_time_sliced, level=850), get_tadv(ds_pl_time_sliced, level=925), get_tadv(ds_pl_time_sliced, level=1000)
            rel_vort_500, rel_vort_850, rel_vort_925, rel_vort_1000 = get_rel_vort(ds_pl_time_sliced, level=500), get_rel_vort(ds_pl_time_sliced, level=850), get_rel_vort(ds_pl_time_sliced, level=925), get_rel_vort(ds_pl_time_sliced, level=1000)
            total_deformation_500, total_deformation_850, total_deformation_925, total_deformation_1000 = get_total_deformation(ds_pl_time_sliced, level=500), get_total_deformation(ds_pl_time_sliced, level=850), get_total_deformation(ds_pl_time_sliced, level=925), get_total_deformation(ds_pl_time_sliced, level=1000)
            shearing_deformation_500, shearing_deformation_850, shearing_deformation_925, shearing_deformation_1000 = get_shearing_deformation(ds_pl_time_sliced, level=500), get_shearing_deformation(ds_pl_time_sliced, level=850), get_shearing_deformation(ds_pl_time_sliced, level=925), get_shearing_deformation(ds_pl_time_sliced, level=1000)
            stretching_deformation_500, stretching_deformation_850, stretching_deformation_925, stretching_deformation_1000 = get_stretching_deformation(ds_pl_time_sliced, level=500), get_stretching_deformation(ds_pl_time_sliced, level=850), get_stretching_deformation(ds_pl_time_sliced, level=925), get_stretching_deformation(ds_pl_time_sliced, level=1000)
            thetae_grad_850, thetae_grad_925, thetae_grad_1000 = get_temp_grad(thetae_850, 'thetae_850'), get_temp_grad(thetae_925, 'thetae_925'), get_temp_grad(thetae_1000, 'thetae_1000')
            t_grad_850, t_grad_925, t_grad_1000 = get_temp_grad(t_850, 't_850'), get_temp_grad(t_925, 't_925'), get_temp_grad(t_1000, 't_1000')

            # Rename variables
            pv_300 = pv_300.rename("pv_300")
            pv_925 = pv_925.rename("pv_925")
            wnd_300 = wnd_300.rename("wnd_300")
            wnd_500 = wnd_500.rename("wnd_500")
            wnd_850 = wnd_850.rename("wnd_850")
            z_250 = z_250.rename("z_250")
            z_500 = z_500.rename("z_500")
            z_850 = z_850.rename("z_850")
            t_250 = t_250.rename("t_250")
            t_500 = t_500.rename("t_500")
            t_850 = t_850.rename("t_850")
            t_925 = t_925.rename("t_925")
            t_1000 = t_1000.rename("t_1000")
            q_850 = q_850.rename("q_850")
            q_925 = q_925.rename("q_925")
            q_1000 = q_1000.rename("q_1000")
            ivt = ivt.rename("ivt")
            thickness_1000_500 = thickness_1000_500.rename("thickness_1000_500")
            qvec_div = qvec_div.rename("qvec_div")
            qvec_magn = qvec_magn.rename("qvec_magn")
            abs_vort = abs_vort.rename("abs_vort")
            thetae_925 = thetae_925.rename("thetae_925")
            fgen_925 = fgen_925.rename("fgen_925")
            pv_700 = pv_700.rename("pv_700")
            pv_850 = pv_850.rename("pv_850")
            pv_1000 = pv_1000.rename("pv_1000")
            z_925 = z_925.rename("z_925")
            z_1000 = z_1000.rename("z_1000")
            thetae_850 = thetae_850.rename("thetae_850")
            thetae_1000 = thetae_1000.rename("thetae_1000")
            fgen_700 = fgen_700.rename("fgen_700")
            fgen_850 = fgen_850.rename("fgen_850")
            fgen_1000 = fgen_1000.rename("fgen_1000")
            tadv_500 = tadv_500.rename("tadv_500")
            tadv_850 = tadv_850.rename("tadv_850")
            tadv_925 = tadv_925.rename("tadv_925")
            tadv_1000 = tadv_1000.rename("tadv_1000")
            rel_vort_500 = rel_vort_500.rename("rel_vort_500")
            rel_vort_850 = rel_vort_850.rename("rel_vort_850")
            rel_vort_925 = rel_vort_925.rename("rel_vort_925")
            rel_vort_1000 = rel_vort_1000.rename("rel_vort_1000")
            total_deformation_500 = total_deformation_500.rename("total_deformation_500")
            total_deformation_850 = total_deformation_850.rename("total_deformation_850")
            total_deformation_925 = total_deformation_925.rename("total_deformation_925")
            total_deformation_1000 = total_deformation_1000.rename("total_deformation_1000")
            shearing_deformation_500 = shearing_deformation_500.rename("shearing_deformation_500")
            shearing_deformation_850 = shearing_deformation_850.rename("shearing_deformation_850")
            shearing_deformation_925 = shearing_deformation_925.rename("shearing_deformation_925")
            shearing_deformation_1000 = shearing_deformation_1000.rename("shearing_deformation_1000")
            stretching_deformation_500 = stretching_deformation_500.rename("stretching_deformation_500")
            stretching_deformation_850 = stretching_deformation_850.rename("stretching_deformation_850")
            stretching_deformation_925 = stretching_deformation_925.rename("stretching_deformation_925")
            stretching_deformation_1000 = stretching_deformation_1000.rename("stretching_deformation_1000")
            thetae_grad_850 = thetae_grad_850.rename("thetae_grad_850")
            thetae_grad_925 = thetae_grad_925.rename("thetae_grad_925")
            thetae_grad_1000 = thetae_grad_1000.rename("thetae_grad_1000")
            t_grad_850 = t_grad_850.rename("t_grad_850")
            t_grad_925 = t_grad_925.rename("t_grad_925")
            t_grad_1000 = t_grad_1000.rename("t_grad_1000")

            # Merge datasets
            final_ds = xr.merge([
                pv_300, pv_700, pv_850, pv_925, pv_1000,
                wnd_300, wnd_500, wnd_850,
                z_250, z_500, z_850, z_925, z_1000,
                t_250, t_500, t_850, t_925, t_1000,
                q_850, q_925, q_1000,
                ivt, thickness_1000_500,
                qvec_div, qvec_magn, abs_vort,
                thetae_850, thetae_925, thetae_1000,
                fgen_700, fgen_850, fgen_925, fgen_1000,
                tadv_500, tadv_850, tadv_925, tadv_1000,
                rel_vort_500, rel_vort_850, rel_vort_925, rel_vort_1000,
                total_deformation_500, total_deformation_850, total_deformation_925, total_deformation_1000,
                shearing_deformation_500, shearing_deformation_850, shearing_deformation_925, shearing_deformation_1000,
                stretching_deformation_500, stretching_deformation_850, stretching_deformation_925, stretching_deformation_1000,
                thetae_grad_850, thetae_grad_925, thetae_grad_1000,
                t_grad_850, t_grad_925, t_grad_1000
            ], compat='override')

            # Extract point data and save to a single CSV
            ds_point = get_point_data(final_ds, lat, lon, buffer)
            ds_point_time = add_time_dimension(ds_point, year, month, day, hour)
            save_to_csv(ds_point_time, year, month, day, label, output_file="all_events.csv")

    print("Script is complete!")

if __name__ == '__main__':
    main()
