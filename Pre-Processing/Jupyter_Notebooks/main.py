import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import gaussian_filter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import time as datetime

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
    temp_grad = np.sqrt(dT_dx**2 + dT_dy**2) * 1000 * 100  # units: K/100 km
    temp_grad_da = xr.DataArray(temp_grad, dims=['latitude', 'longitude'], coords={'latitude': var['latitude'], 'longitude': var['longitude']})
    temp_grad_da.name = f'{var_name}_grad'

    return temp_grad

def get_fgen(ds_pl, level):
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    t_sliced = ds_pl['T'].sel(level=level) # units: K
    theta = mpcalc.potential_temperature(level*units.hPa, t_sliced) # units: K
    fgen = mpcalc.frontogenesis(theta, u_sliced, v_sliced) * 1000*100*3600*3 # units: K per 100 km 3h

    return fgen

def get_point_data(final_ds, lat, lon, buffer):
    # Convert longitude from west to east
    lon_e = 360 - lon 
    ds_area_point = final_ds.sel(latitude=slice(lat + buffer, lat - buffer), longitude=slice(lon_e - buffer, lon_e + buffer))
    ds_area_point_median = ds_area_point.median(dim=['latitude', 'longitude'])
    return ds_area_point_median

def add_time_dimension(final_ds, year, month, day, start_hour):
    formatted_time = np.datetime64(pd.to_datetime(f"{year}-{month:02d}-{day:02d} {start_hour:02d}:00"))
    ds_final = final_ds.expand_dims(time=[formatted_time])
    
    return ds_final

def save_to_csv(ds_point, year, month, day):
    df = ds_point.to_dataframe().reset_index()
    output_file = f"ds_point_{year}-{month:02d}-{day:02d}_hour.csv"
    df.to_csv(output_file, index=False)

def main():
    year = 2019
    month = 2
    first_day = 13
    last_day = 13
    start_hour = 0
    end_hour = start_hour + 1
    level_1 = 1000
    level_2 = 500
    directions = {'North': 55, 
                'East': 250, 
                'South': 20, 
                'West': 200} # units: degrees North, degrees East
    g = 9.81 # units: m/s^2
    lat, lon = 39.5, 130
    buffer = 0.25 # units: degrees

    for day in range(first_day, last_day + 1):
        print(f"Processing data for {year}-{month:02d}-{day:02d}...")
        ds_pl, ds_sfc = load_datasets(year=year, month=month, start_day=day, end_day=day, start_hour=0, end_hour=23)
        ds_pl_sliced, ds_sfc_sliced = slice_dataset_to_domain(ds_pl=ds_pl, ds_sfc=ds_sfc, directions=directions)

        for i in range(start_hour, end_hour):
            print(f"Processing hour {i}...")
            ds_pl_time_sliced = ds_pl_sliced.isel(time=i)
            ds_sfc_time_sliced = ds_sfc_sliced.isel(time=i)

            pv_300, pv_925 = get_pv(ds_pl_time_sliced, level=300), get_pv(ds_pl_time_sliced, level=925)
            wnd_300, wnd_500, wnd_850 = get_wnd(ds_pl_time_sliced, level=300), get_wnd(ds_pl_time_sliced, level=500), get_wnd(ds_pl_time_sliced, level=850)
            z_250, z_500, z_850 = get_geopotential_height(ds_pl_time_sliced, level=250), get_geopotential_height(ds_pl_time_sliced, level=500), get_geopotential_height(ds_pl_time_sliced, level=850)
            t_250, t_500, t_850, t_925, t_1000 = get_temperature(ds_pl_time_sliced, level=250), get_temperature(ds_pl_time_sliced, level=500), get_temperature(ds_pl_time_sliced, level=850), get_temperature(ds_pl_time_sliced, level=925), get_temperature(ds_pl_time_sliced, level=1000)
            q_850, q_925, q_1000 = get_specific_humidity(ds_pl_time_sliced, level=850), get_specific_humidity(ds_pl_time_sliced, level=925), get_specific_humidity(ds_pl_time_sliced, level=1000)
            ivt = get_ivt(ds_pl_time_sliced, g=g)
            thickness_1000_500 = get_thickness(ds_pl_time_sliced, level1=1000, level2=500)
            qvec_div, qvec_magn = get_qvec(ds_pl_time_sliced, g=g)
            abs_vort = get_absolute_vorticity(ds_pl_time_sliced, level=500, g=g)
            thetae_925 = get_thetae(ds_pl_time_sliced, level=925)
            #grad_thetae_925 = get_temp_grad(thetae_925, var_name='thetae_925')
            fgen_925 = get_fgen(ds_pl_time_sliced, level=925)

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

            final_ds = xr.merge([
                pv_300, pv_925, wnd_300, wnd_500, wnd_850,
                z_250, z_500, z_850, t_250, t_500, t_850, t_925, t_1000,
                q_850, q_925, q_1000, ivt, thickness_1000_500,
                qvec_div, qvec_magn, abs_vort, thetae_925, fgen_925
            ], compat='override')         

            print(final_ds)

            ds_point = get_point_data(final_ds, lat, lon, buffer)

            print(ds_point)

            ds_point_time = add_time_dimension(ds_point, year, month, day, start_hour)

            save_to_csv(ds_point_time, year, month, day)

        if ds_pl is None or ds_sfc is None:
            print(f"Skipping {year}-{month:02d}-{day:02d} due to missing datasets.")
            continue
    
    print("Script is complete!")

if __name__ == '__main__':
    main()