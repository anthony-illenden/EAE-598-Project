import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

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

def get_ivt_grad(ivt):
    divt_dx, divt_dy = mpcalc.geospatial_gradient(ivt)  # units: K/m/s / m
    ivt_grad = np.sqrt(divt_dx**2 + divt_dy**2) * 1000  * units.meters / units.kilometers # units: K/m/s / km
    ivt_grad_da = xr.DataArray(ivt_grad, dims=['latitude', 'longitude'], coords={'latitude': ivt['latitude'], 'longitude': ivt['longitude']})

    return ivt_grad_da