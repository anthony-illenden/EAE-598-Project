import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

def get_pv(ds_pl, level):
    """
    Extract the ERA5 PV variable.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The pressure level (hPa) to extract the PV variable from.
    
    Returns
    -------
    xarray.DataArray
        The PV variable at the specified pressure level.

    """
    pv = ds_pl['PV'].sel(level=level) * 1e-6  # Convert PVU to PV (1 PVU = 1e-6 K m^2/kg/s)
    return pv

def get_thickness(ds_pl, level1, level2):
    """
    Calculate the thickness of a layer between two pressure levels.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level1 : int
        The lower pressure level (hPa).
    level2 : int
        The upper pressure level (hPa).
    
    Returns 
    -------
    xarray.DataArray
        The thickness of the layer between the two pressure levels.

    """
    z1 = ds_pl['Z'].sel(level=level1)
    z2 = ds_pl['Z'].sel(level=level2)
    thickness = z2 - z1
    return thickness

def get_wnd(ds_pl, level):
    """
    Calculate the wind speed at a specific pressure level.
    
    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The pressure level (hPa) to calculate the wind speed from.
    
    Returns 
    -------
    xarray.DataArray
        The wind speed at the specified pressure level.

    """
    u = ds_pl['U'].sel(level=level)
    v = ds_pl['V'].sel(level=level)
    wnd_speed = np.sqrt(u**2 + v**2)
    return wnd_speed

def get_ivt(ds_pl, g): 
    """
    Calculate the Integrated Vapor Transport (IVT) using the u- and v-wind components and specific humidity.
    
    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    g : float
        The acceleration due to gravity (m/s^2).

    Returns
    -------
    xarray.DataArray
        The IVT magnitude.

    """
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
    """
    Calculate the Q-vector divergence and magnitude.
    
    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    g : float
        The acceleration due to gravity (m/s^2).

    Returns
    -------
    xarray.DataArray
        The Q-vector divergence and magnitude.

    """
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
    """
    Calculate the absolute vorticity at a specific pressure level.
    
    Parameters 
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate the absolute vorticity.
    g : float
        The acceleration due to gravity (m/s^2).

    Returns 
    -------
    xarray.DataArray
        The absolute vorticity at the specified pressure level.
    
    """
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
    """
    Calculate the geopotential height at a specific pressure level.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate the geopotential height.
    
    Returns
    -------
    xarray.DataArray
        The geopotential height at the specified pressure level.

    """
    z = ds_pl['Z'].sel(level=level)  # Geopotential height in meters
    return z

def get_temperature(ds_pl, level):
    """
    Extract the temperature variable at a specific pressure level.
    
    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to extract the temperature variable.

    Returns 
    -------
    xarray.DataArray
        The temperature variable at the specified pressure level.

    """
    t = ds_pl['T'].sel(level=level)  # Temperature in Kelvin
    return t

def get_specific_humidity(ds_pl, level):
    """
    Extract the specific humidity variable at a specific pressure level.
    
    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to extract the specific humidity variable.

    Returns
    -------
    xarray.DataArray
        The specific humidity variable at the specified pressure level.
    
    """
    q = ds_pl['Q'].sel(level=level)  # Specific humidity in kg/kg
    return q

def get_thetae(ds_pl, level):
    """
    Calculate the equivalent potential temperature (theta-e) at a specific pressure level.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate theta-e.
    
    Returns 
    -------
    xarray.DataArray
        The equivalent potential temperature (theta-e) at the specified pressure level.
    
    """
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
    """
    Calculate the temperature gradient of any given temperature variable.
    
    Parameters
    ----------
    var : xarray.DataArray
        The temperature variable to calculate the gradient for.
    var_name : str
        The name of the temperature variable.
    
    Returns 
    -------
    xarray.DataArray
        The temperature gradient of the specified variable.
    
    """
    dT_dx, dT_dy = mpcalc.geospatial_gradient(var)  # units: K/m
    temp_grad = np.sqrt(dT_dx**2 + dT_dy**2) * 1000 * 100  * units.meters / units.kilometers # units: K/100 km
    temp_grad_da = xr.DataArray(temp_grad, dims=['latitude', 'longitude'], coords={'latitude': var['latitude'], 'longitude': var['longitude']})

    return temp_grad_da

def get_fgen(ds_pl, level):
    """
    Calculate the frontogenesis at a specific pressure level.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate frontogenesis.
    
    Returns 
    -------
    xarray.DataArray
        The frontogenesis at the specified pressure level.
    
    """
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    t_sliced = ds_pl['T'].sel(level=level) # units: K
    theta = mpcalc.potential_temperature(level*units.hPa, t_sliced) # units: K
    fgen = mpcalc.frontogenesis(theta, u_sliced, v_sliced) * 1000*100*3600*3 # units: K per 100 km 3h

    return fgen

def get_rel_vort(ds_pl, level):
    """
    Calculate the relative vorticity at a specific pressure level.
    
    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate relative vorticity.
    
    Returns 
    -------
    xarray.DataArray
        The relative vorticity at the specified pressure level.
    
    """
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    rel_vort = mpcalc.vorticity(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return rel_vort

def get_tadv(ds_pl, level):
    """
    Calculate the temperature advection at a specific pressure level.

    Parameters 
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate temperature advection.
    
    Returns
    -------
    xarray.DataArray
        The temperature advection at the specified pressure level.

    """
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    t_sliced = ds_pl['T'].sel(level=level) # units: K
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m

    tadv = mpcalc.advection(t_sliced, u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 3600 * units.seconds / units.hour # units: k/hr

    return tadv

def get_total_deformation(ds_pl, level):
    """
    Calculate the total deformation at a specific pressure level.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate total deformation.

    Returns 
    -------
    xarray.DataArray
        The total deformation at the specified pressure level.

    """
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    total_deformation = mpcalc.total_deformation(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return total_deformation

def get_shearing_deformation(ds_pl, level):
    """
    Calculate the shearing deformation at a specific pressure level.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate shearing deformation.
    
    Returns
    -------
    xarray.DataArray
        The shearing deformation at the specified pressure level.

    """
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    shearing_deformation = mpcalc.shearing_deformation(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return shearing_deformation

def get_stretching_deformation(ds_pl, level):
    """
    Calculate the stretching deformation at a specific pressure level.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        The pressure level dataset.
    level : int
        The specific pressure level (hPa) to calculate stretching deformation.
    
    Returns
    -------
    xarray.DataArray
        The stretching deformation at the specified pressure level.

    """
    u_sliced = ds_pl['U'].sel(level=level) # units: m/s
    v_sliced = ds_pl['V'].sel(level=level) # units: m/s
    lats, lons = u_sliced['latitude'], u_sliced['longitude']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m
    stretching_deformation = mpcalc.stretching_deformation(u=u_sliced, v=v_sliced, dx=dx, dy=dy) * 1e5 # units: 1/s 

    return stretching_deformation

def get_ivt_grad(ivt):
    """
    Calculate the gradient of the Integrated Vapor Transport (IVT).

    Parameters 
    ----------
    ivt : xarray.DataArray
        The IVT variable to calculate the gradient for.
    
    Returns 
    -------
    xarray.DataArray
        The gradient of the IVT variable.

    """
    # Calculate the IVT gradient
    divt_dx, divt_dy = mpcalc.geospatial_gradient(ivt)  # units: K/m/s / m
    ivt_grad = np.sqrt(divt_dx**2 + divt_dy**2) * 1000  * units.meters / units.kilometers # units: K/m/s / km
    ivt_grad_da = xr.DataArray(ivt_grad, dims=['latitude', 'longitude'], coords={'latitude': ivt['latitude'], 'longitude': ivt['longitude']})

    return ivt_grad_da