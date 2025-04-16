from data_processing import (
    load_datasets,
    load_local_datasets,
    slice_dataset_to_domain,
    get_point_data,
    add_time_dimension,
    save_to_csv)
from calculations import (
    get_pv,
    get_thickness,
    get_wnd,
    get_ivt,
    get_qvec,
    get_absolute_vorticity,
    get_geopotential_height,
    get_temperature,
    get_specific_humidity,
    get_thetae,
    get_temp_grad,
    get_fgen,
    get_rel_vort,
    get_tadv,
    get_total_deformation,
    get_shearing_deformation,
    get_stretching_deformation,
    get_ivt_grad)

import xarray as xr

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
        {"year": 2017, "month": 3, "start_day": 28, "start_hour": 18, "lat": 40, "lon": 140, "label": "noMFW"},
        {"year": 2020, "month": 11, "start_day": 14, "start_hour": 21, "lat": 45, "lon": 128, "label": "noMFW"},
        {"year": 2021, "month": 10, "start_day": 24, "start_hour": 2, "lat": 40, "lon": 135, "label": "noMFW"},
        {"year": 2023, "month": 7, "start_day": 23, "start_hour": 6, "lat": 45, "lon": 150, "label": "noMFW"},
        {"year": 2006, "month": 11, "start_day": 4, "start_hour": 7, "lat": 45, "lon": 135, "label": "MFW"},
        {"year": 2023, "month": 1, "start_day": 14, "start_hour": 19, "lat": 36, "lon": 153.5, "label": "MFW"},
        {"year": 2014, "month": 12, "start_day": 10, "start_hour": 21, "lat": 36, "lon": 138, "label": "MFW"},
        {"year": 2021, "month": 12, "start_day": 18, "start_hour": 13, "lat": 40, "lon": 139, "label": "MFW"},
        {"year": 2015, "month": 8, "start_day": 27, "start_hour": 23, "lat": 32, "lon": 140, "label": "MFW"},
        {"year": 2014, "month": 11, "start_day": 21, "start_hour": 8, "lat": 42.5, "lon": 140, "label": "noMFW"},
        {"year": 2012, "month": 11, "start_day": 19, "start_hour": 19, "lat": 41, "lon": 129, "label": "noMFW"},
        {"year": 2005, "month": 1, "start_day": 18, "start_hour": 11, "lat": 44, "lon": 138, "label": "noMFW"},
        {"year": 2010, "month": 1, "start_day": 20, "start_hour": 6, "lat": 38, "lon": 130, "label": "MFW"},
        {"year": 2018, "month": 2, "start_day": 13, "start_hour": 2, "lat": 40, "lon": 145, "label": "MFW"},
        {"year": 2017, "month": 12, "start_day": 28, "start_hour": 0, "lat": 30, "lon": 148, "label": "MFW"},
        {"year": 2017, "month": 11, "start_day": 19, "start_hour": 3, "lat": 38, "lon": 142.5, "label": "MFW"},
        {"year": 2016, "month": 11, "start_day": 14, "start_hour": 6, "lat": 42, "lon": 130, "label": "noMFW"},
        {"year": 2015, "month": 2, "start_day": 6, "start_hour": 0, "lat": 37.5, "lon": 130, "label": "noMFW"},
        {"year": 2015, "month": 3, "start_day": 25, "start_hour": 6, "lat": 35, "lon": 146, "label": "noMFW"},
        {"year": 2014, "month": 11, "start_day": 5, "start_hour": 21, "lat": 40, "lon": 136.5, "label": "MFW"},
        {"year": 2014, "month": 4, "start_day": 16, "start_hour": 14, "lat": 41, "lon": 148, "label": "noMFW"},
        {"year": 2010, "month": 2, "start_day": 4, "start_hour": 18, "lat": 35, "lon": 128, "label": "noMFW"},
        {"year": 2008, "month": 1, "start_day": 3, "start_hour": 10, "lat": 40, "lon": 130, "label": "noMFW"},
        {"year": 2008, "month": 2, "start_day": 23, "start_hour": 0, "lat": 35, "lon": 140, "label": "noMFW"},
        {"year": 2019, "month": 2, "start_day": 2, "start_hour": 1, "lat": 31.5, "lon": 127, "label": "noMFW"},
        {"year": 2019, "month": 1, "start_day": 17, "start_hour": 3, "lat": 35, "lon": 126, "label": "noMFW"},
        {"year": 2022, "month": 12, "start_day": 30, "start_hour": 20, "lat": 40, "lon": 127, "label": "MFW"},
        {"year": 2024, "month": 12, "start_day": 28, "start_hour": 18, "lat": 39, "lon": 132, "label": "MFW"},
        {"year": 2022, "month": 6, "start_day": 10, "start_hour": 21, "lat": 39, "lon": 135, "label": "MFW"},
        {"year": 2017, "month": 2, "start_day": 6, "start_hour": 2, "lat": 26, "lon": 156.5, "label": "MFW"},
        {"year": 2005, "month": 3, "start_day": 26, "start_hour": 9, "lat": 35, "lon": 138.5, "label": "MFW"}]

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

        if data_mode == "local":
            print(f"Processing local file for {start_hour}:00...")
            ds_pl_time_sliced = ds_pl_sliced  # already single timestep
            ds_sfc_time_sliced = ds_sfc_sliced

            # Process variables
            pv_300, pv_700, pv_850, pv_925, pv_1000 = get_pv(ds_pl_time_sliced, level=300), get_pv(ds_pl_time_sliced, level=700), get_pv(ds_pl_time_sliced, level=850), get_pv(ds_pl_time_sliced, level=925), get_pv(ds_pl_time_sliced, level=1000)
            wnd_300, wnd_500, wnd_850 = get_wnd(ds_pl_time_sliced, level=300), get_wnd(ds_pl_time_sliced, level=500), get_wnd(ds_pl_time_sliced, level=850)
            z_250, z_500, z_850, z_925, z_1000 = get_geopotential_height(ds_pl_time_sliced, level=250), get_geopotential_height(ds_pl_time_sliced, level=500), get_geopotential_height(ds_pl_time_sliced, level=850), get_geopotential_height(ds_pl_time_sliced, level=925), get_geopotential_height(ds_pl_time_sliced, level=1000)
            t_250, t_500, t_850, t_925, t_1000 = get_temperature(ds_pl_time_sliced, level=250), get_temperature(ds_pl_time_sliced, level=500), get_temperature(ds_pl_time_sliced, level=850), get_temperature(ds_pl_time_sliced, level=925), get_temperature(ds_pl_time_sliced, level=1000)
            q_850, q_925, q_1000 = get_specific_humidity(ds_pl_time_sliced, level=850), get_specific_humidity(ds_pl_time_sliced, level=925), get_specific_humidity(ds_pl_time_sliced, level=1000)
            ivt = get_ivt(ds_pl_time_sliced, g=g)
            ivt_grad = get_ivt_grad(ivt)
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
            ivt_grad = ivt_grad.rename("ivt_grad")
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
                ivt, ivt_grad, thickness_1000_500,
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
            ds_point_time = add_time_dimension(ds_point, year, month, day, start_hour)
            save_to_csv(ds_point_time, year, month, day, label)
        
        else: 
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
                ivt_grad = get_ivt_grad(ivt)
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
                ivt_grad = ivt_grad.rename("ivt_grad")
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
                    ivt, ivt_grad, thickness_1000_500,
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
                ds_point_time = add_time_dimension(ds_point, year, month, day, start_hour)
                save_to_csv(ds_point_time, year, month, day, label)

    print("Script is complete!")

if __name__ == '__main__':
    main()
