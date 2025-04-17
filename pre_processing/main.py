import xarray as xr
from data_processing import (
    load_datasets,
    load_local_datasets,
    slice_dataset_to_domain,
    process_variables,
    get_point_data,
    add_time_dimension,
    save_to_csv)

def main():
    data_mode = "local"  # "local" or "download"
    # Dictionary to store event data with year YYYY, month MM, start_day DD, start_hour HH, lat (N), lon (W), label (MFW or noMFW)
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

    # Loop over the events
    for event in events:
        year, month, day, start_hour = event["year"], event["month"], event["start_day"], event["start_hour"]
        lat, lon, label = event["lat"], event["lon"], event["label"]
        end_hour = start_hour + 1

        print(f"Processing data for {year}-{month:02d}-{day:02d} from {start_hour}:00 to {end_hour}:00...")

        # Load datasets
        if data_mode == "local":
            ds_pl, ds_sfc = load_local_datasets(year=year, month=month, day=day, hour=start_hour)
        else:
            ds_pl, ds_sfc = load_datasets(year=year, month=month, start_day=day, end_day=day, start_hour=0, end_hour=23)

        # Check if datasets are loaded successfully
        if ds_pl is None or ds_sfc is None:
            print(f"Skipping {year}-{month:02d}-{day:02d} due to missing datasets.")
            continue

        # Slice datasets to the specified domain
        ds_pl_sliced, ds_sfc_sliced = slice_dataset_to_domain(ds_pl=ds_pl, ds_sfc=ds_sfc, directions=directions)

        # Process variables
        ds_pl_time_sliced = ds_pl_sliced  # Assuming single timestep for local data
        final_ds = process_variables(ds_pl_time_sliced, g)

        # Extract point data and save to CSV
        ds_point = get_point_data(final_ds, lat, lon, buffer)
        ds_point_time = add_time_dimension(ds_point, year, month, day, start_hour)
        save_to_csv(ds_point_time, year, month, day, label)

    print("Script is complete!")

if __name__ == '__main__':
    main()