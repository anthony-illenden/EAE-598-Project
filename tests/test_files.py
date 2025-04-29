import os

# Directory 
data_dir = "C:\\Users\\Tony\\Documents\\GitHub\\EAE-598-Project\\data\\era5"

# List of events 
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

def test_file_existence(events, data_dir):
    """
    Test if the required files exist in the specified directory.
    
    Parameters
    ----------
    events : list of dict
        List of events with year, month, start_day, and start_hour.
    data_dir : str
        Directory where the files are located.
    
    Returns
    -------
    None
        Asserts if the files do not exist.

    """

    missing_files = []

    for event in events:
        year = event["year"]
        month = f"{event['month']:02d}"
        day = f"{event['start_day']:02d}"
        hour = f"{event['start_hour']:02d}"

        pl_file = f"pl_{year}_{month}_{day}_{hour}.nc"
        sfc_file = f"sfc_{year}_{month}_{day}_{hour}.nc"

        if not os.path.exists(os.path.join(data_dir, pl_file)):
            missing_files.append(pl_file)
        if not os.path.exists(os.path.join(data_dir, sfc_file)):
            missing_files.append(sfc_file)

    assert not missing_files, f"Missing files: {missing_files}"

if __name__ == "__main__":
    test_file_existence(events, data_dir)
    print("All files exist!")