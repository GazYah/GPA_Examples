import os
import glob
import re
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, Optional, Set

def extract_date(filename: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extracts the year, month, and day from a filename containing a date in the format YYYYMMDDT.

    Args:
        filename (str): The filename to extract the date from.

    Returns:
        Tuple[Optional[int], Optional[int], Optional[int]]: A tuple containing (year, month, day) as integers,
        or (None, None, None) if the date is not found.
    """
    match = re.search(r'(\d{8})T', filename)
    if match:
        date_str = match.group(1)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        return year, month, day
    return None, None, None

def combine_inca_files(
    input_folder: str,
    output_file: str,
    static_vars: Set[str] = {'lat', 'lon', 'x', 'y', 'lambert_conformal_conic', 'time'}
) -> None:
    """
    Combines multiple INCA NetCDF files by averaging variables for each (month, day) across all years.
    Static variables are copied from the first file. The result is written to a new NetCDF file.
    For example, two years worth of data will be averaged for every day of the year, resulting in 365 files.

    Args:
        input_folder (str): Path to the folder containing input NetCDF files.
        output_file (str): Path to the output NetCDF file.
        static_vars (Set[str], optional): Set of variable names to treat as static (copied, not averaged).
            Defaults to {'lat', 'lon', 'x', 'y', 'lambert_conformal_conic', 'time'}.

    Returns:
        None
    """
    # Gather all files
    nc_files = sorted(glob.glob(os.path.join(input_folder, "*.nc")))
    if not nc_files:
        print("No NetCDF files found in the input folder.")
        return

    # Read static variables from the first file
    with Dataset(nc_files[0], 'r') as ds:
        static_data = {var: ds.variables[var][:] for var in static_vars if var in ds.variables}
        static_attrs = {var: ds.variables[var].__dict__ for var in static_vars if var in ds.variables}
        dims = {dim: len(ds.dimensions[dim]) for dim in ds.dimensions}
        global_attrs = ds.__dict__

    # Group arrays by (var, month, day) across all years
    day_grouped = defaultdict(list)
    for f in tqdm(nc_files, desc="Grouping by (month, day)"):
        year, month, day = extract_date(os.path.basename(f))
        if year is None:
            continue
        with Dataset(f, 'r') as ds:
            for var in ds.variables:
                if var in static_vars:
                    continue
                arr = ds.variables[var][:]
                if arr.ndim == 3:
                    arr = arr[0]
                key = (var, month, day)
                day_grouped[key].append(arr)

    # Average for each (var, month, day)
    averaged_data = {}
    for (var, month, day), arrs in tqdm(day_grouped.items(), desc="Averaging days"):
        arrays = np.stack(arrs)
        avg = np.mean(arrays, axis=0)
        name = f"{var}_{month:02d}_{day:02d}"
        averaged_data[name] = avg

    # Write to new NetCDF
    with Dataset(output_file, 'w') as out_nc:
        # Create dimensions
        for dim, size in dims.items():
            out_nc.createDimension(dim, size)
        # Write static variables
        for var in static_data:
            shape = static_data[var].shape
            if var == 'time':
                dims_tuple = ('time',)
            elif var == 'y':
                dims_tuple = ('y',)
            elif var == 'x':
                dims_tuple = ('x',)
            elif var in ('lat', 'lon'):
                if len(shape) == 2:
                    dims_tuple = ('y', 'x')
                else:
                    dims_tuple = (var,)
            elif var == 'lambert_conformal_conic':
                dims_tuple = ()
            else:
                dims_tuple = ('y', 'x')
            attrs = static_attrs[var].copy()
            fill_value = attrs.pop('_FillValue', None)
            if fill_value is not None:
                v = out_nc.createVariable(var, static_data[var].dtype, dims_tuple, fill_value=fill_value)
            else:
                v = out_nc.createVariable(var, static_data[var].dtype, dims_tuple)
            v[:] = static_data[var]
            for attr, val in attrs.items():
                setattr(v, attr, val)
        # Write averaged variables
        for name, arr in averaged_data.items():
            v = out_nc.createVariable(name, arr.dtype, ('y', 'x'))
            v[:, :] = arr
        # Copy global attributes
        for attr, val in global_attrs.items():
            if attr == "name":
                continue
            out_nc.setncattr(attr, val)

    print(f"Day-of-year averaged NetCDF written to {output_file}")