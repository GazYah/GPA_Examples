import xarray as xr
import numpy as np
from tqdm import tqdm

def trim_netcdf_by_nan(
    input_file: str,
    output_file: str,
    mask_var: str = "cat_1"
) -> None:
    """
    Trims all variables in a NetCDF file by masking out locations where `mask_var` is NaN.

    Args:
        input_file (str): Path to the input NetCDF file.
        output_file (str): Path to save the trimmed NetCDF file.
        mask_var (str, optional): Variable to use as the mask. Defaults to "cat_1".

    Returns:
        None
    """
    ds = xr.open_dataset(input_file)
    mask = ds[mask_var].isnull()
    for var in tqdm(ds.data_vars, desc="Trimming variables"):
        ds[var] = ds[var].where(~mask)
    ds.to_netcdf(output_file)
    print(f"Trimmed NetCDF saved to {output_file}")

# Example usage:
# trim_netcdf_by_nan("input.nc", "output.nc", mask_var="cat_1")