import pandas as pd
import numpy as np
from pyproj import Transformer
import netCDF4
import shutil
from tqdm import tqdm

def filter_and_grid_csv_to_netcdf(
    csv_path: str,
    netcdf_path: str,
    output_netcdf_path: str,
    years: list[int] = [2024]
) -> None:
    """
    Reads a tab-separated CSV file, filters and processes the data, and writes a presence grid to a NetCDF file.

    Args:
        csv_path (str): Path to the input CSV file.
        netcdf_path (str): Path to the template NetCDF file (will be copied).
        output_netcdf_path (str): Path to the output NetCDF file.
        years (list[int], optional): List of years to filter the data. Defaults to [2024].

    Returns:
        None
    """
    # --- 1. Read and filter the CSV ---
    df = pd.read_csv(csv_path, sep='\t', dtype=str)

    # Convert columns to appropriate types
    df['coordinateUncertaintyInMeters'] = pd.to_numeric(df['coordinateUncertaintyInMeters'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
    df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')
    df['individualCount'] = pd.to_numeric(df['individualCount'], errors='coerce')

    # 1) Filter coordinateUncertaintyInMeters <= 1000 or None
    mask_uncertainty = (df['coordinateUncertaintyInMeters'].isna()) | (df['coordinateUncertaintyInMeters'] <= 1000)
    df = df[mask_uncertainty]

    # 2) Filter year in list
    df = df[df['year'].isin(years)]

    # 3) Convert coordinates from EPSG:4326 to EPSG:31287
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:31287", always_xy=True)
    def convert_coords(row):
        if pd.notna(row['decimalLongitude']) and pd.notna(row['decimalLatitude']):
            x, y = transformer.transform(row['decimalLongitude'], row['decimalLatitude'])
            return pd.Series({'x_31287': x, 'y_31287': y})
        else:
            return pd.Series({'x_31287': np.nan, 'y_31287': np.nan})

    df[['x_31287', 'y_31287']] = df.apply(convert_coords, axis=1)

    # Remove rows with missing projected coordinates
    df = df.dropna(subset=['x_31287', 'y_31287'])

    # --- 4. Copy input NetCDF and add new variable ---
    shutil.copy2(netcdf_path, output_netcdf_path)

    with netCDF4.Dataset(output_netcdf_path, 'a') as nc:
        x_var = nc.variables['x'][:]
        y_var = nc.variables['y'][:]

        presence_grid = np.zeros((len(y_var), len(x_var)), dtype=np.float32)

        x_min, x_max = x_var.min(), x_var.max()
        y_min, y_max = y_var.min(), y_var.max()

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            x, y = row['x_31287'], row['y_31287']
            if (x_min <= x <= x_max) and (y_min <= y <= y_max):
                x_idx = np.abs(x_var - x).argmin()
                y_idx = np.abs(y_var - y).argmin()
                if 0 <= x_idx < len(x_var) and 0 <= y_idx < len(y_var):
                    val = row['individualCount'] if not np.isnan(row['individualCount']) else 1
                    presence_grid[y_idx, x_idx] += val

        if 'presence' not in nc.variables:
            presence = nc.createVariable('presence', 'f4', ('y', 'x'), fill_value=0)
        else:
            presence = nc.variables['presence']
        presence[:, :] = presence_grid
        presence.long_name = "Presence or individual count from filtered CSV"

# Example usage:
# filter_and_grid_csv_to_netcdf(csv_path, netcdf_path, output_netcdf_path, years=[2024])