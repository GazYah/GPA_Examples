import netCDF4
import rasterio
import numpy as np
from tqdm import tqdm
from typing import Tuple, Sequence

def squash_land_use(
    in_nc_path: str,
    tiff_path: str,
    out_nc_path: str,
    cell_size: int = 1000,
    all_unique: Sequence[int] = (1, 2, 3, 4, 5, 6)
) -> None:
    """
    Calculates the percentage of each land use category within each cell of a NetCDF grid,
    using a raster (TIFF) as the source of land use categories, and writes the results to a new NetCDF file.
    Currently being used on grids that fit eachother, dont know if it will work on grids that do not fit eachother.

    Args:
        in_nc_path (str): Path to the input NetCDF file containing the grid (must have 'x', 'y', 'lat', 'lon' variables).
        tiff_path (str): Path to the input raster (TIFF) file containing land use categories.
        out_nc_path (str): Path to the output NetCDF file to write the category percentages.
        cell_size (int, optional): Size of each grid cell in the NetCDF (default is 1000).
        all_unique (Sequence[int], optional): Iterable of all unique land use category values to process.

    Returns:
        None
    """
    # --- NetCDF input ---
    nc_in = netCDF4.Dataset(in_nc_path, mode='r')
    x = nc_in.variables['x'][:]
    y = nc_in.variables['y'][:]

    # --- Raster (TIFF) input ---
    with rasterio.open(tiff_path) as src:
        nodata = src.nodata if src.nodata is not None else 15

        # Prepare output arrays for each category
        out_shape = (len(y), len(x))
        cat_arrays = {cat: np.full(out_shape, np.nan, dtype=np.float32) for cat in all_unique}

        # Loop over every cell in the NetCDF grid with tqdm progress bar
        for iy in tqdm(range(len(y)), desc="Processing rows"):
            for ix in range(len(x)):
                x_bounds = (x[ix] - cell_size/2, x[ix] + cell_size/2)
                y_bounds = (y[iy] - cell_size/2, y[iy] + cell_size/2)
                window = rasterio.windows.from_bounds(
                    left=x_bounds[0], bottom=y_bounds[0],
                    right=x_bounds[1], top=y_bounds[1],
                    transform=src.transform
                )
                data = src.read(1, window=window)
                data = data[data != nodata]
                if data.size == 0:
                    continue
                unique, counts = np.unique(data, return_counts=True)
                total = counts.sum()
                percentages = {int(val): (count / total) * 100 for val, count in zip(unique, counts)}
                for cat in all_unique:
                    cat_arrays[cat][iy, ix] = percentages.get(cat, 0.0)

    # --- Copy input NetCDF to output and add new variables ---
    with netCDF4.Dataset(out_nc_path, "w") as nc_out:
        # Copy dimensions
        for name, dim in nc_in.dimensions.items():
            nc_out.createDimension(name, (len(dim) if not dim.isunlimited() else None))
        # Copy all variables except for the new categories
        for name, varin in nc_in.variables.items():
            outVar = nc_out.createVariable(name, varin.datatype, varin.dimensions, zlib=True)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]
        # Add new category variables
        for cat in all_unique:
            var = nc_out.createVariable(f'cat_{cat}', np.float32, ('y', 'x'), zlib=True)
            var[:, :] = cat_arrays[cat]
            var.units = "percent"
            var.long_name = f"Percentage of category {cat} in {cell_size}m cell"
        # Copy global attributes
        nc_out.setncatts({k: nc_in.getncattr(k) for k in nc_in.ncattrs()})

    nc_in.close()
    print(f"Saved output NetCDF: {out_nc_path}")

# Example usage:
# squash_land_use(
#     in_nc_path=r"loading/inca_data_midnight_monthly_avg_01.nc",
#     tiff_path=r"loading/testout.tif",
#     out_nc_path=r"landuse_percentages.nc"
# )