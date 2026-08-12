from typing import Any, Dict, Mapping, Sequence, cast

import numpy as np
import xarray as xr
from cop_ard_core.dem import (
    check_available_transformation,
    compute_shape_and_off,
    extract_vertical_crs,
    get_default_epsg,
    interp_dem,
)
from cop_ard_core.logger import logger, timeit
from pyproj import CRS

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.io import exists, get_storage_options, normpath, open_copdem30m, read, save_empty_dataset, write
from cop_ard_app.utils import GeoEnvPlugin


def dem_reproject(
    dem: xr.DataArray,
    spacing: dict[str, Any],
    outproj: str | int,
    data_path: str,
    chunks: dict = {"x": 2048, "y": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Reproject a Digital Elevation Model (DEM) to a specified CRS.

    This function orchestrates the reprojection of a DEM. It defines the output
    grid, initializes an empty Zarr store for the result, and then maps the
    `interp_dem` function over blocks to perform the interpolation in parallel.

    Parameters
    ----------
    dem : xarray.DataArray
        The input Digital Elevation Model as an xarray.DataArray.
    spacing
        The desired spatial resolution (spacing) in the output CRS units.
    outproj
        The target EPSG code for the output DEM.
    data_path
        The path where the reprojected DEM Zarr store will be created.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    enable_dask : bool
        If True, use Dask for parallel computation.
    **kwargs
        Additional keyword arguments, such as `method` for interpolation.

    Returns
    -------
    xarray.Dataset
        The reprojected DEM as an xarray.Dataset, read back from the Zarr store.
    """
    method = kwargs.pop("method", "cubic")
    # Ensure the input DEM has 'x' and 'y' coordinates.
    if (all([i in dem.coords for i in ["x", "y"]])) is False:
        raise Exception('DEM dataset coordinates must contain "x" and "y"')
    dem_ds = dem.to_dataset(name="z")

    # --- CRS Handling ---
    CRS_EPSG = f"EPSG:{outproj}" if not str(outproj).startswith("EPSG:") else outproj
    if not isinstance(CRS_EPSG, str):
        raise Exception(f"Expected str, got {type(CRS_EPSG)}")
    # Create 3D CRS objects for robust transformation.
    in_crs, out_crs = CRS(dem.rio.crs).to_3d(), CRS(CRS_EPSG).to_3d()
    dem_ds.rio.write_crs(in_crs, inplace=True)
    # Construct a composite CRS that preserves the vertical datum.
    vertical_crs = extract_vertical_crs(in_crs)
    crs_2d = out_crs.to_2d().to_string()
    MID_EPSG = f"{crs_2d}+{vertical_crs}" if vertical_crs != "" else crs_2d
    mid_crs = CRS(MID_EPSG).to_3d()

    # --- Define Output Grid ---
    shape, off = compute_shape_and_off(dem, spacing, CRS_EPSG)
    # Create the coordinate arrays for the output grid.
    x_lin, y_lin = [np.linspace(off[i], off[i] + (shape[i] - 1) * spacing[i], shape[i]) for i in ["x", "y"]]

    # --- Initialize Output ---
    # Create an empty Zarr store for the reprojected DEM.
    dem_out = save_empty_dataset(
        data_vars=["z"],
        dims={"z": ["y", "x"]},
        shape={"z": [y_lin.size, x_lin.size]},
        dtype={"z": dem_ds.z.dtype},
        chunks=chunks,
        crs=out_crs,
        coords={"y": y_lin, "x": x_lin},
        data_path=normpath(f"{data_path}/dem_resampled"),
        **kwargs,
    )

    # --- Run Reprojection in Parallel ---
    # Map the interpolation function over blocks of the output grid.
    save_options = {
        "data_path": normpath(f"{data_path}/dem_resampled"),
        "save_options": {
            "skip_read": True,
            "consolidated": False,
            "storage_options": get_storage_options(kwargs),
        },
    }
    map_blocks(
        interp_dem,
        dem_out.z,
        args=[dem_ds.z, mid_crs, method],
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,
        store_mapping={0: save_options},
    )
    # Read and return the final reprojected dataset.
    return read(
        normpath(f"{data_path}/dem_resampled"), as_type="xr.Dataset", storage_options=get_storage_options(kwargs)
    )


def single_dem(
    burst_id: str,
    slc_attrs: list[dict],
    multiburst: bool,
    dem: xr.Dataset,
    spacing: dict[str, float],
    working_directory_static: str,
    working_directory_tmp: str,
    chunks: dict = {"x": 2048, "y": 2048},
    crs: str | None = None,
    enable_dask: bool = True,
    border: float = 10.0 / 3600.0,  # Approx. 300m border in degrees
    recompute=False,
    **kwargs,
):
    """
    Reproject a DEM for a single burst or scene.

    This worker function calculates the geographic bounding box of a single SAR
    burst, clips the input DEM to that area, and then calls `dem_reproject` to
    perform the reprojection.

    Parameters
    ----------
    burst_id
        The identifier for the burst or scene.
    slc
        The DataTree for the specific burst, used to find its footprint.
    multiburst : bool
        A boolean flag for multiburst mode.
    dem
        The full input Digital Elevation Model.
    spacing
        The desired spatial resolution of the output DEM.
    working_directory
        The root directory for outputs.
    chunks : dict
        A dictionary defining chunk sizes for processing.
    crs
        The target EPSG code. If None, it will be automatically determined.
    enable_dask : bool
        If True, use Dask for parallel computation.
    border
        A border in degrees to add to the burst's bounding box.
    **kwargs
        Additional keyword arguments for the reprojection function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the reprojected DEM Dataset.
    """
    # Define the output path for this burst.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{working_directory_tmp}/{path}")
    simulation_path = f"{working_directory_static}/processed_data/{path}/simulation.ok"
    storage_options_static = get_storage_options(kwargs, key="storage_options_static")
    # --- Check for Existing Results ---
    # Decide whether to run the computation or use a cached result.
    if exists(simulation_path, storage_options=storage_options_static) is False or recompute:
        # --- Calculate Bounding Box ---
        lat = []
        lon = []
        # Collect the corner coordinates from all original images in this burst.
        for attrs in slc_attrs:
            annot = attrs[list(attrs)[0]]
            for v in annot["GeoData"]["ImageCorners"]:
                lat.append(v["Lat"])
                lon.append(v["Lon"])
        # Determine the min/max lat/lon and add a border.
        lat_min, lat_max = min(lat) - border, max(lat) + border
        lon_min, lon_max = min(lon) - border, max(lon) + border

        # --- Clip and Reproject DEM ---
        # Clip the global DEM to the specific area of this burst.
        dem_aoi = dem.rio.clip_box(lon_min, lat_min, lon_max, lat_max, crs="EPSG:4326").chunk(chunks)
        # Determine the target CRS if not provided.
        if crs is None:
            crs = get_default_epsg(dem_aoi)
        # Call the main reprojection function.
        dem_resampled = dem_reproject(
            dem_aoi, spacing, crs, chunks=chunks, data_path=data_path, enable_dask=enable_dask, **kwargs
        )
    else:
        logger.info(f"Found existing DEM file for {burst_id}")
        dem_resampled = read(
            normpath(f"{working_directory_static}/{path}/simulation"),
            as_type="xr.Dataset",
            storage_options=storage_options_static,
        ).h.to_dataset(name="z")

    return (burst_id, dem_resampled)


@timeit
def stack_dem(
    dem: xr.Dataset,
    slcs: xr.DataTree,
    multiburst: bool,
    spacing: dict[str, float],
    working_directory_static: str,
    working_directory_tmp: str,
    chunks: dict = {"x": 2048, "y": 2048},
    crs: str | None = None,
    enable_dask: bool = True,
    recompute: bool = False,
    **kwargs,
):
    """
    Reproject a DEM for each burst or scene in a stack.

    This function orchestrates the reprojection of a DEM for an entire stack.
    It maps the `single_dem` function over each item in the `slcs` DataTree,
    processing each one in parallel, and then assembles the results into a new
    DataTree.

    Parameters
    ----------
    dem : xarray.DataArray
        The input Digital Elevation Model as an xarray.DataArray.
    slcs : xarray.DataTree
        An xarray.DataTree containing the SAR data, used to define the
        geographic extent for each burst.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    spacing
        The desired spatial resolution of the output DEMs.
    working_directory
        The root directory for all processing outputs.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    crs
        The target EPSG code. If None, it will be automatically determined.
    **kwargs
        Additional keyword arguments passed to the reprojection function.

    Returns
    -------
    xarray.DataTree
        An xarray.DataTree containing the reprojected DEM for each burst or scene.
    """
    # --- Determine Target CRS ---
    if crs is None:
        # If no CRS is specified, determine the default UTM/UPS zone.
        epsg = get_default_epsg(dem)
    else:
        epsg = crs
    # Check if the required transformation is available.
    check_available_transformation(dem, epsg)

    # --- Prepare Arguments and Run in Parallel ---
    # Create a list of arguments for each burst/scene to be processed.
    args = [
        (k, [o.measurements.slc.attrs for o in s.orig.values()], multiburst)
        for k, s in (slcs.bursts if multiburst else {"data": slcs}).items()
    ]
    args = list(zip(*args))
    results = cast(
        list,
        map(
            single_dem,
            args,
            # Pass common arguments via kwargs.
            kwargs=dict(
                dem=dem,
                spacing=spacing,
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                chunks=chunks,
                crs=crs,
                enable_dask=enable_dask,
                recompute=recompute,
                **kwargs,
            ),
            combine=False,
            enable_dask=enable_dask,
            Plugin=GeoEnvPlugin(in_proj=dem.rio.crs, out_proj=epsg),
        ),
    )

    # --- Assemble Results ---
    dem_tree = xr.DataTree()
    for k, d in results:
        # Reconstruct the DataTree structure from the list of results.
        if multiburst:
            dem_tree[f"bursts/{k}"] = d
        else:
            dem_tree = xr.DataTree(d)
    return dem_tree


def import_dem(
    bbox: Sequence[float],
    working_dir: str | None = None,
    dem_type: str = "Copernicus",
    chunks: Mapping[Any, int] | None = None,
    border: float = 0.05,
    **kwargs,
) -> xr.DataArray:
    """
    Import and preprocess a Digital Elevation Model (DEM).

    This function handles the import of a DEM from a specified source (e.g.,
    Copernicus GLO-30, or a user-provided file path). It clips the DEM to a
    given bounding box, applies chunking, handles NoData values, and saves
    the result to a Zarr store.
    """

    # Expand the bounding box slightly to avoid edge effects during reprojection.
    minx, miny, maxx, maxy = bbox
    minx = minx - border
    miny = miny - border
    maxx = maxx + border
    maxy = maxy + border

    # --- DEM Source Handling ---
    if dem_type == "Copernicus":
        # Download the Copernicus DEM for the specified bounding box.
        dem = open_copdem30m(minx, miny, maxx, maxy)
    else:
        # Open a user-specified DEM from a local or remote path.
        dem = xr.open_dataset(
            dem_type,
            chunks={},
            engine="zarr",
            decode_coords="all",
            storage_options=get_storage_options(kwargs, key="storage_options_dem"),
        )
        if not dem.rio.crs:
            raise Exception(f"Input DEM projection not defined {dem} {dem.rio.crs}")

        # Standardize coordinate names if the CRS is geographic.
        if dem.rio.crs.is_geographic:
            if all([i in dem.coords for i in ["lon", "lat"]]):
                dem = dem.rename({"lat": "y", "lon": "x"})

        # Preserve original chunking for efficient clipping.
        chunksize = {e: dem.chunksizes[e][0] for e in dem.coords if e in ["x", "y", "lon", "lat"]}

        # Clip the DEM to the area of interest.
        dem = dem.rio.clip_box(minx, miny, maxx, maxy, crs="EPSG:4326")

        # Re-apply chunking after clipping.
        dem = dem.chunk(chunksize)

    # Apply user-specified chunking if provided.
    if chunks:
        dem = dem.chunk(chunks)
        # Drop encoding to prevent issues when writing re-chunked data to Zarr
        dem = dem.drop_encoding()

    # Convert the Dataset to a DataArray named 'z'.
    dem = dem.to_array(name="z").squeeze(drop=True)

    # --- NaN Handling ---
    # Ensure spatial dimensions are strictly recognized
    dem.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    # Write the processed DEM to a Zarr store in the working directory.
    if (dem_type == "Copernicus") or dem_type.startswith("https://"):
        dem = write(dem, normpath(f"{working_dir}/dem"), storage_options=get_storage_options(kwargs))
        if not isinstance(dem, xr.DataArray):
            raise Exception("Error reopening DEM dataarray")

    return dem
