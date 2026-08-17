from typing import Any, Dict, cast

import numpy as np
import xarray as xr
from cop_ard_core.logger import logger, timeit
from cop_ard_core.simul import (
    SPATIAL_DIMS_,
    compute_simulation,
    extract_simul_on_radar_chunk,
    extract_simulation_ml,
    get_geo_from_radar_bbox,
    get_radar_bbox,
    init_simulation,
)
from pyproj import CRS
from scipy.constants import speed_of_light

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.io import (
    delete,
    exists,
    get_storage_options,
    makedirs,
    normpath,
    read,
    touch,
    write,
)
from cop_ard_app.utils import GeoEnvPlugin, get_margin


def dem_to_radar(
    dem: xr.DataArray,
    orbits: xr.DataArray,
    working_dir: str,
    chunks: dict = {"x": 1024, "y": 1024, "acquisition_id": 1},
    skip_read: bool = True,
    enable_dask: bool = True,
    **kwargs,
) -> xr.Dataset | None:
    """
    Compute the radar geometry (slant range, azimuth time) for each pixel of a DEM.

    This function orchestrates the DEM-to-radar-coordinate transformation. It first
    initializes the output data structure, populates it with elevation data, and then
    maps the `compute_simulation` function over blocks of the DEM to perform the
    geometric calculations in parallel.

    Parameters
    ----------
    dem : xr.DataArray
        The input DEM as an xarray.DataArray.
    orbits : xr.DataArray
        The satellite orbit data as an xarray.DataArray.
    working_dir : str
        A path to a directory for storing intermediate and final results.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    skip_read : bool
        If True, the function will not read and return the final dataset,
        useful for workflows that only need to ensure the data is written.
    enable_dask : bool
        If True, use Dask for parallel computation.

    Returns
    -------
    xr.Dataset or None
        The final simulation dataset read from disk if `skip_read` is False, otherwise None.
    """
    if (set(chunks.keys()) == set((*SPATIAL_DIMS_, "acquisition_date"))) is False:
        raise Exception("Chunks must contains x, y and acquisition_date keys")
    # Define the path for the simulation output.
    data_path = f"{working_dir}/simulation"
    acquisition_ids = orbits.acquisition_id.values
    xy_chunks = {k: v for k, v in chunks.items() if k in SPATIAL_DIMS_}
    x = dem.x.values
    y = dem.y.values
    # --- Step 1: Initialize Output Storage ---
    simulation = init_simulation_dataset(acquisition_ids, dem.rio.crs, x, y, xy_chunks, data_path, **kwargs)

    if not isinstance(simulation, xr.Dataset):
        raise Exception(f"Expected xr.Dataset, got {type(simulation)}")

    resampled_dem = dem.to_dataset(name="z")
    # --- Step 2: Populate with Elevation Data ---
    simulation = update_elevation(simulation, resampled_dem, xy_chunks, data_path, **kwargs)
    # --- Step 3: Compute Simulation in Parallel ---
    # Map the core computation function over blocks of the simulation and DEM datasets.
    save_options = {
        "data_path": normpath(data_path),
        "save_options": {
            "consolidated": False,
            "storage_options": get_storage_options(kwargs),
        },
    }
    map_blocks(
        compute_simulation,
        (simulation, resampled_dem.z),
        args=(orbits,),
        combine=False,
        enable_dask=enable_dask,
        store_mapping={0: save_options},
    )
    # Optionally, read and return the final result.
    return (
        read(normpath(data_path), as_type="xr.Dataset", storage_options=get_storage_options(kwargs))
        if not skip_read
        else None
    )


def update_elevation(
    simulation: xr.Dataset, resampled_dem: xr.Dataset, xy_chunks: Dict[str, Any], data_path: str, **kwargs
) -> xr.Dataset:
    """
    Update the simulation dataset with elevation data from the DEM.

    Parameters
    ----------
    simulation : xr.Dataset
        The simulation dataset to be updated.
    resampled_dem : xr.Dataset
        The Digital Elevation Model (DEM) data, resampled to the
        simulation grid, as an xarray.Dataset.
    xy_chunks : Dict[str, Any]
        A dictionary defining the chunk sizes for spatial dimensions.
    data_path : str
        The path to the simulation Zarr store on disk.

    ----------
    simulation : xr.Dataset
        The simulation dataset to be updated.
    resampled_dem : xr.Dataset
        The Digital Elevation Model (DEM) data, resampled to the
        simulation grid, as an xarray.Dataset.
    xy_chunks : Dict[str, Any]
        A dictionary defining the chunk sizes for spatial dimensions.
    data_path : str
        The path to the simulation Zarr store on disk.

    Returns
    -------
    xr.Dataset
        The updated simulation dataset, read back from the Zarr store.
    """
    chunks = {**xy_chunks, "acquisition_date": 1, "axis": -1}
    # Assign the elevation data ('z' from the DEM) to the 'h' variable in the simulation.
    simulation["h"] = resampled_dem.z
    # Re-chunk the simulation dataset to ensure consistency.
    simulation = simulation.chunk(chunks)
    # Write the updated elevation data ('h') back to the Zarr store.
    simulation = cast(
        xr.Dataset,
        write(
            simulation.h.to_dataset(name="h").drop_vars(["spatial_ref"]),
            data_path=normpath(data_path),
            consolidated=False,
            region={"x": slice(None, None), "y": slice(None, None)},  # Write to the full spatial extent.
            mode="r+",  # Open in read/write mode to update.
            storage_options=get_storage_options(kwargs),
        ),
    )

    if not isinstance(simulation, xr.Dataset):
        raise Exception(f"Expected xr.Dataset, got {type(simulation)}")
    return simulation


def init_simulation_dataset(
    acquisition_ids: np.ndarray,
    crs: CRS,
    x: np.ndarray,
    y: np.ndarray,
    xy_chunks: Dict[str, Any],
    data_path: str,
    **kwargs,
) -> xr.Dataset:
    """
    Initialize an empty Zarr store for the DEM-to-radar simulation results.

    This function creates the structure for the simulation output on disk. It sets up
    an xarray.Dataset with the correct dimensions, coordinates, data types, and
    chunking for all the variables that will be computed in the simulation
    (e.g., azimuth_time, slant_range_time, etc.).

    Parameters
    ----------
    acquisition_dates : np.ndarray
        A numpy array of acquisition dates for the stack.
    crs : CRS
        The pyproj CRS object of the output grid.
    x : np.ndarray
        A numpy array of the x-coordinates for the output grid.
    y : np.ndarray
        A numpy array of the y-coordinates for the output grid.
    xy_chunks : Dict[str, Any]
        A dictionary defining the chunk sizes for the spatial dimensions ('x', 'y').
    data_path : str
        The path where the simulation Zarr store will be created.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset representing the initialized (but mostly empty) simulation data,
        read back from the newly created Zarr store.
    """
    simulation = init_simulation(acquisition_ids, crs, x, y)

    # Apply the specified chunking to the dataset.
    simulation = simulation.chunk({**xy_chunks, "acquisition_id": 1, "axis": -1})

    # Save the simulation to a Zarr store on disk.
    simulation = write(
        simulation,
        data_path=normpath(data_path),
        storage_options=get_storage_options(kwargs),
    )

    if not isinstance(simulation, xr.Dataset):
        raise ValueError(f"Expected is xr.Dataset, got {type(simulation)}")
    return simulation


def single_simulation(
    burst_id: str,
    dem: xr.DataArray,
    multiburst: bool,
    orbits: xr.DataArray,
    working_directory: str,
    master_image: bool = False,
    chunks: dict = {"x": 1024, "y": 1024, "acquisition_date": 1},
    recompute: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.Dataset]:
    """
    Run the DEM-to-radar simulation for a single burst or scene.

    This function acts as a wrapper around `dem_to_radar`. It handles the logic for
    path management and decides whether to recompute the simulation or use an
    existing result, which is particularly useful for the master image in a stack.

    Parameters
    ----------
    burst_id : str
        The identifier for the burst or scene being processed.
    dem : xr.DataArray
        The DEM data for this specific burst.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    orbits : xr.DataArray
        The satellite orbit data.
    working_directory : str
        The root directory for all processing outputs.
    master_image : bool, optional
        If True, this is the master image of the stack. Its
        simulation can be cached to speed up subsequent runs. Defaults to False.
    chunks : dict, optional
        A dictionary defining the chunk sizes for Dask processing.
        Defaults to {"x": 1024, "y": 1024, "acquisition_date": 1}.
    recompute : bool, optional
        If True, force recomputation even if a cached result exists.
        Defaults to False.
    enable_dask : bool, optional
        If True, use Dask for parallel computation. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing the burst_id (str) and the resulting simulation Dataset (xr.Dataset).
    """
    # Define the relative path for this burst's data.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{working_directory}/{path}")
    # Define a path for a "done" file to check for completion.
    simulation_path = f"{working_directory}/processed_data/{path}/simulation.ok"
    # --- Check for Existing Results ---
    # Decide whether to run the computation or use a cached result.
    if (
        (master_image is False)
        or (exists(simulation_path, storage_options=get_storage_options(kwargs)) is False)
        or recompute
    ):
        # If this is the master image and we're recomputing, delete the "done" file.
        if master_image:
            delete(
                simulation_path,
                allow_non_existent=True,
                storage_options=get_storage_options(kwargs),
            )
        # --- Run the Main Simulation ---
        dem_to_radar(
            dem=dem,
            orbits=orbits,
            working_dir=data_path,
            chunks=chunks,
            skip_read=True,  # We will read the data later.
            enable_dask=enable_dask,
            **kwargs,
        )
        # If this was the master image, create the "done" file to cache the result.
        if master_image:
            makedirs(
                normpath(f"{working_directory}/processed_data/{path}"),
                exist_ok=True,
                **kwargs,
            )
            touch(simulation_path, **kwargs)
    else:
        logger.info(f"Found existing master image simulation file for {burst_id}")
    # Read the final simulation data from disk.
    simulation = read(
        normpath(f"{data_path}/simulation"),
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs),
    )
    return (burst_id, simulation)


@timeit
def stack_simulation(
    dem_tree: xr.DataTree,
    orbits: xr.DataArray,
    multiburst: bool,
    working_directory: str,
    chunks: dict = {"x": 1024, "y": 1024, "acquisition_date": 1},
    master_image: bool = False,
    recompute: bool = False,
    **kwargs,
) -> xr.DataTree:
    """
    Run the DEM-to-radar simulation for an entire stack of bursts or scenes.

    This function maps the `single_simulation` function over all the items
    in a DataTree, processing each burst or scene in parallel. It then assembles
    the results back into a single output DataTree.

    Parameters
    ----------
    dem_tree : xr.DataTree
        An xarray.DataTree where each node contains the DEM for a burst.
    orbits : xr.DataArray
        The satellite orbit data for the entire stack.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    working_directory : str
        The root directory for all processing outputs.
    chunks : dict, optional
        A dictionary defining the chunk sizes for Dask processing.
        Defaults to {"x": 1024, "y": 1024, "acquisition_date": 1}.
    master_image : bool, optional
        If True, indicates that this is the master image's simulation.
        Defaults to False.
    recompute : bool, optional
        If True, force recomputation of all simulations. Defaults to False.

    Returns
    -------
    xr.DataTree
        An xarray.DataTree containing the simulation results for all bursts/scenes.
    """
    # Prepare the arguments for the parallel map operation.
    for k, d in (dem_tree.bursts if multiburst else {"data": dem_tree}).items():
        in_proj = d.z.rio.crs
        break
    args = [(k, d.z, multiburst) for k, d in (dem_tree.bursts if multiburst else {"data": dem_tree}).items()]
    args = list(zip(*args))
    # --- Parallel Execution ---
    results = cast(
        list,
        map(
            single_simulation,
            args,
            # Pass common arguments via kwargs.
            kwargs=dict(
                orbits=orbits,
                working_directory=working_directory,
                master_image=master_image,
                chunks=chunks,
                recompute=recompute,
                **kwargs,
            ),
            enable_dask=kwargs.get("enable_dask", True),
            Plugin=GeoEnvPlugin(in_proj=in_proj, out_proj="EPSG:4978"),
            combine=False,
        ),
    )
    # --- Assemble Results ---
    simulation_tree = xr.DataTree()
    for k, sim in results:
        # Reconstruct the DataTree structure from the list of results.
        if multiburst:
            simulation_tree[f"bursts/{k}"] = sim
        else:
            simulation_tree = xr.DataTree(sim)
    return simulation_tree


@timeit
def stack_simulation_ml(
    simulation_tree: xr.DataTree,
    reference_id: str,
    multiburst: str,
    working_directory: str,
    recompute: bool = False,
    **kwargs,
):
    """
    Create multilooked simulation data for an entire stack of bursts or scenes.

    This function maps `single_simulation_ml` over all items in a DataTree to
    generate the downsampled simulation overviews for each burst/scene in parallel.

    Parameters
    ----------
    simulation_tree : xr.Datatree
        The DataTree containing the full-resolution simulations.
    reference_date : str
        The reference acquisition date.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    working_directory :str
        The root directory for all outputs.
    recompute : bool
        If True, force recomputation.

    Returns
    -------
    dict
        A dictionary (or nested dictionary for multiburst) containing the
        multilooked simulation data for the entire stack.
    """
    # Prepare the arguments for the parallel map operation.
    args = [
        (k, s.to_dataset(), multiburst)
        for k, s in (simulation_tree.bursts if multiburst else {"data": simulation_tree}).items()
    ]
    args = list(zip(*args))
    # --- Parallel Execution ---
    results = cast(
        list,
        map(
            single_simulation_ml,
            args,
            kwargs=dict(
                reference_id=reference_id,
                working_directory=working_directory,
                recompute=recompute,
                **kwargs,
            ),
            combine=False,
            enable_dask=kwargs.get("enable_dask", True),
        ),
    )
    # --- Assemble Results ---
    simulation_ml_tree: dict = {"bursts": {}} if multiburst else {}
    for k, sim_ml in results:
        # Reconstruct the nested dictionary structure from the list of results.
        if multiburst:
            simulation_ml_tree["bursts"][k] = sim_ml
        else:
            simulation_ml_tree = sim_ml
    return simulation_ml_tree


def initialize_simulation(
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    chunk_length: int,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Initialize and subset simulation data based on SLC extent and calculate global parameters.

    This function determines the relevant geographic subset of the simulation data
    that corresponds to the spatial extent of a given Single Look Complex (SLC)
    image. It calculates various global parameters needed for subsequent rasterization
    and RTC processing, such as grid dimensions, pixel area, and bounding boxes.

    Parameters
    ----------
    slc : xr.DataArray
        The input SLC image DataArray, used to define the
        radar grid extent.
    simulation : xr.Dataset
        The full simulation dataset containing radar geometry
        (azimuth_time, slant_range_time) mapped onto the geographic grid.
    simulation_ml : dict
        A dictionary containing the downsampled ("multilooked")
        simulation data, used for quickly finding the geographic bounds.
    chunk_length : int
        The desired chunk size for the spatial dimensions (x, y)
        of the processed simulation subset.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing:

        - **selection** (`xr.Dataset`): The subset of the simulation data relevant to the
          SLC, re-chunked and with added radar pixel coordinates (``az_px``, ``rg_px``).
        - **global_values** (`xr.Dataset`): A dataset containing important global parameters
          and coordinate vectors derived from the SLC and simulation data.
    """
    # --- Extract Metadata and Calculate Parameters ---
    # Get metadata from the first polarization's attributes.
    annot = slc.attrs[list(slc.attrs.keys())[0]]
    # Calculate azimuth time step (in seconds) and slant range time step.
    dt_az = annot["Grid"]["Col"]["SS"] * annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1]
    dt_rg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light
    # Get reference times for coordinate conversion.
    az_ref, rg_ref = slc.azimuth_time[0].values, slc.slant_range_time[0].values
    # Calculate the area of a single pixel in the radar grid.
    radar_grid_pixel_area = annot["Grid"]["Col"]["SS"] * annot["Grid"]["Row"]["SS"]
    # Calculate time steps as timedelta64 and float for margin calculation.
    margin = get_margin(
        time_ca_polynomial_coefficients=annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"],
        azimuth_time_spacing=annot["Grid"]["Col"]["SS"],
        slant_range_time_spacing=annot["Grid"]["Row"]["SS"],
    )

    # --- Select Relevant Simulation Subset ---
    # Use the helper function to extract the simulation data covering the SLC's footprint plus margin.
    azimuth_time = slc.azimuth_time
    slant_range_time = slc.slant_range_time
    bbox_radar = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(simulation_ml, bbox_radar)
    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    selection = extract_simul_on_radar_chunk(
        simulation, simulation.acquisition_date.values[0], bbox_radar, bbox_geo
    ).squeeze()  # Remove singleton dimensions.
    # Get the geographic bounding box of the selected simulation subset.
    bbox_geo = [selection.x.min().values, selection.y.min().values, selection.x.max().values, selection.y.max().values]

    # --- Calculate Radar Pixel Coordinates ---
    # Convert the simulation's azimuth and range times into pixel coordinates relative to the SLC grid origin.
    az = (selection.azimuth_time - az_ref).astype(float) / (dt_az * 1e9)
    rg = (selection.slant_range_time - rg_ref).astype(float) / (dt_rg)

    # --- Clean Up and Rechunk Selection ---
    # Keep only essential coordinates and variables in the selection.
    keep_coords = ["x", "y"]
    keep_vars = ["dem_distance", "h"]
    selection = selection.drop_vars([i for i in selection.coords if i not in keep_coords])
    selection = selection.drop_vars([i for i in selection.data_vars if i not in keep_vars])
    # Assign the calculated radar pixel coordinates.
    selection = selection.assign(az_px=az, rg_px=rg)
    # Apply the specified chunking to the selected subset.
    selection = selection.chunk(chunk_length)

    # --- Calculate Global Grid Dimensions ---
    # Get geographic grid spacing.
    spacing_x = abs(selection.x[1] - selection.x[0]).values
    spacing_y = abs(selection.y[1] - selection.y[0]).values
    # Calculate the total size of the geographic grid covering the selection.
    dim_x_global = np.round((bbox_geo[2] - bbox_geo[0]) / spacing_x).astype(int)
    dim_y_global = np.round((bbox_geo[3] - bbox_geo[1]) / spacing_y).astype(int)

    # Determine the extent of the radar grid covered by the selection in pixel coordinates.
    az_px = selection.az_px
    rg_px = selection.rg_px
    ymin_global, xmin_global = np.floor(az_px.min().values).astype(int), np.floor(rg_px.min().values).astype(int)
    ymax_global, xmax_global = np.ceil(az_px.max().values).astype(int), np.ceil(rg_px.max().values).astype(int)
    # Calculate the total size of the relevant radar grid portion.
    xsize_global = np.ceil(xmax_global - xmin_global).astype(int)
    ysize_global = np.ceil(ymax_global - ymin_global).astype(int)

    # --- Assemble Global Values Dataset ---
    # Store all calculated global parameters and coordinate vectors in a separate dataset.
    global_values = {
        "radar_grid_pixel_area": {"dims": ("point"), "data": [radar_grid_pixel_area]},
        "chunk_length": {"dims": ("point"), "data": [chunk_length]},
        "chunks_slc": {  # Store original SLC chunk sizes
            "dims": ("chunk"),
            "data": [slc.chunksizes["azimuth_time"][0], slc.chunksizes["slant_range_time"][0]],
        },
        "radar_xsize": {"dims": ("point"), "data": [xsize_global]},
        "radar_ysize": {"dims": ("point"), "data": [ysize_global]},
        "radar_xmin": {"dims": ("point"), "data": [xmin_global]},
        "radar_ymin": {"dims": ("point"), "data": [ymin_global]},
        "radar_xmax": {"dims": ("point"), "data": [xmax_global]},  # Max extent needed?
        "geo_xsize": {"dims": ("point"), "data": [dim_x_global]},
        "geo_ysize": {"dims": ("point"), "data": [dim_y_global]},
        "bbox": {"dims": ("values"), "data": [0, 0, slc.slant_range_time.size, slc.azimuth_time.size]},  # Full SLC bbox
        "x_coords": {"dims": ("x"), "data": selection.x.values},
        "y_coords": {"dims": ("y"), "data": selection.y.values},
        "rg_coords": {"dims": ("slant_range_time"), "data": slc.slant_range_time.values},
        "az_coords": {"dims": ("azimuth_time"), "data": slc.azimuth_time.values},
    }
    global_values = xr.Dataset.from_dict(global_values).rio.write_crs(simulation.rio.crs)
    return selection, global_values


def single_simulation_ml(
    burst_id: str,
    simulation: xr.Dataset,
    multiburst: bool,
    reference_id: str,
    working_directory: str,
    recompute: bool = False,
    **kwargs,
):
    """
    Create and manage multilooked simulation data for a single burst or scene.

    This function wraps `extract_simulation_ml` and adds logic to check for
    existing results, allowing it to skip recomputation if the data already exists.

    Parameters
    ----------
    burst_id : str
        The identifier for the burst or scene.
    simulation : xr.Dataset
        The full-resolution simulation dataset for this burst.
    multiburst : bool
        A boolean flag for multiburst mode.
    reference_date : str
        The reference acquisition date.
    working_directory : str
        The root directory for outputs.
    recompute : bool
        If True, force recomputation.

    Returns
    -------
    tuple
        A tuple containing the burst_id and a dictionary with the 'min' and 'max'
        multilooked datasets.
    """
    # Define the relative path for this burst's data.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{working_directory}/{path}")
    # Define a path for a "done" file to check for completion.
    simulation_ml_path = f"{working_directory}/processed_data/{path}/simulation_ml.ok"
    # --- Check for Existing Results ---
    if (exists(simulation_ml_path, **kwargs) is False) or recompute:
        # If recomputing, delete the "done" file.
        delete(simulation_ml_path, allow_non_existent=True, **kwargs)
        # --- Run the Multilooking Extraction ---
        chunks = dict(x=simulation.chunksizes["x"][0], y=simulation.chunksizes["y"][0])
        simulation_ml = extract_simulation_ml(
            simulation=simulation,
            reference_id=reference_id,
            coarsen_config=chunks,
        )
        for k, v in simulation_ml.items():
            write(
                v.chunk(chunks),
                normpath(f"{data_path}/simulation_ml_{k}"),
                skip_read=True,
                storage_options=get_storage_options(kwargs),
            )

        # Create the "done" file to cache the result.
        touch(simulation_ml_path, **kwargs)
    else:
        logger.info(f"Found existing simulation multilook file for {burst_id}")

    # --- Read and Return the Result ---
    simulation_ml = {
        "min": read(
            normpath(f"{data_path}/simulation_ml_min"),
            as_type="xr.Dataset",
            storage_options=get_storage_options(kwargs),
        ),
        "max": read(
            normpath(f"{data_path}/simulation_ml_max"),
            as_type="xr.Dataset",
            storage_options=get_storage_options(kwargs),
        ),
    }
    return (burst_id, simulation_ml)
