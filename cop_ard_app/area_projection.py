from typing import Any, Hashable, Mapping

import botocore.exceptions
import dask.array as da
import numpy as np
import xarray as xr
from cop_ard_core.area_projection import compute_geocoded_mask, compute_rasterization, compute_rtc_ap, compute_shadow
from cop_ard_core.dem import get_resolution_in_meters
from cop_ard_core.griddata import init_interpolate
from cop_ard_core.kernels import update_dense_array
from cop_ard_core.logger import logger, timeit
from scipy.constants import speed_of_light

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.io import (
    delete,
    exists,
    get_storage_options,
    glob,
    makedirs,
    normpath,
    read,
    save_empty_dataset,
    touch,
    write,
)
from cop_ard_app.simulation import initialize_simulation
from cop_ard_app.utils import GeoEnvPlugin, compute_chunks, get_margin


def map_shadow_projection(
    slc: xr.DataArray,
    data_path_static: str,
    threshold: float = 0.3,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Computes the geocoded Gamma Naught image by mapping radar data to the geogrid.

    This function orchestrates the geocoding process. It first reads global
    grid parameters, creates an empty Zarr dataset for the output geocoded
    image (`gamma0`), and then uses `map_blocks` to apply the `compute_geo_ap`
    function to each chunk in parallel (if Dask is enabled).

    Parameters
    ----------
    slc : xr.DataArray
        The input Single Look Complex (SLC) radar data,
        which will be read in chunks.
    data_path_static : str
        Path to the directory containing static data,
        such as global grid values and weights (w).
    data_path_tmp : str
        Path to the directory where temporary and final
        output Zarr stores (like 'geo_grid') are written.
    threshold : float, optional
        A threshold value passed to `compute_geo_ap`
        used for ANF normalization. Defaults to 1e-3.
    enable_dask : bool, optional
        Whether to enable Dask for parallel
        computation. Defaults to True.
    recompute : bool, optional
        A flag passed to `compute_geo_ap`, primarily
        to control mask (layover/shadow) re-computation.
        Defaults to True.
    skip_read : bool, optional
        If True, the function returns None after
        computation. If False, it reads the final
        'gamma0' dataset from disk and returns it.
        Defaults to True.
    chunksizes : dict[str, int] or None, optional
        Specific Dask chunk sizes
        to use for the computation.
        Defaults to None.

    Returns
    -------
    xr.DataArray or None
        The geocoded 'gamma0' values if `skip_read` is False,
        otherwise None.
    """
    # Load global metadata saved from the previous simulation step
    global_values = read(
        data_path=normpath(f"{data_path_static}/global_values"),
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs),
    )
    dim_x_global = global_values.geo_xsize.values[0]
    dim_y_global = global_values.geo_ysize.values[0]
    # Get the chunk_length used during the weight computation
    chunk_length = global_values.chunk_length.values[0]

    # Define the dimensions, shape, and chunking for the output Zarr store
    dims = ["y", "x"]
    shape = [dim_y_global, dim_x_global]
    chunks = {"y": chunk_length, "x": chunk_length}

    # Create an empty Dask array with the correct shape and chunks to map over
    # Note: The +1 is because the grid coordinates define cell edges, not centers
    empty_data = xr.DataArray(da.empty(shape=(shape[0] + 1, shape[1] + 1), dtype="float32"), dims=dims).chunk(chunks)

    # Get the final computed chunk structure
    chunks = compute_chunks(dict(empty_data.chunksizes))

    # Define the arguments to be passed to the worker function `compute_geocoded_mask`
    args = [chunk_length, slc.shape[-1], data_path_static, threshold]

    def region_callback(region: Mapping[str, slice], kwargs: dict) -> Mapping[str, slice]:
        out = {}
        for k, v in region.items():
            if k in ["y", "x"]:
                out[k] = slice(v.start, v.stop - 1)
            else:
                raise ValueError(f"Unexpected dimension {k}")
        return out

    save_options = {
        "data_path": f"{data_path_static}/geo_grid",
        "save_options": {
            "consolidated": False,
            "skip_read": True,
            "storage_options": get_storage_options(kwargs),
        },
    }

    # Map the `compute_geo_ap` function over all chunks of the `empty_data`
    map_blocks(
        func=blockwise_geocoded_mask,
        obj=empty_data,
        args=args,
        kwargs=kwargs,
        chunks=chunks,
        enable_dask=enable_dask,
        combine=False,
        region_callback={0: region_callback},
        store_mapping={0: save_options},
    )

    return


def map_shadow(
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    data_path: str,
    eps: float,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Maps the shadow computation over the radar grid chunks.

    This function sets up an empty Dask array matching the radar grid and
    maps the `compute_shadow` function over its chunks.

    Parameters
    ----------
    slc : xr.DataArray
        The reference SLC DataArray (used for metadata
        and dimensions).
    simulation : xr.Dataset
        The radar simulation dataset.
    simulation_ml : dict
        The multi-looked simulation data.
    data_path : str
        Path to the directory where results will be saved
        and global values are read from.
    eps : float
        Epsilon value passed to `compute_shadow`.
    enable_dask : bool, optional
        Whether to enable Dask for parallel
        computation. Defaults to True.

    Returns
    -------
    None
        This function saves the shadow mask to disk.
    """

    # Load global values (chunk sizes, bounding box) from the static path
    # Extract annotation metadata from the SLC
    annot = slc.attrs[list(slc.attrs.keys())[0]]

    # Calculate time differentials for azimuth and range
    dt = {
        "azimuth_time": annot["Grid"]["Col"]["SS"] * annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1],
        "slant_range_time": 2 * annot["Grid"]["Row"]["SS"] / speed_of_light,
    }
    # Calculate time steps (used for margins)
    margin = get_margin(
        time_ca_polynomial_coefficients=annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"],
        azimuth_time_spacing=annot["Grid"]["Col"]["SS"],
        slant_range_time_spacing=annot["Grid"]["Row"]["SS"],
    )
    # Compile Numba functions in griddata
    init_interpolate()

    # Create an empty Dask array with the shape and chunks of the radar grid
    if "polarization" in slc.dims:
        radar_data = slc.isel(acquisition_date=0, polarization=0)
    else:
        radar_data = slc
    # Define arguments for the worker function
    local_incidence_angle = read(
        f"{data_path}/geo_grid", consolidate_metadata=False, storage_options=get_storage_options(kwargs)
    ).lia
    simulation_h = simulation.h
    simulation_dem_distance = simulation.dem_distance
    simulation_azimuth_time = simulation.azimuth_time
    simulation_slant_range_time = simulation.slant_range_time

    # Map the `compute_shadow` function over all chunks
    save_options = {
        "data_path": normpath(f"{data_path}/radar_grid"),
        "save_options": {
            "consolidated": False,
            "skip_read": True,
            "storage_options": get_storage_options(kwargs, "storage_options"),
        },
    }
    map_blocks(
        func=compute_shadow,
        obj=radar_data,
        args=[
            simulation_h,
            simulation_dem_distance,
            simulation_azimuth_time,
            simulation_slant_range_time,
            simulation_ml,
            local_incidence_angle,
            eps,
            dt,
            margin,
        ],
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,  # Results are written to disk
        store_mapping={0: save_options},
    )
    return


def mask_shadow_single_burst(
    burst_id: str,
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    wdir_path_static: str,
    eps: float = 1e-4,
    threshold: float = 0.3,
    enable_dask: bool = True,
    recompute: bool = False,
    **kwargs,
) -> None:
    """
    Computes and saves the radar-grid shadow mask for a single burst.

    This function orchestrates the shadow computation by calling `map_shadow`.
    It checks for a completion file ('mask_radar.ok') to skip computation
    if `recompute` is False.

    Parameters
    ----------
    burst_id : str
        The identifier for the burst (e.g., "IW1_001").
    slc : xr.DataArray
        The reference Single Look Complex (SLC) data array
        for this burst.
    simulation : xr.Dataset
        The radar simulation dataset corresponding
        to this burst.
    simulation_ml : dict
        The multi-looked simulation data.
    multiburst : bool
        Flag indicating if processing is per-burst.
    wdir_path_static : str
        Path to the static working directory.
    eps : float, optional
        Epsilon value used in shadow computation.
        Defaults to 1e-4.
    threshold : float, optional
        Threshold value used in geocoded mask computation.
        Defaults to 0.3.
    enable_dask : bool, optional
        Flag to enable Dask for sub-tasks.
        Defaults to True.
    recompute : bool, optional
        Flag to force re-computation.
        Defaults to False.

    Returns
    -------
    None
        This function saves the shadow mask to disk.
    """
    # Define the data path based on whether it's multiburst or single data mode
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{wdir_path_static}/{path}")

    # Define the path for the completion file
    mask_path = f"{wdir_path_static}/processed_data/{path}/mask_radar.ok"

    # Check if computation should be skipped
    if (exists(mask_path, **kwargs) is False) or recompute:
        # If recomputing, delete the old completion file
        delete(mask_path, allow_non_existent=True, **kwargs)

        # Call the mapping function to compute the shadow
        map_shadow(
            slc=slc,
            simulation=simulation,
            simulation_ml=simulation_ml,
            data_path=data_path,
            eps=eps,
            enable_dask=enable_dask,
            **kwargs,
        )
        map_shadow_projection(
            slc=slc,
            data_path_static=data_path,
            threshold=threshold,
            enable_dask=enable_dask,
            **kwargs,
        )
        # Create the 'processed_data' directory if it doesn't exist
        makedirs(normpath(f"{wdir_path_static}/processed_data/{path}"), exist_ok=True, **kwargs)
        # "Touch" the completion file to signify success
        touch(mask_path, **kwargs)
    else:
        # Skip computation if completion file exists and recompute is False
        logger.info(f"Found existing mask files for burst {burst_id}")
    return


@timeit
def stack_mask(
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict[str, xr.Dataset],
    working_directory_static: str,
    reference_date: str,
    multiburst: bool = True,
    recompute: bool = False,
    **kwargs,
) -> None:
    """
    Orchestrates the shadow mask computation for a stack of SLCs.

    This function prepares arguments for each burst (or a single data block)
    and maps the `mask_shadow_single_burst` function across them.

    Parameters
    ----------
    slcs : xr.DataTree
        An xarray DataTree containing the SLC burst data.
    simulation_tree : xr.Dataset
        An xarray Dataset (or DataTree-like
        structure) containing the simulation results.
    simulation_ml_tree : dict[str, xr.Dataset]
        A dictionary containing the
        multi-looked simulation results.
    working_directory_static : str
        Path to the static working directory
        for saving persistent results.
    reference_date : str
        The acquisition date string used to select the
        reference SLC.
    multiburst : bool, optional
        Flag indicating whether to process data as
        individual bursts. Defaults to True.
    recompute : bool, optional
        Flag to force re-computation even if
        completion files exist. Defaults to False.

    Returns
    -------
    None
        This function executes the computation and saves results to disk.
    """
    args = []
    # Prepare arguments for each burst if in multiburst mode
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        v = v.orig[reference_date].measurements.slc
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        a = (k, v, s, s_ml, multiburst)
        in_proj = s.rio.crs
        args.append(a)

    # Transpose the list of tuples for the map function
    args = list(zip(*args))

    # Map the `mask_shadow_single_burst` function across all argument sets
    map(
        mask_shadow_single_burst,
        args,
        kwargs=dict(wdir_path_static=working_directory_static, recompute=recompute, **kwargs),
        combine=False,  # Results are written to disk
        enable_dask=kwargs.get("enable_dask", True),
        Plugin=GeoEnvPlugin(in_proj=in_proj, out_proj="EPSG:4978"),
    )

    return


def map_rasterization_and_rtc_ap(
    simulation: xr.Dataset,
    global_values: xr.Dataset,
    data_path_static: str,
    data_path_tmp: str,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Orchestrate the rasterization and Radiometric Terrain Correction (RTC) using area projection.

    This function manages the parallel execution of the core rasterization and RTC
    algorithms. It reads global parameters, computes overlapping chunks, initializes
    output datasets, and then maps the `compute_rasterization_and_rtc_ap` worker
    function over blocks of the simulation data. Finally, it triggers the conversion
    of sparse results back to dense output grids.

    Parameters
    ----------
    simulation : xr.Dataset
        The simulation dataset containing radar geometry mapped
        onto the geographic grid.
    data_path_static : str
        Path to the directory for storing persistent static outputs
        (like the final `radar_grid` and `geo_grid`).
    data_path_tmp : str
        Path to the directory for storing temporary intermediate
        files (like sparse matrices).
    enable_dask : bool, optional
        If True, use Dask for parallel computation. Defaults to True.

    Returns
    -------
    None
        Results are written to Zarr stores specified by the paths.
    """
    # --- Clean Up Previous Results ---
    # Delete temporary sparse matrix directory and old weight matrix if they exist.
    for f, t in zip([f"{data_path_static}/w", f"{data_path_tmp}/sparse_dir"], ["static", "dynamic"]):
        try:
            delete(
                path=normpath(f),
                recursive=True,
                allow_non_existent=True,
                storage_options=get_storage_options(kwargs, key=f"storage_options_{t}"),
            )
        except botocore.exceptions.NoCredentialsError as e:
            raise PermissionError(
                f"S3 credentials not found or invalid. Cannot access {f}. "
                "Please provide 'storage_options' or set environment variables (AWS_ACCESS_KEY_ID, etc.)."
            ) from e
    # --- Load Global Parameters ---
    dim_x_global, dim_y_global = global_values.geo_xsize.values[0], global_values.geo_ysize.values[0]
    xmin_global, ymin_global = global_values.radar_xmin.values[0], global_values.radar_ymin.values[0]
    xsize_global, chunk_length = global_values.radar_xsize.values[0], global_values.chunk_length.values[0]
    radar_grid_pixel_area, bbox = global_values.radar_grid_pixel_area.values[0], global_values.bbox.values
    chunks_slc = global_values.chunks_slc.values
    az = global_values.az_coords.values
    rg = global_values.rg_coords.values
    crs = global_values.rio.crs

    # --- Prepare for Parallel Execution ---
    # Compute overlapping chunks for the simulation grid.
    chunks, x, y = compute_chunks(dict(simulation.chunksizes)), simulation.x.values, simulation.y.values
    # Determine winding order based on grid orientation.
    y_sign, x_sign = (1 if y[0] < y[1] else 0, 1 if x[0] < x[1] else 0)
    # Define output coordinates at pixel centers.
    normal_order, x, y = np.logical_xor(y_sign, x_sign), (x[0:-1] + x[1:]) / 2, (y[0:-1] + y[1:]) / 2
    # --- Initialize Output Datasets ---
    # Initialize the empty Zarr store for the final geographic grid outputs (lia, anf, nlooks, masks).
    data_vars = ["lia", "anf", "nlooks", "mask"]

    save_empty_dataset(
        data_vars=data_vars,
        dims={k: ["y", "x"] for k in data_vars},
        coords={"y": y, "x": x},
        crs=crs,
        shape={k: [dim_y_global, dim_x_global] for k in data_vars},
        chunks={"y": chunk_length, "x": chunk_length},
        dtype={k: "float32" if k not in ["mask"] else "int32" for k in data_vars},
        skip_read=True,
        data_path=f"{data_path_static}/geo_grid",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )
    # Get the resolution in meters for RTC calculation.
    resolution = get_resolution_in_meters(simulation)

    # --- Map Rasterization and RTC Worker ---
    # Apply the core worker function `compute_rasterization_and_rtc_ap` to each chunk.
    # Write the geogrid dataset to the static path, placing it in the correct region
    store_mapping = {
        0: {
            "data_path": normpath(f"{data_path_static}/geo_grid"),
            "dims": ["y", "x"],
            "save_options": {
                "skip_read": True,
                "consolidated": False,
                "storage_options": get_storage_options(kwargs, "storage_options_static"),
            },
        },
        1: {
            "data_path": normpath(f"{data_path_static}/w"),
            "drop_coords": False,
            "drop_attrs": False,
            "save_options": {
                "consolidated": False,
                "mode": "a",
                "skip_read": True,
                "storage_options": get_storage_options(kwargs, "storage_options_static"),
            },
        },
        2: {
            "data_path": normpath(f"{data_path_tmp}/sparse_dir"),
            "drop_coords": False,
            "drop_attrs": False,
            "save_options": {
                "consolidated": True,
                "mode": "a",
                "skip_read": True,
                "storage_options": get_storage_options(kwargs, "storage_options_dynamic"),
            },
        },
    }

    def output_callback(region: dict[str, slice], kwargs: dict[str, Any]) -> dict[str, slice] | None:
        # Determine the chunk's row and column position in the global grid using the job_id
        row_number = kwargs["job_id"] // (np.ceil(dim_x_global / chunk_length).astype(int))
        column_number = kwargs["job_id"] % (np.ceil(dim_x_global / chunk_length).astype(int))

        # Define the geogrid region (slice) this chunk corresponds to, for writing output
        region["y"] = slice(
            row_number * chunk_length, row_number * chunk_length + region["y"].stop - region["y"].start - 1
        )
        region["x"] = slice(
            column_number * chunk_length, column_number * chunk_length + region["x"].stop - region["x"].start - 1
        )
        return region

    region_callback = {
        0: output_callback,
        1: lambda _, __: None,
        2: lambda _, __: None,
        3: lambda _, __: None,
    }
    map_blocks(
        func=blockwise_rasterization,
        obj=simulation,
        args=[
            xmin_global,
            ymin_global,
            xsize_global,
            dim_x_global,
            radar_grid_pixel_area,
            bbox,
            chunk_length,
            resolution,
        ],
        chunks=chunks,
        enable_dask=enable_dask,
        combine=False,
        store_mapping=store_mapping,
        region_callback=region_callback,
    )
    store_mapping = {
        0: {
            "data_path": normpath(f"{data_path_static}/geo_grid"),
            "dims": ["y", "x"],
            "save_options": {
                "skip_read": True,
                "consolidated": False,
                "storage_options": get_storage_options(kwargs, "storage_options_static"),
            },
        },
        1: {
            "data_path": normpath(f"{data_path_tmp}/sparse_dir/A_block"),
            "drop_coords": False,
            "drop_attrs": False,
            "save_options": {
                "consolidated": True,
                "mode": "a",
                "skip_read": True,
                "storage_options": get_storage_options(kwargs, "storage_options_dynamic"),
            },
        },
    }

    map_blocks(
        func=blockwise_rtc_ap,
        obj=simulation,
        args=[
            dim_x_global,
            radar_grid_pixel_area,
            normal_order,
            bbox,
            chunk_length,
            data_path_static,
        ],
        kwargs=kwargs,
        chunks=chunks,
        enable_dask=enable_dask,
        combine=False,
        store_mapping=store_mapping,
        region_callback=region_callback,
    )

    # --- Convert Sparse Results to Dense ---
    # Find all the intermediate sparse matrix files generated by the workers.
    files_A = glob(
        path=normpath(f"{data_path_tmp}/sparse_dir/A_block/*"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    files_l1 = glob(
        path=normpath(f"{data_path_tmp}/sparse_dir/layover_block_0/*"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    files_l2 = glob(
        path=normpath(f"{data_path_tmp}/sparse_dir/layover_block_1/*"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    files_A = [file for file in files_A if not file.endswith(".json")]
    files_l1 = [file for file in files_l1 if not file.endswith(".json")]
    files_l2 = [file for file in files_l2 if not file.endswith(".json")]
    # Initialize the empty Zarr store for the final dense radar grid outputs (anf, masks).
    data_vars = ["anf", "layover", "shadow"]
    data_path = f"{data_path_static}/radar_grid"
    A = save_empty_dataset(
        data_vars=data_vars,
        dims={k: ["azimuth_time", "slant_range_time"] for k in data_vars},
        shape={k: [bbox[3], bbox[2]] for k in data_vars},  # Use full radar grid dimensions
        coords={"azimuth_time": az, "slant_range_time": rg},
        chunks={"azimuth_time": chunks_slc[0], "slant_range_time": chunks_slc[1]},
        dtype={k: "bool" if k != "anf" else "float32" for k in data_vars},
        skip_read=False,
        data_path=data_path,
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )

    # Map the `sparse_to_dense` worker function over chunks of the empty output grid `A`.
    save_options = {
        "data_path": data_path,
        "save_options": {
            "consolidated": False,
            "skip_read": True,
            "storage_options": get_storage_options(kwargs, "storage_options_static"),
        },
    }
    map_blocks(
        sparse_to_dense,
        A,
        args=(files_A, files_l1, files_l2, data_path_static),
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,
        store_mapping={0: save_options},
    )
    return


def single_burst_geometry(
    burst_id: str,
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    chunk_length: int,
    wdir_path_static: str,
    wdir_path_tmp: str,
    recompute: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Runs the geometry computation (rasterization and RTC-AP) for a single
    burst or data block.

    This function initializes the simulation, runs the
    `map_rasterization_and_rtc_ap` process, and writes a 'area_projection.ok'
    file to mark completion.

    Parameters
    ----------
    burst_id : str
        The identifier for the burst (e.g., "IW1_001").
    slc : xr.DataArray
        The reference Single Look Complex (SLC) data array
        for this burst.
    simulation : xr.Dataset
        The radar simulation dataset corresponding
        to this burst.
    simulation_ml : dict
        The multi-looked simulation data.
    multiburst : bool
        Flag indicating if processing is per-burst.
    chunk_length : int
        The edge size for geogrid chunks.
    wdir_path_static : str
        Path to the static working directory.
    wdir_path_tmp : str
        Path to the temporary working directory.
    recompute : bool, optional
        Flag to force re-computation.
        Defaults to False.
    enable_dask : bool, optional
        Flag to enable Dask for sub-tasks
        (passed to `map_rasterization_and_rtc_ap`).
        Defaults to True.

    Returns
    -------
    None
        This function executes the computation and saves results to disk.
    """
    # Type check for the SLC input
    if not isinstance(slc, xr.DataArray):
        raise Exception(f"Expected xr.DataArray, got {type(slc)}")

    # Define paths based on whether it's multiburst or single data mode
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path_static = normpath(f"{wdir_path_static}/{path}")
    data_path_tmp = normpath(f"{wdir_path_tmp}/{path}")

    # Define the path for the completion file
    area_projection_path = normpath(f"{wdir_path_static}/processed_data/{path}/area_projection.ok")

    # Check if computation should be skipped
    if (
        exists(area_projection_path, storage_options=get_storage_options(kwargs, key="storage_options_static")) is False
    ) or recompute:
        # If recomputing, delete the old completion file
        delete(
            area_projection_path,
            allow_non_existent=True,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )

        # Initialize the simulation to get the relevant chunk and global values
        selection, global_values = initialize_simulation(
            slc=slc,
            simulation=simulation,
            simulation_ml=simulation_ml,
            chunk_length=chunk_length,
        )

        # Write the global values (grid dims, CRS, etc.) to disk
        write(
            global_values,
            normpath(f"{data_path_static}/global_values"),
            mode="w",
            skip_read=True,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )

        # Run the main rasterization and RTC-AP mapping process
        map_rasterization_and_rtc_ap(
            selection, global_values, data_path_static, data_path_tmp, enable_dask=enable_dask, **kwargs
        )

        # Create the 'processed_data' directory
        makedirs(
            normpath(f"{wdir_path_static}/processed_data/{path}"),
            exist_ok=True,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )
        # "Touch" the completion file to signify success
        touch(
            area_projection_path,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )
    else:
        # Skip computation if completion file exists and recompute is False
        logger.info(f"Found existing area_projection files for burst {burst_id}")
    return


def stack_geometry(
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict[str, xr.Dataset],
    working_directory_static: str,
    working_directory_tmp: str,
    reference_date: str,
    chunk_length: int = 1024,
    multiburst: bool = True,
    recompute: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Orchestrates the geometry calculation (rasterization and RTC-AP) for a
    stack of SLCs, processing either individual bursts or a merged dataset.

    This function prepares and maps the `single_burst_geometry` function
    across all bursts (if `multiburst` is True) or a single data block.

    Parameters
    ----------
    slcs : xr.DataTree
        An xarray DataTree containing the SLC burst data.
    simulation_tree : xr.Dataset
        An xarray Dataset (or DataTree-like
        structure accessed within) containing
        the simulation results corresponding
        to the SLCs.
    simulation_ml_tree : dict[str, xr.Dataset]
        A dictionary containing the
        multi-looked simulation results,
        keyed similarly to `slcs`
        (e.g., by 'bursts').
    working_directory_static : str
        Path to the static working directory
        for saving persistent results
        (e.g., weights, final geo_grid).
    working_directory_tmp : str
        Path to the temporary working directory
        for intermediate files.
    reference_date : str
        The acquisition date string used to select the
        reference SLC for geometry calculation.
    chunk_length : int, optional
        The edge size for geogrid chunks.
        Defaults to 1024.
    multiburst : bool, optional
        Flag indicating whether to process data as
        individual bursts. Defaults to True.
    recompute : bool, optional
        Flag to force re-computation even if
        intermediate files exist. Defaults to False.

    Returns
    -------
    None
        This function executes the computation and saves results to disk.
    """
    args = []
    # Prepare arguments for each burst if in multiburst mode
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        v = v.orig[reference_date].measurements.slc
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        a = (k, v, s, s_ml, multiburst)
        args.append(a)
        in_proj = s.rio.crs

    # Transpose the list of tuples for the map function
    args = list(zip(*args))

    # Map the `single_burst_geometry` function across all argument sets
    map(
        single_burst_geometry,
        args,
        kwargs=dict(
            chunk_length=chunk_length,
            wdir_path_static=working_directory_static,
            wdir_path_tmp=working_directory_tmp,
            recompute=recompute,
            enable_dask=enable_dask,
            **kwargs,
        ),
        Plugin=GeoEnvPlugin(in_proj=in_proj, out_proj="EPSG:4978"),
        combine=False,  # Results are written to disk, not combined
        enable_dask=enable_dask,
    )

    return


def blockwise_rtc_ap(
    simulation: xr.DataArray,
    dim_x_global: int,
    radar_grid_pixel_area: float,
    normal_order: bool,
    bbox: np.ndarray,
    chunk_length: int,
    data_path_static: str,
    job_id=None,
    **kwargs,
) -> tuple[xr.Dataset | None, xr.DataTree | None]:
    """
    Compute rasterization weights and the accumulated gamma-naught area (RTC-AP)
    for a block of the radar data simulation.

    This function performs triangulation on the geogrid, computes weights
    for mapping from radar grid to geogrid, and then calculates the
    Gamma Naught Area (GNA), local incidence angle (lia), and number of looks (nlooks).

    Parameters
    ----------
    simulation : xr.DataArray
        The radar data simulation chunk, containing
        coordinates (az_px, rg_px) and DEM height (h).
    data_path_static : str
        Path to the directory where static results
        (geo_grid, weights) will be saved.
    data_path_tmp : str
        Path to the directory where temporary results
        (sparse layover/area blocks) will be saved.
    xmin_global : int
        Global minimum of the radar x-coordinate (range).
    ymin_global : int
        Global minimum of the radar y-coordinate (azimuth).
    xsize_global : int
        Total radar grid size along the x-axis (range).
    dim_x_global : int
        Total geogrid size along the x-axis.
    radar_grid_pixel_area : float
        The area of a single radar grid pixel.
    normal_order : bool
        The order needed to compute the surface normal correctly.
    bbox : np.ndarray
        Bounding box coordinates of the full radar scene
        [min_rg, min_az, max_rg, max_az].
    chunk_length : int
        Size (dimension) of a square geogrid chunk.
    resolution : tuple[float, float]
        Output resolution (az_res, rg_res)
        used to calculate expected number of looks.
    job_id : int or None, optional
        Identifier for the job, used to determine the
        position of this chunk in the global grid.

    Returns
    -------
    tuple[xr.Dataset, xr.DataTree | None, xr.DataTree | None, xr.DataTree | None]
        A tuple containing the computed geogrid data (nlooks, anf, lia) and the temporary sparse area matrices.
    """
    so = get_storage_options(kwargs)
    # Get local geogrid dimensions (number of cells per row and per column)
    # The size is the number of points, so dimensions are size - 1
    dim_x, dim_y = simulation.x.size - 1, simulation.y.size - 1

    # Determine the chunk's row and column position in the global grid using the job_id
    row_number = job_id // (np.ceil(dim_x_global / chunk_length).astype(int))
    column_number = job_id % (np.ceil(dim_x_global / chunk_length).astype(int))

    # Load the sparse weights file corresponding to this chunk
    ablock_dt = None
    f = normpath(f"{data_path_static}/w/{row_number}.{column_number}")
    if exists(f, storage_options=so):
        # Read sparse matrix components (CSR format)
        w = read(f, as_type="xr.Dataset", consolidated=False, storage_options=so)
        # --- Start Radiometric Terrain Correction (RTC) Area Projection (AP) ---
        anf, lia, a_block = compute_rtc_ap(
            csr_matrix=w,
            simulation=simulation,
            radar_grid_pixel_area=radar_grid_pixel_area,
            normal_order=normal_order,
            bbox=bbox,
        )
        bbox_radar = np.array([bbox[3], 0, bbox[2], 0])
        ablock_dt = xr.DataTree(children={".".join([str(i) for i in bbox_radar]): xr.DataTree(dataset=a_block)})
    else:
        # If no weights were computed (e.g., empty chunk), create empty arrays
        anf = xr.DataArray(np.zeros((dim_y, dim_x), dtype="float32"), dims=["y", "x"])
        lia = xr.DataArray(np.zeros((dim_y, dim_x), dtype="float32"), dims=["y", "x"])

    # Combine geogrid results into a single dataset
    geo_data = xr.Dataset(data_vars={"anf": anf, "lia": lia})

    return geo_data, ablock_dt


def sparse_to_dense(
    A: xr.Dataset, files_A, files_l1, files_l2, data_path: str, chunk: dict = {}, **kwargs
) -> xr.Dataset:
    """
    Converts and accumulates sparse matrix blocks into a single dense xr.DataArray
    for a specified chunk of the radar grid.

    This function is intended to be used with a `map_blocks` operation. It reads
    multiple sparse files (Area, Layover1, Layover2), finds those that
    overlap with the target `chunk`, and sums them into a dense array.

    Parameters
    ----------
    A : xr.Dataset
        An empty xr.Dataset, required by the `map_blocks`
        framework to determine the `chunk` argument.
    files_A : list of str
        A list of file paths to the sparse 'A' (Area) blocks.
    files_l1 : list of str
        A list of file paths to the sparse 'l1' (layover) blocks.
    files_l2 : list of str
        A list of file paths to the sparse 'l2' (layover) blocks.
    data_path : str
        The directory where the resulting dense radar grid
        dataset will be saved.
    chunk : dict or None, optional
        The chunk definition (e.g., {'azimuth_time': slice(...),
        'slant_range_time': slice(...)}) associated with this job.

    Returns
    -------
    xr.Dataset
        The accumulated dense 'anf' and 'layover' arrays as an xr.Dataset.
    """
    # Extract the radar grid boundaries for this target dense chunk
    rg_min = chunk["slant_range_time"].start
    rg_max = chunk["slant_range_time"].stop
    az_min = chunk["azimuth_time"].start
    az_max = chunk["azimuth_time"].stop

    # Initialize empty dense arrays to accumulate results
    A_blocks = np.zeros((az_max - az_min, rg_max - rg_min), dtype="float32")
    l1_blocks = np.zeros((az_max - az_min, rg_max - rg_min), dtype="float32")
    l2_blocks = np.zeros((az_max - az_min, rg_max - rg_min), dtype="float32")

    # Loop over the three file lists and their corresponding destination arrays
    for list_files, dense_array in zip([files_A, files_l1, files_l2], [A_blocks, l1_blocks, l2_blocks]):
        # Iterate through each sparse file
        for f in list_files:
            file = normpath(f)
            # Parse the sparse block's bounding box from its file name
            # Format: .../{az_min}.{az_max}.{rg_min}.{rg_max}
            line = file.split("/")[-1].split(".")
            block_az_min = int(line[0])
            block_az_max = int(line[1])
            block_rg_min = int(line[2])
            block_rg_max = int(line[3])

            # Check if the sparse block overlaps with the target dense chunk
            if (
                (block_az_max >= az_min)
                & (block_az_min <= az_max)
                & (block_rg_max >= rg_min)
                & (block_rg_min <= rg_max)
            ):
                # If it overlaps, read the sparse data
                ds = read(
                    file,
                    consolidated=False,
                    storage_options=get_storage_options(kwargs, "storage_options_dynamic"),
                )
                data = ds.data.values
                indices = ds.indices.values
                indptr = ds.indptr.values

                # Call the kernel to efficiently update the dense array with sparse data
                # This handles adding values to the correct sub-region
                update_dense_array(
                    array=dense_array.ravel(),
                    data=data,
                    indices=indices,
                    indptr=indptr,
                    xmin=rg_min,
                    xmax=rg_max,
                    ymin=az_min,
                    ymax=az_max,
                )

    # Convert layover contribution counts to boolean masks
    # Any pixel with a contribution > 1 is considered layover
    l1_blocks = l1_blocks > 1 + 1e-7  # Use epsilon for float comparison
    l2_blocks = l2_blocks > 1 + 1e-7

    # Convert the dense ANF array to an xr.DataArray with correct dims
    A_blocks = xr.DataArray(A_blocks, dims=["azimuth_time", "slant_range_time"])
    # Combine the two layover masks
    layover_blocks = xr.DataArray(l1_blocks + l2_blocks, dims=["azimuth_time", "slant_range_time"])

    # Create the final dataset for the radar grid chunk
    radar_data = xr.Dataset(data_vars={"anf": A_blocks, "layover": layover_blocks})

    # Write the dense dataset to disk for the specified chunk region
    return radar_data


def blockwise_geocoded_mask(
    empty_data: xr.Dataset,
    chunk_length: int,
    xsize_global: int,
    data_path_static: str,
    threshold: float = 0.3,
    chunk: dict = {},
    **kwargs,
) -> xr.Dataset:
    """
    Computes the geocoded Gamma Naught for a single chunk.

    This is the worker function called by `map_blocks`. It loads the
    pre-computed sparse weights (`w`) for this chunk, reads the corresponding
    section of the SLC and radar-grid ANF, normalizes the SLC intensity by
    the ANF, performs the sparse matrix multiplication (geocoding), and
    writes the resulting geocoded chunk to the 'geo_grid' Zarr store.

    Parameters
    ----------
    empty_data : xr.Dataset
        The template chunk (used by `map_blocks`
        to determine dimensions).
    slc : xr.DataArray
        The *entire* SLC DataArray. This function will
        slice the required portion.
    chunk_length : int
        The edge size of a square geogrid chunk.
    xsize_global : int
        The total global size of the radar grid along
        the x-axis (range).
    data_path_static : str
        Path to static data (weights 'w', 'radar_grid').
    data_path_tmp : str
        Path to the output 'geo_grid' Zarr store.
    recompute : bool, optional
        If True, triggers re-computation of
        geogrid masks (layover, shadow).
        Defaults to False.
    threshold : float, optional
        Minimum ANF value; pixels below this
        are set to 0 to avoid division by zero.
        Defaults to 1e-3.
    chunk : dict or None, optional
        A dictionary injected by `map_blocks`
        defining the slices for this chunk.

    Returns
    -------
    xr.Dataset
        Geocoded mask data for this chunk.
    """
    so = get_storage_options(kwargs)
    # Get the geogrid dimensions for this chunk (e.g., 512x512)
    dim_x, dim_y = empty_data.x.size - 1, empty_data.y.size - 1

    # Determine the chunk's row/column index from its start position
    row_number, column_number = chunk["y"].start // chunk_length, chunk["x"].start // chunk_length

    # Load the sparse weights file corresponding to this chunk
    f = normpath(f"{data_path_static}/w/{row_number}.{column_number}")
    if exists(f, storage_options=so):
        # Read sparse matrix components (CSR format)
        ds = read(f, as_type="xr.Dataset", consolidated=False, storage_options=so)
        # Get the bounding box of the radar pixels this chunk depends on
        bbox_radar = ds.attrs["radar_bbox"]
        radar_data = read(
            normpath(f"{data_path_static}/radar_grid"),
            mask_and_scale=False,
            storage_options=so,
        ).isel(
            azimuth_time=slice(bbox_radar[0], bbox_radar[1] + 1),
            slant_range_time=slice(bbox_radar[2], bbox_radar[3] + 1),
        )
        geocoded_mask = compute_geocoded_mask(
            csr_matrix=ds,
            layover=radar_data.layover,
            shadow=radar_data.shadow,
            xsize_global=xsize_global,
            bbox_radar=bbox_radar,
            output_shape=(dim_y, dim_x),
            threshold=threshold,
        )
        geocoded_mask = xr.Dataset(data_vars={"mask": (["y", "x"], geocoded_mask)})
    else:
        # If no weights were provided, create an empty boolean mask
        size = {k: v.stop - v.start for k, v in chunk.items()}
        geocoded_mask = xr.Dataset(
            data_vars={"mask": (["y", "x"], np.zeros((size["y"] - 1, size["x"] - 1), dtype="int32"))}
        )

    return geocoded_mask


def blockwise_rasterization(
    simulation: xr.DataArray,
    xmin_global: int,
    ymin_global: int,
    xsize_global: int,
    dim_x_global: int,
    radar_grid_pixel_area: float,
    bbox: np.ndarray,
    chunk_length: int,
    resolution: tuple[float, float],
    job_id=None,
) -> tuple[xr.Dataset, xr.DataTree | None, xr.DataTree | None]:
    """
    Compute rasterization weights and the accumulated gamma-naught area (RTC-AP)
    for a block of the radar data simulation.

    This function performs triangulation on the geogrid, computes weights
    for mapping from radar grid to geogrid, and then calculates the
    Gamma Naught Area (GNA), local incidence angle (lia), and number of looks (nlooks).

    Parameters
    ----------
    simulation : xr.DataArray
        The radar data simulation chunk, containing
        coordinates (az_px, rg_px) and DEM height (h).
    data_path_static : str
        Path to the directory where static results
        (geo_grid, weights) will be saved.
    data_path_tmp : str
        Path to the directory where temporary results
        (sparse layover/area blocks) will be saved.
    xmin_global : int
        Global minimum of the radar x-coordinate (range).
    ymin_global : int
        Global minimum of the radar y-coordinate (azimuth).
    xsize_global : int
        Total radar grid size along the x-axis (range).
    dim_x_global : int
        Total geogrid size along the x-axis.
    radar_grid_pixel_area : float
        The area of a single radar grid pixel.
    normal_order : bool
        The order needed to compute the surface normal correctly.
    bbox : np.ndarray
        Bounding box coordinates of the full radar scene
        [min_rg, min_az, max_rg, max_az].
    chunk_length : int
        Size (dimension) of a square geogrid chunk.
    resolution : tuple[float, float]
        Output resolution (az_res, rg_res)
        used to calculate expected number of looks.
    job_id : int or None, optional
        Identifier for the job, used to determine the
        position of this chunk in the global grid.

    Returns
    -------
    tuple[xr.Dataset, xr.DataTree | None, xr.DataTree | None, xr.DataTree | None]
        A tuple containing the computed geogrid data (nlooks, anf, lia) and the temporary sparse area matrices.
    """

    # Determine the chunk's row and column position in the global grid using the job_id
    row_number = job_id // (np.ceil(dim_x_global / chunk_length).astype(int))
    column_number = job_id % (np.ceil(dim_x_global / chunk_length).astype(int))
    block_id = f"{row_number}.{column_number}"
    nlooks, w, layover = compute_rasterization(
        simulation=simulation,
        xmin_global=xmin_global,
        ymin_global=ymin_global,
        xsize_global=xsize_global,
        radar_grid_pixel_area=radar_grid_pixel_area,
        bbox=bbox,
        resolution=resolution,
        block_id=block_id,
    )
    if w is not None:
        w_dt = xr.DataTree(children={block_id: xr.DataTree(dataset=w)})
    if layover is not None:
        layover_dt = xr.DataTree(children={f"layover_block_{i}": xr.DataTree() for i in range(2)})
        for i, (layover_bbox, layover_matrix) in enumerate(layover):
            layover_dt[f"layover_block_{i}/{layover_bbox}"] = xr.DataTree(dataset=layover_matrix)

    # Combine geogrid results into a single dataset
    geo_data = xr.Dataset(data_vars={"nlooks": nlooks})

    return geo_data, w_dt, layover_dt
