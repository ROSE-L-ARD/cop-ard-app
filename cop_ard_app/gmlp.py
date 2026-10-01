# Copyright 2024 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.dem import compute_shape_and_off
from cop_ard_core.gmlp import get_radar_spacing, warp_cmlp
from cop_ard_core.griddata import interp_dataset, interp_dataset_scaled
from cop_ard_core.logger import logger, timeit
from cop_ard_core.simul import extract_simul_on_radar_chunk, get_geo_from_radar_bbox, get_radar_bbox
from cop_ard_core.warp import sinc

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.io import get_storage_options, normpath, read, save_empty_dataset

DIMS = ["polarization", "y", "x"]
LONG_DIMS = ["polarization", "past_reference_date", "y", "x"]
COORDS = ["polarization", "past_reference_date", "acquisition_date"]


@timeit
def stack_gmlp(
    cmlp_tree: xr.DataTree,
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict,
    spacing: dict,
    reference_id: str,
    working_directory_static: str,
    working_directory_tmp: str,
    enable_dask: bool = True,
    chunks: dict | None = None,
    **kwargs,
):
    """
    Geocode a stack of coregistered SLCs in parallel.

    This function is the main entry point for the geocoding phase. It prepares
    the necessary arguments for each burst (or a single scene) and then maps the
    `single_geocode_phase` worker function across a Dask cluster to perform the
    geocoding in parallel. Finally, it assembles the geocoded results into a
    new DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The input DataTree containing the coregistered SLC stack (`coreg` group).
    simulation_tree : xr.DataTree
        A DataTree containing the high-resolution simulation data
        (radar geometry on the DEM grid) for each burst.
    simulation_ml_tree : dict
        A dictionary with the downsampled ("multilooked") simulation
        data, used for quickly finding overlaps.
    working_directory_static : str
        The path to a directory for storing persistent static data.
    working_directory_tmp : str
        The path to a directory for temporary processing data.
    gamma_flattening : bool, optional
        If True, apply a radiometric terrain correction to produce
        gamma nought backscatter.
    do_reramp : bool, optional
        If True, re-apply the phase ramp to the geocoded data.
    client_options : Dict[str, str], optional
        A dictionary of options for configuring the Dask client.
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.
    chunks : dict, optional
        Chunksizes to be used in GMLP processing.
        Defaults to {"acquisition_date": 1, "past_reference_date": 1, "polarization": 1, "y": 2048, "x": 2048}
    **kwargs
        Additional keyword arguments passed to the processing functions,
        such as `warp_kernel_shape`.

    Returns
    -------
    xr.DataTree
        A new DataTree containing the geocoded SLC (GSLC) data for each burst.
    """
    chunks = chunks or {"acquisition_date": 1, "past_reference_date": 1, "polarization": 1, "y": 2048, "x": 2048}
    multiburst = "bursts" in cmlp_tree
    args = []
    # --- Prepare Arguments for Parallel Mapping ---
    for k, v in (cmlp_tree.bursts if multiburst else {"data": cmlp_tree}).items():
        slc_node = slcs["bursts"][k] if multiburst else slcs
        conditions = slc_node["coreg/conditions"]
        etad = conditions.get("ETAD")  # Safely returns None if ETAD isn't present

        # Extract only 'phase' variables
        measurements_ds = slc_node["coreg/measurements"].to_dataset()
        phase_vars = [var for var in list(measurements_ds.data_vars) if "phase" in str(var)]
        phase = measurements_ds[phase_vars]
        dates_ids = {k: v for k, v in zip(phase.acquisition_date.values, phase.acquisition_id.values)}
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        for date, cmlp in v.cmlp.items():
            id = dates_ids[date]
            a = (
                k,
                date,
                cmlp,
                s,
                s_ml,
                etad.to_dataset().sel(acquisition_id=[id]) if etad is not None else None,
                phase.sel(acquisition_id=id),
            )
            args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Geocoding in Parallel ---
    # Map the `single_geocode_phase` worker function over the list of arguments.
    results = cast(
        list,
        map(
            single_gmlp,
            args,
            kwargs=dict(
                multiburst=multiburst,
                reference_id=reference_id,
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                spacing=spacing,
                chunks=chunks,
                enable_dask=enable_dask,
                **kwargs,
            ),
            enable_dask=enable_dask,
            combine=False,
        ),
    )

    # --- Assemble Results ---
    gmlp_tree = xr.DataTree()
    # Reconstruct the DataTree from the list of results returned by the workers.
    for burst_id, date, gmlp in results:
        if multiburst:
            gmlp_tree[f"bursts/{burst_id}/{date}"] = gmlp
        else:
            gmlp_tree[date] = gmlp
    return gmlp_tree


def single_gmlp(
    burst_id: str,
    date: str,
    cmlp: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    etad: xr.Dataset | None,
    phase: xr.Dataset,
    multiburst: bool,
    spacing: dict,
    reference_id: str,
    working_directory_static: str,
    working_directory_tmp: str,
    chunks: dict,
    enable_dask: bool = True,
    **kwargs,
):
    """
    Execute the end-to-end geocoding workflow for a single burst.

    This worker function prepares the interpolation kernel and calls `map_geocode_phase`
    to perform the core geocoding computation. It then merges the result with
    other geometric grid data (like incidence angles) to create the final
    geocoded product for the burst.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst being processed.
    cslc : xr.DataTree
        The input DataTree for the coregistered burst.
    simulation : xr.Dataset
        The high-resolution simulation data for this burst.
    simulation_ml : dict
        The multilooked simulation data for this burst.
    multiburst : bool
        A boolean flag for multiburst mode.
    working_directory_static : str
        The path for storing persistent static data.
    working_directory_tmp : str
        The path for temporary processing data.
    gamma_flattening : bool
        If True, apply radiometric terrain correction.
    do_reramp : bool
        If True, re-apply the phase ramp.
    client_options : dict, optional
        A dictionary of options for the Dask client.
    enable_dask : bool, optional
        If True, enable Dask for sub-tasks.
    **kwargs
        Additional keyword arguments for the warping function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the final geocoded xarray.Dataset for that burst.
    """
    # Define the output paths for this burst.
    path = f"bursts/{burst_id}" if multiburst else f"data/{reference_id}"
    data_path_tmp = normpath(f"{working_directory_tmp}/{path}")
    data_path_static = normpath(f"{working_directory_static}/{path}")

    # --- Prepare Interpolation Kernel ---
    # Get warp parameters from kwargs or use default values.
    ov_fact = kwargs.get("warp_ov_fact", 100_000)
    kernel_shape = kwargs.get("warp_kernel_shape", np.array([11, 11]))
    if not (isinstance(kernel_shape, np.ndarray)):
        kernel_shape = np.array(kernel_shape)
    kaiser_beta = kwargs.get("warp_kaiser_beta", 6)

    # Pre-compute the sinc kernels for x and y directions.
    sinc_x = sinc(kernel_shape[1], ov_fact, kaiser_beta)
    sinc_y = sinc(kernel_shape[0], ov_fact, kaiser_beta)

    kernel = {
        "kernel_shape": kernel_shape,
        "ov_fact": ov_fact,
        "kaiser_beta": kaiser_beta,
        "sinc_x": sinc_x,
        "sinc_y": sinc_y,
    }

    # --- Run Core Geocoding ---
    # This call orchestrates the block-wise warping from slant range to ground range.
    result = map_geocoding_cmlp(
        burst_id,
        date,
        cmlp,
        simulation,
        simulation_ml,
        etad,
        phase,
        data_path_static,
        data_path_tmp,
        kernel,
        spacing,
        chunks,
        enable_dask,
        **kwargs,
    )

    # --- Merge with Geometric Data ---
    # Read the pre-computed geometric grid data (e.g., LIA, anf).
    return result


def map_geocoding_cmlp(
    burst_id: str,
    date: str,
    cmlp: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    etad: xr.Dataset | None,
    phase: xr.Dataset,
    data_path_static: str,
    data_path_tmp: str,
    kernel: dict,
    spacing: dict,
    chunks: dict,
    enable_dask: bool = True,
    border: int = 10,
    **kwargs,
) -> tuple[str, str, xr.DataTree]:
    """
    Orchestrate the block-wise geocoding of a single coregistered SLC.

    This function initializes an empty Zarr store for the final Geocoded SLC (GSLC)
    product and then uses `map_blocks` to apply the `warp_gslc` function to
    chunks of the simulation grid.

    Parameters
    ----------
    cslc : xr.DataTree
        The input DataTree for the coregistered burst.
    simulation : xr.Dataset
        The high-resolution simulation data for this burst.
    simulation_ml : dict
        The multilooked simulation data.
    data_path_static : str
        Path for persistent static data.
    data_path_tmp : str
        Path for temporary processing data.
    coreg_path : str
        Path to the source coregistered data.
    kernel : dict
        A dictionary with parameters for the sinc interpolation kernel.
    gamma_flattening : bool
        If True, apply radiometric terrain correction.
    do_reramp : bool, optional
        If True, re-apply the phase ramp. Defaults to True.
    client_options : dict, optional
        A dictionary of options for the Dask client. Defaults to {"type": "LOCAL"}.
    enable_dask : bool, optional
        If True, use Dask for sub-tasks. Defaults to True.
    skip_read : bool, optional
        If True, do not read and return the final dataset. Defaults to True.

    Returns
    -------
    xr.DataTree or None
        The geocoded Dataset if `skip_read` is False, otherwise None.
    """

    # --- Prepare Simulation Subset ---
    # Calculate grid spacing for margin calculation.
    product = next(iter(cmlp.values()))
    azimuth_time = product.azimuth_time
    slant_range_time = product.slant_range_time

    radar_spacing_cmlp = get_radar_spacing(azimuth_time, slant_range_time)
    radar_spacing_slc = get_radar_spacing(phase.azimuth_time, phase.slant_range_time)
    margin = {e: radar_spacing_cmlp[e] * border for e in ["azimuth_time", "slant_range_time"]}
    bbox_radar = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(simulation_ml, bbox_radar)
    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    selection = (
        (
            extract_simul_on_radar_chunk(
                simulation,
                simulation.acquisition_id.values[0],
                bbox_radar,
                bbox_geo,
            )
        )
        .squeeze()
        .drop_vars(["axis", "dem_distance", "acquisition_id", "h"])
    )
    # Define the output coordinates at pixel centers.
    shape, off = compute_shape_and_off(selection, spacing)
    # Create the coordinate arrays for the output grid.
    x, y = [np.linspace(off[i], off[i] + (shape[i] - 1) * spacing[i], shape[i]) for i in ["x", "y"]]

    # --- Initialize Output GSLC Dataset ---
    # Create the empty Zarr store for the Geocoded SLC.
    cmlp_ds = cmlp.to_dataset()
    mask = read(
        f"{data_path_static}/geo_grid", as_type="xr.Dataset", storage_options=get_storage_options(kwargs, "static")
    ).mask.to_dataset()
    all_vars = {**cmlp_ds.data_vars, **phase.data_vars, **mask.data_vars}
    logger.debug(f"all_vars: {list(all_vars)}")

    # 2. Assegnazione dimensioni pulita
    dims = {}
    for var_name, da in all_vars.items():
        if (var_name in (phase.data_vars)) or (var_name in (mask.data_vars)):
            dims[var_name] = ["y", "x"]
        elif "past_reference_date" in da.dims:
            dims[var_name] = LONG_DIMS
        else:
            dims[var_name] = DIMS

    # 3. Shape e Dtype trasparenti per 2D, 3D o 4D (zero KeyError)
    dataset_shape = {k: [*da.shape[:-2], y.size, x.size] for k, da in all_vars.items()}
    dtype = {k: da.dtype.name for k, da in all_vars.items()}
    dtype["mask"] = np.int8

    # 4. Coords & Chunks
    coords = {k: v.values for k, v in cmlp_ds.coords.items() if k in COORDS}
    coords["acquisition_id"] = phase.acquisition_id.values
    coords.update({"y": y, "x": x})

    ch = {k: chunks[k] for k in cmlp_ds.chunksizes if k in LONG_DIMS}
    ch.update({"y": chunks["y"], "x": chunks["x"]})

    image = save_empty_dataset(
        data_vars=list(all_vars),
        dims=dims,
        shape=dataset_shape,
        chunks=ch,
        dtype=dtype,
        coords=coords,
        crs=selection.rio.crs.to_string(),
        data_path=normpath(f"{data_path_tmp}/gmlp/{date}"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        skip_read=False,
    )
    logger.debug(f"image: {image}")
    image = image.phase.isel({k: 0 for k in image.phase.dims if k not in ["y", "x"]})
    # --- Map the Geocoding Worker Function ---
    # This call applies the `warp_cmlp` function to each block of the simulation grid.
    save_options = {
        0: {
            "data_path": normpath(f"{data_path_tmp}/gmlp/{date}"),
            "save_options": {
                "consolidated": False,
                "storage_options": get_storage_options(kwargs, key="storage_options_dynamic"),
            },
        }
    }
    map_blocks(
        blockwise_interp_cmlp,
        image,
        args=(selection, cmlp_ds, phase, mask, kernel),
        kwargs=kwargs
        | {
            "dtaz_cmlp": radar_spacing_cmlp["azimuth_time"].astype(float),
            "dtrg_cmlp": radar_spacing_cmlp["slant_range_time"],
            "dtaz_slc": radar_spacing_slc["azimuth_time"].astype(float),
            "dtrg_slc": radar_spacing_slc["slant_range_time"],
        },
        combine=False,
        enable_dask=enable_dask,
        store_mapping=save_options,
    )
    gmlp = xr.DataTree()
    gmlp["measurements"] = read(
        normpath(f"{data_path_tmp}/gmlp/{date}"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        as_type="xr.Dataset",
    )
    gmlp = update_conditions_gmlp(gmlp, simulation, etad)

    return (burst_id, date, gmlp)


def update_conditions_gmlp(
    gmlp: xr.DataTree,
    simulation: xr.Dataset,
    etad: xr.Dataset | None = None,
    grid_size: tuple[int, int] = (200, 200),
) -> xr.DataTree:
    """Interpolate look vectors and optional ETAD timing corrections onto a coarse (y, x) ground grid

    and assign them to the DataTree conditions.

    Parameters
    ----------
    gmlp : xr.DataTree
        Target DataTree container storing processed SAR conditions.
    simulation : xr.Dataset
        Static simulation dataset containing spatial coordinates ('y', 'x'),
        'dem_distance', 'azimuth_time', and 'slant_range_time'.
    etad : xr.Dataset or None, optional
        Extended Timing and Atmosphere Correction (ETAD) dataset defined on
        radar time coordinates.
    grid_size : Tuple[int, int], default=(200, 200)
        Target resolution (y_size, x_size) for the interpolated coarse grid.

    Returns
    -------
    xr.DataTree
        Updated DataTree with populated 'conditions/look_vector' and optional
        'conditions/etad' nodes.
    """
    # --- 1. Define Coarse Spatial Grid (y, x) ---
    y = simulation.y
    x = simulation.x

    idx_y = np.linspace(0, len(y) - 1, grid_size[0], dtype=int)
    idx_x = np.linspace(0, len(x) - 1, grid_size[1], dtype=int)
    y_sampled = y[idx_y]
    x_sampled = x[idx_x]

    target_coords = {"y": y_sampled, "x": x_sampled}
    y_grid, x_grid = np.meshgrid(y_sampled.values, x_sampled.values, indexing="ij")
    xi_spatial = (y_grid, x_grid)
    points_spatial = (y.values, x.values)

    # --- 2. Compute and Interpolate Unit Look Vector ---
    dem_distance = simulation.dem_distance.squeeze()  # shape: (y, x, 3)
    norm = np.linalg.norm(dem_distance.values, ord=2, axis=-1, keepdims=True)

    # Reshape look vector to (3, y, x)
    look_vector_arr = np.moveaxis(dem_distance.values / norm, -1, 0)
    ds_look = xr.Dataset(
        data_vars={"look_vector": (("axis", "y", "x"), look_vector_arr)},
        coords={"axis": [0, 1, 2]},
    )

    # Interpolate look vectors directly on 1D rectilinear spatial coordinates
    ds_look_coarse = interp_dataset_scaled(
        data=ds_look,
        xi=xi_spatial,
        common_dims=("y", "x"),
        points=points_spatial,
        target_coords=target_coords,
        method="linear",
    )

    # Re-normalize vector magnitudes back to unit norm after interpolation
    look_vals = ds_look_coarse["look_vector"].values  # shape: (3, ny, nx)
    norm_coarse = np.linalg.norm(look_vals, ord=2, axis=0, keepdims=True)
    norm_coarse[norm_coarse == 0] = 1.0  # Prevent potential division by zero
    ds_look_coarse["look_vector"][...] = look_vals / norm_coarse

    gmlp["conditions/look_vector"] = ds_look_coarse.astype("float64")

    # --- 3. Interpolate ETAD Corrections (Radar Time -> Coarse Geo Grid) ---
    if etad is not None and len(etad.data_vars) > 0:
        # Reference origin timestamp for high-precision floating point conversion
        t0 = simulation.azimuth_time.min().values

        # Convert simulation radar times to relative nanoseconds (float64)
        sim_az_float = (simulation.azimuth_time - t0).astype("timedelta64[ns]").astype(float)
        sim_rg_float = simulation.slant_range_time.astype(float)

        ds_sim_times = xr.Dataset(
            data_vars={
                "azimuth_time": sim_az_float,
                "slant_range_time": sim_rg_float,
            }
        )

        # Interpolate simulation radar times onto the coarse (y, x) grid
        ds_times_coarse = interp_dataset(
            data=ds_sim_times,
            xi=xi_spatial,
            common_dims=("y", "x"),
            points=points_spatial,
            target_coords=target_coords,
            method="linear",
        )

        # Convert native ETAD time coordinates to relative nanoseconds (float64)
        etad_az_float = (etad.azimuth_time.values - t0).astype("timedelta64[ns]").astype(float)
        etad_rg_float = etad.slant_range_time.values.astype(float)
        points_etad = (etad_az_float, etad_rg_float)

        # Extract 2D sampled radar times to use as query coordinates for ETAD
        az_query = ds_times_coarse.azimuth_time.squeeze().values
        rg_query = ds_times_coarse.slant_range_time.squeeze().values
        xi_etad = (az_query, rg_query)
        etad_interpolated_list = []

        for acq_id in etad.acquisition_id.values:
            etad_sub = etad.sel(acquisition_id=acq_id)
            interp_sub = interp_dataset_scaled(
                data=etad_sub,
                xi=xi_etad,
                common_dims=("azimuth_time", "slant_range_time"),
                points=points_etad,
                rename={"azimuth_time": "y", "slant_range_time": "x"},
                target_coords={"y": y_sampled, "x": x_sampled},
                method="cubic",
            )

            interp_sub = interp_sub.expand_dims("acquisition_id").assign_coords(acquisition_id=[acq_id])
            etad_interpolated_list.append(interp_sub)

        etad_yx = (
            xr.concat(etad_interpolated_list, dim="acquisition_id")
            .astype("float32")
            .chunk({"acquisition_id": 1, "y": -1, "x": -1})
        )

        gmlp["conditions/etad"] = etad_yx

    return gmlp


def blockwise_interp_cmlp(
    image: xr.DataArray,
    simulation: xr.Dataset,
    cmlp: xr.Dataset,
    phase: xr.Dataset,
    mask: xr.Dataset,
    kernel: dict,
    dtaz_cmlp: float,
    dtrg_cmlp: float,
    dtaz_slc: float,
    dtrg_slc: float,
    margin: int = 5,
    method: str = "linear",
    chunk: dict | None = None,
    **kwargs,
) -> xr.Dataset:
    """Warp coregistered SLC and phase data to a ground-projected geographical grid.

    This function performs blockwise geocoding by projecting radar-space
    coordinates (slant range and azimuth times) onto a regular output spatial
    grid (y, x). Multilooked CMLP variables are resampled using high-precision
    sinc interpolation, while full-resolution phase model variables are
    interpolated using scaled grid interpolation (`interp_dataset_scaled`).

    Parameters
    ----------
    image : xr.DataArray
        Reference target image block defining output spatial dimensions (y, x)
        and geotransform resolution.
    simulation : xr.Dataset
        Simulation dataset mapping output (y, x) grid coordinates to absolute
        radar coordinates (`azimuth_time`, `slant_range_time`).
    cmlp : xr.Dataset
        Coregistered multilooked SLC dataset containing SAR data variables
        (e.g., `phase`, `short_term_coh`).
    phase : xr.Dataset
        Full-resolution phase model dataset (e.g., `phase_model`).
    mask : xr.Dataset, optional
        Quality mask dataset in ground dimensions. Defaults to None.
    kernel : dict, optional
        Configuration dictionary containing parameters for the sinc
        interpolator (e.g., `kernel_shape`).
    dtaz_cmlp : float
        Azimuth sampling step for CMLP data in nanoseconds.
    dtrg_cmlp : float
        Slant range sampling step for CMLP data in seconds.
    dtaz_slc : float
        Azimuth sampling step for full-resolution SLC data in nanoseconds.
    dtrg_slc : float
        Slant range sampling step for full-resolution SLC data in seconds.
    margin : int, optional
        Bounding box expansion margin in pixels for simulation slicing.
        Defaults to 5.
    method : str, optional
        Interpolation method for spatial grid coordinates ('linear', 'cubic').
        Defaults to "linear".
    chunk : dict, optional
        Dask chunking specifications if applicable. Defaults to None.
    **kwargs
        Additional keyword arguments passed to underlying functions.

    Returns
    -------
    xr.Dataset
        Geocoded output dataset aligned to the target (y, x) spatial grid.
    """
    chunk = chunk or {}
    shape = image.shape

    # --- 1. Simulation Bounding Box & Subsetting ---
    # Extract pixel resolution from target image (ground space)
    spacing_out = np.abs(image.rio.resolution(recalc=True))

    xmin_sim = image.x.min().values - margin * spacing_out[0]
    xmax_sim = image.x.max().values + margin * spacing_out[0]
    ymin_sim = image.y.min().values - margin * spacing_out[1]
    ymax_sim = image.y.max().values + margin * spacing_out[1]

    # Check actual axis direction (ascending vs. descending) to prevent empty slices
    sim_x_asc = simulation.x.values[0] < simulation.x.values[-1]
    sim_y_asc = simulation.y.values[0] < simulation.y.values[-1]

    xslice = slice(xmin_sim, xmax_sim) if sim_x_asc else slice(xmax_sim, xmin_sim)
    yslice = slice(ymin_sim, ymax_sim) if sim_y_asc else slice(ymax_sim, ymin_sim)

    simulation_sub = simulation.sel(x=xslice, y=yslice)
    x_grid, y_grid = np.meshgrid(image.x, image.y)

    # --- 2. High-Precision Time Conversion (Sub-nanosecond accuracy) ---
    # Convert absolute azimuth timestamps into relative float seconds from t0
    t0 = simulation_sub.azimuth_time.min().values
    az_sec = (simulation_sub.azimuth_time.values - t0).astype("timedelta64[ns]").astype(float) / 1e9
    simulation_sub["azimuth_time_sec"] = (
        simulation_sub.azimuth_time.dims,
        az_sec,
    )

    # --- 3. Spatial Coordinate Interpolation onto Target Grid ---
    simul_out = interp_dataset(
        data=simulation_sub[["azimuth_time_sec", "slant_range_time"]],
        xi=(y_grid, x_grid),
        common_dims=("y", "x"),
        method=method,
        fill_value=np.nan,
    )

    az_sec_interp = simul_out.azimuth_time_sec.values
    rg_time_interp = simul_out.slant_range_time.values

    # --- 4. Convert Radar Times to Relative Pixel Coordinates ---
    dtaz_cmlp_sec = dtaz_cmlp / 1e9
    dtaz_slc_sec = dtaz_slc / 1e9

    offset_az_cmlp = (t0 - cmlp.azimuth_time.min().values).astype("timedelta64[ns]").astype(float) / 1e9
    offset_az_phase = (t0 - phase.azimuth_time.min().values).astype("timedelta64[ns]").astype(float) / 1e9

    # Floating-point pixel coordinates for CMLP (multilooked) grid
    az_cmlp = (offset_az_cmlp + az_sec_interp) / dtaz_cmlp_sec
    rg_cmlp = (rg_time_interp - cmlp.slant_range_time.min().values) / dtrg_cmlp

    # Floating-point pixel coordinates for Phase (full-resolution) grid
    az_phase = (offset_az_phase + az_sec_interp) / dtaz_slc_sec
    rg_phase = (rg_time_interp - phase.slant_range_time.min().values) / dtrg_slc

    # --- 5. Valid Pixel Masking (Excluding Interpolation Kernel Margins) ---
    kernel_shape = np.array(kernel.get("kernel_shape", [2, 2]))
    kernel_margin = kernel_shape // 2 + 1

    valid_mask = (
        (rg_cmlp >= kernel_margin[1])
        & (az_cmlp >= kernel_margin[0])
        & (rg_cmlp < (cmlp.slant_range_time.size - kernel_margin[1]))
        & (az_cmlp < (cmlp.azimuth_time.size - kernel_margin[0]))
    )
    ind = np.where(valid_mask)

    # --- 6. Initialize Output Dataset Container ---
    ds = xr.Dataset()
    for name, var in cmlp.data_vars.items():
        var_shape = [*var.shape[:-2], shape[0], shape[1]]
        var_dims = [*var.dims[:-2], "y", "x"]
        ds[name] = xr.DataArray(np.zeros(var_shape, dtype=var.dtype), dims=var_dims)

    for name, var in phase.data_vars.items():
        var_shape = [shape[0], shape[1]]
        var_dims = ["y", "x"]
        ds[name] = xr.DataArray(np.zeros(var_shape, dtype=var.dtype), dims=var_dims)

    # --- 7. Execution of Warping & Interpolation ---
    if ind[0].size > 0:
        # A. Resample CMLP variables using Sinc Warping
        x_pts, y_pts = rg_cmlp[ind], az_cmlp[ind]

        # Safe bounding box cropping using integer pixel indices
        xmin = max(0, int(np.floor(x_pts.min() - kernel_margin[1])))
        xmax = min(
            cmlp.slant_range_time.size,
            int(np.ceil(x_pts.max() + kernel_margin[1])),
        )
        ymin = max(0, int(np.floor(y_pts.min() - kernel_margin[0])))
        ymax = min(
            cmlp.azimuth_time.size,
            int(np.ceil(y_pts.max() + kernel_margin[0])),
        )

        x_rel = x_pts - xmin
        y_rel = y_pts - ymin

        cmlp_sub = cmlp.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))

        # Flatten 2D target indices for 1D view indexing in warp_cmlp
        flat_ind = ind[0] * shape[1] + ind[1]

        for name in cmlp.data_vars:
            name_str = str(name)
            warp_cmlp(
                target=ds[name_str],
                target_indexes=flat_ind,
                source=cmlp_sub[name_str].compute(),
                x=x_rel,
                y=y_rel,
                kernel=kernel,
                is_phase=(name_str == "phase"),
                is_coherence=("coh" in name_str),
            )

    # B. Resample Phase Dataset variables using Scaled Grid Interpolation
    if len(phase.data_vars) > 0:
        # Safety boundary margin to prevent NaN/zeros at edges during linear/cubic interp
        phase_margin = 2 if method == "linear" else 4

        # Calculate safe integer index bounds
        az_min_idx = max(0, int(np.floor(az_phase.min())) - phase_margin)
        az_max_idx = min(
            phase.azimuth_time.size,
            int(np.ceil(az_phase.max())) + phase_margin,
        )
        rg_min_idx = max(0, int(np.floor(rg_phase.min())) - phase_margin)
        rg_max_idx = min(
            phase.slant_range_time.size,
            int(np.ceil(rg_phase.max())) + phase_margin,
        )

        # Slice phase dataset using integer index positions (.isel)
        phase_sub = phase.isel(
            azimuth_time=slice(az_min_idx, az_max_idx),
            slant_range_time=slice(rg_min_idx, rg_max_idx),
        )

        # Relative grid points matching phase_sub dimensions
        phase_grid_points = (
            np.arange(phase_sub.azimuth_time.size, dtype=float),
            np.arange(phase_sub.slant_range_time.size, dtype=float),
        )

        # Shift query pixel coordinates to match phase_sub relative origin
        az_phase_rel = az_phase - az_min_idx
        rg_phase_rel = rg_phase - rg_min_idx

        # Interpolate full-res phase variables directly onto target (y, x) coordinates
        phase_interp = interp_dataset_scaled(
            data=phase_sub,
            xi=(az_phase_rel, rg_phase_rel),
            common_dims=("azimuth_time", "slant_range_time"),
            points=phase_grid_points,
            method=method,
            fill_value=0.0,
        ).rename({"azimuth_time": "y", "slant_range_time": "x"})

        for name in phase.data_vars:
            name_str = str(name)
            ds[name_str] = phase_interp[name_str]

    # --- 8. Mask Interpolation ---
    if mask is not None:
        # Check coordinate direction for mask dataset
        mask_x_asc = mask.x.values[0] < mask.x.values[-1]
        mask_y_asc = mask.y.values[0] < mask.y.values[-1]

        mask_xslice = slice(xmin_sim, xmax_sim) if mask_x_asc else slice(xmax_sim, xmin_sim)
        mask_yslice = slice(ymin_sim, ymax_sim) if mask_y_asc else slice(ymax_sim, ymin_sim)

        mask_sub = mask.sel(x=mask_xslice, y=mask_yslice)

        mask_interp = interp_dataset(
            data=mask_sub,
            xi=(y_grid, x_grid),
            common_dims=("y", "x"),
            method="nearest",
            fill_value=0,
        )
        ds["mask"] = mask_interp["mask"].astype(np.int8)

    return ds
