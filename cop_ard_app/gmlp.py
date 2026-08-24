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
from typing import Generator, cast

import numpy as np
import xarray as xr
from cop_ard_core.dem import compute_shape_and_off
from cop_ard_core.gmlp import get_radar_spacing, warp_cmlp
from cop_ard_core.griddata import interp_dataset
from cop_ard_core.logger import timeit
from cop_ard_core.simul import extract_simul_on_radar_chunk, get_geo_from_radar_bbox, get_radar_bbox
from cop_ard_core.warp import extract_rg_az, sinc

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.io import get_storage_options, normpath, read, save_empty_dataset

DIMS = ["polarization", "y", "x"]
LONG_DIMS = ["polarization", "past_reference_date", "y", "x"]
COORDS = ["polarization", "past_reference_date", "acquisition_date"]


@timeit
def stack_gmlp(
    cmlp_tree: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict,
    spacing: dict,
    working_directory_tmp: str,
    enable_dask: bool = True,
    chunks: dict = {"acquisition_date": 1, "past_reference_date": 1, "polarization": 1, "y": 2048, "x": 2048},
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
    **kwargs
        Additional keyword arguments passed to the processing functions,
        such as `warp_kernel_shape`.

    Returns
    -------
    xr.DataTree
        A new DataTree containing the geocoded SLC (GSLC) data for each burst.
    """
    multiburst = "bursts" in cmlp_tree
    args = []
    # --- Prepare Arguments for Parallel Mapping ---
    for k, v in (cmlp_tree.bursts if multiburst else {"data": cmlp_tree}).items():
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        for date,cmlp in v.cmlp.items():
            a = (k, date,cmlp, s, s_ml)
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
    for (burst_id,date,gmlp) in results:
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
    multiburst: bool,
    spacing: dict,
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
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path_tmp = normpath(f"{working_directory_tmp}/{path}")

    # --- Prepare Interpolation Kernel ---
    # Get warp parameters from kwargs or use default values.
    ov_fact = kwargs.get("warp_ov_fact", 100_000)
    kernel_shape = kwargs.get("warp_kernel_shape", np.array([11, 11]))
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
    burst_id:str,
    date:str,
    cmlp: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    data_path_tmp: str,
    kernel: dict,
    spacing: dict,
    chunks: dict,
    enable_dask: bool = True,
    border: int = 10,
    **kwargs,
    ) -> tuple[str,str,xr.Dataset]:
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
    xr.Dataset or None
        The geocoded Dataset if `skip_read` is False, otherwise None.
    """

    # --- Prepare Simulation Subset ---
    # Calculate grid spacing for margin calculation.
    product = next(iter(cmlp.values()))
    azimuth_time = product.azimuth_time
    slant_range_time = product.slant_range_time

    radar_spacing = get_radar_spacing(azimuth_time, slant_range_time)
    margin = {e: radar_spacing[e] * border for e in ["azimuth_time", "slant_range_time"]}
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
    data = cmlp.ds
    data_vars = list(data.data_vars)
    dims = {k: DIMS if "past_reference_date" not in data[k].dims else LONG_DIMS for k in data_vars}
    shape = {k: [*data[k].shape[:-2], y.size, x.size] for k in data_vars}
    dtype = {k: data[k].dtype.name for k in data_vars}
    coords = {k: v.values for k, v in data.coords.items() if k in COORDS}
    coords["y"] = y
    coords["x"] = x
    ch = {k: chunks[k] for k in data.chunksizes.keys() if k in LONG_DIMS}
    ch["y"] = chunks["y"]
    ch["x"] = chunks["x"]
    image = save_empty_dataset(
        data_vars=data_vars,
        dims=dims,
        shape=shape,
        chunks=ch,
        dtype=dtype,
        coords=coords,
        crs=selection.rio.crs.to_string(),
        data_path=normpath(f"{data_path_tmp}/gmlp/{date}"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        skip_read=False
    )
    
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
        blockwise_warp_cmlp,
        image,
        args=(selection, data,  kernel),
        kwargs=kwargs
        | {
            "dtaz": radar_spacing["azimuth_time"].astype(float),
            "dtrg": radar_spacing["slant_range_time"],
        },
        combine=False,
        enable_dask=enable_dask,
        store_mapping=save_options,
    )
    gmlp = read(
        normpath(f"{data_path_tmp}/gmlp/{date}"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        as_type="xr.Dataset"
    )

    return (burst_id,date,gmlp)


def blockwise_warp_cmlp(
    image: xr.DataArray,
    simulation: xr.Dataset,
    cmlp: xr.Dataset,
    kernel: dict,
    dtaz: float,
    dtrg: float,
    margin: int = 5,
    method: str = "linear",
    chunk: dict = {},
    **kwargs,
    ) -> xr.Dataset:
    """
    Warp coregistered SLC data to a ground-projected grid (Geocoded SLC).

    This function performs the final geocoding step, resampling the data from
    the radar's slant range geometry to a regular grid on the ground (e.g.,
    UTM coordinates). It uses sinc interpolation for high accuracy and can
    apply final phase corrections (reramping) and radiometric corrections
    (gamma flattening).

    Parameters
    ----------
    simulation : xr.Dataset
        The dataset defining the ground-projected grid (y, x coordinates)
        and the corresponding radar coordinates (azimuth_time, slant_range_time).
    coreg_path : str
        The path to the coregistered SLC data.
    kernel : dict
        A dictionary with parameters for the sinc interpolator.
    data_path_static : str
        Path to static data like radar grid masks.
    data_path_tmp : str
        Path to a temporary directory for writing results.
    do_reramp : bool
        If True, apply a final phase ramp to the geocoded data.
    gamma_flattening : bool
        If True, apply a radiometric flattening correction.
    eps : float, optional
        A small epsilon value for safe division in gamma flattening.
        Defaults to 1e-3.
    chunk : object, optional
        The chunk configuration for processing. Defaults to None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    None
    """
    # --- Initialization and Data Loading ---
    shape = image.shape

    spacing_in = np.abs(simulation.rio.resolution(recalc=True))
    spacing_out = image.rio.resolution(recalc=True)

    xmin = image.x.min().values - margin * spacing_in[0]
    xmax = image.x.max().values + margin * spacing_in[0]
    ymin = image.y.min().values - margin * spacing_in[1]
    ymax = image.y.max().values + margin * spacing_in[1]

    xslice = slice(xmin, xmax) if spacing_out[0] > 0 else slice(xmax, xmin)
    yslice = slice(ymin, ymax) if spacing_out[1] > 0 else slice(ymax, ymin)

    simulation = simulation.sel(x=xslice, y=yslice)
    rg, az = extract_rg_az(cmlp, simulation, dtaz=dtaz, dtrg=dtrg)
    # Average coordinates to find the center of each pixel.
    simulation["azimuth_time"].data = az
    simulation["slant_range_time"].data = rg
    x_grid, y_grid = np.meshgrid(image.x, image.y)

    # Call the generic dataset interpolation function.
    simul_out = interp_dataset(
        data=simulation, xi=(y_grid, x_grid), common_dims=("y", "x"), method=method, fill_value=0.0
    )

    rg_interp = simul_out.slant_range_time.values.ravel()
    az_interp = simul_out.azimuth_time.values.ravel()

    # Find the indices of points that can be safely interpolated (i.e., not too close to the edge).
    kernel_margin = kernel["kernel_shape"] // 2 + 1
    (ind,) = np.where(
        (rg_interp >= kernel_margin[1])
        & (az_interp >= kernel_margin[0])
        & (rg_interp < (cmlp.slant_range_time.size - kernel_margin[1]))
        & (az_interp < (cmlp.azimuth_time.size - kernel_margin[0]))
    )

    # Define the subset of radar coordinates needed for interpolation.
    x, y = rg_interp[ind], az_interp[ind]

    # Calculate the bounding box to read from the coregistered data.
    xmin, xmax = (
        np.round(x.min() - kernel_margin[1]).astype(int),
        np.round(x.max() + kernel_margin[1]).astype(int),
    )
    ymin, ymax = (
        np.round(y.min() - kernel_margin[0]).astype(int),
        np.round(y.max() + kernel_margin[0]).astype(int),
    )

    # Translate coordinates to be relative to the bounding box origin.
    x, y = x - xmin, y - ymin
    ds = xr.Dataset()
    for name, var in cmlp.data_vars.items():
        var_shape = list(var.shape)
        var_shape[-2], var_shape[-1] = shape[0], shape[1]
        var_dims = list(var.dims)
        var_dims[-2], var_dims[-1] = "y", "x"
        ds[name] = xr.DataArray(np.zeros(var_shape, dtype=var.dtype), dims=var_dims)
    # Select the required portion of the input data.
    cmlp = cmlp.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))
    if ind.size > 0:
        # --- Prepare Data for Interpolation ---
        # --- Loop over each acquisition date ---
        for name in cmlp.data_vars:
            name = str(name)
            # 1. Evaluate string conditions ONCE outside the loop for speed
            is_phase = name == "phase"
            is_coh = "coh" in name
            warp_cmlp(
                source=cmlp[name].compute(),
                target=ds[name],
                target_indexes=ind,
                x=x,
                y=y,
                kernel=kernel,
                is_phase=is_phase,
                is_coherence=is_coh,
            )

    return ds
