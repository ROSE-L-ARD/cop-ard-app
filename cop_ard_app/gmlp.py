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
from cop_ard_core.simul import extract_simul_on_radar_chunk
from cop_ard_app.utils import get_margin, compute_chunks
from cop_ard_core.warp import sinc, sinc_warp
from cop_ard_app.io import normpath, read, save_empty_dataset, get_storage_options, write
from cop_ard_core.logger import timeit
from cop_ard_app.computation import map, map_blocks
from typing import Dict, cast

import dask.array as da
import numpy as np
import xarray as xr
from scipy.constants import speed_of_light


DIMS = ["polarization", "y", "x"]
LONG_DIMS = ["polarization", "past_reference_date", "y", "x"]
COORDS = ["polarization", "past_reference_date", "acquisition_date"]


@timeit
def stack_geocoding(
    cmlp_tree: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict,
    working_directory_static: str,
    working_directory_tmp: str,
    multilook: dict,
    gamma_flattening: bool = False,
    do_reramp: bool = True,
    client_options: Dict[str, str] = {"type": "LOCAL"},
    multiburst: bool = True,
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
    args = []
    # --- Prepare Arguments for Parallel Mapping ---
    # In multiburst mode, iterate through each burst in the DataTree.
    if multiburst:
        for k in cmlp_tree.bursts:
            # Package all necessary data and parameters for one burst into a tuple.
            r = (
                k,  # burst_id
                cmlp_tree.bursts[k].cmlp,
                simulation_tree.bursts[k].to_dataset(),
                simulation_ml_tree["bursts"][k],
                multiburst,
            )
            args.append(r)
    else:
        # For single-scene data, create a single argument tuple.
        r = (
            "data",  # scene_id
            cmlp_tree.cmlp,
            simulation_tree.to_dataset(),
            simulation_ml_tree,
            multiburst,
        )
        args.append(r)
    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Geocoding in Parallel ---
    # Map the `single_geocode_phase` worker function over the list of arguments.
    results = cast(
        list,
        map(
            single_geocoding,
            args,
            kwargs=dict(
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                gamma_flattening=gamma_flattening,
                do_reramp=do_reramp,
                **kwargs,
            ),
            client_options=client_options,
            combine=False,
        ),
    )

    # --- Assemble Results ---
    gslcs = xr.DataTree()
    # Reconstruct the DataTree from the list of results returned by the workers.
    for k, gslc in results:
        if multiburst:
            gslcs[f"bursts/{k}"] = gslc
        else:
            gslcs = xr.DataTree(gslc)
    return gslcs


def single_geocoding(
    burst_id: str,
    cmlp_path: str,
    acquisition_dates: list[str],
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    multilook: dict,
    working_directory_static: str,
    working_directory_tmp: str,
    gamma_flattening: bool,
    do_reramp: bool,
    client_options: dict = {},
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
    coreg_path = normpath(f"{working_directory_tmp}/{path}/coreg")

    data_path_static = normpath(f"{working_directory_static}/{path}")
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
    dt = map_geocoding(
        cmlp_path,
        acquisition_dates,
        simulation,
        simulation_ml,
        slc,
        multilook,
        data_path_static,
        data_path_tmp,
        coreg_path,
        kernel,
        gamma_flattening,
        do_reramp,
        client_options,
        enable_dask,
        **kwargs,
    )

    # --- Merge with Geometric Data ---
    # Read the pre-computed geometric grid data (e.g., LIA, anf).
    return (burst_id, dt)


def map_geocoding(
    cmlp_path: str,
    acquisition_dates: list[str],
    simulation: xr.Dataset,
    simulation_ml: dict,
    slc: xr.DataArray,
    multilook: dict,
    data_path_static: str,
    data_path_tmp: str,
    coreg_path: str,
    kernel: dict,
    gamma_flattening: bool,
    do_reramp: bool = True,
    client_options: dict = {"type": "LOCAL"},
    enable_dask: bool = True,
    **kwargs,
) -> xr.DataTree:
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
    annot = slc.attrs
    annot = annot[list(annot.keys())[0]]
    # Calculate grid spacing for margin calculation.
    margin = get_margin(
        time_ca_polynomial_coefficients=annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"],
        azimuth_time_spacing=annot["Grid"]["Col"]["SS"],
        slant_range_time_spacing=annot["Grid"]["Row"]["SS"],
    )
    # Extract the subset of the simulation grid relevant to this CSLC.
    selection = extract_simul_on_radar_chunk(
        simulation=simulation, simulation_ml, margin, simulation.acquisition_date.values[0]
    ).drop_vars(["axis", "dem_distance"])
    # NOTE: do we want this in rose-L?? this was needed to be aligned with shiroma. probably yes
    # Define the output coordinates at pixel centers.
    x, y = selection.x.values, selection.y.values
    x, y = (x[0:-1] + x[1:]) / 2, (y[0:-1] + y[1:]) / 2

    # --- Initialize Output GSLC Dataset ---
    # Create the empty Zarr store for the Geocoded SLC.
    for date in acquisition_dates:
        data = read(normpath(f"{cmlp_path}/{date}"), as_type="xr.Dataset", consolidate_metadata=False)
        data_vars = list(data.data_vars)
        dims = {k: DIMS if "past_reference_date" not in data[k].dims else LONG_DIMS for k in data_vars}
        chunks = {k: v[0] for k, v in data.chunksizes.items() if k in LONG_DIMS}
        chunks["y"] = selection.chunksizes["y"][0]
        chunks["x"] = selection.chunksizes["x"][0]
        shape = {k: [*data[k].shape[:-2], y.size, x.size] for k in data_vars}
        dtype = {k: data[k].dtype.name for k in data_vars}
        coords = {k: v.values for k, v in data.coords.items() if k in COORDS}
        coords["y"] = y
        coords["x"] = x
        save_empty_dataset(
            data_vars=data_vars,
            dims=dims,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            coords=coords,
            crs=selection.rio.crs.to_string(),
            data_path=normpath(f"{data_path_tmp}/GMLP/{date}"),
            storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        )

    shape_ = (len(acquisition_dates), y.size, x.size)
    chunks = {"acquisition_date": 1, "y": selection.chunksizes["y"][0], "x": selection.chunksizes["x"][0]}
    empty_dataset = xr.DataArray(
        da.empty(shape_), dims=["acquisition_date", "y", "x"], coords={"acquisition_date": acquisition_dates}
    ).chunk(chunks)
    chunks = compute_chunks(dict(empty_dataset.chunksizes))
    # --- Map the Geocoding Worker Function ---
    # This call applies the `warp_gslc` function to each block of the simulation grid.
    map_blocks(
        warp_cmlp,
        empty_dataset,
        args=(
            cmlp_path,
            slc,
            selection,
            kernel,
            multilook,
            data_path_static,
            data_path_tmp,
            do_reramp,
            gamma_flattening,
        ),
        kwargs=kwargs,
        chunks=chunks,
        client_options=client_options,
        combine=False,
        enable_dask=enable_dask,
    )
    dt = xr.DataTree()
    for date in acquisition_dates:
        data = read(
            normpath(f"{data_path_tmp}/GMLP/{date}"),
            consolidate_metadata=False,
            storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        )
        dt[date] = data
    return dt


def extract_rg_az(slc: xr.DataArray, simulation: xr.Dataset, multilook: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract range and azimuth coordinates from a simulation dataset in the SLC's pixel reference system.

    Parameters
    ----------
    slc : xr.DataArray
        The input SLC DataArray, used for its metadata and
        as the reference for the pixel coordinate system origin
        (minimum azimuth and range times).
    simulation : xr.Dataset
        The simulation dataset containing `slant_range_time` and
        `azimuth_time` coordinates to be converted.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - The range coordinates in relative pixels (`rg`).
        - The azimuth coordinates in relative pixels (`az`).
    """
    # Extract metadata from the SLC attributes.
    annot = slc.attrs
    annot = annot[list(annot)[0]]
    # Calculate the time interval per pixel in azimuth and range.
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9 * multilook["azimuth_time"]
    dtrg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light * multilook["slant_range_time"]

    # Transform the simulation's absolute time coordinates into the SLC's relative pixel coordinates.
    rg = ((simulation.slant_range_time - slc.slant_range_time.min()) / dtrg).values
    az = ((simulation.azimuth_time - slc.azimuth_time.min()).astype(float) / dtaz).values
    # --- Type Checking ---
    if not isinstance(rg, np.ndarray):
        raise Exception(f"Expected np.ndarray, got {type(rg)}")
    if not isinstance(az, np.ndarray):
        raise Exception(f"Expected np.ndarray, got {type(az)}")
    return rg, az


@timeit
def warp_cmlp(
    empty_dataset: xr.DataArray,
    cmlp_path: str,
    slc: xr.DataArray,
    simulation: xr.Dataset,
    kernel: dict,
    multilook: dict,
    data_path_static: str,
    data_path_tmp: str,
    do_reramp: bool,
    gamma_flattening: bool,
    eps: float = 1e-3,
    chunk: dict = {},
    **kwargs,
) -> None:
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
    acquisition_date = empty_dataset.acquisition_date.values[0]
    simul = simulation.isel({k: v for k, v in chunk.items() if k in ["y", "x"]}).squeeze()
    cmlp = read(
        f"{cmlp_path}/{acquisition_date}",
        consolidate_metadata=False,
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    # Use the first acquisition's simulation as the reference grid.
    # Define paths to the different components of the coregistered data.
    # Define the output region for the current chunk.
    region = {
        "y": slice(chunk["y"].start, chunk["y"].stop - 1),
        "x": slice(chunk["x"].start, chunk["x"].stop - 1),
    }
    shape, margin = (simul.y.size - 1, simul.x.size - 1), kernel["kernel_shape"] // 2 + 1
    # --- Coordinate Transformation ---
    # Convert the ground-projected simulation coordinates into the radar's pixel coordinate system.
    rg, az = extract_rg_az(slc, simul, multilook=multilook)
    # Average coordinates to find the center of each pixel.
    rg, az = (rg[1:, :] + rg[:-1, :]) / 2, (az[1:, :] + az[:-1, :]) / 2
    rg, az = ((rg[:, 1:] + rg[:, :-1]) / 2).ravel(), ((az[:, 1:] + az[:, :-1]) / 2).ravel()
    # Find the indices of points that can be safely interpolated (i.e., not too close to the edge).
    (ind,) = np.where(
        (rg >= margin[1])
        & (az >= margin[0])
        & (rg < (cmlp.slant_range_time.size - margin[1]))
        & (az < (cmlp.azimuth_time.size - margin[0]))
    )
    # --- Initialize Output Arrays ---
    ds = xr.Dataset()
    for name, var in cmlp.data_vars.items():
        sh = list(var.shape)
        sh[-2], sh[-1] = shape[0], shape[1]
        dims = list(var.dims)
        dims[-2], dims[-1] = "y", "x"
        ds[name] = xr.DataArray(np.zeros(sh, dtype=var.dtype), dims=dims)
    # --- Main Geocoding Loop ---
    if ind.size > 0:
        # Define the subset of radar coordinates needed for interpolation.
        x, y = rg[ind], az[ind]
        # Calculate the bounding box to read from the coregistered data.
        xmin, xmax = np.round(x.min() - margin[1]).astype(int), np.round(x.max() + margin[1]).astype(int)
        ymin, ymax = np.round(y.min() - margin[0]).astype(int), np.round(y.max() + margin[0]).astype(int)
        # Translate coordinates to be relative to the bounding box origin.
        x, y = x - xmin, y - ymin
        # --- Prepare Data for Interpolation ---
        # Select the required portion of the input data.
        cmlp = cmlp.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))
        # --- Loop over each acquisition date ---
        for name in cmlp.data_vars.keys():
            # 1. Evaluate string conditions ONCE outside the loop for speed
            is_phase = name == "phase"
            is_coh = "coh" in name

            target = ds[name].data
            source = cmlp[name].compute().data

            # 2. Create memory-mapped VIEWS
            spatial_size = target.shape[-2] * target.shape[-1]
            # target_view is now (N_leading, Y * X)
            target_view = target.reshape(-1, spatial_size)
            # source_view is (N_leading, Y, X)
            source_view = source.reshape(-1, source.shape[-2], source.shape[-1])
            # 3. Loop and insert
            for i in range(target_view.shape[0]):
                inp = source_view[i]

                # 2. Apply the appropriate warping logic
                if is_phase:
                    # Wrap phase to complex phasor to avoid interpolation artifacts
                    complex_phasor = np.exp(1j * inp)
                    out = np.angle(sinc_warp(complex_phasor, x, y, kernel))
                else:
                    # Standard warp for amplitude/coherence/other
                    out = sinc_warp(inp, x, y, kernel)

                    if is_coh:
                        # np.clip instantly bounds the array between 0 and 1,
                        out = np.clip(out, 0.0, 1.0)
                # 3. Write back into the target view
                target_view[i, ind] = out
    # --- Finalize and Write Output ---
    # Write the resulting GMLP chunk to the temporary Zarr store.
    write(
        ds,
        normpath(f"{data_path_tmp}/GMLP/{acquisition_date}"),
        region=region,
        consolidated=False,
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )

    return
