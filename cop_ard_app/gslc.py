from cop_ard_core.logger import timeit
from cop_ard_app.computation import map, map_blocks
from scipy.constants import speed_of_light
from typing import Mapping, cast
from cop_ard_core.dem import compute_shape_and_off
from cop_ard_core.simul import get_radar_bbox, get_geo_from_radar_bbox, extract_simul_on_radar_chunk
from cop_ard_app.utils import get_margin
from cop_ard_core.warp import sinc, warp_gslc
from cop_ard_app.io import normpath, read, get_storage_options, save_empty_dataset
import numpy as np
import xarray as xr


def single_geocode_phase(
    burst_id: str,
    cslc: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    working_directory_static: str,
    working_directory_tmp: str,
    gamma_flattening: bool,
    spacing: dict,
    do_reramp: bool,
    chunks: dict,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.DataTree]:
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
    g = map_geocode_phase(
        cslc,
        simulation,
        simulation_ml,
        data_path_static,
        data_path_tmp,
        kernel,
        gamma_flattening,
        spacing,
        chunks,
        do_reramp,
        enable_dask,
        skip_read=False,  # We need the result back in memory.
        **kwargs,
    )

    # --- Merge with Geometric Data ---
    # Read the pre-computed geometric grid data (e.g., LIA, anf).
    geo_data = read(
        normpath(f"{data_path_static}/geo_grid"),
        mask_and_scale=False,
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )

    gslc = xr.DataTree.from_dict({"measurements": g, "conditions": geo_data})

    return (burst_id, gslc)


def map_geocode_phase(
    cslc: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    data_path_static: str,
    data_path_tmp: str,
    kernel: dict,
    gamma_flattening: bool,
    spacing: dict,
    chunks: dict,
    do_reramp: bool = True,
    enable_dask: bool = True,
    skip_read: bool = True,
    **kwargs,
) -> xr.Dataset | None:
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
    annot = cslc.measurements.cstack.attrs
    annot = annot[list(annot.keys())[0]]
    # Calculate grid spacing for margin calculation.
    margin = get_margin(
        time_ca_polynomial_coefficients=annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"],
        azimuth_time_spacing=annot["Grid"]["Col"]["SS"],
        slant_range_time_spacing=annot["Grid"]["Row"]["SS"],
    )
    # Extract the subset of the simulation grid relevant to this CSLC.
    azimuth_time = cslc.measurements.azimuth_time
    slant_range_time = cslc.measurements.slant_range_time
    bbox_radar = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(simulation_ml, bbox_radar)

    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    selection = (
        (
            extract_simul_on_radar_chunk(
                simulation,
                simulation.acquisition_date.values[0],
                bbox_radar,
                bbox_geo,
            )
        )
        .squeeze()
        .drop_vars(["axis", "dem_distance", "acquisition_date", "h"])
    )

    # Define the output coordinates at pixel centers.
    shape, off = compute_shape_and_off(selection, spacing)
    # Create the coordinate arrays for the output grid.
    x, y = [np.linspace(off[i], off[i] + (shape[i] - 1) * spacing[i], shape[i]) for i in ["x", "y"]]

    # --- Initialize Output GSLC Dataset ---
    # Create the empty Zarr store for the Geocoded SLC.
    image = save_empty_dataset(
        data_vars=["gstack", "phase_model"],
        dims={
            "gstack": ["polarization", "acquisition_date", "y", "x"],
            "phase_model": ["acquisition_date", "y", "x"],
        },
        shape={
            "gstack": [
                cslc.measurements.polarization.size,
                cslc.measurements.acquisition_date.size,
                y.size,
                x.size,
            ],
            "phase_model": [
                cslc.measurements.acquisition_date.size,
                y.size,
                x.size,
            ],
        },
        chunks=chunks,
        dtype={
            "gstack": cslc.measurements.cstack.dtype,
            "phase_model": cslc.measurements.phase_model.dtype,
        },
        coords={
            "polarization": cslc.measurements.polarization.values,
            "acquisition_date": cslc.measurements.acquisition_date.values,
            "y": y,
            "x": x,
        },
        crs=selection.rio.crs.to_string(),
        data_path=normpath(f"{data_path_tmp}/geo_sinc"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        skip_read=False,
    )

    # --- Map the Geocoding Worker Function ---
    # This call applies the `warp_gslc` function to each block of the simulation grid.
    chunksizes = {k: v[0] for k, v in image.chunksizes.items()}
    chunksizes["acquisition_date"] = 10
    chunksizes["polarization"] = image.polarization.size
    radar_path = normpath(f"{data_path_static}/radar_grid")
    radar_data = read(
        radar_path,
        consolidate_metadata=False,
        mask_and_scale=False,
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )
    # Define paths to the different components of the coregistered data.
    save_options = {
        "data_path": normpath(f"{data_path_tmp}/geo_sinc"),
        "save_options": {
            "skip_read": True,
            "consolidated": False,
            "storage_options": get_storage_options(kwargs, key="storage_options_dynamic"),
        },
    }

    def region_callback(region: Mapping[str, slice], kwargs: dict) -> Mapping[str, slice]:
        out = {}
        for k, v in region.items():
            if k in ["y", "x"]:
                out[k] = slice(v.start, v.stop - 1)
            elif k == "polarization":
                out[k] = slice(0, cslc.measurements.polarization.size)
            elif k == "acquisition_date":
                out[k] = v
            else:
                raise ValueError(f"Unexpected dimension {k}")
        return out

    # Extract metadata from the SLC attributes.
    annot = cslc.measurements.cstack.attrs
    annot = annot[list(annot)[0]]
    # Calculate the time interval per pixel in azimuth and range.
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
    dtrg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light
    scp_idx = annot["ImageData"]["SCPPixel"]["Col"], annot["ImageData"]["SCPPixel"]["Row"]

    map_blocks(
        warp_gslc,
        image,
        args=(
            selection,
            kernel,
            cslc.measurements.cstack,
            cslc.measurements.phase_model,
            cslc.conditions.phase_demod.phase_demod,
            radar_data.anf,
            do_reramp,
            gamma_flattening,
            dtaz,
            dtrg,
            scp_idx,
        ),
        kwargs=kwargs,
        chunksizes=chunksizes,
        combine=False,
        enable_dask=enable_dask,
        region_callback={0: region_callback},
        store_mapping={0: save_options},
    )

    # Optionally read and return the final dataset.
    return (
        read(
            normpath(f"{data_path_tmp}/geo_sinc"),
            as_type="xr.Dataset",
            storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        )
        if not skip_read
        else None
    )


@timeit
def stack_geocoding(
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict,
    working_directory_static: str,
    working_directory_tmp: str,
    spacing: dict,
    gamma_flattening: bool = False,
    do_reramp: bool = True,
    multiburst: bool = True,
    enable_dask: bool = True,
    chunks: dict = {"acquisition_date": 1, "polarization": 1, "y": 2048, "x": 2048},
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
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        v = v.coreg.drop_nodes("quality")
        unwanted_nodes = [child for child in v["conditions"].children if child != "phase_demod"]
        v["conditions"] = v["conditions"].drop_nodes(unwanted_nodes)
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        a = (k, v, s, s_ml, multiburst)
        args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Geocoding in Parallel ---
    # Map the `single_geocode_phase` worker function over the list of arguments.
    results = cast(
        list,
        map(
            single_geocode_phase,
            args,
            kwargs=dict(
                spacing=spacing,
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                gamma_flattening=gamma_flattening,
                do_reramp=do_reramp,
                enable_dask=enable_dask,
                chunks=chunks,
                **kwargs,
            ),
            enable_dask=enable_dask,
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
            gslcs = gslc
    return gslcs

