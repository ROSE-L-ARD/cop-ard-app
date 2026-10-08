"""
Coregistration of SAR image stacks for interferometric ARD products.

Warps original SLC images onto a common reference radar grid, updates the
lookup tables (LUTs) for radiometric and phase-demodulation parameters, and
maintains the coregistration conditions and quality metadata.
"""

import os
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.griddata import init_interpolate
from cop_ard_core.logger import logger, timeit
from cop_ard_core.warp import GRID, coreg_qlt, getLUT, sinc, warp_cslc, warp_dazdrg
from scipy.constants import speed_of_light

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.io import (
    exists,
    generate_empty_dataset,
    get_storage_options,
    normpath,
    path_to_dict,
    read,
    save_empty_dataset,
    write,
)
from cop_ard_app.model import get_coefficients
from cop_ard_app.slc import updateETAD, updateLookVector, updateOrbits
from cop_ard_app.utils import get_best_pol, get_margin


def init_single_coreg_stack(
    slcs: xr.DataTree,
    ref_id: str,
    wdir_path: str,
    recompute: bool = False,
    **kwargs,
) -> xr.DataTree:
    """
    Initialize the empty 'coreg' group for a single burst or stripmap scene.

    This function creates the structure for the coregistered stack. It uses the
    dimensions and coordinates of the reference date image and creates empty
    Dask arrays (`cstack`, `phase_model`) that will be filled in during processing.

    Args:
        slcs: The input DataTree containing the 'orig' data.
        ref_date: The reference date for coregistration.
        store_path: The path where the 'coreg' group should be created.

    Returns:
        The newly created 'coreg' DataTree node, read back from disk.
    """
    if wdir_path is None:
        raise ValueError("Output SLCs path is required.")
    if (ref_id in slcs.orig) is False:
        raise ValueError("Reference image not available.")

    # --- Write the Empty Structure to Disk ---
    crg = xr.DataTree.from_dict({k: xr.Dataset() for k in ["conditions", "quality"]})
    acquisition_ids = list(slcs.orig.keys())
    acquisition_dates = [str(slcs.orig[i].measurements.acquisition_date.values[0]) for i in acquisition_ids]
    if (recompute is False) and len(acquisition_ids) != 1:
        acquisition_dates = [date for i, date in zip(acquisition_ids, acquisition_dates) if i != ref_id]
        acquisition_ids = [i for i in acquisition_ids if i != ref_id]

    # 1. Gather polarizations directly as sets
    pol_sets = [{str(p) for p in slcs.orig[d].measurements.polarization.values} for d in acquisition_ids]
    # 2. Find the total unique polarizations across all dates
    all_pols = set.union(*pol_sets) if pol_sets else set()

    # 3. Check that every date has the complete set of polarizations
    for d, p_set in zip(acquisition_ids, pol_sets):
        if p_set != all_pols:
            raise ValueError(f"For acquisition date {d} got polarizations {p_set}, expected {all_pols}")

    slc = slcs.orig[ref_id].measurements.slc
    # ref_date = slc.acquisition_date.values
    slc_sec = slcs.orig[acquisition_ids[0]].measurements.slc
    shape = [len(all_pols), len(acquisition_ids), slc.azimuth_time.size, slc.slant_range_time.size]
    measurements_path = os.path.join(wdir_path, "coreg", "measurements")
    data_vars = ["cstack", "phase_model"]
    dims = {
        "cstack": ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
        "phase_model": ["acquisition_id", "azimuth_time", "slant_range_time"],
    }
    shape_dict = {"cstack": shape, "phase_model": shape[1:]}
    dtype = {"cstack": "complex64", "phase_model": "float64"}
    chunks = {k: v[0] for k, v in slc_sec.chunksizes.items()}

    coords = {
        "polarization": ("polarization", list(all_pols)),
        "acquisition_id": ("acquisition_id", acquisition_ids),
        "acquisition_date": ("acquisition_id", acquisition_dates),
        "azimuth_time": ("azimuth_time", slc.azimuth_time.values),
        "slant_range_time": ("slant_range_time", slc.slant_range_time.values),
        "azimuth_m": ("azimuth_time", slc.azimuth_m.values),
        "slant_range_m": ("slant_range_time", slc.slant_range_m.values),
    }

    measurements = generate_empty_dataset(
        data_vars=data_vars,
        dims=dims,
        shape=shape_dict,
        chunks=chunks,
        dtype=dtype,
        coords=coords,
        attrs={"coregistration_reference_id": ref_id},
        attrs_data_vars={"cstack": slc.attrs},
    )

    write(
        measurements.drop_vars(["cstack", "phase_model"]),
        measurements_path,
        mode="w",
        compute=True,
        storage_options=get_storage_options(kwargs),
        skip_read=True,
    )
    write(
        measurements,
        measurements_path,
        mode="a",
        compute=False,
        storage_options=get_storage_options(kwargs),
        skip_read=True,
    )
    measurements = read(measurements_path, as_type="xr.Dataset", storage_options=get_storage_options(kwargs))

    crg["measurements"] = measurements

    data_vars = ["daz", "drg"]
    dims = {
        "daz": ["acquisition_id", "azimuth_time", "slant_range_time"],
        "drg": ["acquisition_id", "azimuth_time", "slant_range_time"],
    }
    shape_dict = {"daz": shape[1:], "drg": shape[1:]}
    dtype = {"daz": "timedelta64[ns]", "drg": "float64"}

    daz_drg = save_empty_dataset(
        data_vars=data_vars,
        dims=dims,
        shape=shape_dict,
        chunks=chunks,
        dtype=dtype,
        data_path=f"{wdir_path}/daz_drg",
        coords=coords,
        skip_read=False,
        compute=True,
        **kwargs,
    )

    recompute_daz_drg = True

    if daz_drg is not None:
        crg["measurements/daz"] = daz_drg["daz"]
        crg["measurements/drg"] = daz_drg["drg"]
        crg["measurements/daz"].attrs["recompute"] = recompute_daz_drg
        crg["measurements/daz"].attrs["store_path"] = f"{wdir_path}/daz_drg"

    if ref_id not in acquisition_ids:
        logger.info(f"Removing reference date for burst : {slcs.name}")
        del slcs.orig[ref_id]
    # Add the new 'coreg' node to the in-memory DataTree.
    slcs["coreg"] = crg

    if not isinstance(slcs, xr.DataTree):
        raise TypeError(f"Expected xr.Datatree got {type(slcs)}")
    return slcs


def initialize_LUT_coords(
    acquisition_id: xr.DataArray,
    acquisition_date: xr.DataArray,
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    grid: tuple[int, int] = GRID,
) -> xr.Dataset:
    """
    Initialize a coarse coordinate grid for the Lookup Tables (LUTs).

    This function creates a downsampled version of the full-resolution measurement
    grid. This coarse grid is used as the coordinate system for all the LUTs
    (e.g., Doppler, noise) to save space and computation time. It also initializes
    the `daz` and `drg` variables that will be filled in later.

    Parameters
    ----------
    measurements
        The full-resolution coregistered measurements dataset.

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset representing the coarse LUT grid with empty `daz`
        and `drg` data variables.
    """
    # Calculate the step size to achieve the target coarse grid size (e.g., 11x11).
    az_step, rg_step = (
        np.ceil(azimuth_time.size / grid[0]).astype(int),
        np.ceil(slant_range_time.size / grid[1]).astype(int),
    )
    # Create the coarse grid by decimating the full-resolution coordinates.
    LUT_coords = xr.Dataset(
        coords={
            "acquisition_id": acquisition_id,
            "acquisition_date": acquisition_date,
            "azimuth_time": azimuth_time,
            "slant_range_time": slant_range_time,
        }
    )
    LUT_coords = LUT_coords.isel(
        azimuth_time=slice(None, None, az_step),
        slant_range_time=slice(None, None, rg_step),
    )

    # --- Initialize daz and drg Data Variables ---
    # Assign empty arrays to hold the azimuth and range shift values on this coarse grid.
    LUT_coords = LUT_coords.assign(
        {
            "drg": (
                ["acquisition_id", "azimuth_time", "slant_range_time"],
                np.full(
                    (
                        LUT_coords.acquisition_id.size,
                        LUT_coords.azimuth_time.size,
                        LUT_coords.slant_range_time.size,
                    ),
                    0,
                    dtype=float,
                ),
            ),
            "daz": (
                ["acquisition_id", "azimuth_time", "slant_range_time"],
                np.full(
                    (
                        LUT_coords.acquisition_id.size,
                        LUT_coords.azimuth_time.size,
                        LUT_coords.slant_range_time.size,
                    ),
                    0,
                    dtype="timedelta64[ns]",
                ),
            ),
        }
    )

    return LUT_coords


def coreg_single_burst(
    burst_id: str,
    slc: xr.DataTree,
    simulation_reference: xr.Dataset,
    simulation_secondary: xr.Dataset | None,
    simulation_ml: dict,
    multiburst: bool,
    daz_drg: xr.Dataset | None,
    reference_id: str,
    working_directory: str,
    recompute: bool = False,
    save_daz_drg: bool = False,
    do_reramp: bool = True,
    do_flattening: bool = True,
    enable_dask: bool = True,
    x_poly: xr.DataArray | None = None,
    y_poly: xr.DataArray | None = None,
    **kwargs,
) -> tuple[str, xr.DataTree]:
    """
    Execute the end-to-end coregistration workflow for a single burst.

    This is the main worker function that is mapped across the Dask cluster.
    It initializes the output structure, runs the geometric warping process by
    calling `warp_dazdrg_cslc`, and then updates all the ancillary metadata
    (conditions and quality) for the newly coregistered burst.

    Parameters
    ----------
    burst_id
        The identifier for the burst being processed.
    slc
        The input DataTree for this burst, containing the `orig` data.
    simulation_reference
        The reference simulation data for this burst.
    simulation_secondary
        An optional secondary simulation dataset.
    simulation_ml
        The multilooked simulation data for this burst.
    multiburst
        A boolean flag for multiburst mode.
    reference_date
        The reference acquisition date.
    working_directory
        The root directory for all outputs.
    enable_dask
        If True, enable Dask for sub-tasks.
    **kwargs
        Additional keyword arguments for the warping function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the resulting `coreg` DataTree node.
    """
    # Define the output path for this burst.
    path = f"bursts/{burst_id}" if multiburst else f"data/{reference_id}"

    # --- Step 1: Initialize Output Structure ---
    # Create the empty 'coreg' group on disk.
    coreg_path = normpath(f"{working_directory}/{path}")
    slc = init_single_coreg_stack(
        slc,
        reference_id,
        coreg_path,
        recompute=recompute,
        **kwargs,
    )

    # --- Prepare Data Structures ---
    measurements = slc.coreg.measurements.to_dataset()
    conditions = slc.coreg.conditions
    quality = slc.coreg.quality

    # Initialize the coarse grid for the Lookup Tables.
    LUT_coords = initialize_LUT_coords(
        acquisition_id=measurements.acquisition_id,
        acquisition_date=measurements.acquisition_date,
        azimuth_time=measurements.azimuth_time,
        slant_range_time=measurements.slant_range_time,
    )

    # --- Prepare Interpolation Kernel ---
    # Define the parameters for the sinc interpolation kernel.
    ov_fact = kwargs.get("warp_ov_fact", 100_000)
    kernel_shape = np.array(kwargs.get("warp_kernel_shape", [11, 11]))
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

    # --- Step 2: Perform Warping in Parallel ---
    # Use `map_blocks` to apply the `warp_dazdrg_cslc` function to chunks of the data.
    chunkable_dazdrg = []
    placeholder_dazdrg = []
    if "daz" in measurements:
        compute_daz_drg = measurements.daz.attrs.get("recompute", True)
        daz_drg_store_path = measurements.daz.attrs["store_path"]
        chunkable_dazdrg = [measurements.daz, measurements.drg]
    else:
        placeholder_dazdrg = [None, None]
        compute_daz_drg = True

    store_mapping = {
        0: {
            "data_path": os.path.join(coreg_path, "daz_drg"),
            "dims": ["acquisition_date", "azimuth_time", "slant_range_time"],
            "save_options": {
                "consolidated": False,
                "storage_options": get_storage_options(kwargs),
                "skip_read": True,
            },
        }
    }
    orig_slcs = xr.DataTree.from_dict({f"orig/{k}/slc": v.measurements.slc for k, v in slc.orig.children.items()})
    orig_etad = {}
    for k, v in slc["orig"].children.items():
        if "ETAD" in v.conditions:
            child_etad = v.conditions.ETAD.to_dataset()
            orig_etad[k] = child_etad
        else:
            orig_etad[k] = None

    orig_etad_dt = xr.DataTree.from_dict(orig_etad)

    # --- Calculate Grid Spacing and Margins ---
    annotation = measurements.cstack.attrs
    single_pol_annotation = annotation[next(iter(annotation.keys()))]
    # Calculate azimuth and range sampling intervals.
    margin = get_margin(
        time_ca_polynomial_coefficients=single_pol_annotation["RMA"]["INCA"]["TimeCAPoly"]["Coefs"],
        azimuth_time_spacing=single_pol_annotation["Grid"]["Col"]["SS"],
        slant_range_time_spacing=single_pol_annotation["Grid"]["Row"]["SS"],
    )
    azimuth_grid_spacing = (
        single_pol_annotation["Grid"]["Col"]["SS"] * single_pol_annotation["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1]
    )
    range_grid_spacing = 2 * single_pol_annotation["Grid"]["Row"]["SS"] / speed_of_light
    chunksizes = {
        "azimuth_time": measurements.chunksizes["azimuth_time"][0],
        "slant_range_time": measurements.chunksizes["slant_range_time"][0],
        "acquisition_id": 10,
    }

    map_blocks(
        warp_dazdrg,
        obj=[measurements.cstack, *chunkable_dazdrg],  # if available, chunk daz and drg; otherwise, pass placeholders
        args=(
            *placeholder_dazdrg,
            simulation_reference,
            simulation_secondary,
            simulation_ml,
            margin,
            measurements.coregistration_reference_id,
            azimuth_grid_spacing,
            range_grid_spacing,
        ),
        kwargs={
            "compute_daz_drg": compute_daz_drg,
        },
        combine=False,
        enable_dask=enable_dask,
        chunksizes=chunksizes,
        store_mapping=store_mapping,
    )

    daz_drg = read(
        daz_drg_store_path,
        storage_options=get_storage_options(kwargs),
        as_type="xr.Dataset",
    )

    store_mapping = {
        0: {
            "data_path": measurements.attrs["store_path"],
            "save_options": {
                "consolidated": False,
                "storage_options": get_storage_options(kwargs),
            },
        },
    }

    map_blocks(
        blockwise_warp_cslc,
        obj=[measurements.cstack, measurements.phase_model, daz_drg.daz, daz_drg.drg, measurements.acquisition_id],
        args=(
            measurements.coregistration_reference_id,
            orig_slcs,
            orig_etad_dt,
            kernel,
        ),
        kwargs={
            "do_reramp": do_reramp,
            "do_flattening": do_flattening,
            "x_poly": x_poly,
            "y_poly": y_poly,
        },
        combine=False,
        enable_dask=enable_dask,
        chunksizes=chunksizes,
        store_mapping=store_mapping,
    )

    list_daz_drg_sel = cast(
        list,
        map_blocks(
            update_LUT,
            obj=daz_drg,
            kwargs={"LUT_coords": LUT_coords},
            combine=False,
            enable_dask=enable_dask,
            chunksizes={"acquisition_id": daz_drg.acquisition_id.size},
        ),
    )
    # --- Step 3: Update Metadata ---
    # Merge the list of daz/drg offsets computed by each chunk.
    LUT_daz_drg = sum([k for k in list_daz_drg_sel if k is not None])
    if not isinstance(LUT_daz_drg, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(LUT_daz_drg)}")

    measurements = read(
        measurements.store_path,
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs),
    )
    if daz_drg is not None:
        measurements["daz"] = daz_drg["daz"]
        measurements["drg"] = daz_drg["drg"]
    # Update the 'conditions' group (orbits, ETAD, LUTs) and write to disk.
    conditions = update_conditions(slc, measurements, simulation_reference, LUT_daz_drg, conditions)
    conditions = write(
        conditions,
        normpath(f"{working_directory}/{path}/coreg/conditions"),
        mode="w",
        storage_options=get_storage_options(kwargs),
    )
    # Update the 'quality' group (accuracy, radiometric LUTs) and write to disk.
    quality = update_quality(slc, LUT_daz_drg, quality)
    quality = write(
        quality,
        normpath(f"{working_directory}/{path}/coreg/quality"),
        mode="w",
        storage_options=get_storage_options(kwargs),
    )

    coreg = xr.DataTree.from_dict({"measurements": measurements, "conditions": conditions, "quality": quality})
    return (burst_id, coreg)


@timeit
def stack_coregistration(
    slcs: xr.DataTree,
    simulation_reference_tree: xr.DataTree,
    simulation_ml_tree: dict,
    reference_id: str,
    working_directory: str,
    simulation_secondary_tree: xr.DataTree | None = None,
    save_daz_drg: bool = False,
    daz_drg: xr.DataTree | None = None,
    do_reramp: bool = True,
    do_flattening: bool = True,
    multiburst: bool = True,
    recompute: bool = False,
    enable_dask: bool = True,
    x_poly: xr.DataArray | None = None,
    y_poly: xr.DataArray | None = None,
    **kwargs,
) -> xr.DataTree:
    """
    Coregister a stack of SAR images in parallel.

    This function orchestrates the coregistration of an entire stack of SLCs
    (either as multiple bursts or a single scene). It prepares the necessary
    arguments for each burst and then maps the `coreg_single_burst` function
    across a Dask cluster to process them in parallel.

    Parameters
    ----------
    slcs
        The input DataTree containing the stack of original SLC images.
    simulation_reference_tree
        A DataTree with the simulated radar geometry
        for the reference DEM.
    simulation_ml_tree
        A dictionary containing the downsampled ("multilooked")
        simulation data, used for quickly finding overlaps.
    reference_date
        The acquisition date of the reference image for the coregistration.
    working_directory
        The root directory for all processing outputs.
    simulation_secondary_tree
        An optional DataTree with simulation data for a
        secondary DEM, used for differential measurements.
    multiburst
        A boolean flag indicating if the data is in multiburst format.
    **kwargs
        Additional keyword arguments passed down to the processing functions,
        such as `warp_kernel_shape` and `warp_kaiser_beta`.

    Returns
    -------
    DataTree
        The updated `slcs` DataTree, with the newly created `coreg` nodes containing
        the coregistered stack and associated metadata.
    """
    args = []
    if daz_drg is not None and save_daz_drg is True:
        daz_drg = None
    # --- Prepare Arguments for Parallel Mapping ---
    # In multiburst mode, iterate over each burst in the DataTree.
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        s = (simulation_reference_tree.bursts[k] if multiburst else simulation_reference_tree).ds
        s2 = (
            (simulation_secondary_tree.bursts[k] if multiburst else simulation_secondary_tree).ds
            if simulation_secondary_tree is not None
            else None
        )
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        dz = (daz_drg.bursts[k] if multiburst else daz_drg).ds if daz_drg is not None else None
        a = (k, v, s, s2, s_ml, multiburst, dz)
        args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # JIT-compile the Numba interpolation function before starting the main computation.
    init_interpolate()

    # --- Execute Coregistration in Parallel ---
    # Map the `coreg_single_burst` function over the list of arguments.
    coregs = cast(
        list,
        map(
            coreg_single_burst,
            args,
            kwargs=dict(
                reference_id=reference_id,
                working_directory=working_directory,
                enable_dask=enable_dask,
                recompute=recompute,
                save_daz_drg=save_daz_drg,
                do_reramp=do_reramp,
                do_flattening=do_flattening,
                x_poly=x_poly,
                y_poly=y_poly,
                **kwargs,
            ),
            enable_dask=enable_dask,
        ),
    )

    # --- Assemble Results ---
    # Integrate the results from each worker back into the main DataTree.
    for k, coreg in coregs:
        if multiburst:
            slcs[f"bursts/{k}/coreg"] = coreg
        else:
            slcs["coreg"] = coreg

    return slcs


def update_conditions(
    slc: xr.DataTree,
    measurements: xr.Dataset,
    simulation: xr.Dataset,
    daz_drg_LUT: xr.Dataset,
    conditions: xr.DataTree,
) -> xr.DataTree:
    """
    Update and populate the 'conditions' group of the coregistered stack.

    This function acts as a high-level manager for updating all ancillary data
    related to the processing conditions. It calls helper functions to update
    orbit data, ETAD corrections, and the phase demodulation (Doppler) LUT.

    Parameters
    ----------
    slc
        The DataTree for a single burst, containing the `orig` data.
    daz_drg
        The dataset of computed azimuth and range shifts, used for LUT generation.
    conditions
        The target `conditions` DataTree node to be updated.

    Returns
    -------
    DataTree
        The updated `conditions` DataTree node.
    """
    # Consolidate and update the orbit information.
    conditions["orbit"] = updateOrbits(slc)
    # Consolidate and update ETAD information if available.
    if all(("conditions" in v) and ("ETAD" in v.conditions) for _, v in slc.orig.items()):
        conditions["ETAD"] = updateETAD(slc, measurements)

    conditions["look_vector"] = updateLookVector(measurements, simulation)
    # Generate the updated Doppler (phase_demod) LUT for the coregistered grid.
    xLUT = updateLut(slc, daz_drg_LUT, "Grid.Col.DeltaKCOAPoly.Coefs", "phase_demod")
    conditions["phase_demod"] = xLUT
    return conditions


def update_quality(slc: xr.DataTree, daz_drg: xr.Dataset, quality: xr.Dataset) -> xr.Dataset:
    """
    Update and populate the 'quality' group of the coregistered stack.

    This function computes and stores all data related to the quality of the
    coregistered product, including the estimated coregistration accuracy and
    Lookup Tables for radiometric quantities (noise, sigma0, gamma0, etc.).

    Parameters
    ----------
    slc
        The DataTree for a single burst, containing `orig` and `coreg` data.
    daz_drg
        The dataset of computed azimuth and range shifts, used for LUT generation.
    quality
        The target `quality` Dataset to be updated.

    Returns
    -------
    xarray.Dataset
        The updated `quality` Dataset.
    """
    # Mapping from official calibration names to variable names in the output.
    name_mapping = {
        "SigmaZero": "sigma0",
        "GammaZero": "gamma0",
        "BetaZero": "beta0",
        "RCS": "RCS",
        "NoiseLevel": "noise",
    }
    calib_ok = ["SigmaZero", "GammaZero", "BetaZero", "RCS"]
    NoiseLevel = "NoiseLevel"

    # --- Compute and Assign Quality Metrics ---
    # Compute the estimated coregistration accuracy.
    annot = slc.coreg.measurements.cstack.attrs
    annot = annot[next(iter(annot.keys()))]
    scpcoa_incidence_angle = annot["SCPCOA"]["IncidenceAng"]
    coreg_arp_coeffs = {e: get_coefficients(f"Position.ARPPoly.{e}.Coefs", annot) for e in ["X", "Y", "Z"]}
    if None in coreg_arp_coeffs.values():
        raise ValueError("Missing ARP polynomial coefficients in metadata.")
    t0 = np.datetime64(annot["Timeline"]["CollectStart"].rstrip("Z|z"), "ns")
    acquisition_date_arp_coeffs = {}
    acquisition_date_t0 = {}
    acquisition_date_orbit_type = {}
    for acquisition_date in slc.coreg.measurements.acquisition_id.values:
        acquisition_date_annot = slc.orig[acquisition_date].measurements.slc.attrs
        acquisition_date_annot = acquisition_date_annot[next(iter(acquisition_date_annot.keys()))]
        acquisition_date_arp_coeffs[acquisition_date] = {
            e: get_coefficients(f"Position.ARPPoly.{e}.Coefs", acquisition_date_annot) for e in ["X", "Y", "Z"]
        }
        acquisition_date_t0[acquisition_date] = np.datetime64(annot["Timeline"]["CollectStart"].rstrip("Z|z"), "ns")
        if any(coeff is None for coeff in acquisition_date_arp_coeffs[acquisition_date].values()):
            raise ValueError("Missing ARP polynomial coefficients in metadata.")
        # Determine the type of orbit data used for this acquisition.
        if ("conditions" in slc.orig[acquisition_date]) and ("orbit" in slc.orig[acquisition_date].conditions):
            orb_type = slc.orig[acquisition_date].conditions.orbit.attrs["orbit_type"]
        else:
            orb_type = "AUX_ANNOTATED"
        acquisition_date_orbit_type[acquisition_date] = orb_type

    quality["crg_accuracy"] = coreg_qlt(
        slc.coreg.measurements.ds,
        daz_drg,
        scpcoa_incidence_angle=scpcoa_incidence_angle,
        # coeffs cannot be None, ValueError would have been raised above
        coreg_arp_coeffs=coreg_arp_coeffs,  # type: ignore
        t0=t0,
        acquisition_date_arp_coeffs=acquisition_date_arp_coeffs,
        acquisition_date_orbit_type=acquisition_date_orbit_type,
        acquisition_date_t0=acquisition_date_t0,
    )

    # Generate the updated Noise LUT for the coregistered grid.

    if path_to_dict(annot, "Radiometric.NoiseLevel.NoisePoly.Coefs") is not None:
        quality = quality.assign(
            updateLut(
                slc,
                daz_drg,
                "Radiometric.NoiseLevel.NoisePoly.Coefs",
                name_mapping[NoiseLevel],
            )
        )
    # Generate updated LUTs for all specified radiometric calibration types.
    for calib in calib_ok:
        if path_to_dict(annot, f"Radiometric.{calib}SFPoly.Coefs") is not None:
            quality = quality.assign(
                updateLut(
                    slc,
                    daz_drg,
                    f"Radiometric.{calib}SFPoly.Coefs",
                    name_mapping[calib],
                )
            )
    return quality


def update_LUT(daz_drg: xr.Dataset, LUT_coords: xr.Dataset):
    """
    Scatter one chunk of computed daz/drg shifts into the coarse LUT grid.

    Selects the coarse-grid points falling inside the spatial extent of the
    given daz/drg chunk and copies the corresponding azimuth and range
    displacements into an otherwise-zero dataset shaped like ``LUT_coords``.
    The per-chunk results are summed by the caller to build the full lookup
    table.

    Parameters
    ----------
    daz_drg : xr.Dataset
        Chunk of the computed azimuth (``daz``) and range (``drg``)
        displacement grids on the fine grid.
    LUT_coords : xr.Dataset
        Coarse lookup-table coordinate grid to be updated.

    Returns
    -------
    xr.Dataset
        Zero-valued dataset shaped like ``LUT_coords`` carrying the
        ``daz``/``drg`` values of this chunk at the overlapping points.
    """
    mask = (
        (LUT_coords.azimuth_time >= daz_drg.azimuth_time.min())
        & (LUT_coords.azimuth_time <= daz_drg.azimuth_time.max())
        & (LUT_coords.slant_range_time >= daz_drg.slant_range_time.min())
        & (LUT_coords.slant_range_time <= daz_drg.slant_range_time.max())
    ).values.ravel()
    (ind,) = np.where(mask)
    # If there are valid points, update the LUT with the computed daz and drg values.
    LUT_assign = xr.zeros_like(LUT_coords)
    if ind.size:
        az_ind, rg_ind = np.unravel_index(ind, (LUT_coords.azimuth_time.size, LUT_coords.slant_range_time.size))
        az_ind_da = xr.DataArray(az_ind, dims="point")
        rg_ind_da = xr.DataArray(rg_ind, dims="point")
        # Update daz values in the LUT for the current acquisition date chunk.
        LUT_assign.daz[..., az_ind_da, rg_ind_da] = daz_drg.daz.sel(
            azimuth_time=LUT_coords.azimuth_time[az_ind_da],
            slant_range_time=LUT_coords.slant_range_time[rg_ind_da],
        )
        # Update drg values in the LUT for the current acquisition date chunk.
        LUT_assign.drg[..., az_ind_da, rg_ind_da] = daz_drg.drg.sel(
            azimuth_time=LUT_coords.azimuth_time[az_ind_da],
            slant_range_time=LUT_coords.slant_range_time[rg_ind_da],
        )
    return LUT_assign


def blockwise_warp_cslc(
    coreg_cstack: xr.DataArray,
    coreg_phase_model: xr.DataArray,
    daz: xr.DataArray,
    drg: xr.DataArray,
    coreg_acquisition_id: xr.DataArray,
    coregistration_reference_id: np.datetime64,
    orig_slcs: xr.DataTree,
    orig_etad: xr.DataTree,
    kernel: dict,
    do_reramp: bool = True,
    do_flattening: bool = True,
    x_poly: xr.DataArray | None = None,
    y_poly: xr.DataArray | None = None,
    chunk: dict | None = None,
):
    """
    Perform sinc interpolation and processing of a stack of SAR images.

    This function iterates through each acquisition in a coregistered SAR
    dataset, applies geometric and polynomial corrections, and warps the
    original SLC (Single Look Complex) image to the reference grid using
    sinc interpolation. It also handles optional phase adjustments like
    flattening and reramping.

    Parameters
    ----------
    coreg : xr.Dataset
        Coregistered SAR dataset containing the target grid.
    daz_drg : xr.Dataset
        A dataset with azimuth and slant-range adjustment values.
    orig : list
        The original SAR dataset used as a reference.
    kernel : dict
        A dictionary containing parameters for the sinc interpolation kernel.
    do_reramp : bool, optional
        If True, apply phase reramping to the warped image. Defaults to True.
    do_flattening : bool, optional
        If True, apply phase flattening. Defaults to False.
    x_poly : xr.DataArray or None, optional
        Optional polynomial corrections for the range (x dimension).
    y_poly : xr.DataArray or None, optional
        Optional polynomial corrections for the azimuth (y dimension).
    store_path : str or None, optional
        Path to save the processed SAR stack.
    chunk : object, optional
        The chunk configuration for saving data.

    Returns
    -------
    xr.Dataset
        The updated coregistered dataset with the warped cstack.
    """
    chunk = chunk or {}
    # Define a margin around the processing area based on the kernel size to avoid edge effects.
    margin = kernel["kernel_shape"] // 2 + 1

    # --- Main Loop: Process each acquisition date ---
    for i, acquisition_id in enumerate(coreg_acquisition_id.values):
        # Extract the original SLC image and its metadata for the current date.
        slc = orig_slcs.orig[acquisition_id].slc
        single_pol_annotation = slc.attrs[next(iter(slc.attrs.keys()))]
        min_tx_frequency = single_pol_annotation["RadarCollection"]["TxFrequency"]["Min"]
        max_tx_frequency = single_pol_annotation["RadarCollection"]["TxFrequency"]["Max"]
        dkcoa_polynomial_coeffs = get_coefficients("Grid.Col.DeltaKCOAPoly.Coefs", single_pol_annotation)
        timeca_polynomial_coeffs = get_coefficients("RMA.INCA.TimeCAPoly.Coefs", single_pol_annotation)
        if dkcoa_polynomial_coeffs is None or timeca_polynomial_coeffs is None:
            raise ValueError(f"Failed to extract polynomial coefficients for acquisition date {acquisition_id}")

        dtaz = timeca_polynomial_coeffs[1] * single_pol_annotation["Grid"]["Col"]["SS"] * 1e9
        dtrg = 2 * single_pol_annotation["Grid"]["Row"]["SS"] / speed_of_light
        scp_pixel_row = single_pol_annotation["ImageData"]["SCPPixel"]["Row"]
        scp_pixel_col = single_pol_annotation["ImageData"]["SCPPixel"]["Col"]
        etad_correction_grid = orig_etad[acquisition_id].ds
        evaluation_offset = {k: v.start for k, v in chunk.items()}
        warp_cslc(
            coreg_cstack=coreg_cstack,
            coreg_phase_model=coreg_phase_model,
            slc=slc,
            daz=daz.sel(acquisition_id=acquisition_id),
            drg=drg.sel(acquisition_id=acquisition_id),
            acquisition_id=acquisition_id,
            acquisition_date_index=i,
            coregistration_reference_id=coregistration_reference_id,
            etad_correction_grid=etad_correction_grid,
            kernel=kernel,
            azimuth_grid_spacing=dtaz,
            range_grid_spacing=dtrg,
            margin=margin,
            max_tx_frequency=max_tx_frequency,
            min_tx_frequency=min_tx_frequency,
            scp_pixel_col=scp_pixel_col,
            scp_pixel_row=scp_pixel_row,
            timeca_polynomial_coeffs=timeca_polynomial_coeffs,
            dkcoa_polynomial_coeffs=dkcoa_polynomial_coeffs,
            do_reramp=do_reramp,
            do_flattening=do_flattening,
            x_poly=x_poly[acquisition_id] if x_poly is not None else None,
            y_poly=y_poly[acquisition_id] if y_poly is not None else None,
            evaluation_offset=evaluation_offset,
        )
    coreg = xr.Dataset({"cstack": coreg_cstack, "phase_model": coreg_phase_model})
    return coreg


def updateLut(
    slcs: xr.DataTree, daz_drg: xr.Dataset, path_to_coeff: str, out_name: str, grid: tuple[int, int] = GRID, **kwargs
) -> xr.Dataset:
    """
    Update or create a Lookup Table (LUT) for a specific parameter.

    This function orchestrates the creation of a LUT (e.g., for Doppler
    frequency, sigma0, etc.) for each acquisition in a stack. It retrieves
    the necessary polynomial coefficients from the metadata, generates the
    LUT for the coregistered grid using `getLUT`, and stores it in a new
    xarray.Dataset with appropriate coordinates and attributes.

    Parameters
    ----------
    slcs : xr.DataTree
        The input DataTree containing original and coregistered data.
    daz_drg : xr.Dataset
        The dataset with azimuth and range displacement grids.
    path_to_coeff : str
        A string path (e.g., "a.b.c") to locate the
        polynomial coefficients within the metadata dictionary.
    out_name : str
        The name for the output data variable in the new Dataset (e.g., "doppler").
    **kwargs
        Additional parameters, including `GRID` for decimation.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset containing the newly computed LUT.
    """
    # Extract metadata from the coregistered stack.
    pols = slcs.coreg.measurements.polarization.values
    # --- Initialize Output LUT DataArray ---
    # Handle cases with and without a polarization dimension.
    if out_name in ["sigma0", "gamma0", "noise"]:
        xLUT = xr.DataArray(
            np.zeros((pols.size, slcs.coreg.measurements.acquisition_id.size, *grid)),
            dims=["polarization", "acquisition_id", "slant_range_time", "azimuth_time"],
            coords={"acquisition_id": slcs.coreg.measurements.acquisition_id.values, "polarization": pols},
        )
    else:
        xLUT = xr.DataArray(
            np.zeros((slcs.coreg.measurements.acquisition_id.size, *grid)),
            dims=["acquisition_id", "slant_range_time", "azimuth_time"],
            coords={"acquisition_id": slcs.coreg.measurements.acquisition_id.values},
        )
    # --- Main Loop: Compute LUT for each acquisition ---
    for i, dd in enumerate(slcs.coreg.measurements.acquisition_id.values):
        # Handle per-polarization coefficients if necessary.
        if "polarization" in xLUT.dims:
            for p, pol in enumerate(xLUT.polarization.values):
                annot = slcs.orig[dd].measurements.slc.attrs[pol]
                coeffs = get_coefficients(path_to_coeff, annot)
                if coeffs is None:
                    raise ValueError(f"Could not find coefficients for polarization {pol}")
                # compute Doppler on the new reference system
                xLUT[p, i, :, :], az_ref, rg_ref = getLUT(
                    slcs.coreg.measurements.sel(acquisition_id=dd),
                    daz_drg.sel(acquisition_id=dd),
                    slcs.orig[dd].measurements.to_dataset(),
                    coeffs,
                    grid,
                )
        else:
            annot = slcs.orig[dd].measurements.slc.attrs
            annot = annot[next(iter(annot.keys()))]
            coeffs = get_coefficients(path_to_coeff, annot)
            if coeffs is None:
                raise ValueError("Could not find coefficients for polarization")
            # compute Doppler on the new reference system
            xLUT[i, :, :], az_ref, rg_ref = getLUT(
                slcs.coreg.measurements.sel(acquisition_id=dd),
                daz_drg.sel(acquisition_id=dd),
                slcs.orig[dd].measurements.to_dataset(),
                coeffs,
                grid,
            )
    if coeffs is None:
        raise ValueError("Could not find coefficients")
    # --- Set Coordinates and Attributes ---
    # Extract reference point and scaling factors for the coregistered grid.
    best_pol = get_best_pol(pols)
    annot_ref = slcs.coreg.measurements.cstack.attrs[best_pol]
    try:
        rg0 = slcs.coreg.measurements.isel(
            slant_range_time=annot_ref["ImageData"]["SCPPixel"]["Row"]
        ).slant_range_time.values
        az0 = slcs.coreg.measurements.isel(azimuth_time=annot_ref["ImageData"]["SCPPixel"]["Col"]).azimuth_time.values
    except Exception as e:  # noqa: BLE001
        logger.warning(f"type is: {e.__class__.__name__}, fix for tests")
        rg0 = slcs.coreg.measurements.slant_range_time[0].values
        az0 = slcs.coreg.measurements.azimuth_time[0].values
    # Range scaling (meters).
    drg = speed_of_light / 2
    # Azimuth scaling (seconds).
    daz = 1.0 / (1e9 * annot_ref["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1])

    # Assign the correct coordinates to the LUT DataArray.
    xLUT = xLUT.assign_coords({"slant_range_time": rg_ref.values, "azimuth_time": az_ref.values})

    # Store the reference point, scaling factors, and polynomial degree as attributes.
    xLUT = xLUT.assign_attrs(
        {
            "reference": {"azimuth_time": az0.astype(str), "slant_range_time": float(rg0)},
            "scale": {"azimuth_time": daz, "slant_range_time": drg},
            "deg": {"azimuth_time": coeffs.shape[1], "slant_range_time": coeffs.shape[0]},
        }
    )

    # Convert the final DataArray to a Dataset.
    return xLUT.to_dataset(name=out_name)


def finalize_coreg(
    slcs: xr.DataTree,
    working_directory: str,
    reference_id: str,
    do_image_matching: bool = False,
    do_split_spectrum: bool = False,
    do_esd: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> xr.DataTree:
    """
    Inject the stored current reference image into the coregistered stack.

    When at least one of the ``do_image_matching``, ``do_split_spectrum`` or
    ``do_esd`` flags is enabled, loads the previously stored current
    reference for each burst or slice and concatenates it along the
    ``acquisition_id`` dimension into the coregistration measurements,
    sorted by acquisition date. If the reference id already exists in a
    node, the previous entry is renamed to ``{reference_id}_sec`` to avoid
    a collision, including its node under ``orig``. The reference image is
    also added as an ``orig`` SLC entry. Bursts without a current
    reference are dropped from the DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        Coregistered SLC DataTree to be extended with the reference image.
    working_directory : str
        Root directory containing the product stores, including the
        ``current_reference`` stores.
    reference_id : str
        Acquisition id to be assigned to the loaded current reference.
    do_image_matching : bool, optional
        Feature flag; when all flags are false the step is skipped.
        Defaults to False.
    do_split_spectrum : bool, optional
        Feature flag for the split-spectrum step. Defaults to False.
    do_esd : bool, optional
        Feature flag for the ESD step. Defaults to False.
    enable_dask : bool, optional
        Whether Dask may be used for sub-tasks. Defaults to True.
    **kwargs : dict
        Extra storage options used for reading the product stores.

    Returns
    -------
    xr.DataTree
        The DataTree with the current reference appended to the coregistration
        measurements and registered as an original SLC input.

    Raises
    ------
    ValueError
        If no current reference can be found for any burst or for the
        single data take, leaving an empty DataTree.
    """
    if not (do_image_matching or do_split_spectrum or do_esd):
        logger.info("Image matching and split spectrum disabled. Skipping current_reference loading.")
        return slcs

    storage_options = get_storage_options(kwargs)
    multiburst = "bursts" in slcs

    bursts_to_remove = []
    nodes = slcs.bursts if multiburst else {"data": slcs}

    for k, v in nodes.items():
        path = f"bursts/{k}" if multiburst else f"data/{reference_id}"
        current_reference_path = normpath(f"{working_directory}/{path}/current_reference")

        if exists(current_reference_path, storage_options=storage_options):
            coreg_ds = v.coreg.measurements.to_dataset()
            cr_ds = read(current_reference_path, as_type="xr.Dataset", storage_options=storage_options)
            target_node = slcs[path] if multiburst else slcs
            # -------------------------------------------------------------
            # HANDLE COLLISION: Rename reference_id -> reference_id_sec + WARNING
            # -------------------------------------------------------------
            sec_ref_id = f"{reference_id}_sec"
            c_ids = [str(x) for x in coreg_ds.acquisition_id.values]

            if reference_id in c_ids:
                logger.warning(
                    f"Reference ID '{reference_id}' already present in node"
                    f" '{k}'. Renaming previous reference entry to"
                    f" '{sec_ref_id}' to prevent collision."
                )

                # 1. Update acquisition_id list
                c_ids = [sec_ref_id if x == reference_id else x for x in c_ids]

                # 2. Rename child node under orig DataTree if it exists
                if reference_id in target_node.orig.children:
                    target_node.orig[sec_ref_id] = target_node.orig[reference_id]
                    del target_node.orig[reference_id]

            # Format new incoming reference dataset
            r_val = str(
                cr_ds.acquisition_date.values.item()
                if cr_ds.acquisition_date.ndim == 0
                else cr_ds.acquisition_date.values[0]
            )

            cr_ds = cr_ds.assign_coords(
                acquisition_id=("acquisition_id", [reference_id]),
                acquisition_date=("acquisition_id", [r_val]),
                azimuth_m=("azimuth_time", coreg_ds.azimuth_m.values),
                slant_range_m=("slant_range_time", coreg_ds.slant_range_m.values),
            )

            c_dates = [str(x) for x in coreg_ds.acquisition_date.values]
            coreg_ds = coreg_ds.assign_coords(
                acquisition_id=("acquisition_id", c_ids), acquisition_date=("acquisition_id", c_dates)
            )
            # Safely concatenate without coordinate duplicate issues
            new_coreg_ds = xr.concat([coreg_ds, cr_ds], dim="acquisition_id").sortby("acquisition_date")

            if multiburst:
                slcs[path].coreg.measurements = new_coreg_ds
                slcs[f"{path}/orig/{reference_id}/measurements"] = cr_ds.rename({"cstack": "slc"})
            else:
                slcs.coreg.measurements = new_coreg_ds
                slcs[f"orig/{reference_id}/measurements"] = cr_ds.rename({"cstack": "slc"})
        else:
            bursts_to_remove.append(k)

    if multiburst:
        for k in bursts_to_remove:
            logger.warning(f"Current reference missing for burst {k}. Dropping burst from DataTree.")
            del slcs.bursts[k]

        if len(slcs.bursts.children) == 0:
            raise ValueError("Current reference missing for all bursts. DataTree is empty.")
    else:
        if bursts_to_remove:
            raise ValueError("Current reference missing for the datatake. DataTree might be empty.")

    return slcs
