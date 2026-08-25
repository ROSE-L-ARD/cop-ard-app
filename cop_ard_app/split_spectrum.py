import os
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.logger import logger, timeit
from cop_ard_core.split_spectrum import fit_ionosphere, joint_unwrap, reconstruct_iono_surface, split_spectrum_core
from cop_ard_core.utils import get_azimuth_time_raw, get_best_pol

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.esd import estimate, find_global_shift
from cop_ard_app.interf import get_mosaic_interf
from cop_ard_app.io import Path, read, write
from cop_ard_app.model import get_carrier_frequency
from cop_ard_app.utils import get_bounding_box


@timeit
def separate_dispersive_spectrum_components(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    frequency_offset: float,
    bandwidth: float,
    cohe_th: float,
    a_min: float,
    r_min: float,
    y_poly: xr.Dataset,
    wdir_path: Path,
    multilook: dict | None = None,
    enable_dask: bool = True,
    force_compute: bool = True,
):

    # define center band frequencies
    f0 = get_carrier_frequency(reference_slcs.bursts, "coreg/measurements/cstack")
    f_l_tot = f0 - frequency_offset
    f_h_tot = f0 + frequency_offset

    # lower band interferogram
    mosaic_l = get_mosaic_interf(
        reference_slcs,
        secondary_slcs,
        f0=-frequency_offset,
        bw=bandwidth,
        a_min=a_min,
        r_min=r_min,
        multilook=multilook,
        y_poly=y_poly,
        enable_dask=enable_dask,
        force_compute=force_compute,
    )

    # upper band interferogram
    mosaic_h = get_mosaic_interf(
        reference_slcs,
        secondary_slcs,
        f0=frequency_offset,
        bw=bandwidth,
        a_min=a_min,
        r_min=r_min,
        multilook=multilook,
        y_poly=y_poly,
        enable_dask=enable_dask,
        force_compute=force_compute,
    )

    if wdir_path is not None:
        # save to disk
        mosaic_l_path = wdir_path.append("mosaic_l")
        mosaic_l = cast(xr.Dataset, write(mosaic_l, data_path=mosaic_l_path, mode="w"))
        mosaic_h_path = wdir_path.append("mosaic_h")
        mosaic_h = cast(xr.Dataset, write(mosaic_h, data_path=mosaic_h_path, mode="w"))

    # joint interferogram phase unwrapping
    phase_unw_l, phase_unw_h, coherence = joint_unwrap(
        mosaic_l.cohe, mosaic_h.cohe, mosaic_l.phase, mosaic_h.phase, cohe_th=cohe_th
    )
    phase_tot_l = phase_unw_l + mosaic_l.phase_ramp if "phase_ramp" in mosaic_l.data_vars else phase_unw_l
    phase_tot_h = phase_unw_h + mosaic_h.phase_ramp if "phase_ramp" in mosaic_h.data_vars else phase_unw_h

    # separate dispersive and non-dispersive components
    phase_iono, phase_nd = split_spectrum_core(phase_tot_l, phase_tot_h, f0, f_l_tot, f_h_tot)
    phase_iono = phase_iono.to_dataset(name="phase")
    # assign azimuth time of raw image in order to enable interpolation
    phase_iono["azimuth_time_raw"] = mosaic_l["azimuth_time_raw"]
    phase_iono["coherence"] = coherence

    if wdir_path is not None:
        # save to disk
        phase_iono_path = wdir_path.append("phase_iono")
        logger.debug(f"save phase_iono to file {phase_iono_path}")
        phase_iono = write(phase_iono, data_path=phase_iono_path, mode="w")
        phase_nd_path = wdir_path.append("phase_nd")
        logger.debug(f"save phase_nd to file {phase_nd_path}")
        phase_nd = write(phase_nd, data_path=phase_nd_path, mode="w")

    return phase_iono, phase_nd


def remove_iono_surface_chunk(
    cstack: xr.DataArray,
    coeffs: xr.DataArray,
    TimeCOAPoly: np.ndarray,
    CollectStart: np.ndarray,
    H_iono: xr.DataArray | float,
    H_orb: xr.DataArray | float,
    degree: int = 1,
) -> tuple[xr.DataArray, xr.DataArray] | None:

    # compute raw  azimuth time
    azimuth_time_raw = get_azimuth_time_raw(cstack, TimeCOAPoly, CollectStart)

    surf = reconstruct_iono_surface(
        cstack.azimuth_time, cstack.slant_range_time, azimuth_time_raw, coeffs, H_iono, H_orb, degree=degree
    )

    cstack *= np.exp(-1j * surf)

    return xr.Dataset(
        {"cstack": cstack, "phase_iono": surf.expand_dims({"acquisition_id": cstack.acquisition_id.values})}
    )


def remove_iono_surface_single_burst(
    burst_id: str,
    burst: xr.Dataset,
    coeffs: xr.DataArray,
    H_iono: xr.DataArray | float,
    H_orb: xr.DataArray | float,
    outfolder: Path,
    degree: int = 1,
    enable_dask: bool = True,
) -> tuple[str, xr.DataTree] | None:

    # recontruct the surface over all points
    cstack = burst.cstack

    # select preferred polarization
    pol = get_best_pol(cstack.polarization.values)
    annot_pol = cstack.attrs[pol]

    TimeCOAPoly = annot_pol["Grid"]["TimeCOAPoly"]["Coefs"]
    CollectStart = np.datetime64(annot_pol["Timeline"]["CollectStart"][:-1], "ns")

    # init output file structure
    burst["phase_iono"] = xr.zeros_like(burst["cstack"].isel(polarization=0), dtype="float32")
    write(burst, outfolder.append(f"bursts/{burst_id}/coreg/measurements"), skip_read=True)

    store_mapping = {
        0: {
            "data_path": outfolder.append(f"bursts/{burst_id}/coreg/measurements"),
            # "dims": ["acquisition_date", "azimuth_time", "slant_range_time"],
            "save_options": {
                "skip_read": True,
                "consolidated": False,
            },
        },
    }

    map_blocks(
        remove_iono_surface_chunk,
        cstack,
        args=(coeffs, TimeCOAPoly, CollectStart, H_iono, H_orb, degree),
        enable_dask=enable_dask,
        combine=False,
        store_mapping=store_mapping,  # store_directory=outfolder.append(f"bursts/{burst_id}/coreg/measurements"),
    )

    return


@timeit
def remove_iono_surface(
    slcs: xr.DataTree,
    coeffs: xr.DataArray,
    H_iono: float,
    H_orb: float,
    outfolder: Path,
    degree: int = 1,
    enable_dask: bool = True,
) -> xr.DataTree:

    # init out datatree
    outdatatree = xr.DataTree()
    for burst_id in slcs.bursts.keys():
        outdatatree[f"bursts/{burst_id}/coreg/measurements"] = xr.DataTree()
    logger.debug(outfolder)
    write(outdatatree, data_path=outfolder, skip_read=True)

    args = [(burst_id, slc.coreg.measurements.to_dataset()) for burst_id, slc in slcs.bursts.items()]
    args = list(zip(*args))

    # update all bursts

    map(
        remove_iono_surface_single_burst,
        args=args,
        kwargs={
            "coeffs": coeffs,
            "H_iono": H_iono,
            "H_orb": H_orb,
            "degree": degree,
            "outfolder": outfolder,
            "enable_dask": enable_dask,
        },
        enable_dask=enable_dask,
        combine=False,
    )

    return read(data_path=outfolder, consolidate_metadata=True)


def split_spectrum(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    multilook: dict,
    frequency_offset: float,
    bandwidth: float,
    cohe_th: float,
    H_iono: float,
    H_orb: float,
    wdir_path: Path,
    esd_poly_degree: int = 2,
    split_spectrum_poly_degree: int = 2,
    norm: int = 1,
    enable_dask: bool = True,
    force_compute: bool = True,
):

    # evaluate bounding box
    bbox = tuple(get_bounding_box(reference_slcs.bursts, "coreg/measurements"))

    # estimate ESD at full-band

    esd = estimate(
        reference_slcs,
        secondary_slcs,
        poly_degree=esd_poly_degree,
        enable_dask=enable_dask,
        force_compute=force_compute,
    )
    logger.debug(f"esd: {esd}")

    # fit polynomial function to ESD and remove it from the interferogram
    y_poly = find_global_shift(esd, poly_degree=esd_poly_degree)
    # y_poly = None
    # separate dispersive and non-dispersive components
    logger.debug("separate dispersive and non-dispersive components")
    phase_iono, phase_nd = separate_dispersive_spectrum_components(
        reference_slcs,
        secondary_slcs,
        frequency_offset=frequency_offset,
        bandwidth=bandwidth,
        cohe_th=cohe_th,
        a_min=bbox[0],
        r_min=bbox[2],
        y_poly=y_poly,
        multilook=multilook,
        wdir_path=wdir_path,
        enable_dask=enable_dask,
        force_compute=force_compute,
    )

    # fit surface at ionospheric height
    logger.debug("fit surface at ionospheric height")
    coeffs = fit_ionosphere(
        phase_iono.phase,
        phase_iono.azimuth_time_raw,
        H_iono,
        H_orb,
        degree=split_spectrum_poly_degree,
        norm=norm,
        weights=phase_iono.coherence,
    )

    # remove ionosphere from cslc and update structure
    logger.debug("remove ionosphere from cslc and update structure")
    secondary_slcs_filt = remove_iono_surface(
        secondary_slcs,
        coeffs,
        H_iono,
        H_orb,
        degree=split_spectrum_poly_degree,
        outfolder=Path(os.path.join(str(wdir_path), "slc_filt")),
        enable_dask=enable_dask,
    )

    return secondary_slcs_filt
