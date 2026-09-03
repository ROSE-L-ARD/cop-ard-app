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
from cop_ard_app.io import Path, read, write, normpath
from cop_ard_app.model import get_carrier_frequency
from cop_ard_app.utils import (
    get_bounding_box,
    split_slc_datatrees,
    compute_iono_and_orb_height,
    recompose_slc_datatrees,
)


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
    multiburst: bool = True,
    enable_dask: bool = True,
    chunks={"azimuth_time": 4096, "slant_range_time": 4096},
    reference_id: str | None = None,
):

    # define center band frequencies
    f0 = get_carrier_frequency(reference_slcs, "coreg/measurements/cstack", multiburst=multiburst)
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
        multiburst=multiburst,
        chunks=chunks,
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
        multiburst=multiburst,
        chunks=chunks,
    )

    if wdir_path is not None:
        # save to disk
        mosaic_l_path = wdir_path.append("mosaic_l")
        mosaic_l = cast(xr.Dataset, write(mosaic_l, data_path=mosaic_l_path, mode="w"))
        mosaic_h_path = wdir_path.append("mosaic_h")
        mosaic_h = cast(xr.Dataset, write(mosaic_h, data_path=mosaic_h_path, mode="w"))

    # joint interferogram phase unwrapping
    phase_unw_l, phase_unw_h, coherence = joint_unwrap(
        mosaic_l.coherence, mosaic_h.coherence, mosaic_l.phase, mosaic_h.phase, cohe_th=cohe_th
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
        phase_iono = write(phase_iono, data_path=phase_iono_path, mode="w")
        phase_nd_path = wdir_path.append("phase_nd")
        phase_nd = write(phase_nd, data_path=phase_nd_path, mode="w")

    return phase_iono, phase_nd


def remove_iono_surface_chunk(
    cstack: xr.DataArray,
    poly_corr: xr.DataArray,
    TimeCOAPoly: np.ndarray,
    CollectStart: np.datetime64,
    H_iono: xr.DataArray | float,
    H_orb: xr.DataArray | float,
) -> xr.Dataset:

    # compute raw  azimuth time
    azimuth_time_raw = get_azimuth_time_raw(cstack, TimeCOAPoly, CollectStart)

    surf = reconstruct_iono_surface(
        cstack.azimuth_time, cstack.slant_range_time, azimuth_time_raw, poly_corr, H_iono, H_orb
    )

    cstack *= np.exp(-1j * surf)

    ds = xr.Dataset(
        {
            "cstack": cstack,
            "phase_iono": surf.expand_dims({"acquisition_id": cstack.acquisition_id.values}),
        }
    )

    return ds


def remove_iono_surface_single_burst(
    burst_id: str,
    coreg: xr.Dataset,
    poly_corr: xr.DataArray,
    H_iono: xr.DataArray | float,
    H_orb: xr.DataArray | float,
    wdir_path: Path,
    multiburst: bool,
    enable_dask: bool = True,
    reference_id: str | None = None,
) -> tuple[str, xr.Dataset]:

    if (not multiburst) and (reference_id is None):
        raise ValueError(f"In single burst case, the reference_id is mandatory")
    path = f"bursts/{burst_id}" if multiburst else f"data/{reference_id}"
    data_path = wdir_path.append(path).append("coreg_ss/measurements")
    # recontruct the surface over all points
    cstack = coreg.cstack

    # select preferred polarization
    pol = get_best_pol(cstack.polarization.values)
    annot_pol = cstack.attrs[pol]

    TimeCOAPoly = annot_pol["Grid"]["TimeCOAPoly"]["Coefs"]
    CollectStart = np.datetime64(annot_pol["Timeline"]["CollectStart"].rstrip("Z|z"), "ns")

    # init output file structure
    coreg_ss = xr.Dataset({"cstack": cstack})
    coreg_ss["phase_iono"] = xr.zeros_like(cstack.isel(polarization=0), dtype="float32")
    write(coreg_ss, data_path, skip_read=True)

    store_mapping = {
        0: {
            "data_path": f"{wdir_path}/{path}/coreg_ss/measurements",
            "save_options": {
                "skip_read": True,
                "consolidated": False,
            },
        },
    }

    map_blocks(
        remove_iono_surface_chunk,
        cstack,
        args=(poly_corr, TimeCOAPoly, CollectStart, H_iono, H_orb),
        combine=False,
        store_mapping=store_mapping,
        enable_dask=enable_dask,
    )

    coreg_ss = read(data_path, as_type="xr.Dataset")

    return (burst_id, coreg_ss)


@timeit
def remove_iono_surface(
    slcs: xr.DataTree,
    poly_corr: xr.DataArray,
    H_iono: float,
    H_orb: float,
    wdir_path: Path,
    multiburst: bool,
    enable_dask: bool = True,
    reference_id: str | None = None,
) -> xr.DataTree:

    args = [
        (burst_id, slc.coreg.measurements.to_dataset())
        for burst_id, slc in (slcs.bursts if multiburst else {"data": slcs}).items()
    ]
    args = list(zip(*args))

    # update all bursts
    results = map(
        remove_iono_surface_single_burst,
        args=args,
        kwargs={
            "poly_corr": poly_corr,
            "H_iono": H_iono,
            "H_orb": H_orb,
            "wdir_path": wdir_path,
            "enable_dask": enable_dask,
            "multiburst": multiburst,
            "reference_id": reference_id,
        },
        enable_dask=enable_dask,
        combine=False,
    )

    for k, v in results:
        if multiburst:
            slcs.bursts[k].coreg.measurements["cstack"] = v.cstack
            slcs.bursts[k].coreg.measurements["phase_iono"] = v.phase_iono
        else:
            slcs.coreg.measurements["cstack"] = v.cstack
            slcs.coreg.measurements["phase_iono"] = v.phase_iono

    return slcs


def split_spectrum(
    slcs: xr.DataTree,
    multilook: dict,
    frequency_offset: float,
    bandwidth: float,
    cohe_th: float,
    wdir_path: str,
    multiburst: bool,
    reference_id: str,
    esd_poly_degree: int = 2,
    split_spectrum_poly_degree: int = 2,
    h_iono: float | None = None,
    enable_dask: bool = True,
):
    wdir_path = Path(wdir_path)
    reference_slcs, secondary_slcs = split_slc_datatrees(
        slcs=slcs,
        reference_id=reference_id,
        multiburst=multiburst,
    )
    h_orb, h_iono = compute_iono_and_orb_height(
        reference_slcs=reference_slcs,
        secondary_slcs=secondary_slcs,
        wdir_path_tmp=wdir_path,
        acquisition_id=reference_id,
        iono_height_input=h_iono,
    )

    # evaluate bounding box
    bbox = tuple(get_bounding_box(reference_slcs, "coreg/measurements", multiburst=multiburst))

    # estimate ESD at full-band
    if multiburst:
        esd = estimate(
            reference_slcs,
            secondary_slcs,
            poly_degree=esd_poly_degree,
            enable_dask=enable_dask,
        )

        # fit polynomial function to ESD and remove it from the interferogram
        y_poly = find_global_shift(esd, poly_degree=esd_poly_degree)
        # y_poly = None
        # separate dispersive and non-dispersive components
        logger.debug("separate dispersive and non-dispersive components")
    else:
        y_poly = None

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
        reference_id=reference_id,
        multiburst=multiburst,
        enable_dask=enable_dask,
    )
    # fit surface at ionospheric height
    logger.debug("fit surface at ionospheric height")
    poly_corr = fit_ionosphere(
        phase_iono.phase,
        phase_iono.azimuth_time_raw,
        h_iono,
        h_orb,
        degree=split_spectrum_poly_degree,
        weights=phase_iono.coherence,
    )

    # remove ionosphere from cslc and update structure
    logger.debug("remove ionosphere from cslc and update structure")
    secondary_slcs = remove_iono_surface(
        secondary_slcs,
        poly_corr,
        h_iono,
        h_orb,
        wdir_path=wdir_path,
        multiburst=multiburst,
        enable_dask=enable_dask,
        reference_id=reference_id,
    )
    slcs = recompose_slc_datatrees(
        reference_slcs,
        secondary_slcs,
        multiburst=multiburst,
    )
    return slcs
