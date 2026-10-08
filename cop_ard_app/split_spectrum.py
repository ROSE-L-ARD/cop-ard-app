from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.logger import logger, timeit
from cop_ard_core.split_spectrum import (
    fit_ionosphere,
    joint_unwrap,
    joint_unwrap_delta_mean,
    reconstruct_iono_surface,
    split_spectrum_core,
    split_spectrum_core_delta_mean,
)
from cop_ard_core.utils import get_azimuth_time_raw
from numpy.polynomial.polynomial import polygrid2d

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.esd import estimate, find_global_shift
from cop_ard_app.interf import get_mosaic_interf
from cop_ard_app.io import Path, read, write
from cop_ard_app.model import get_carrier_frequency
from cop_ard_app.utils import (
    build_coreg_poly_dict,
    compute_iono_and_orb_height,
    get_best_pol,
    get_bounding_box,
    recompose_slc_datatrees,
    split_slc_datatrees,
)


def get_phase_corr_from_x_poly(
    azimuth_time: xr.DataArray, slant_range_time: xr.DataArray, x_poly: xr.DataArray, freq: float
) -> xr.DataArray:

    out_coords = {"azimuth_time": azimuth_time.values, "slant_range_time": slant_range_time.values}

    az_sc = (
        (azimuth_time - np.datetime64(x_poly.attrs["off_y"], "ns")).astype(float) - x_poly.offset[0]
    ) / x_poly.scale_fact[0]
    rg_sc = (slant_range_time - x_poly.offset[1]) / x_poly.scale_fact[1]

    rg_corr = xr.DataArray(
        polygrid2d(az_sc.values, rg_sc.values, x_poly.values),
        dims=["azimuth_time", "slant_range_time"],
        coords=out_coords,
    )
    return 2 * np.pi * freq * rg_corr


@timeit
def separate_dispersive_spectrum_components(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    frequency_offset: float,
    bandwidth: float,
    cohe_th: float,
    a_min: float,
    r_min: float,
    x_poly: xr.Dataset,
    y_poly: xr.Dataset,
    wdir_path: Path,
    multilook: dict | None = None,
    multiburst: bool = True,
    enable_dask: bool = True,
    chunks=None,
    reference_id: str | None = None,
):
    if chunks is None:
        chunks = {"azimuth_time": 4096, "slant_range_time": 4096}
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

    if x_poly is not None:
        # compute phase correction to be applied to each subband
        ph_corr = get_phase_corr_from_x_poly(
            mosaic_l.azimuth_time,
            mosaic_l.slant_range_time,
            x_poly[mosaic_l.acquisition_id.values[0]],
            frequency_offset,
        )
        logger.debug(f"APPLY CORRECTIONS {ph_corr.min().values, ph_corr.max().values}")
    else:
        ph_corr = 0.0

    mosaic_l["phase"] = mosaic_l["phase"] + ph_corr

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
    mosaic_h["phase"] = mosaic_h["phase"] - ph_corr

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
    phase_iono_ds = phase_iono.to_dataset(name="phase")
    # assign azimuth time of raw image in order to enable interpolation
    phase_iono_ds["azimuth_time_raw"] = mosaic_l["azimuth_time_raw"]
    phase_iono_ds["coherence"] = coherence

    if wdir_path is not None:
        # save to disk
        phase_iono_path = wdir_path.append("phase_iono")
        write(phase_iono_ds, data_path=phase_iono_path, mode="w", skip_read=True)
        phase_iono_ds = read(phase_iono_path, as_type="xr.Dataset")
        phase_nd_path = wdir_path.append("phase_nd")
        write(phase_nd, data_path=phase_nd_path, mode="w")
        phase_nd = read(phase_nd_path, as_type="xr.DataArray")

    return phase_iono_ds, phase_nd


@timeit
def separate_dispersive_spectrum_components_delta_mean(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    frequency_offset: float,
    bandwidth: float,
    cohe_th: float,
    a_min: float,
    r_min: float,
    x_poly: xr.Dataset,
    y_poly: xr.Dataset,
    wdir_path: Path,
    multilook: dict | None = None,
    multiburst: bool = True,
    enable_dask: bool = True,
    chunks=None,
    reference_id: str | None = None,
):
    """
    Estimate dispersive and non-dispersive phase components using the
    split-spectrum method and the delta/mean phase formulation.

    This workflow generates low-frequency and high-frequency sub-band
    interferograms, jointly unwraps their differential and mean phases,
    and separates the resulting phase into dispersive (ionospheric) and
    non-dispersive components. Optionally, frequency-dependent phase
    corrections can be applied before the split-spectrum inversion.

    Parameters
    ----------
    reference_slcs
        Reference SLC stack used to form the interferograms.
    secondary_slcs
        Secondary SLC stack used to form the interferograms.
    frequency_offset
        Frequency offset between the carrier frequency and the center
        frequency of each sub-band [Hz].
    bandwidth
        Bandwidth of the sub-bands used for split-spectrum processing [Hz].
    cohe_th
        Coherence threshold used during phase unwrapping.
    a_min
        Minimum azimuth oversampling factor or spacing passed to the
        interferogram generation routine.
    r_min
        Minimum range oversampling factor or spacing passed to the
        interferogram generation routine.
    x_poly
        Polynomial phase-correction model applied with opposite signs to
        the two sub-bands in order to compensate frequency-dependent phase
        errors. If ``None``, no correction is applied.
    y_poly
        Polynomial model used during interferogram generation.
    wdir_path
        Working directory used to persist intermediate products.
        If ``None``, no products are written to disk.
    multilook
        Optional multilooking specification passed to interferogram
        generation.
    multiburst
        If ``True``, process data as a mosaic of bursts. Default is
        ``True``.
    enable_dask
        Enable Dask-backed processing. Default is ``True``.
    chunks
        Chunking scheme used for Dask processing.
        Defaults to {"azimuth_time": 4096, "slant_range_time": 4096}
    reference_id
        Optional identifier of the reference acquisition.

    Returns
    -------
    tuple[xr.Dataset, xr.DataArray]
        A tuple containing:

        - **phase_iono**: Dispersive (ionospheric) phase component stored
          as a dataset. The dataset contains the estimated phase, coherence,
          and raw azimuth-time coordinates.
        - **phase_nd**: Non-dispersive phase component.

    Notes
    -----
    The processing chain consists of the following steps:

    1. Determine the carrier frequency and the center frequencies of the
       low-band and high-band sub-spectra.
    2. Generate low-band and high-band interferograms.
    3. Apply optional frequency-dependent phase corrections.
    4. Jointly unwrap the mean and differential phases of the two
       sub-bands.
    5. Separate dispersive and non-dispersive phase contributions using
       :func:`split_spectrum_core_delta_mean`.
    6. Store coherence and auxiliary coordinates required for subsequent
       interpolation and modeling steps.

    The returned dispersive phase is intended to represent the ionospheric
    contribution, while the non-dispersive phase mainly contains geometric,
    deformation, and other frequency-independent signals.
    """
    if chunks is None:
        chunks = {"azimuth_time": 4096, "slant_range_time": 4096}
    # define center band frequencies
    f0 = get_carrier_frequency(
        reference_slcs,
        "coreg/measurements/cstack",
        multiburst=multiburst,
    )
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

    if x_poly is not None:
        # compute phase correction to be applied to each subband
        ph_corr = get_phase_corr_from_x_poly(
            mosaic_l.azimuth_time,
            mosaic_l.slant_range_time,
            x_poly[mosaic_l.acquisition_id.values[0]],
            frequency_offset,
        )
        logger.debug(f"APPLY CORRECTIONS {ph_corr.min().values, ph_corr.max().values}")
    else:
        ph_corr = 0.0

    mosaic_l["phase"] = mosaic_l["phase"] + ph_corr

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
    mosaic_h["phase"] = mosaic_h["phase"] - ph_corr

    if wdir_path is not None:
        # save to disk
        mosaic_l_path = wdir_path.append("mosaic_l")
        mosaic_l = cast(
            xr.Dataset,
            write(mosaic_l, data_path=mosaic_l_path, mode="w"),
        )

        mosaic_h_path = wdir_path.append("mosaic_h")
        mosaic_h = cast(
            xr.Dataset,
            write(mosaic_h, data_path=mosaic_h_path, mode="w"),
        )

    # joint interferogram phase unwrapping
    phase_delta, phase_mean, coherence = joint_unwrap_delta_mean(
        mosaic_l.coherence,
        mosaic_h.coherence,
        mosaic_l.phase,
        mosaic_h.phase,
        cohe_th=cohe_th,
    )

    if "phase_ramp" in mosaic_l.data_vars:
        phase_mean = phase_mean + mosaic_l.phase_ramp

    # separate dispersive and non-dispersive components
    phase_iono, phase_nd = split_spectrum_core_delta_mean(
        phase_delta,
        phase_mean,
        f0,
        f_l_tot,
        f_h_tot,
    )

    phase_iono_ds = phase_iono.to_dataset(name="phase")

    # assign azimuth time of raw image in order to enable interpolation
    phase_iono_ds["azimuth_time_raw"] = mosaic_l["azimuth_time_raw"]
    phase_iono_ds["coherence"] = coherence

    if wdir_path is not None:
        # save to disk
        phase_iono_path = wdir_path.append("phase_iono")
        write(
            phase_iono_ds,
            data_path=phase_iono_path,
            mode="w",
        )
        phase_iono_ds = read(phase_iono_path, as_type="xr.Dataset")

        phase_nd_path = wdir_path.append("phase_nd")
        write(
            phase_nd,
            data_path=phase_nd_path,
            mode="w",
            skip_read=True,
        )
        phase_nd = read(phase_nd_path, as_type="xr.DataArray")

    return phase_iono_ds, phase_nd


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
        raise ValueError("In single burst case, the reference_id is mandatory")
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
    if not results or not isinstance(results[0], tuple) or len(results[0]) != 2:
        raise ValueError("remove_iono_surface_single_burst returned invalid results")

    for k, v in cast(list[tuple[str, xr.Dataset]], results):
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
    x_poly: xr.Dataset | None = None,
    dbfile: str | None = None,
):
    wdir_path = Path(wdir_path)  # type: ignore[assignment]
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
        dbfile=dbfile,
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

        for id in y_poly.data_vars:
            logger.debug(f"ESD {id}: {y_poly[id].values}")
    else:
        y_poly = None

    # separate dispersive and non-dispersive components
    phase_iono, _ = separate_dispersive_spectrum_components_delta_mean(
        reference_slcs,
        secondary_slcs,
        frequency_offset=frequency_offset,
        bandwidth=bandwidth,
        cohe_th=cohe_th,
        a_min=bbox[0],
        r_min=bbox[2],
        x_poly=x_poly,
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

    attrs = {
        "split_spectrum": build_coreg_poly_dict(
            description="Split Spectrum",
            y_poly=y_poly,
            x_poly=x_poly,
            iono_poly=poly_corr.to_dataset(name=reference_id),
        )
    }

    slcs.attrs.setdefault("coregistration_polynomials", {}).update(attrs)
    return slcs
