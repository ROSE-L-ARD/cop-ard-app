import json
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.logger import timeit
from cop_ard_core.utils import compute_interf_norm, time_to_pixel, wrap, get_azimuth_time_raw

from cop_ard_app.computation import map
from cop_ard_app.utils import (
    extract_subband,
    get_best_pol,
    get_stack_phramp_diff,
)


# @timeit
def update_interf_coordinates(interf: xr.Dataset, multilook: dict) -> xr.Dataset:

    # compute pixel coordinates
    y0 = multilook["azimuth_time"] / 2 - 0.5
    x0 = multilook["slant_range_time"] / 2 - 0.5

    # set multilooked coordinates
    interf = interf.assign_coords(
        {
            "y": (["azimuth_time"], np.round((interf.y - y0) / multilook["azimuth_time"]).values.astype(int)),
            "x": (["slant_range_time"], np.round((interf.x - x0) / multilook["slant_range_time"]).values.astype(int)),
        }
    )

    return interf


# @timeit
def get_interf_single_burst(cslc_ref: xr.DataArray, cslc_sec: xr.DataArray, multilook: dict) -> xr.Dataset:
    """
    Compute the interferometric phase and coherence with respect to a reference date
    applying a boxcar filter
    """

    # multilook of complex interferogram
    interf_norm = compute_interf_norm(cslc_ref.squeeze(), cslc_sec, multilook)

    # init empty dataset
    interf_ds = xr.Dataset()

    # extract phase
    interf_ds["phase"] = xr.ufuncs.angle(interf_norm)

    # extract coherence
    interf_ds["coherence"] = xr.ufuncs.abs(interf_norm)

    return interf_ds


# @timeit
def extract_coreg_aoi(coreg: xr.Dataset, a_min: float, r_min: float, multilook: dict) -> xr.Dataset:

    # extract imput chuncking
    chunk_burst = {k: v[0] for k, v in coreg.chunksizes.items() if k in ["azimuth_time", "slant_range_time"]}

    # add pixel coordinates wrt the AOI
    coreg_sel = cast(xr.Dataset, time_to_pixel(coreg, a_min, r_min))

    x_min = np.ceil(coreg_sel.x.min() / multilook["slant_range_time"]).astype(int) * multilook["slant_range_time"]
    x_max = np.floor(coreg_sel.x.max() / multilook["slant_range_time"]).astype(int) * multilook["slant_range_time"]
    y_min = np.ceil(coreg_sel.y.min() / multilook["azimuth_time"]).astype(int) * multilook["azimuth_time"]
    y_max = np.floor(coreg_sel.y.max() / multilook["azimuth_time"]).astype(int) * multilook["azimuth_time"]

    # select and force chuncking and swap back coordinates
    coreg_sel = coreg_sel.sel(x=slice(x_min, x_max + 1), y=slice(y_min, y_max + 1))
    coreg_sel = coreg_sel.swap_dims(y="azimuth_time", x="slant_range_time").chunk(chunk_burst)

    return coreg_sel


@timeit
def extract_interf_single(
    burst_id: str,
    annot: str,
    coreg_ref: xr.Dataset,
    coreg_sec: xr.Dataset,
    a_min: float,
    r_min: float,
    multilook: dict,
    y_poly: xr.Dataset | None = None,
    f0: float | None = None,
    bw: float | None = None,
):
    # decompress attributes
    annot_dict = json.loads(annot)

    # extract aoi
    coreg_ref_sel = extract_coreg_aoi(coreg_ref, a_min, r_min, multilook)
    coreg_sec_sel = extract_coreg_aoi(coreg_sec, a_min, r_min, multilook)

    # extract subbands
    # logger.debug(f"cslc shape: {coreg_sel.cstack.shape}")
    if (f0 is not None) and (bw is not None):
        # logger.debug("Computing full range spectrum...")
        cslc_ref = extract_subband(coreg_ref_sel.cstack, bw, f0)
        cslc_sec = extract_subband(coreg_sec_sel.cstack, bw, f0)
    else:
        cslc_ref = coreg_ref_sel.cstack
        cslc_sec = coreg_sec_sel.cstack

    # remove uselsess attributes
    cslc_ref = cslc_ref.drop_attrs()
    cslc_sec = cslc_sec.drop_attrs()

    # logger.debug(f"cslc shape: {cslc.shape}")
    # logger.debug(f"extract multilooked interferogram and coherence")
    # extract multilooked interferogram and coherence
    interf = get_interf_single_burst(cslc_ref, cslc_sec, multilook)

    # force coordinates
    interf = update_interf_coordinates(interf, multilook)

    # add raw image azimuth times
    TimeCOAPoly = annot_dict[interf.acquisition_id.values[0]]["Grid"]["TimeCOAPoly"]["Coefs"]
    CollectStart = np.datetime64(annot_dict[interf.acquisition_id.values[0]]["Timeline"]["CollectStart"][:-1], "ns")
    azimuth_time_raw = get_azimuth_time_raw(coreg_ref_sel["cstack"], TimeCOAPoly, CollectStart, multilook)
    interf["azimuth_time_raw"] = (["azimuth_time", "slant_range_time"], azimuth_time_raw.values)

    if y_poly is not None:
        # logger.debug("save phase_ramp")
        # compute multilooked phase corrections
        ph_ramp_diff_ml = get_stack_phramp_diff(
            interf.acquisition_id,
            interf.azimuth_time,
            interf.slant_range_time,
            coreg_sec_sel.drg,
            coreg_sec_sel.daz,
            annot_dict,
            y_poly,
            multilook,
        )
        ph_ramp_diff_ml["slant_range_time"] = interf["slant_range_time"]
        ph_ramp_diff_ml["azimuth_time"] = interf["azimuth_time"]

        # apply correction
        interf["phase"] = wrap(interf["phase"] - ph_ramp_diff_ml)
        interf["phase_ramp"] = ph_ramp_diff_ml

    return (burst_id, interf)


@timeit
def extract_interf_multi(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    enable_dask: bool = True,
    force_compute: bool = True,
    **kwargs,
):
    name = list(reference_slcs.bursts)[0]
    available_pols_ref = {str(i) for i in reference_slcs.bursts[name].coreg.measurements.polarization.values}
    available_pols_sec = {str(i) for i in secondary_slcs.bursts[name].coreg.measurements.polarization.values}
    pol = get_best_pol(list(set(available_pols_ref).intersection(set(available_pols_sec))))

    args = [
        (
            burst_id,
            json.dumps({k: v.measurements.slc.attrs[pol] for k, v in slc_ref.orig.items()}),
            slc_ref.coreg.measurements.ds.unify_chunks(),
            slc_sec.coreg.measurements.ds.unify_chunks(),
        )
        for burst_id, slc_ref, slc_sec in zip(
            reference_slcs.bursts.keys(),
            reference_slcs.bursts.values(),
            secondary_slcs.bursts.values(),
        )
    ]
    args = list(zip(*args))

    results = cast(
        list[tuple[str, xr.Dataset]],
        map(
            extract_interf_single,
            args=args,
            kwargs={**kwargs},
            enable_dask=enable_dask,
            force_compute=force_compute,
            combine=False,
        ),
    )

    interf = xr.DataTree()
    for id, (burst_id, ds) in enumerate(results):
        ds = ds.assign(
            {
                "id": xr.full_like(
                    ds.phase.isel(polarization=0, acquisition_id=0),
                    fill_value=id,
                    dtype="int8",
                )
            }
        )
        interf[f"bursts/{burst_id}"] = ds

    return interf


@timeit
def mosaic_interf(interf: xr.DataTree, force_compute: bool = True) -> xr.Dataset:

    data = interf.bursts[list(interf.bursts.children)[0]].to_dataset()
    chunks = {k: v[0] for k, v in data.chunksizes.items()}
    # TODO: margins for mosaic do be adapted to the multilook and the valid points
    mosaic: xr.Dataset = data.isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8))
    mosaic = mosaic.swap_dims({"azimuth_time": "y", "slant_range_time": "x"})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])

    for k, layer in interf.bursts.items():
        layer_sel = layer.isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8)).to_dataset()
        layer_sel = layer_sel.swap_dims({"azimuth_time": "y", "slant_range_time": "x"})
        layer_sel = layer_sel.reset_coords([e for e in layer_sel.coords if e not in layer_sel.dims]).compute()

        mosaic = mosaic.combine_first(layer_sel)

    mosaic = mosaic.swap_dims({"y": "azimuth_time", "x": "slant_range_time"})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])
    mosaic["phase"] = mosaic.phase.transpose("polarization", "acquisition_id", "azimuth_time", "slant_range_time")
    mosaic["coherence"] = mosaic.coherence.transpose("polarization", "acquisition_id", "azimuth_time", "slant_range_time")
    if force_compute:
        mosaic = mosaic.compute()

    return mosaic.chunk(chunks)


@timeit
def get_mosaic_interf(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    enable_dask: bool = True,
    force_compute: bool = True,
    **kwargs,
) -> xr.Dataset:

    # compute interferograms in correspondence of each burst
    interf = extract_interf_multi(
        reference_slcs,
        secondary_slcs,
        enable_dask=enable_dask,
        force_compute=force_compute,
        **kwargs,
    )

    # mosaic all interferograms
    mosaic = mosaic_interf(interf, force_compute=force_compute)

    return mosaic
