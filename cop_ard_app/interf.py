from typing import cast
import json

import numpy as np
import xarray as xr
from cop_ard_core.logger import timeit

from cop_ard_app.computation import map
from cop_ard_app.utils import (
    extract_subband,
    get_azimuth_time_raw,
    get_stack_phramp_diff,
    time_to_pixel,
    wrap,
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
def compute_interf(
    reference: xr.DataArray,
    secondary: xr.DataArray,
    multilook: dict,
) -> xr.DataArray:
    # complex multiplication and average #
    return (reference.conj() * secondary).coarsen(multilook, boundary="trim").mean()  # type: ignore  mean is injected dynamically


# @timeit
def get_power(slc: xr.DataArray, multilook: dict):
    # compute the image power and average
    return (slc * slc.conj()).real.coarsen(multilook, boundary="trim").mean()  # type: ignore mean is injected dynamically


# @timeit
def compute_interf_norm(
    reference: xr.DataArray,
    secondary: xr.DataArray,
    multilook: dict,
) -> xr.DataArray:
    # compute interferogram #
    num = compute_interf(reference, secondary, multilook)

    # compute power of the images
    pow_ref = get_power(reference, multilook)
    pow_sec = get_power(secondary, multilook)

    # compute the normalization factor
    den = np.sqrt(pow_ref * pow_sec)

    # fill invalid values with 0 (null-coherence)
    return (num / den).fillna(0j)


# @timeit
def get_interf_single_burst(cslc: xr.DataArray, reference_date: str, multilook: dict) -> xr.Dataset:
    """
    Compute the interferometric phase and coherence with respect to a reference date
    applying a boxcar filter
    """

    # multilook of complex interferogram
    interf_norm = compute_interf_norm(cslc.sel(acquisition_date=reference_date), cslc, multilook)

    # init empty dataset
    interf_ds = xr.Dataset()

    # extract phase
    interf_ds["phase"] = xr.apply_ufunc(np.angle, interf_norm, dask="parallelized", output_dtypes=(float,))

    # extract coherence
    interf_ds["cohe"] = xr.apply_ufunc(np.abs, interf_norm, dask="parallelized", output_dtypes=(float,))

    return interf_ds


# @timeit
def extract_coreg_aoi(coreg: xr.Dataset, a_min: float, r_min: float, multilook: dict) -> xr.Dataset:

    # extract imput chuncking
    chunk_burst = {k: v[0] for k, v in coreg.chunksizes.items() if k in ["azimuth_time", "slant_range_time"]}

    # add pixel coordinates wrt the AOI
    coreg_sel = time_to_pixel(coreg.copy(), a_min, r_min)

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
    coreg: xr.Dataset,
    bbox: tuple,
    multilook: dict,
    reference_date: str,
    y_poly: xr.Dataset | None = None,
    f0: float | None = None,
    bw: float | None = None,
):
    # decompress attributes
    annot = json.loads(annot)
    # logger.debug(f"processing burst {burst_id}")
    a_min, a_max, r_min, r_max = bbox
    # extract aoi
    coreg_sel = extract_coreg_aoi(coreg, a_min, r_min, multilook)
    # extract subbands
    # logger.debug(f"cslc shape: {coreg_sel.cstack.shape}")
    if (f0 is not None) and (bw is not None):
        # logger.debug("Computing full range spectrum...")
        cslc = extract_subband(coreg_sel.cstack, bw, f0)
    else:
        cslc = coreg_sel.cstack

    # remove uselsess attributes
    cslc = cslc.drop_attrs()
    # logger.debug(f"cslc shape: {cslc.shape}")
    # logger.debug(f"extract multilooked interferogram and coherence")
    # extract multilooked interferogram and coherence
    interf = get_interf_single_burst(cslc, reference_date, multilook)

    # force coordinates
    interf = update_interf_coordinates(interf, multilook)

    # add raw image azimuth times
    azimuth_time_raw = get_azimuth_time_raw(coreg_sel["cstack"], multilook)
    interf["azimuth_time_raw"] = (["azimuth_time", "slant_range_time"], azimuth_time_raw.values)

    if y_poly is not None:
        # logger.debug("save phase_ramp")
        # compute multilooked phase corrections
        ph_ramp_diff_ml = get_stack_phramp_diff(
            interf.drop_vars(interf.data_vars), coreg_sel.drg, coreg_sel.daz, annot, y_poly, multilook
        )
        ph_ramp_diff_ml["slant_range_time"] = interf["slant_range_time"]
        ph_ramp_diff_ml["azimuth_time"] = interf["azimuth_time"]

        # apply correction
        interf["phase"] = wrap(interf["phase"] - ph_ramp_diff_ml)
        interf["phase_ramp"] = ph_ramp_diff_ml

    return (burst_id, interf.compute())


def extract_interf_multi(slcs: xr.DataTree, reference_date: str, enable_dask: bool = True, **kwargs):
    name = list(slcs.bursts)[0]
    available_pols = {str(i) for i in slcs.bursts[name].coreg.measurements.polarization.values}
    priorities = ["VV", "HH", "VH", "HV"]
    pol = next((p for p in priorities if p in available_pols), None)
    if pol is None:
        raise Exception(f"No polarization has been found {available_pols}")
    args = [
        (
            burst_id,
            json.dumps({k: v.measurements.slc.attrs[pol] for k, v in slc.orig.items()}),
            slc.coreg.measurements.ds.unify_chunks(),
        )
        for burst_id, slc in slcs.bursts.items()
    ]
    args = list(zip(*args))

    results = cast(
        list[tuple[str, xr.Dataset]],
            map(
            extract_interf_single,
            args=args,
            kwargs={"reference_date": reference_date, **kwargs},
            enable_dask=enable_dask,
            combine=False,
        )
    )

    interf = xr.DataTree()
    for id, (burst_id, ds) in enumerate(results):
        ds = ds.assign(
            {"id": xr.full_like(ds.phase.isel(polarization=0, acquisition_date=0), fill_value=id, dtype="int8")}
        )
        interf[f"bursts/{burst_id}"] = ds

    return interf


def mosaic_interf(interf: xr.DataTree) -> xr.Dataset:

    mosaic: xr.Dataset = (
        interf.bursts[list(interf.bursts.children)[0]]
        .to_dataset()
        .isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8))
    )
    mosaic = mosaic.swap_dims({"azimuth_time": "y", "slant_range_time": "x"})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])

    for k, layer in interf.bursts.items():
        layer_sel = layer.isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8)).to_dataset()
        layer_sel = layer_sel.swap_dims({"azimuth_time": "y", "slant_range_time": "x"})
        layer_sel = layer_sel.reset_coords([e for e in layer_sel.coords if e not in layer_sel.dims]).compute()

        mosaic = mosaic.combine_first(layer_sel).compute()

    mosaic = mosaic.swap_dims({"y": "azimuth_time", "x": "slant_range_time"})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])

    return mosaic


def get_mosaic_interf(slcs: xr.DataTree, reference_date: str, **kwargs) -> xr.Dataset:

    # compute interferograms in correspondence of each burst
    interf = extract_interf_multi(slcs, reference_date, **kwargs)

    # mosaic all interferograms
    mosaic = mosaic_interf(interf)

    return mosaic
