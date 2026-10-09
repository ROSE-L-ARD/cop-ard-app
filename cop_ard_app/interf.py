"""
Interferometric phase and coherence computations for ARD products.

Derives multilooked interferometric information from COREG-stacked SLC data,
including subband extraction and mosaicking over an Area of Interest.
"""

import json
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.logger import timeit
from cop_ard_core.utils import compute_interf_norm, get_azimuth_time_raw, time_to_pixel, wrap

from cop_ard_app.computation import map
from cop_ard_app.utils import (
    extract_subband,
    get_best_pol,
    get_stack_phramp_diff,
)


# @timeit
def update_interf_coordinates(interf: xr.Dataset, multilook: dict) -> xr.Dataset:
    """
    Re-index the interferometric pixel coordinates on the multilook grid.

    Offsets the `x` and `y` coordinates by half a multilook window and
    divides them by the `slant_range_time` and `azimuth_time` window sizes,
    rounding the result to integer indices of consecutive multilook windows.

    Parameters
    ----------
    interf : xr.Dataset
        The interferometric dataset with `x` and `y` pixel coordinates.
    multilook : dict
        The multilook configuration with `azimuth_time` and
        `slant_range_time` window sizes.

    Returns
    -------
    xr.Dataset
        The same dataset with integer multilooked `x` and `y` coordinates.
    """

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
    """
    Extract the Area of Interest (AoI) from a COREG dataset.

    Converts the acquisition and range times into pixel coordinates with
    respect to `a_min` and `r_min`, then selects the data bounded by those
    coordinates. The selection bounds are snapped to multiples of the
    multilook strides so that whole multilook windows are kept, and the
    original burst chunking is preserved.

    Parameters
    ----------
    coreg : xr.Dataset
        The COREG dataset to subset.
    a_min : float
        Minimum acquisition time of the Area of Interest.
    r_min : float
        Minimum range time of the Area of Interest.
    multilook : dict
        The multilook configuration providing the `azimuth_time` and
        `slant_range_time` strides.

    Returns
    -------
    xr.Dataset
        The subset of `coreg` over the Area of Interest, with pixel
        coordinates and original burst dimensions restored.
    """

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


# @timeit
def extract_interf_single(
    burst_id: str,
    annot: str,
    coreg_ref: xr.Dataset,
    coreg_sec: xr.Dataset,
    polarization: str,
    a_min: float,
    r_min: float,
    multilook: dict,
    y_poly: xr.Dataset | None = None,
    f0: float | None = None,
    bw: float | None = None,
):
    """
    Compute the interferometric phase and coherence for a single burst.

    Extracts the Area of Interest (and, when requested, a spectral subband)
    from the reference and secondary COREG datasets, computes the multilooked
    interferometric phase and coherence, re-indexes the coordinates on the
    multilook grid and attaches the raw image azimuth times. When `y_poly`
    is provided, the phase is corrected by the multilooked phase-ramp
    difference and the correction is also stored.

    Parameters
    ----------
    burst_id : str
        Identifier of the burst being processed.
    annot : str
        JSON-encoded acquisition annotation information.
    coreg_ref : xr.Dataset
        The reference-date COREG dataset.
    coreg_sec : xr.Dataset
        The secondary-date COREG dataset.
    polarization : str
        The polarization tag to select from the COREG attributes.
    a_min : float
        Minimum acquisition time of the Area of Interest.
    r_min : float
        Minimum range time of the Area of Interest.
    multilook : dict
        The multilook configuration (window sizes along azimuth and range).
    y_poly : xr.Dataset | None, optional
        Optional polynomial model used to compute and apply phase-ramp
        corrections. Defaults to None.
    f0 : float | None, optional
        Central frequency of the spectral subband to extract. Defaults to None.
    bw : float | None, optional
        Bandwidth of the spectral subband to extract. Defaults to None.

    Returns
    -------
    tuple
        A tuple `(burst_id, interf)` where `burst_id` is the input burst
        identifier and `interf` is an `xr.Dataset` holding the `phase`,
        `coherence` and `azimuth_time_raw` fields (plus `phase_ramp` when
        a phase correction is applied).
    """
    # decompress attributes
    annot_dict = json.loads(annot)
    TimeCOAPoly = coreg_ref.cstack.attrs[polarization]["Grid"]["TimeCOAPoly"]["Coefs"]
    CollectStart = np.datetime64(coreg_ref.cstack.attrs[polarization]["Timeline"]["CollectStart"][:-1], "ns")

    # extract aoi
    coreg_ref_sel = extract_coreg_aoi(coreg_ref, a_min, r_min, multilook)
    coreg_sec_sel = extract_coreg_aoi(coreg_sec, a_min, r_min, multilook)

    # extract subbands
    if (f0 is not None) and (bw is not None):
        cslc_ref = extract_subband(coreg_ref_sel.cstack, bw, f0)
        cslc_sec = extract_subband(coreg_sec_sel.cstack, bw, f0)
    else:
        cslc_ref = coreg_ref_sel.cstack
        cslc_sec = coreg_sec_sel.cstack

    # remove uselsess attributes
    cslc_ref = cslc_ref.drop_attrs()
    cslc_sec = cslc_sec.drop_attrs()

    # extract multilooked interferogram and coherence
    interf = get_interf_single_burst(cslc_ref, cslc_sec, multilook)
    # force coordinates
    interf = update_interf_coordinates(interf, multilook)

    # add raw image azimuth times

    azimuth_time_raw = get_azimuth_time_raw(coreg_ref_sel["cstack"], TimeCOAPoly, CollectStart, multilook)
    interf["azimuth_time_raw"] = (["azimuth_time", "slant_range_time"], azimuth_time_raw.values)
    if y_poly is not None:
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
    multiburst: bool,
    enable_dask: bool = True,
    **kwargs,
):
    """
    Compute interferometric datasets for many bursts in parallel.

    Determines the best polarization common to the reference and secondary
    SLC products, maps `extract_interf_single` over every burst using the
    Dask-enabled `map` of `cop_ard_app.computation`, and assembles the
    per-burst interferometric datasets (each tagged with a per-burst integer
    `id`) into a single tree.

    Parameters
    ----------
    reference_slcs : xr.DataTree
        The reference-date SLC product (a tree of bursts, or a single
        dataset when `multiburst` is False).
    secondary_slcs : xr.DataTree
        The secondary-date SLC product, paired with `reference_slcs`.
    multiburst : bool
        Whether the products are multiburst (ESA) products with one node
        per burst.
    enable_dask : bool, optional
        Whether to run the per-burst computations in parallel with Dask.
        Defaults to True.
    **kwargs
        Additional keyword arguments forwarded to `extract_interf_single`
        (e.g. `a_min`, `r_min`, `multilook`, `y_poly`, `f0`, `bw`).

    Returns
    -------
    xr.DataTree
        A tree with one `bursts/<burst_id>` node per processed burst, each
        holding its interferometric dataset with an added `id` variable.
    """
    # Determine the node to inspect for polarization availability
    if multiburst:
        first_burst_key = next(iter(reference_slcs.bursts))
        ref_node = reference_slcs.bursts[first_burst_key]
        sec_node = secondary_slcs.bursts[first_burst_key]
    else:
        ref_node = reference_slcs
        sec_node = secondary_slcs

    # Extract common polarizations
    available_pols_ref = {str(i) for i in ref_node.coreg.measurements.polarization.values}
    available_pols_sec = {str(i) for i in sec_node.coreg.measurements.polarization.values}
    pol = get_best_pol(list(available_pols_ref.intersection(available_pols_sec)))

    # Prepare arguments mapping
    if multiburst:
        args = [
            (
                burst_id,
                json.dumps({k: v.measurements.slc.attrs[pol] for k, v in slc_ref.orig.items()}),
                slc_ref.coreg.measurements.ds.sel(polarization=pol),
                slc_sec.coreg.measurements.ds.sel(polarization=pol),
            )
            for burst_id, slc_ref, slc_sec in zip(
                reference_slcs.bursts.keys(),
                reference_slcs.bursts.values(),
                secondary_slcs.bursts.values(),
            )
        ]
    else:
        # Non-multiburst case (e.g., StripMap): Single dataset wrapped into a list
        args = [
            (
                "data",  # Standard dummy unit ID for non-multiburst modes
                json.dumps({k: v.measurements.slc.attrs[pol] for k, v in reference_slcs.orig.items()}),
                reference_slcs.coreg.measurements.ds.sel(polarization=pol),
                secondary_slcs.coreg.measurements.ds.sel(polarization=pol),
            )
        ]

    args = list(zip(*args))

    results = cast(
        list[tuple[str, xr.Dataset]],
        map(
            extract_interf_single,
            args=args,
            kwargs={"polarization": pol, **kwargs},
            enable_dask=enable_dask,
            force_compute=True,
            combine=False,
        ),
    )

    interf = xr.DataTree()
    for id, (burst_id, ds) in enumerate(results):
        ds = ds.assign(
            {
                "id": xr.full_like(
                    ds.phase.isel(acquisition_id=0),
                    fill_value=id,
                    dtype="int8",
                )
            }
        )
        interf[f"bursts/{burst_id}"] = ds

    return interf


@timeit
def mosaic_interf(interf: xr.DataTree, chunks: dict) -> xr.Dataset:
    """
    Mosaic the per-burst interferometric datasets into a single dataset.

    Crops fixed margins from every burst layer, overlays the layers with
    `combine_first` (earlier burst layers take precedence where values are
    missing) and assembles a single dataset indexed back by acquisition and
    range time, computed and chunked as requested.

    Parameters
    ----------
    interf : xr.DataTree
        The tree of per-burst interferometric datasets.
    chunks : dict
        The desired chunking of the final mosaicked dataset.

    Returns
    -------
    xr.Dataset
        The mosaicked interferometric dataset with computed, chunked
        `phase` and `coherence` fields.
    """

    data = interf.bursts[next(iter(interf.bursts.children))].to_dataset()
    # TODO: margins for mosaic do be adapted to the multilook and the valid points
    mosaic: xr.Dataset = data.isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8))
    mosaic = mosaic.swap_dims({"azimuth_time": "y", "slant_range_time": "x"})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])

    for layer in interf.bursts.values():
        layer_sel = layer.isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8)).to_dataset()
        layer_sel = layer_sel.swap_dims({"azimuth_time": "y", "slant_range_time": "x"})
        layer_sel = layer_sel.reset_coords([e for e in layer_sel.coords if e not in layer_sel.dims]).compute()

        mosaic = mosaic.combine_first(layer_sel)

    mosaic = mosaic.swap_dims({"y": "azimuth_time", "x": "slant_range_time"})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])
    mosaic["phase"] = mosaic.phase.transpose("acquisition_id", "azimuth_time", "slant_range_time")
    mosaic["coherence"] = mosaic.coherence.transpose("acquisition_id", "azimuth_time", "slant_range_time")
    return mosaic.compute().chunk(chunks)


@timeit
def get_mosaic_interf(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    multiburst: str,
    chunks: dict,
    enable_dask: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Compute the mosaicked interferometric phase and coherence fields.

    High-level driver: computes the interferometric datasets for every burst
    with `extract_interf_multi` and then mosaics them into a single dataset
    with `mosaic_interf`.

    Parameters
    ----------
    reference_slcs : xr.DataTree
        The reference-date SLC product.
    secondary_slcs : xr.DataTree
        The secondary-date SLC product.
    multiburst : str
        Whether the products are multiburst (ESA) products.
    chunks : dict
        The desired chunking of the final mosaicked dataset.
    enable_dask : bool, optional
        Whether to enable Dask for the per-burst computations. Defaults to True.
    **kwargs
        Additional keyword arguments forwarded to `extract_interf_multi`
        (e.g. `a_min`, `r_min`, `multilook`, `y_poly`, `f0`, `bw`).

    Returns
    -------
    xr.Dataset
        The mosaicked interferometric dataset.
    """

    # compute interferograms in correspondence of each burst
    interf = extract_interf_multi(
        reference_slcs,
        secondary_slcs,
        multiburst=multiburst,
        enable_dask=enable_dask,
        **kwargs,
    )

    # mosaic all interferograms
    mosaic = mosaic_interf(interf, chunks)

    return mosaic
