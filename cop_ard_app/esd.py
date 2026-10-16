import datetime
import json
import os
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core import detrend
from cop_ard_core.esd import compute_azimuth_shifts, optimize_curvature
from cop_ard_core.logger import logger

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.io import Path, generate_empty_dataset, get_storage_options, normpath, read, write
from cop_ard_app.utils import (
    build_coreg_poly_dict,
    get_best_pol,
    get_stack_phramp_diff,
    recompose_slc_datatrees,
    split_slc_datatrees,
)


def esd_single_burst(
    burst_id: str,
    current_burst_ref: xr.Dataset,
    next_burst_ref: xr.Dataset,
    current_burst_sec: xr.Dataset,
    next_burst_sec: xr.Dataset,
    multilook: dict | None = None,
    coherence_threshold: float = 0.7,
    enable_dask: bool = True,
    **kwargs,
) -> xr.Dataset:
    """

    Compute the residual shift in the azimuth direction applying the ESD procedure.
    It returns the estimeted shift (in pixel), the corresponding reliability index
    and the position of the overlap area used to perform the estimate
    (i.e the azimuth and slant_range coordinates of the baricenter of the area
    of overlap between the two consecutive bursts used to perform the ESD)

    :param slcs_path: path to the input datatree
    :param burst_list: list of burst identifier the datatree contains
    :param c_id: identifier (e.g iw1_254) of the burst for which the ESD must be computed
    :param multilook: multilook params, defaults to {"azimuth_pixel": 4, "slant_range_time": 16}

    :return: dictionary with the following structure:
             {
               (reference_date_0, x, y): shifts_0, mean_coherence_0,
               (reference_date_1, x, y): shifts_1, mean_coherence_1,
               ...
               (reference_date_n-1, x, y): shifts_n-1, mean_coherence_n-1,
             }
             where shifts_i and mean_coherence_i are the dataarrays of shifts
             and of mean coherence values, respectively, obtained using the
             reference_date_i image as ESD reference

    """
    multilook = multilook or {"azimuth_pixel": 4, "slant_range_time": 16}

    def time_to_pixel_az(stack: xr.Dataset, dtaz: float, reference_time: datetime.datetime | None = None):
        if reference_time is None:
            reference_time = stack.azimuth_time[0].values
        azimuth_pixel = np.round((stack.azimuth_time - reference_time).astype("float64") / 1e9 / dtaz).astype("int32")
        stack = stack.assign_coords({"azimuth_pixel": azimuth_pixel})

        return stack.swap_dims(azimuth_time="azimuth_pixel"), reference_time

    annot = current_burst_ref.cstack.attrs

    # select preferred polarization
    pol = get_best_pol(current_burst_ref.polarization.values)

    annot = annot[pol]
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"]

    # swap azimuth_time dimension with a new azimuth time in pixel #
    current_burst_ref, reference_time_ref = time_to_pixel_az(current_burst_ref, dtaz)
    current_burst_sec, reference_time_sec = time_to_pixel_az(current_burst_sec, dtaz)
    next_burst_ref, _ = time_to_pixel_az(next_burst_ref, dtaz, reference_time_ref)
    next_burst_sec, _ = time_to_pixel_az(next_burst_sec, dtaz, reference_time_sec)

    # now the overlap area between burst can be easily computed #
    sel_ref = next_burst_ref.drop_vars(next_burst_ref.data_vars) + current_burst_ref.drop_vars(
        current_burst_ref.data_vars
    )
    sel_sec = next_burst_sec.drop_vars(next_burst_sec.data_vars) + current_burst_sec.drop_vars(
        current_burst_sec.data_vars
    )

    v_dims = [j for j in sel_ref.dims if j not in ["i", "j"]]

    # overlap area extraction from current and next bursts #
    next_burst_ref = next_burst_ref.sel({k: sel_ref.coords[k] for k in v_dims}).cstack.sel(polarization=pol)
    current_burst_ref = current_burst_ref.sel({k: sel_ref.coords[k] for k in v_dims}).cstack.sel(polarization=pol)
    next_burst_sec = next_burst_sec.sel({k: sel_sec.coords[k] for k in v_dims}).cstack.sel(polarization=pol)
    current_burst_sec = current_burst_sec.sel({k: sel_sec.coords[k] for k in v_dims}).cstack.sel(polarization=pol)

    # returns the estimated azimuth shift and the corresponding quality index #
    # new dimension id and corresponding coords added to the shifts #
    coords = {k: (["id"], [current_burst_ref[k].mean().values]) for k in ["azimuth_time", "slant_range_time"]}
    esd = map_blocks(
        compute_azimuth_shifts,
        (current_burst_sec, next_burst_sec, current_burst_ref, next_burst_ref),
        kwargs={
            "multilook": multilook,
            "coherence_threshold": coherence_threshold,
        },
        chunksizes={"acquisition_id": 100},
        enable_dask=enable_dask,
        force_compute=True,
    )
    if not isinstance(esd, xr.Dataset):
        raise TypeError("compute_azimuth_shifts returned wrong data type")

    esd = esd.expand_dims({"id": [burst_id]}).assign_coords(coords)
    return esd


def get_next_burst_id(burst_id):
    swath, id = burst_id.split("_")
    next_burst_id = f"{swath}_{int(id) + 1}"
    return next_burst_id


# @timeit
def estimate(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    poly_degree: int = 0,
    enable_dask: bool = True,
    multilook: dict | None = None,
    coherence_threshold: float = 0.7,
    **kwargs,
):

    args = []
    for k_ref, k_sec in zip(reference_slcs.bursts.keys(), secondary_slcs.bursts.keys()):
        if k_ref != k_sec:
            raise ValueError(f"Bursts {k_ref} and {k_sec} do not match")
        next_id = get_next_burst_id(k_ref)
        if next_id in list(reference_slcs.bursts):
            a = (
                k_ref,
                reference_slcs[f"bursts/{k_ref}/coreg/measurements"].to_dataset(),
                reference_slcs[f"bursts/{next_id}/coreg/measurements"].to_dataset(),
                secondary_slcs[f"bursts/{k_sec}/coreg/measurements"].to_dataset(),
                secondary_slcs[f"bursts/{next_id}/coreg/measurements"].to_dataset(),
            )
            args.append(a)
    args = list(zip(*args))
    # Residual azimuth shifts estimation for each bursts
    esd = cast(
        list[xr.Dataset],
        map(
            esd_single_burst,
            args=args,
            kwargs={"enable_dask": enable_dask, "multilook": multilook, "coherence_threshold": coherence_threshold},
            combine=False,
            enable_dask=enable_dask,
            force_compute=True,
            **kwargs,
        ),
    )

    # Structure re-formatting in order to solve the global problem
    to_merge = [e.drop_vars("acquisition_date", errors="ignore") for e in esd if e is not None and len(e) > 0]
    esd_ds = xr.merge(
        to_merge,
        compat="no_conflicts",
        combine_attrs="drop",
        join="outer",
    )

    args = [esd_ds.sel(acquisition_id=d) for d in esd_ds.acquisition_id.values]  # type: ignore
    esd_opt = cast(
        list,
        map(
            optimize_curvature,
            args=args,
            kwargs={"poly_degree": poly_degree},
            combine=False,
            **kwargs,
        ),
    )
    for e in esd_opt:
        d = e.acquisition_id.values
        esd_ds["mean_phase"].loc[:, d] = e.mean_phase
        esd_ds["mean_shift"].loc[:, d] = e.mean_shift

    # Auxiliary data saved as attributes of the dataset
    return esd_ds


def find_global_shift(esd: xr.Dataset, poly_degree: int = 0, norm: int = 1):
    """
    Starting from the shifts estimated by applying the ESD between
    different bursts and (if requested) by using multiple references,
    a system of linear equation is constructed and solved to obtain
    a polynomial function (of degree poly_degree) that describes the azimuth
    residual shift between each image of the stack and a reference date

    """

    # set the reference point as the mean values of the slant_range_time and azimuth_time coordinates
    r0, t0 = [esd[e].mean().values for e in ["slant_range_time", "azimuth_time"]]

    # scaling coefficients for the slant_range_time and azimuth_time coordinates
    s_x, s_y = [
        (esd[e].max() - esd[e].min()).astype("float64").values / 2.0 for e in ["slant_range_time", "azimuth_time"]
    ]
    if s_x == 0.0:
        s_x = np.array([1.0])
    if s_y == 0.0:
        s_y = np.array([1.0])

    # rescale and center the coordinates of the points where the model is evaluated
    az = ((esd.azimuth_time - t0).astype("float64") / s_y).values

    # call the global surface estimation function
    coeffs = detrend.estimate_global_surface(
        np.atleast_2d(az).reshape(-1, 1),
        esd.mean_shift.values.ravel(),
        esd.coherence.values.ravel(),
        degree=poly_degree,
        norm=norm,
    ).x

    return to_dataset_1D(coeffs, esd.acquisition_id, t0, r0, s_y, s_x)


def to_dataset(X, acquisition_id, t0, r0, s_y, s_x):

    n = len(X)
    deg = round((-3 + np.sqrt(1 + 8 * n)) / 2)

    def triangular_to_polygrid(coef):

        C = np.zeros((deg + 1, deg + 1))

        k = 0
        for d in range(deg + 1):
            for i in range(d, -1, -1):
                j = d - i
                C[i, j] = coef[n - k - 1]
                k += 1

        return C

    axis = np.arange(deg + 1)
    da = [
        xr.DataArray(
            name=str(dd),
            data=triangular_to_polygrid(X),
            dims=["i", "j"],
            coords={"i": axis, "j": axis},
            attrs={"deg": deg, "offset": (t0, r0), "scale_fact": (s_y, s_x)},
        )
        for i, dd in enumerate(acquisition_id.values)
    ]
    return xr.Dataset({e.name: e for e in da})


def to_dataset_1D(X, acquisition_id, t0, r0, s_y, s_x):

    deg = len(X) - 1

    def _to_polygrid(coef):

        C = np.zeros((deg + 1, deg + 1))

        C[:, 0] = np.flip(X)

        return C

    axis = np.arange(deg + 1)
    da = [
        xr.DataArray(
            name=str(dd),
            data=_to_polygrid(X),
            dims=["i", "j"],
            coords={"i": axis, "j": axis},
            attrs={"deg": deg, "offset": (0.0, r0), "scale_fact": (s_y, s_x), "off_y": str(t0)},
        )
        for i, dd in enumerate(acquisition_id.values)
    ]
    return xr.Dataset({e.name: e for e in da})


def stack_esd(
    slcs: xr.DataTree,
    wdir_path: str | Path,
    multiburst: bool,
    reference_id: str,
    esd_poly_degree: int = 2,
    multilook: dict | None = None,
    coherence_threshold: float = 0.7,
    enable_dask: bool = True,
    **kwargs,
):
    wdir_path = Path(str(wdir_path))

    # estimate ESD at full-band
    if multiburst:
        reference_slcs, secondary_slcs = split_slc_datatrees(
            slcs=slcs,
            reference_id=reference_id,
            multiburst=multiburst,
        )
        esd = estimate(
            reference_slcs,
            secondary_slcs,
            poly_degree=esd_poly_degree,
            multilook=multilook,
            coherence_threshold=coherence_threshold,
            enable_dask=enable_dask,
        )

        # fit polynomial function to ESD and remove it from the interferogram
        y_poly = find_global_shift(esd, poly_degree=esd_poly_degree)

        for id in y_poly.data_vars:
            logger.debug(f"ESD {id}: {y_poly[id].values}")
        first_burst_key = next(iter(reference_slcs.bursts))
        ref_node = reference_slcs.bursts[first_burst_key]
        sec_node = secondary_slcs.bursts[first_burst_key]

        # Extract common polarizations
        available_pols_ref = {str(i) for i in ref_node.coreg.measurements.polarization.values}
        available_pols_sec = {str(i) for i in sec_node.coreg.measurements.polarization.values}
        pol = get_best_pol(list(available_pols_ref.intersection(available_pols_sec)))
        args = []

        for k, v in secondary_slcs.bursts.items():
            a = json.dumps({k: v.measurements.slc.attrs[pol] for k, v in reference_slcs.bursts[k].orig.items()})
            args.append((k, v.coreg.measurements.to_dataset(), a))

        args = list(zip(*args))
        results = cast(
            list[tuple[str, xr.Dataset]],
            map(
                map_single_phase_ramp,
                args=args,
                kwargs=dict(
                    y_poly=y_poly,
                    working_directory=wdir_path,
                    enable_dask=enable_dask,
                    **kwargs,
                ),
                enable_dask=enable_dask,
                combine=False,
            ),
        )
        for k, v in results:
            secondary_slcs.bursts[k].coreg.measurements["cstack"] = v.cstack

        slcs = recompose_slc_datatrees(
            reference_slcs,
            secondary_slcs,
            multiburst=multiburst,
        )
    else:
        logger.warning(f"Skipping ESD, got multiburst = {multiburst}")
    attrs = {
        "esd": build_coreg_poly_dict(
            description="Enhanced Spectral Diversity",
            y_poly=y_poly,
        )
    }
    slcs.attrs.setdefault("coregistration_polynomials", {}).update(attrs)
    return slcs


def map_single_phase_ramp(
    burst_id: str,
    coreg_sec: xr.Dataset,
    annot: str,
    y_poly: xr.Dataset,
    working_directory: str,
    enable_dask: bool,
    **kwargs,
):
    # Define the output path for this burst.
    path = f"bursts/{burst_id}"
    coreg_path = normpath(f"{working_directory}/{path}")
    measurements_path = os.path.join(coreg_path, "coreg", "measurements")
    store_mapping = {
        0: {
            "data_path": measurements_path,
            "save_options": {
                "consolidated": False,
                "storage_options": get_storage_options(kwargs),
            },
        },
    }
    cstack = coreg_sec.cstack
    data_vars = ["cstack"]
    dims = {"cstack": cstack.dims}
    shape_dict = {"cstack": cstack.shape}
    dtype = {"cstack": "complex64"}
    chunks = {k: v[0] for k, v in cstack.chunksizes.items()}
    coords = cstack.coords
    measurements = generate_empty_dataset(
        data_vars=data_vars,
        dims=dims,
        shape=shape_dict,
        chunks=chunks,
        dtype=dtype,
        coords=coords,
        attrs_data_vars={"cstack": cstack.attrs},
    )

    write(
        measurements.drop_vars(["cstack"]),
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
    measurements = read(
        measurements_path,
        storage_options=get_storage_options(kwargs),
        as_type="xr.Dataset",
    )

    chunksizes = {
        "azimuth_time": measurements.chunksizes["azimuth_time"][0],
        "slant_range_time": measurements.chunksizes["slant_range_time"][0],
        "acquisition_id": 10,
    }

    map_blocks(
        compute_phase_ramp,
        obj=[coreg_sec],
        args=(y_poly, annot),
        kwargs=kwargs,
        combine=False,
        enable_dask=enable_dask,
        chunksizes=chunksizes,
        store_mapping=store_mapping,
    )

    measurements = read(measurements_path, as_type="xr.Dataset", storage_options=get_storage_options(kwargs))

    return (burst_id, measurements)


def compute_phase_ramp(coreg_sec: xr.Dataset, y_poly: xr.Dataset, annot: str):
    annot_dict = json.loads(annot)
    ph_ramp_diff_ml = get_stack_phramp_diff(
        coreg_sec.acquisition_id,
        coreg_sec.azimuth_time,
        coreg_sec.slant_range_time,
        coreg_sec.drg,
        coreg_sec.daz,
        annot_dict,
        y_poly,
    )

    cstack = coreg_sec.cstack
    cstack *= np.exp(-1j * ph_ramp_diff_ml)
    ds = cstack.to_dataset(name="cstack")
    return ds
