import datetime
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.esd import compute_azimuth_shifts, optimize_curvature
from cop_ard_core.utils import get_best_pol

from cop_ard_app.computation import map, map_blocks
from cop_ard_core import detrend


def esd_single_burst(
    burst_id: str,
    current_burst_ref: xr.Dataset,
    next_burst_ref: xr.Dataset,
    current_burst_sec: xr.Dataset,
    next_burst_sec: xr.Dataset,
    multilook: dict = {"azimuth_pixel": 4, "slant_range_time": 16},
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
            kwargs={"enable_dask": enable_dask},
            combine=False,
            enable_dask=enable_dask,
            force_compute=True,
            **kwargs,
        ),
    )

    # Structure re-formatting in order to solve the global problem
    esd = xr.merge(
        [e for e in esd if e is not None and len(e) > 0],
        compat="no_conflicts",
        combine_attrs="drop",
        join="outer",
    )

    args = [esd.sel(acquisition_id=d) for d in esd.acquisition_id.values]
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
        esd["mean_phase"].loc[:, d] = e.mean_phase
        esd["mean_shift"].loc[:, d] = e.mean_shift

    # Auxiliary data saved as attributes of the dataset
    return esd


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
    rg = ((esd.slant_range_time - r0).astype("float64") / s_x).values

    # call the global surface estimation function
    coeffs = detrend.estimate_global_surface(
        np.atleast_2d(az).reshape(-1, 1),
        esd.mean_shift.values.ravel(),
        esd.coherence.values.ravel(),
        degree=poly_degree,
        norm=norm,
    ).x

    return to_dataset_1D(coeffs, esd.acquisition_id, t0, r0, s_y, s_x)


def to_dataset(X, acquisition_date, t0, r0, s_y, s_x):

    n = len(X)
    deg = int(round((-3 + np.sqrt(1 + 8 * n)) / 2))

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
        for i, dd in enumerate(acquisition_date.values)
    ]
    return xr.Dataset({e.name: e for e in da})


def to_dataset_1D(X, acquisition_date, t0, r0, s_y, s_x):

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
            attrs={"deg": deg, "offset": (0., r0), "scale_fact": (s_y, s_x), "off_y": str(t0)},
        )
        for i, dd in enumerate(acquisition_date.values)
    ]
    return xr.Dataset({e.name: e for e in da})
