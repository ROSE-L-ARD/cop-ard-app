import datetime
from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.esd import compute_azimuth_shifts, optimize_curvature

from cop_ard_app.computation import map, map_blocks


def esd_single_burst(
    burst_id: str,
    current_burst: xr.Dataset,
    next_burst: xr.Dataset,
    reference_dates: list[str],
    multilook: dict = {"azimuth_pixel": 4, "slant_range_time": 16},
    coherence_th: float = 0.7,
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
    :param reference_dates: a list or a single date that corresponds to the image(s) to be
                            used as reference by the shift estimate procedure. In case of a
                            list of images, the shifts will be estimated with respect to
                            all the images in the list

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
    # logger.info(f"Processing overlap area between bursts: {burst_id} and {get_next_burst_id(burst_id)}")

    def time_to_pixel(stack: xr.Dataset, dtaz: float, reference_time: datetime.datetime | None = None):
        if reference_time is None:
            reference_time = stack.azimuth_time[0].values
        azimuth_pixel = np.round((stack.azimuth_time - reference_time).astype("float64") / 1e9 / dtaz).astype("int32")
        stack = stack.assign_coords({"azimuth_pixel": azimuth_pixel})
        return stack.swap_dims(azimuth_time="azimuth_pixel"), reference_time

    annot = current_burst.cstack.attrs
    annot = annot[list(annot)[0]]
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"]

    # swap azimuth_time dimension with a new azimuth time in pixel #
    current_burst, reference_time = time_to_pixel(current_burst, dtaz)
    next_burst, _ = time_to_pixel(next_burst, dtaz, reference_time)

    # now the overlap area between burst can be easily computed #
    sel = next_burst.drop_vars(next_burst.data_vars) + current_burst.drop_vars(current_burst.data_vars)
    v_dims = [j for j in sel.dims if j not in ["i", "j"]]

    available_pols = {str(i) for i in current_burst.polarization.values}
    priorities = ["VV", "HH", "VH", "HV"]
    pol = next((p for p in priorities if p in available_pols), None)
    if pol is None:
        raise Exception(f"No polarization has been found {available_pols}")
    # overlap area extraction from current and next bursts #

    next_burst = next_burst.sel({k: sel.coords[k] for k in v_dims}).cstack.sel(polarization=pol)
    current_burst = current_burst.sel({k: sel.coords[k] for k in v_dims}).cstack.sel(polarization=pol)
    # returns the estimated azimuth shift and the corresponding quality index #
    next_reference = next_burst.sel(acquisition_date=reference_dates)
    current_reference = current_burst.sel(acquisition_date=reference_dates)
    # new dimension id and corresponding coords added to the shifts #
    coords = {k: (["id"], [current_burst[k].mean().values]) for k in ["azimuth_time", "slant_range_time"]}
    esd = map_blocks(
        compute_azimuth_shifts,
        (current_burst, next_burst),
        args=(current_reference, next_reference),
        kwargs={
            "reference_dates": reference_dates,
            "multilook": multilook,
            "coherence_th": coherence_th,
        },
        chunksizes={"acquisition_date": 100},
        enable_dask=enable_dask,
    )

    esd = esd.expand_dims({"id": [burst_id]}).assign_coords(coords)
    return esd


def get_next_burst_id(burst_id):
    swath, id = burst_id.split("_")
    next_burst_id = f"{swath}_{int(id) + 1}"
    return next_burst_id


# @timeit
def estimate(
    slcs: xr.DataTree,
    reference_dates: list[str] | str,
    poly_degree: int = 0,
    enable_dask: bool = True,
    **kwargs,
):

    if not isinstance(reference_dates, (list, tuple)):
        reference_dates = [reference_dates]
    args = []
    for k, v in slcs.bursts.items():
        next_id = get_next_burst_id(k)
        if next_id in list(slcs.bursts):
            a = (k, v["coreg/measurements"].to_dataset(), slcs[f"bursts/{next_id}/coreg/measurements"].to_dataset())
            args.append(a)
    args = list(zip(*args))
    # Residual azimuth shifts estimation for each bursts
    esd = cast(
        list[xr.Dataset],
        map(
            esd_single_burst,
            args=args,
            kwargs={"reference_dates": reference_dates, "enable_dask": enable_dask},
            combine=False,
            enable_dask=enable_dask,
            **kwargs,
        ),
    )

    # Structure re-formatting in order to solve the global problem
    esd = xr.merge(
        [e for e in esd if e is not None],
        compat="no_conflicts",
        combine_attrs="drop",
        join="outer",
    )

    args = [
        esd.sel(acquisition_date=d, reference_date=rd)
        for d in esd.acquisition_date.values
        for rd in reference_dates
        if d != rd
    ]

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
        d, rd = e.acquisition_date.values, e.reference_date.values
        esd["mean_phase"].loc[:, rd, d] = e.mean_phase
        esd["mean_shift"].loc[:, rd, d] = e.mean_shift

    # Auxiliary data saved as attributes of the dataset
    return esd
