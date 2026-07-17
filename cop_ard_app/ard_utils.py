import numpy as np
import scipy.fft
import xarray as xr
from scipy.constants import speed_of_light

from cop_ard_app.model import get_az_rg_times, get_coefficients
from cop_ard_core.utils import get_phramp_diff


# @timeit
def extract_subband(cslc: xr.DataArray, bw: float, f0: float, dim_transf: str = "slant_range_time") -> xr.DataArray:

    dim = cslc.dims
    freq_dim = [f"freq_{e}" if e == dim_transf else e for e in cslc.dims]
    axis = cslc.get_axis_num(dim_transf)
    N = cslc.sizes[dim_transf]
    t_coords = cslc[dim_transf].values
    coord_trasf = (np.arange(N) - N // 2) / (t_coords[-1] - t_coords[0])
    # coord_trasf = (np.arange(cslc[dim_trasf].size) - cslc[dim_trasf].size // 2)  / (cslc[dim_trasf][-1] - cslc[dim_trasf][0]).values
    freq_coords = {
        (f"freq_{e}" if e == dim_transf else e): (coord_trasf if e == dim_transf else cslc.coords[e].values)
        for e in cslc.dims
    }

    frange = xr.DataArray(
        scipy.fft.fftshift(scipy.fft.fft(cslc.values, norm="ortho", axis=axis), axes=(axis,)),
        dims=freq_dim,
        coords=freq_coords,
    )

    mask = (frange.freq_slant_range_time > f0 - bw / 2) * (frange.freq_slant_range_time < f0 + bw / 2)

    return xr.DataArray(
        scipy.fft.ifft((frange * mask).values, norm="ortho", axis=axis),
        dims=dim,
        coords=cslc.drop_vars(["slant_range_m", "azimuth_m"]).coords,
    )


def get_bounding_box(slcs: xr.DataTree, path: str) -> list:
    """
    Compute the bounding box of a burst stack in azimuth time and slant range time.

    This function iterates over all bursts in a stack of SLCs and extracts the minimum and maximum
    values of `azimuth_time` and `slant_range_time` across all bursts. The result is a bounding box
    that encompasses the full spatial and temporal extent of the merged image.

    Parameters
    ----------
    slcs : xr.DataTree
        A DataTree containing multiple SAR bursts, each with `azimuth_time` and `slant_range_time` coordinates.

    Returns
    -------
    list
        A list of four elements: `[az_min, az_max, rg_min, rg_max]`, representing the bounding box
        in azimuth time and slant range time.
    """
    azimuth_time, slant_range_time = [], []

    for burst in slcs.values():
        if burst[path].azimuth_time.size > 0:
            azimuth_time += [
                burst[path].azimuth_time.min().values,
                burst[path].azimuth_time.max().values,
            ]
        if burst[path].slant_range_time.size > 0:
            slant_range_time += [
                burst[path].slant_range_time.min().values,
                burst[path].slant_range_time.max().values,
            ]

    if len(azimuth_time) <= 0 or len(slant_range_time) <= 0:
        raise ValueError("Null slc in input")

    return [
        np.min(azimuth_time),
        np.max(azimuth_time),
        np.min(slant_range_time),
        np.max(slant_range_time),
    ]


def get_azimuth_time_raw(burst: xr.DataArray, multilook: dict | None = None) -> xr.DataArray:

    # retrieve image annotation
    # the polarization is equivalent
    _, annot = next(iter(burst.attrs.items()))

    if multilook is not None:
        az_slice = slice(multilook["azimuth_time"] // 2, None, multilook["azimuth_time"])
        rg_slice = slice(multilook["slant_range_time"] // 2, None, multilook["slant_range_time"])
    else:
        az_slice = slice(None, None, None)
        rg_slice = slice(None, None, None)

    burst_ = burst.isel(azimuth_time=az_slice, slant_range_time=rg_slice)

    # retrieve axis and coefficients
    slant_range_m = burst_.slant_range_m.values
    azimuth_m = burst_.azimuth_m.values
    coeffs = annot["Grid"]["TimeCOAPoly"]["Coefs"]

    # compute the relative azimuth time
    timeCOA = np.polynomial.polynomial.polygrid2d(slant_range_m, azimuth_m, coeffs).T

    # convert to timedelta in nanoseconds
    timeCOA = (timeCOA * 1e9).astype("timedelta64[ns]")

    # compute absolute time
    timeCOA = timeCOA + np.datetime64(annot["Timeline"]["CollectStart"][:-1])

    # return as dataarray
    return xr.DataArray(
        timeCOA,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": burst_.azimuth_time.values, "slant_range_time": burst_.slant_range_time.values},
    )


def get_carrier_frequency(slcs: xr.DataTree, path: str) -> float:
    """
    Compute the mean carrier frequency of the bursts.

    This function iterates over all bursts in a stack of SLCs and extracts the carrier frequency attribute.

    Parameters
    ----------
    slcs : xr.DataTree
        A DataTree containing multiple SAR bursts, each with `azimuth_time` and `slant_range_time` coordinates.

    Returns
    -------
    float
        Average carrier frequency.
    """
    f0 = []

    for burst in slcs.values():
        _, annot = next(iter(burst[path].attrs.items()))
        f0.append((annot["RadarCollection"]["TxFrequency"]["Max"] + annot["RadarCollection"]["TxFrequency"]["Min"]) / 2)

    if len(f0) <= 0:
        raise ValueError("Null slc in input")

    return np.mean(f0)


def get_poly(poly, poly_type: str, dd: str):
    poly.attrs["offset"][0] = np.datetime64(poly.attrs[dd]["offset"][0], "ns")
    poly.attrs[dd]["offset"][1] = np.array(poly.attrs[dd]["offset"][1])
    poly.attrs[dd]["scale_fact"][0] = np.array(int(poly.attrs[dd]["scale_fact"][0].split()[0]))
    poly.attrs[dd]["scale_fact"][1] = np.array(poly.attrs[dd]["scale_fact"][1])

    return poly


# @timeit
def get_stack_phramp_diff(
    interf: xr.Dataset, drg_full: xr.DataArray, daz_full: xr.DataArray, annot: dict, y_poly: xr.Dataset, multilook: dict
):
    ph_ramp_diff_list = []

    for k, dd in enumerate(interf.acquisition_date.values):
        #  call core function
        y_poly[dd].attrs["type"] = y_poly.type

        acquisition_date_annot = annot[dd]
        # centre scene coordinates
        az_time, rg_time = get_az_rg_times(acquisition_date_annot)

        rg_mid = rg_time[acquisition_date_annot["ImageData"]["SCPPixel"]["Row"]]
        az_mid = az_time[acquisition_date_annot["ImageData"]["SCPPixel"]["Col"]]

        timeca_polynomial_coeffs = get_coefficients("RMA.INCA.TimeCAPoly.Coefs", acquisition_date_annot)
        if timeca_polynomial_coeffs is None:
            raise ValueError(f"Missing TimeCAPoly coefficients for acquisition date {dd} in annotation.")
        dkcoa_polynomial_coeffs = get_coefficients("Grid.Col.DeltaKCOAPoly.Coefs", acquisition_date_annot)
        if dkcoa_polynomial_coeffs is None:
            raise ValueError(f"Missing DeltaKCOA polynomial coefficients for acquisition date {dd} in annotation.")
        azimuth_grid_spacing = acquisition_date_annot["Grid"]["Col"]["SS"]

        res = get_phramp_diff(
            azimuth_time=interf.azimuth_time,
            slant_range_time=interf.slant_range_time,
            daz=daz_full.sel(acquisition_date=dd).coarsen(multilook, boundary="trim").mean(),  # type: ignore mean is injected dynamically
            drg=drg_full.sel(acquisition_date=dd).coarsen(multilook, boundary="trim").mean(),  # type: ignore mean is injected dynamically
            timeca_polynomial_coeffs=timeca_polynomial_coeffs,
            dkcoa_polynomial_coeffs=dkcoa_polynomial_coeffs,
            azimuth_grid_spacing=azimuth_grid_spacing,
            scp_pixel_azimuth_time=az_mid,
            scp_pixel_slant_range_time=rg_mid,
            y_poly=y_poly[dd],
        )
        ph_ramp_diff_list.append(res)

    # return the concatenated dataarray
    return xr.concat(ph_ramp_diff_list, dim="acquisition_date").assign_coords(
        {"acquisition_date": interf.acquisition_date}
    )


def wrap(v: np.ndarray, w: float = 2 * np.pi):
    w_ = w / 2.0
    return ((v + w_) % w) - w_


def sparse_multilook(data: xr.Dataset, cohe_th: float, multilook: dict):
    coords = list(multilook.keys())
    if not np.all([e in data.dims for e in coords]):
        raise Exception(f"Not all {coords} in {data.dims}")

    # set temporary linear coords
    azimuth_time = data.azimuth_time.values
    slant_range_time = data.slant_range_time.values
    az0 = azimuth_time[0]
    rg0 = slant_range_time[0]
    dtaz = (azimuth_time[-1] - azimuth_time[0]).astype(float) / (azimuth_time.size - 1)
    dtrg = (slant_range_time[-1] - slant_range_time[0]) / (slant_range_time.size - 1)
    y_ = (azimuth_time - az0).astype(float) / dtaz
    x_ = (slant_range_time - rg0) / dtrg

    data_copy = data.copy(deep=True)
    data_copy = data_copy.assign_coords({"y_": ("azimuth_time", y_), "x_": ("slant_range_time", x_)})
    data_copy = data_copy.swap_dims({"azimuth_time": "y_", "slant_range_time": "x_"})

    # pewrform multilook considering temporary coords
    multilook_yx = {
        "y_": multilook["azimuth_time"],
        "x_": multilook["slant_range_time"],
    }
    data_ml = sparse_multilook_core(data_copy, cohe_th=cohe_th, multilook=multilook_yx)

    # restore original coords
    az_ml = (data_ml.y_.data * dtaz).astype("timedelta64[ns]") + az0
    rg_ml = data_ml.x_.data * dtrg + rg0
    data_ml = data_ml.assign_coords({"azimuth_time": ("y_", az_ml), "slant_range_time": ("x_", rg_ml)})
    data_ml = data_ml.swap_dims({"y_": "azimuth_time", "x_": "slant_range_time"})
    data_ml = data_ml.drop_vars(["y_", "x_"])

    return data_ml


def sparse_multilook_core(data: xr.Dataset, cohe_th: float, multilook: dict):
    coords = list(multilook.keys())

    x_size_ml = data[coords[1]].size // multilook[coords[1]] + 1

    data_stack = data.stack(points=coords)
    data_stack = data_stack.where((data_stack.cohe > cohe_th).compute(), drop=True)

    x_ml = np.round(data_stack[coords[1]] // multilook[coords[1]]).astype(int)
    y_ml = np.round(data_stack[coords[0]] // multilook[coords[0]]).astype(int)
    ind = (x_ml + y_ml * x_size_ml).astype(int)
    data_stack = data_stack.assign_coords(ind=ind).compute()

    v_mean = data_stack.groupby(["ind"]).median()

    coords_ml = {e: ("points", data_stack[e].astype(float).groupby(["ind"]).mean().data) for e in coords}

    data_ml = xr.Dataset(
        v_mean,
        coords=coords_ml,
    )

    return data_ml


def griddata_interp(data: xr.Dataset, out_coords):
    from scipy.interpolate import griddata

    # set temporary linear coords
    azimuth_time = data.azimuth_time.values
    slant_range_time = data.slant_range_time.values
    az0 = azimuth_time[0]
    rg0 = slant_range_time[0]
    dtaz = (azimuth_time[-1] - azimuth_time[0]).astype(float) / (azimuth_time.size - 1)
    dtrg = (slant_range_time[-1] - slant_range_time[0]) / (slant_range_time.size - 1)
    y_ = (azimuth_time - az0).astype(float) / dtaz
    x_ = (slant_range_time - rg0) / dtrg
    points = np.vstack([x_, y_]).T

    data_copy = data.copy(deep=True)
    data_copy = data_copy.assign_coords({"y_": ("azimuth_time", y_), "x_": ("slant_range_time", x_)})
    data_copy = data_copy.swap_dims({"azimuth_time": "y_", "slant_range_time": "x_"})

    gridx, gridy = out_coords["slant_range_time"], out_coords["azimuth_time"]
    gridx = (gridx - rg0) / dtrg
    gridy = (gridy - az0).astype(float) / dtaz
    gridxx, gridyy = np.meshgrid(gridx, gridy)
    out_points = np.vstack((gridxx.ravel(), gridyy.ravel())).T

    res = griddata(points, data.phase.values, out_points, method="cubic")

    return xr.Dataset(
        data_vars={"phase": (["azimuth_time", "slant_range_time"], res.reshape(gridxx.shape))},
        coords=out_coords,
    )


def get_interf(a, b, ml_factor):
    return (a * b.conj()).coarsen(ml_factor, boundary="trim").mean()


# def get_normalized_interf(a, b, ml_factor, rel_thresh=None, abs_thresh=1e-6):
#     cross = get_interf(a, b, ml_factor)
#     p_a = get_interf(a, a, ml_factor)
#     p_b = get_interf(b, b, ml_factor)

#     coh = cross / np.sqrt(p_a * p_b)

#     valid_mask = (p_a > abs_thresh) & (p_b > abs_thresh)

#     if rel_thresh is not None:
#         is_actively_updated = p_a > (rel_thresh * p_b)
#         valid_mask = valid_mask & is_actively_updated

#     return coh.where(valid_mask, 0j).fillna(0j)


def get_normalized_interf(a, b, ml_factor):
    return (get_interf(a, b, ml_factor) / np.sqrt(get_interf(a, a, ml_factor) * get_interf(b, b, ml_factor))).fillna(0j)


def time_to_pixel(stack: xr.Dataset, y0: float | None = None, x0: float | None = None) -> xr.Dataset:
    """

    Convert from times to pixel dimensions of the input stack

    """

    if y0 is None:
        y0 = stack.azimuth_time[0].values
    if x0 is None:
        x0 = stack.slant_range_time[0].values

    # spacing computation #
    y_spacing, x_spacing = [(stack[e][1] - stack[e][0]).values for e in ["azimuth_time", "slant_range_time"]]

    # times to pixel coordinates #
    x = np.round((stack.slant_range_time - x0) / x_spacing).values.astype("int32")
    y = np.round((stack.azimuth_time - y0) / y_spacing).values.astype("int32")

    stack = stack.assign_coords({"y": (["azimuth_time"], y), "x": (["slant_range_time"], x)})
    return stack.swap_dims(azimuth_time="y", slant_range_time="x")
