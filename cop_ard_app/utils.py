"""
Utility helpers shared by the cop_ard_app processing modules.

It provides data selection, chunk, and multilook aggregation helpers, SAR
metadata and geodesy conversions, coregistration polynomial serialization, and
reference database management for the ARD pipeline.
"""

import itertools
import os

import dask.distributed
import numpy as np
import pyproj
import pyproj.network
import rasterio
import scipy.fft
import xarray as xr
from cop_ard_core.dem import extract_vertical_crs
from cop_ard_core.iono_height import convert_ecef_to_llh, get_iono_height
from cop_ard_core.logger import logger
from cop_ard_core.utils import get_phramp_diff
from pyproj import CRS, Transformer, datadir
from pyproj.transformer import TransformerGroup
from scipy.constants import speed_of_light

from cop_ard_app.computation import map
from cop_ard_app.io import write
from cop_ard_app.model import get_az_rg_times, get_coefficients


class GeoEnvPlugin(dask.distributed.WorkerPlugin):
    """
    Dask `WorkerPlugin` provisioning and validating the geospatial stack.

    On each worker it pins the pyproj data directory, forces offline mode,
    sets `GDAL_DATA` from the installed rasterio package, and strictly checks
    that the PROJ grids for the configured CRS transformation are available and
    that a vertical (geoid) shift is actually applied, raising `RuntimeError`
    otherwise.
    """

    name = "patch_gdal_proj_env"

    def __init__(
        self, in_proj: str = "EPSG:9518", out_proj: str = "EPSG:4979", test_coords: tuple = (12.5, 41.9, 50.0)
    ):
        """
        Initializes the plugin.
        """
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.test_coords = test_coords

    def setup(self, worker):
        """Runs on the worker immediately when it starts."""

        # 1. Base Environment Setup (Locking paths)
        pyproj_dir = datadir.get_data_dir()
        datadir.set_data_dir(pyproj_dir)

        # STRICTLY OFFLINE: Workers are not allowed to use the internet
        pyproj.network.set_network_enabled(False)

        rasterio_base = os.path.dirname(rasterio.__file__)
        rasterio_gdal_dir = os.path.join(rasterio_base, "gdal_data")
        if os.path.exists(rasterio_gdal_dir):
            os.environ["GDAL_DATA"] = rasterio_gdal_dir

        # 2. Strict Metadata Check
        in_crs = CRS(self.in_proj).to_3d()
        out_crs = CRS(self.out_proj).to_3d()
        tg = TransformerGroup(in_crs, out_crs)

        if tg.best_available is False:
            error_msg = (
                f"Worker {worker.address} is missing required PROJ grids for {self.in_proj} -> {self.out_proj}. "
                "Ensure the environment was provisioned properly before starting the worker."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        in_vertical_crs = extract_vertical_crs(in_crs)
        out_vertical_crs = extract_vertical_crs(out_crs)
        if in_vertical_crs != out_vertical_crs:
            # 3. Strict Mathematical Validation
            transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)
            test_x, test_y, test_z = self.test_coords

            # Execute the transformation; errcheck=True catches missing grid errors
            _, _, new_z = transformer.transform(test_x, test_y, test_z, errcheck=True)

            if np.allclose(test_z, new_z, atol=1e-5):
                error_msg = (
                    f"Worker {worker.address} validation failed! Z coordinate did not change. "
                    "The geoid grid is missing or being ignored."
                )
                logger.critical(error_msg)
                raise RuntimeError(error_msg)


def get_margin(
    time_ca_polynomial_coefficients: np.ndarray,
    azimuth_time_spacing: float,
    slant_range_time_spacing: float,
    border: int = 10,
):
    """
    Computes the data selection margins needed to avoid edge effects.

    The azimuth margin comes from the TimeCAPoly linear coefficient times the
    column spacing, converted to a time step; the range margin comes from the
    row spacing converted to a two-way light time. Both are scaled by the
    requested border width in pixels.

    Parameters
    ----------
    time_ca_polynomial_coefficients : np.ndarray
        Time caption polynomial coefficients (`RMA.INCA.TimeCAPoly.Coefs`);
        the linear coefficient gives the azimuth time increment per scene
        coordinate unit.
    azimuth_time_spacing : float
        Column scene-coordinate spacing (`Grid.Col.SS`), in meters.
    slant_range_time_spacing : float
        Row scene-coordinate spacing (`Grid.Row.SS`), in meters.
    border : int, optional
        Width of the margin, in pixels. Defaults to 10.

    Returns
    -------
    dict
        Dictionary with keys `azimuth_time` (`numpy.timedelta64`) and
        `slant_range_time` (float, seconds) holding the margin of each
        dimension.
    """
    step = {
        "azimuth_time": np.timedelta64(
            round((time_ca_polynomial_coefficients[1] * azimuth_time_spacing) * 1e9),
            "ns",
        ),
        "slant_range_time": 2 * slant_range_time_spacing / speed_of_light,
    }
    # Define a margin for data selection to avoid edge effects.
    margin = {e: step[e] * border for e in ["azimuth_time", "slant_range_time"]}
    return margin


def compute_chunks(chunks: dict, overlap: dict | None = None) -> dict:
    """
    Compute overlapping chunk boundaries for use with `map_blocks`.

    Given a dictionary defining the chunk sizes along each dimension, this function
    calculates the start and end indices for each chunk, adding a specified overlap
    between adjacent chunks. This is necessary for algorithms that require access
    to neighboring data during block processing.

    Parameters
    ----------
    chunks : dict
        A dictionary mapping dimension names to tuples representing
        the size of each chunk along that dimension (e.g., ``{'y': (1024, 1024, 512)}``).
    overlap : dict, optional
        A dictionary mapping dimension names (``'x'``, ``'y'``) to
        the desired overlap size in pixels. Defaults to ``{"y": 1, "x": 1}``.

    Returns
    -------
    Dict
        A dictionary where keys are dimension names and values are lists of
        ``[start, end]`` index pairs for each overlapping chunk along that dimension.
        This output format is suitable for the `chunks` argument of `map_blocks`.

    Raises
    ------
    Exception
        If the requested overlap is larger than the chunk size.
    """
    if overlap is None:
        overlap = {"y": 1, "x": 1}
    # --- Input Validation ---
    if (overlap["x"] < chunks["x"][0]) is False:
        raise ValueError(f"Overlap {overlap['x']} is bigger than chunk length {chunks['x'][0]}")
    if (overlap["y"] < chunks["y"][0]) is False:
        raise ValueError(f"Overlap {overlap['y']} is bigger than chunk length {chunks['y'][0]}")

    ch = {}
    # Iterate through each dimension specified in the input chunks.
    for k, v in chunks.items():
        # Calculate the cumulative chunk boundaries.
        li = list(v)
        li.insert(0, 0)
        # Check if overlap should be applied to this dimension and if there's more than one chunk.
        if k in overlap and len(v) != 1:
            delta = overlap[k]
            # Calculate [start, end + overlap] for all chunks except the last one.
            ch[k] = [[en * s, en * s + e + delta] for en, (s, e) in enumerate(zip(li[0:-2], li[1:-1]))]
            # Handle the last chunk separately: [previous_end - overlap, previous_end + last_chunk_size].
            if v[-1] != 1:  # Avoid issues if the last chunk has size 1
                ch[k].append([ch[k][-1][1] - delta, ch[k][-1][1] - delta + v[-1]])
        else:
            # If no overlap or only one chunk, calculate simple [start, end] boundaries.
            ch[k] = [[en * s, en * s + e] for en, (s, e) in enumerate(zip(li[:-1], li[1:]))]

    # --- Generate Product of Chunk Boundaries ---
    # Use itertools.product to create all combinations of chunk boundaries across dimensions.
    r = np.array(list(itertools.product(*[v for v in ch.values()])))
    # Restructure the result into the dictionary format required by `map_blocks`.
    return {k: r[:, en].tolist() for en, k in enumerate(chunks.keys())}


# @timeit
def extract_subband(cslc: xr.DataArray, bw: float, f0: float, dim_transf: str = "slant_range_time") -> xr.DataArray:
    """
    Extracts a frequency subband along one dimension via a Fourier transform.

    The input complex raster is FFT-shifted along `dim_transf`, masked to keep
    only the frequencies within `[f0 - bw/2, f0 + bw/2]`, and inverse
    transformed back to the original domain.

    Parameters
    ----------
    cslc : xr.DataArray
        Complex raster data (e.g. a co-registered SLC stack) to be filtered.
    bw : float
        Width of the frequency subband to keep.
    f0 : float
        Center frequency of the subband, in frequency units of the
        transformed dimension.
    dim_transf : str, optional
        Name of the dimension to transform. Defaults to "slant_range_time".

    Returns
    -------
    xr.DataArray
        The filtered data with the original dimensions; the `slant_range_m`
        and `azimuth_m` coordinates are dropped from the result.
    """
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
        scipy.fft.fftshift(scipy.fft.fft(cslc.fillna(0j).values, norm="ortho", axis=axis), axes=(axis,)),
        dims=freq_dim,
        coords=freq_coords,
    )

    mask = (frange.freq_slant_range_time > f0 - bw / 2) * (frange.freq_slant_range_time < f0 + bw / 2)

    return xr.DataArray(
        scipy.fft.ifft((frange * mask).values, norm="ortho", axis=axis),
        dims=dim,
        coords=cslc.drop_vars(["slant_range_m", "azimuth_m"]).coords,
    )


def get_bounding_box(slcs: xr.DataTree, path: str, multiburst: bool = True) -> list:
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
    for burst in (slcs.bursts if multiburst else {"data": slcs}).values():
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


def get_spacing(slcs: xr.DataTree, path: str) -> list:
    """
    Compute the spacing of a burst stack in azimuth time and slant range time.

    This function iterates over all bursts in a stack of SLCs and extracts the minimum and maximum
    values of `azimuth_time` and `slant_range_time` across all bursts and derive the corresponding spacing.

    Parameters
    ----------
    slcs : xr.DataTree
        A DataTree containing multiple SAR bursts, each with `azimuth_time` and `slant_range_time` coordinates.

    Returns
    -------
    list
        A list of two elements: `[dt_az, dt_rg]`, representing the spacing in azimuth time and slant range time.
    """
    azimuth_time, slant_range_time = [], []

    for burst in slcs.values():
        if burst[path].azimuth_time.size > 0:
            azimuth_time += [
                (burst[path].azimuth_time.max().values - burst[path].azimuth_time.min().values).astype(float)
                / (burst[path].azimuth_time.size - 1)
            ]
        if burst[path].slant_range_time.size > 0:
            slant_range_time += [
                (burst[path].slant_range_time.max().values - burst[path].slant_range_time.min().values)
                / (burst[path].slant_range_time.size - 1)
            ]

    if len(azimuth_time) <= 0 or len(slant_range_time) <= 0:
        raise ValueError("Null slc in input")

    return [
        np.mean(azimuth_time),
        np.mean(slant_range_time),
    ]


def get_poly(poly, poly_type: str, dd: str):
    """
    Normalizes the attributes of a polynomial dataset parsed from JSON.

    The time offset stored in the top-level `offset` attribute and in the
    `dd` attribute group is converted to `numpy.datetime64` (nanoseconds),
    while the offsets and scale factors are converted to numpy arrays.

    Parameters
    ----------
    poly : xr.Dataset
        Dataset holding coregistration polynomials and their attributes.
    poly_type : str
        Type of the polynomial. Currently unused by the implementation.
    dd : str
        Name of the polynomial attribute group to normalize.

    Returns
    -------
    xr.Dataset
        The same dataset with numpy and datetime64 attribute values.
    """
    poly.attrs["offset"][0] = np.datetime64(poly.attrs[dd]["offset"][0], "ns")
    poly.attrs[dd]["offset"][1] = np.array(poly.attrs[dd]["offset"][1])
    poly.attrs[dd]["scale_fact"][0] = np.array(int(poly.attrs[dd]["scale_fact"][0].split()[0]))
    poly.attrs[dd]["scale_fact"][1] = np.array(poly.attrs[dd]["scale_fact"][1])

    return poly


# @timeit
def get_stack_phramp_diff(
    acquisition_id: xr.DataArray,
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    drg_full: xr.DataArray,
    daz_full: xr.DataArray,
    annot: dict,
    y_poly: xr.Dataset,
    multilook: dict | None = None,
):
    """
    Computes the ESD phase ramp difference for every acquisition in a stack.

    For each acquisition ID, the corresponding SICD annotation is looked up,
    the azimuth and slant range times of the SCP pixel are evaluated with
    `get_az_rg_times`, the TimeCAPoly and DeltaKCOA coefficients are read
    from the annotation, and `cop_ard_core.get_phramp_diff` is called on the
    `daz`/`drg` displacement fields (optionally coarsened and averaged over
    the multilook window) together with the matching ESD polynomial.

    Parameters
    ----------
    acquisition_id : xr.DataArray
        One-dimensional coordinate listing the acquisition IDs of the stack.
    azimuth_time : xr.DataArray
        Azimuth time coordinate of the stack.
    slant_range_time : xr.DataArray
        Slant range time coordinate of the stack.
    drg_full : xr.DataArray
        Range component of the geometric displacement fields for the full
        stack, indexed by `acquisition_id`.
    daz_full : xr.DataArray
        Azimuth component of the geometric displacement fields for the full
        stack, indexed by `acquisition_id`.
    annot : dict
        SICD-format annotations keyed by acquisition date, holding the
        `RMA`, `Grid` and `ImageData` metadata sections.
    y_poly : xr.Dataset
        Azimuth (ESD) polynomial data variables keyed by acquisition date.
    multilook : dict, optional
        Coarsening window used to average `daz` and `drg` over multilook
        cells. Defaults to ``None`` (no coarsening).

    Returns
    -------
    xr.DataArray
        Phase ramp differences (radians) concatenated along a new
        `acquisition_id` dimension.

    Raises
    ------
    ValueError
        If the TimeCAPoly or DeltaKCOA polynomial coefficients are missing
        in the annotation for an acquisition date.
    """
    ph_ramp_diff_list = []

    for k, dd in enumerate(acquisition_id.values):
        #  call core function

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
        daz = daz_full.sel(acquisition_id=dd)
        drg = drg_full.sel(acquisition_id=dd)

        if multilook is not None:
            daz = daz.coarsen(multilook, boundary="trim").mean()  # type: ignore
            drg = drg.coarsen(multilook, boundary="trim").mean()  # type: ignore

        res = get_phramp_diff(
            azimuth_time=azimuth_time,
            slant_range_time=slant_range_time,
            daz=daz,
            drg=drg,
            timeca_polynomial_coeffs=timeca_polynomial_coeffs,
            dkcoa_polynomial_coeffs=dkcoa_polynomial_coeffs,
            azimuth_grid_spacing=azimuth_grid_spacing,
            scp_pixel_azimuth_time=az_mid,
            scp_pixel_slant_range_time=rg_mid,
            y_poly=y_poly[dd],
        )
        ph_ramp_diff_list.append(res)

    # return the concatenated dataarray
    return xr.concat(ph_ramp_diff_list, dim="acquisition_id").assign_coords({"acquisition_id": acquisition_id})


def sparse_multilook(data: xr.Dataset, cohe_th: float, multilook: dict):
    """
    Performs a sparse multilook aggregation over an acquisition-parameter stack.

    The `azimuth_time` and `slant_range_time` coordinates are converted to
    temporary linear pixel coordinates (`y_`, `x_`), `sparse_multilook_core`
    aggregates the cells of the multilook window, and the original
    coordinates are reconstructed from the aggregated cell positions.

    Parameters
    ----------
    data : xr.Dataset
        Stack of per-acquisition parameters, with `azimuth_time`,
        `slant_range_time` and a `coherence` variable.
    cohe_th : float
        Minimum coherence of the pixels aggregated by multilook.
    multilook : dict
        Dictionary mapping the dimensions `azimuth_time` and
        `slant_range_time` to the multilook window sizes.

    Returns
    -------
    xr.Dataset
        Multilooked dataset indexed by the reconstructed
        `azimuth_time`/`slant_range_time` coordinates.

    Raises
    ------
    ValueError
        If the multilook dimensions are not all dimensions of `data`.
    """
    coords = list(multilook.keys())
    if not np.all([e in data.dims for e in coords]):
        raise ValueError(f"Not all {coords} in {data.dims}")

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
    """
    Aggregates, per multilook cell, the parameters of coherent stack points.

    Points whose coherence is above `cohe_th` are stacked along a `points`
    dimension, mapped to the multilook cell index `ind` obtained by integer
    division of the linear coordinates by the window sizes, and grouped: each
    variable is reduced to the median of its cell and the coordinates to
    their mean.

    Parameters
    ----------
    data : xr.Dataset
        Dataset indexed by the two coordinates listed in `multilook` (in the
        wrapped helper these are the linear coordinates `y_` and `x_`) and
        holding a `coherence` variable.
    cohe_th : float
        Coherence threshold; only points above it are aggregated.
    multilook : dict
        Dictionary mapping the coordinate names to the window sizes.

    Returns
    -------
    xr.Dataset
        Dataset with a stacked `points` dimension, one row per non-empty
        multilook cell, holding the per-cell medians and mean coordinates.
    """
    coords = list(multilook.keys())

    x_size_ml = data[coords[1]].size // multilook[coords[1]] + 1

    data_stack = data.stack(points=coords)
    data_stack = data_stack.where((data_stack.coherence > cohe_th).compute(), drop=True)

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


def slice_and_drop_coordinates(
    ds: xr.DataArray,
    dimension_coordinates_to_keep: list[str],
    slices: dict[str, slice] | None = None,
) -> xr.DataArray:
    """Helper to apply slices and drop unwanted coordinates."""
    coords_to_drop = [k for k in ds.coords if k not in dimension_coordinates_to_keep]
    if slices is not None:
        ds = ds.isel(slices)
    return ds.drop_vars(coords_to_drop)


def get_bbox(slcs, reference_date: str | None = None):
    """
    Get the bounding box of the SAR data.

    Parameters
    ----------
    slcs : xr.DataTree
        DataTree of SAR data.
    reference_date : str, optional
        Reference date for selecting specific data.

    Returns
    -------
    list
        Bounding box [min_lon, min_lat, max_lon, max_lat].
    """
    lat, lon = [], []

    # Iterate through all scenes.
    for slc in (slcs.bursts if "bursts" in slcs else {"data": slcs}).values():
        # Optionally filter by a reference date.
        for date in slc.orig if reference_date is None else [reference_date]:
            # Extract corner coordinates from the metadata.
            annot = slc.orig[date].measurements.slc.attrs
            annot = annot[next(iter(annot.keys()))]
            for v in annot["GeoData"]["ImageCorners"]:
                lat.append(v["Lat"])
                lon.append(v["Lon"])

    # Calculate the min/max latitude and longitude, rounding to one decimal place.
    return [
        np.floor(np.min(lon) * 10) / 10,
        np.floor(np.min(lat) * 10) / 10,
        np.ceil(np.max(lon) * 10) / 10,
        np.ceil(np.max(lat) * 10) / 10,
    ]


def save_single_burst(
    burst_id,
    current_reference,
    stable_reference,
    reference_db_path,
    multiburst,
    reference_id: str | None = None,
):
    """
    Writes the current and stable references of a single burst to disk.

    The current reference is always written to
    `<reference_db_path>/<path>/current_reference`; when a stable reference
    is provided, it is sliced per `acquisition_date` and each slice is
    written under `<reference_db_path>/<path>/stable_reference/<acq_date>`.
    The node path is `bursts/<burst_id>` in the multiburst case and
    `data/<reference_id>` otherwise.

    Parameters
    ----------
    burst_id : str
        Identifier of the burst (used as the node path in a multiburst tree).
    current_reference : xr.Dataset
        Current reference cstack to write.
    stable_reference : xr.Dataset
        Stable (EOPF) reference cstack indexed by `acquisition_date`; it is
        skipped when ``None``.
    reference_db_path : str
        Root path of the reference database.
    multiburst : bool
        Whether the data are organized in a multiburst tree.
    reference_id : str, optional
        Data node identifier, mandatory when `multiburst` is False. Defaults
        to ``None``.

    Raises
    ------
    ValueError
        If `multiburst` is False and no `reference_id` is given.
    """
    if (not multiburst) and (reference_id is None):
        raise ValueError("In single burst case, the reference_id is mandatory")
    path = f"bursts/{burst_id}" if multiburst else f"data/{reference_id}"
    current_reference_path = f"{reference_db_path}/{path}/current_reference"
    stable_reference_path = f"{reference_db_path}/{path}/stable_reference"

    # 1. Always write the current reference
    write(current_reference, current_reference_path)

    # 2. Write stable references only if they exist
    if stable_reference is not None:
        # Use acquisition_id to iterate and slice safely, avoiding datetime string formatting KeyError
        for acq_date in stable_reference.acquisition_date.values:
            # Safely extract the slice for this specific acquisition
            s_ref_slice = stable_reference.sel(acquisition_date=[acq_date])

            write(s_ref_slice, f"{stable_reference_path}/{acq_date}")


def update_reference_database(
    current_reference_tree, stable_references_tree, reference_db_path, reference_id: str | None = None
):
    """
    Dispatches `save_single_burst` over every node of the reference trees.

    The `measurements/cstack` of each current reference and of the matching
    stable reference are extracted; the stable cstack falls back to ``None``
    when the stable tree is the dummy placeholder created to bypass EOPF
    errors (flagged by the `is_empty` attribute of its `status` node) or the
    corresponding node is missing. The collected arguments are then mapped
    over `save_single_burst`, optionally with dask.

    Parameters
    ----------
    current_reference_tree : xr.DataTree
        DataTree of current references, either a multiburst tree or a single
        node.
    stable_references_tree : xr.DataTree
        DataTree of stable (EOPF) references, or ``None``.
    reference_db_path : str
        Root path of the reference database.
    reference_id : str, optional
        Data node identifier used in the single-node case. Defaults to
        ``None``.
    """
    multiburst = "bursts" in current_reference_tree

    # 1. Check if the stable_references_tree is the dummy one we created to bypass EOPF errors
    is_stable_dummy = False
    if stable_references_tree is not None and "status" in stable_references_tree:
        is_stable_dummy = stable_references_tree["status"].attrs.get("is_empty", False)

    args = []
    # Iterate over the valid current references
    for k, v in (current_reference_tree.bursts if multiburst else {"data": current_reference_tree}).items():
        c_ref = v.measurements.cstack

        # 2. Safely extract s_ref, falling back to None if it's dummy or missing
        s_ref = None
        if not is_stable_dummy and stable_references_tree is not None:
            if multiburst:
                # Safely traverse the multiburst tree without causing KeyErrors
                if "bursts" in stable_references_tree and k in stable_references_tree.bursts:
                    node = stable_references_tree.bursts[k]
                    if "measurements" in node and "cstack" in node.measurements:
                        s_ref = node.measurements.cstack
            else:
                # Safely traverse the single data tree
                if "measurements" in stable_references_tree and "cstack" in stable_references_tree.measurements:
                    s_ref = stable_references_tree.measurements.cstack

        a = (k, c_ref, s_ref)
        args.append(a)

    # 3. Transpose arguments and pass them to the map function
    args = list(zip(*args))
    map(
        save_single_burst,
        args,
        kwargs={"reference_db_path": reference_db_path, "multiburst": multiburst, "reference_id": reference_id},
        enable_dask=True,
    )


def split_slc_datatrees(
    slcs,
    reference_id: str,
    multiburst: bool = True,
) -> tuple[xr.DataTree, xr.DataTree]:
    """Splits an SLC dataset or multi-burst object into reference and secondary DataTrees.

    Parameters
    ----------
    slcs : object
        The input SLC container containing `.bursts` or direct xarray data.
    reference_id : str
        Acquisition ID for the reference SLC dataset.
    multiburst : bool, default True
        Flag indicating if the input contains multiple bursts.

    Returns
    -------
    tuple[xr.DataTree, xr.DataTree]
        A tuple of (reference_slcs, secondary_slcs) DataTrees.
    """
    reference_slcs = xr.DataTree()
    secondary_slcs = xr.DataTree()
    reference_slcs.attrs = slcs.attrs
    # Determine dictionary items based on multiburst flag
    burst_items = slcs.bursts.items() if multiburst else {"data": slcs}.items()

    for k, v in burst_items:
        prefix = f"bursts/{k}/" if multiburst else ""

        # Populate reference tree
        ref_path = f"{prefix}coreg"
        reference_slcs[ref_path] = v.coreg
        m = v.coreg.measurements
        if not isinstance(m, xr.Dataset):
            m = m.to_dataset()
        reference_slcs[ref_path]["measurements"] = m.sel(acquisition_id=[reference_id])
        reference_slcs[f"{prefix}orig"] = v.orig

        # Populate secondary tree
        sec_path = f"{prefix}coreg/measurements"
        secondary_slcs[sec_path] = m.drop_sel(acquisition_id=reference_id)

    return reference_slcs, secondary_slcs


def recompose_slc_datatrees(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    multiburst: bool = True,
    keep_reference: bool = False,
) -> xr.DataTree:
    """Recomposes a single SLC DataTree from reference and secondary DataTrees.

    Parameters
    ----------
    reference_slcs : xr.DataTree
        DataTree containing reference measurements and 'orig' groups.
    secondary_slcs : xr.DataTree
        DataTree containing secondary measurements.
    multiburst : bool, default True
        Flag indicating if the dataset contains multiple bursts.

    Returns
    -------
    xr.DataTree
        Recomposed SLC DataTree containing combined measurements and original data.
    """
    slcs = reference_slcs.copy()

    if multiburst and "bursts" in reference_slcs.children:
        burst_keys = list(reference_slcs["bursts"].children.keys())
    else:
        burst_keys = [None]

    for k in burst_keys:
        prefix = f"bursts/{k}/" if (multiburst and k is not None) else ""
        meas_path = f"{prefix}coreg/measurements"

        sec_ds = secondary_slcs[meas_path].ds

        if keep_reference:
            ref_ds = reference_slcs[meas_path].ds
            combined_ds = xr.concat([ref_ds, sec_ds], dim="acquisition_id")
            slcs[meas_path] = combined_ds.sortby("acquisition_date")
        else:
            slcs[meas_path] = sec_ds

        slcs[meas_path] = sanitize_coordinates(slcs[meas_path].to_dataset())

    return slcs


def compute_iono_and_orb_height(
    reference_slcs: xr.DataTree,
    secondary_slcs: xr.DataTree,
    wdir_path_tmp: str,
    acquisition_id: str,
    iono_height_input: float | None = None,
    default_iono_height: float = 350_000.0,
    dbfile: str | None = None,
) -> tuple[float, float]:
    """Calculates orbital height (h_orb) and determines ionospheric height (h_iono).

    Parameters
    ----------
    reference_slcs : xr.DataTree
        DataTree container containing reference SLC coregistration and origin metadata.
    secondary_slcs : xr.DataTree
        DataTree container containing secondary SLC coregistration and origin metadata.
    wdir_path_tmp : str
        Temporary working directory path for ionosphere calculation.
    acquisition_id : str
        Target acquisition ID to select measurements from.
    iono_height_input : float or int or None, default None
        Pre-defined ionospheric height. If None, `get_iono_height()` is called.
    default_iono_height : float, default 350000.0
        Fallback height (in meters) if `get_iono_height()` raises an exception.

    Returns
    -------
    tuple[float, float]
        A tuple containing (h_orb, h_iono) in meters.
    """
    # Resolve single vs multi-burst object structure
    multiburst = "bursts" in reference_slcs
    burst_reference = next(iter(reference_slcs.bursts.values())) if multiburst else reference_slcs
    burst_secondary = next(iter(secondary_slcs.bursts.values())) if multiburst else secondary_slcs

    # Extract metadata attributes for secondary scene
    best_pol = get_best_pol(burst_secondary.coreg.measurements.cstack.polarization.values)

    acq_dates = burst_secondary.coreg.measurements.acquisition_date.values

    if acq_dates.size != 1:
        raise ValueError(f"Expected exactly 1 acquisition date for '{acquisition_id}', got {acq_dates.size}.")

    acq_date = str(acq_dates[0])
    logger.debug(f"Resolved secondary acquisition date: {acq_date}")

    # Calculate orbital height (h_orb) using burst_reference node
    scp_dict = burst_reference.orig[acquisition_id].measurements.slc.attrs[best_pol]["SCPCOA"]["ARPPos"]
    scp_ecef = np.array([scp_dict[k] for k in ["X", "Y", "Z"]])
    h_orb = float(convert_ecef_to_llh(scp_ecef)[2])
    logger.info(f"Calculated h_orb: {h_orb} m")

    # Determine ionospheric height (h_iono) logic
    if iono_height_input is None:
        try:
            h_iono = float(get_iono_height(acq_date, scp_dict, wdir_path_tmp, dbfile))
        except RuntimeError as err:
            logger.warning(
                f"Failed to fetch iono height automatically ({err}). Falling back to default: {default_iono_height} m"
            )
            h_iono = float(default_iono_height)
    elif isinstance(iono_height_input, (float, int)):
        h_iono = float(iono_height_input)
    else:
        raise TypeError(f"Invalid iono_height_input type: {type(iono_height_input)}. Expected float, int, or None.")

    logger.info(f"Final h_iono: {h_iono} m")

    return h_orb, h_iono


def sanitize_coordinates(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Re-assigns the non-dimensional coordinates of a dataset or array.

    Coordinates whose name is not one of the object's dimensions are rebuilt
    from their dimensions and values, leaving the index (dimension)
    coordinates untouched.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input object whose coordinates are to be sanitized.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The same object with sanitized non-dimensional coordinates.
    """
    clean_coords = {}
    for coord_name, coord_var in ds.coords.items():
        if coord_name not in ds.dims:
            clean_coords[coord_name] = (coord_var.dims, coord_var.values)

    return ds.assign_coords(clean_coords)


def get_best_pol(polarization_values: list) -> str:
    """
    Select the preferred polarization based on a priority list.

    Parameters
    ----------
    polarization_values : list
        List of available polarizations.

    Returns
    -------
    str
        The selected polarization.
    """
    # select preferred polarization
    available_pols = {str(i) for i in polarization_values}
    priorities = ["VV", "HH", "VH", "HV"]
    pol = next((p for p in priorities if p in available_pols), None)
    if pol is None:
        raise ValueError(f"No polarization has been found {available_pols}")

    return pol


def build_coreg_poly_dict(description, y_poly=None, x_poly=None, iono_poly=None):
    """
    Builds a nested dictionary for coregistration metadata, handling both
    azimuth (y_poly) and range (x_poly) polynomials.

    Args:
        description (str): A brief explanation of the correction.
        application_method (str): How the polynomial is applied (e.g., "phase_ramp", "warp").
        y_poly (xr.Dataset, optional): Dataset containing azimuth polynomial variables.
        x_poly (xr.Dataset, optional): Dataset containing range polynomial variables.

    Returns:
        dict: A structured dictionary ready for attributes.
    """
    out_dict = {
        "description": description,
        "acquisitions": {},
    }

    def _to_float(val) -> float:
        """
        Returns a scalar-like value as a Python float.
        """
        return float(np.asarray(val).flat[0])

    # Funzione interna per processare un singolo dataset e aggiornare out_dict
    def _process_dataset(poly_dataset, direction_name):
        """
        Copies one direction of polynomial data into the coregistration dict.

        Each data variable is taken as the polynomial of one acquisition
        (identified by its name) and its flattened coefficients, shape,
        offsets, scale factors and time reference are stored in
        `out_dict["acquisitions"]` under the given direction name.

        Parameters
        ----------
        poly_dataset : xr.Dataset
            Dataset of polynomial data variables named by acquisition ID.
        direction_name : str
            Direction key used in the output dictionary ("azimuth", "range"
            or "ionosphere").
        """
        for acq_id, poly_data in poly_dataset.data_vars.items():
            acq_id_str = str(acq_id)
            time_ref = str(poly_data.attrs.get("off_y", ""))
            offset = poly_data.attrs.get("offset", (0.0, 0.0))
            scale = poly_data.attrs.get("scale_fact", (1.0, 1.0))
            logger.critical(f"time_Ref: {time_ref}")
            # Se l'acquisizione non esiste ancora nel dizionario, la creiamo
            if acq_id_str not in out_dict["acquisitions"]:
                out_dict["acquisitions"][acq_id_str] = {}

            # Aggiungiamo i dati per la specifica direzione
            out_dict["acquisitions"][acq_id_str][direction_name] = {
                "coefficients": poly_data.values.flatten().tolist(),
                "shape": list(poly_data.shape),
                "azimuth_offset": _to_float(offset[0]),
                "range_offset": _to_float(offset[1]),
                "azimuth_scale": _to_float(scale[0]),
                "range_scale": _to_float(scale[1]),
                "azimuth_time_reference_ns": time_ref,
            }

    if y_poly is not None:
        _process_dataset(y_poly, "azimuth")

    if x_poly is not None:
        _process_dataset(x_poly, "range")
    if iono_poly is not None:
        _process_dataset(iono_poly, "ionosphere")

    return out_dict
