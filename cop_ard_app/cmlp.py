import numpy as np
import xarray as xr
from cop_ard_core.logger import logger, timeit

import cop_ard_app.utils
from cop_ard_app.io import (
    get_storage_options,
    normpath,
    read,
    save_empty_dataset,
    write,
)

SHORT_DIMS = ["polarization", "azimuth_time", "slant_range_time"]
DIMS = ["polarization", "acquisition_date", "azimuth_time", "slant_range_time"]
LONG_DIMS = ["polarization", "past_reference_date", "azimuth_time", "slant_range_time"]


@timeit
def single_cmlp(
    burst_id: str,
    cstack: xr.DataArray,
    multilook: dict,
    working_directory: str,
    revisit_cycle: np.timedelta64,
    memory_param: float = 0.95,
    coherence_threshold: float = 0.1,
    stable_reference_step: int = 10,
    max_past_stable_history: int = 10,
    last_reference: xr.DataArray | None = None,
    stable_reference: xr.DataArray | None = None,
    **kwargs,
):
    logger.info(f"Setting memory param : {memory_param}")
    logger.info(f"Setting coherence_threshold : {coherence_threshold}")
    logger.info(f"Setting revisit_cycle : {revisit_cycle.astype('timedelta64[D]')}")
    logger.info(f"Setting stable_reference_step : {stable_reference_step}")
    logger.info(f"Setting max_past_stable_history : {max_past_stable_history}")
    # 1. Setup paths and validation
    expected = set(DIMS[2:])
    if set(multilook) != expected:
        raise ValueError(f"Wrong keys: {set(multilook)}. Expected {expected}")

    path = f"bursts/{burst_id}"
    data_path = normpath(f"{working_directory}/{path}")

    # 2. Prepare Data (Slicing, Padding, Chunking)
    cstack_sel, slices, pad, chunks = _prepare_cstack(cstack, multilook)
    acquisition_dates = cstack_sel.acquisition_date.values

    # Handle inputs for resumption (Apply same slicing to inputs)
    if last_reference is not None:
        last_reference = last_reference.isel(slices).drop_vars([k for k in last_reference.coords if k not in DIMS])
    if stable_reference is not None:
        stable_reference = stable_reference.isel(slices).drop_vars(
            [k for k in stable_reference.coords if k not in DIMS]
        )

    # 3. Calculate Date Milestones
    stable_reference_dates = _calculate_stable_dates(
        acquisition_dates, revisit_cycle, stable_reference_step, stable_reference
    )

    # 4. Initialize Output Stores
    _initialize_cmlp_zarr_stores(
        data_path,
        cstack,
        acquisition_dates,
        stable_reference_dates,
        multilook,
        chunks,
        kwargs,
    )

    # 5. Get Starting State
    current_reference, start_date = get_current_reference(cstack_sel, last_reference, stable_reference)
    stable_cstack = stable_reference
    for date_idx, date in enumerate(cstack_sel.acquisition_date.values):
        logger.debug(f"{'*' * 50}")
        logger.info(f"Processing: {date}")

        slc = cstack_sel.sel(acquisition_date=date).fillna(0j)
        normalized_interf = _compute_and_save_interferogram(
            target_slc=slc,
            reference_slc=current_reference,
            multilook=multilook,
            output_path=f"{data_path}/cmlp/{date}",
            compute_phase=True,
            name="short_term_coh",
        )

        # Step 2: Update the Reference State
        current_reference = _update_recursive_reference(
            slc=slc,
            current_reference=current_reference,
            normalized_interf=normalized_interf,
            memory_param=memory_param,
            chunks=chunks,
        )

        if date in stable_reference_dates:
            stable_cstack = _update_stable_reference(
                current_reference=current_reference,
                date=date,
                stable_reference_dates=stable_reference_dates,
                data_path=data_path,
                pad=pad,
                chunks=chunks,
                slices=slices,
                initial_stable_reference=stable_reference,
            )

        if stable_cstack is not None:
            stable_cstack = stable_cstack.sortby("acquisition_date").isel(
                acquisition_date=slice(-max_past_stable_history, None)
            )
            current_reference = _correct_drift(
                stable_cstack=stable_cstack,
                current_reference=current_reference,
                multilook=multilook,
                coherence_threshold=coherence_threshold,
                chunks=chunks,
                shape=normalized_interf.shape,
                output_path=f"{data_path}/cmlp/{date}",
            )

    return current_reference


def _prepare_cstack(cstack: xr.DataArray, multilook: dict) -> tuple[xr.DataArray, dict, dict, dict]:
    """
    Slices and chunks the input stack to align with multilook factors.

    Parameters
    ----------
    cstack : xr.DataArray
        The input complex stack.
    multilook : dict
        Multilook factors.

    Returns
    -------
    cstack_sel : xr.DataArray
        The sliced and sorted data array.
    slices : dict
        The slice objects used for cropping.
    pad : dict
        Padding required to restore original size.
    chunks : dict
        Chunk sizes for dask arrays.
    """
    # Create slices that are integer multiples of the multilook factors
    slices = {k: slice(0, cstack[k].size // v * v) for k, v in multilook.items()}

    # Calculate chunks (preserving original chunk structure)
    chunks = {k: v[0] for k, v in cstack.chunksizes.items() if k in SHORT_DIMS}
    # Drop unnecessary coordinates and sort
    cstack_sel = cstack.isel(slices).drop_vars([k for k in cstack.coords if k not in DIMS]).sortby("acquisition_date")

    # Calculate padding needed to restore original shape later
    pad = {k: (0, cstack[k].size - cstack_sel[k].size) for k in multilook.keys()}

    return cstack_sel, slices, pad, chunks


def _calculate_stable_dates(
    acquisition_dates: np.ndarray,
    revisit_cycle: np.timedelta64,
    stable_reference_step: int,
    stable_reference: xr.DataArray | None = None,
) -> np.ndarray:
    """
    Determines the dates that serve as stable reference milestones.

    Parameters
    ----------
    acquisition_dates : np.ndarray
        Array of all acquisition dates.
    revisit_cycle : np.timedelta64
        Time delta representing one revisit cycle.
    stable_reference_step : int
        Number of cycles between stable references.

    Returns
    -------
    stable_reference_dates : np.ndarray
        Subset of dates selected as stable references.
    start_date : np.datetime64
        The very first date in the stack (used for offset calculations).
    """
    dates_ns = acquisition_dates.astype("datetime64[ns]")
    start_date = np.min(dates_ns)
    if stable_reference is not None:
        start_date = np.min(stable_reference.acquisition_date.astype("datetime64[ns]"))
    # Vectorized calculation of elapsed cycles
    elapsed_cycles = (dates_ns - start_date) / revisit_cycle.astype("timedelta64[ns]")

    # Determine bins/milestones
    # We use the iterative approach to match original logic exactly,
    # though vectorization is possible.
    stable_refs = []
    threshold = stable_reference_step

    for i, cycle in enumerate(elapsed_cycles):
        if cycle > threshold:
            stable_refs.append(acquisition_dates[i])
            threshold += stable_reference_step

    return np.array(stable_refs)


def _initialize_cmlp_zarr_stores(
    data_path: str,
    cstack: xr.DataArray,
    acquisition_dates: np.ndarray,
    stable_reference_dates: np.ndarray,
    multilook: dict,
    chunks: dict,
    kwargs: dict,
) -> None:
    """
    Creates empty Zarr/NetCDF stores on disk for the results.

    Parameters
    ----------
    data_path : str
        Base directory for output.
    cstack_sel : xr.DataArray
        The prepared input stack.
    stable_reference_dates : np.ndarray
        Dates for the stable reference store.
    acquisition_date_offset : int
        Offset index for long-term storage.
    multilook : dict
        Multilook factors.
    dims : tuple
        Dimensions of the data.
    chunks : dict
        Chunk configuration.
    kwargs : dict
        Additional arguments for storage options.

    Returns
    -------
    y : np.ndarray
        Indices for upsampling azimuth.
    x : np.ndarray
        Indices for upsampling range.
    """
    storage_opts = get_storage_options(kwargs)

    # 1. Initialize Stable Reference Store
    coords_stable = {k: v.values for k, v in cstack.coords.items() if k in SHORT_DIMS}
    # Define shape for stable ref
    shape_stable = [coords_stable[i].size for i in SHORT_DIMS]
    for date in stable_reference_dates:
        coords_stable["acquisition_date"] = date
        save_empty_dataset(
            data_vars=["cstack"],
            dims={"cstack": SHORT_DIMS},
            coords=coords_stable,
            shape={"cstack": shape_stable},
            chunks=chunks,
            dtype={"cstack": "complex64"},
            skip_read=True,
            data_path=f"{data_path}/stable_reference/{date}",
            storage_options=storage_opts,
        )
    # 2. Initialize Short/Long Term Stores (Coarsened Grids)
    # Calculate coarsened coordinates
    # Note: Using .mean() computes the values, ensuring we have concrete coords
    coords_coarse = {k: cstack[k].coarsen({k: v}, boundary="trim").mean().values for k, v in multilook.items()}  # type: ignore mean is injected dynamically
    coords_coarse["polarization"] = cstack.polarization.values

    for date in acquisition_dates:
        data_vars = ["phase", "short_term_coh"]
        dims_coarse = {k: SHORT_DIMS for k in data_vars}
        s0 = [coords_coarse[i].size for i in SHORT_DIMS]
        shape_coarse = {k: s0 for k in data_vars}
        ch = chunks.copy()
        coords_coarse["acquisition_date"] = date

        sd = stable_reference_dates.astype("datetime64[ns]")
        d = date.astype("datetime64[ns]")
        (ind,) = np.where(sd <= d)
        if ind.size != 0:
            data_vars.append("long_term_coh")
            dims_coarse["long_term_coh"] = LONG_DIMS
            coords_coarse["past_reference_date"] = stable_reference_dates[ind]
            shape_coarse["long_term_coh"] = [coords_coarse[i].size for i in LONG_DIMS]
            ch["past_reference_date"] = 1
        save_empty_dataset(
            data_vars=data_vars,
            dims=dims_coarse,
            coords=coords_coarse,
            shape=shape_coarse,
            chunks=ch,
            dtype={k: "float32" for k in data_vars},
            skip_read=True,
            data_path=f"{data_path}/cmlp/{date}",
            storage_options=storage_opts,
        )

    return


@timeit
def _compute_and_save_interferogram(
    target_slc: xr.DataArray,
    reference_slc: xr.DataArray,
    multilook: dict,
    output_path: str,
    compute_phase: bool,
    name: str,
) -> np.ndarray:
    """
    Computes a normalized interferogram between a target and reference,
    extracts coherence and phase, and writes the result to a Zarr store.

    Parameters
    ----------
    target_slc : xr.DataArray
        The 'slave' image (e.g., current date or oldest stable reference).
    reference_slc : xr.DataArray
        The 'master' image (e.g., current running reference).
    multilook : dict
        Multilook factors.
    output_path : str
        Full path to the output Zarr store (e.g., ".../long_term_cmlc").
    region_slice : slice
        The slice object defining where to write in the 'acquisition_date' dimension.
    dims : tuple
        Dimensions for the output dataset (e.g., ('polarization', 'acquisition_date', ...)).

    Returns
    -------
    xr.DataArray
        The computed normalized interferogram (complex).
    """
    # 1. Compute Normalized Interferogram
    normalized_interf = cop_ard_app.utils.get_normalized_interf(target_slc, reference_slc, multilook).values

    # 2. Extract Statistics
    data = {}
    data[name] = (SHORT_DIMS, np.abs(normalized_interf))
    if compute_phase:
        data["phase"] = (SHORT_DIMS, np.angle(normalized_interf))
    ds = xr.Dataset(data)

    # 3. Write to Disk
    write(
        ds,
        output_path,
        mode="r+",
        skip_read=True,
        consolidated=False,
    )

    return normalized_interf


@timeit
def _update_recursive_reference(
    slc: xr.DataArray,
    current_reference: xr.DataArray,
    normalized_interf: xr.DataArray,
    memory_param: float,
    chunks: dict,
) -> xr.DataArray:
    """
    Updates the recursive reference using the current SLC and the
    upsampled phasor from the normalized interferogram.
    """
    # 1. Extract unit phasor
    magnitude = np.abs(normalized_interf)
    safe_magnitude = np.where(magnitude > 0, magnitude, 1.0)
    phasor_interf = normalized_interf / safe_magnitude

    # 2. Explicitly unpack dimensions for readability
    _, coarse_az, coarse_rg = phasor_interf.shape
    n_bands, full_az, full_rg = slc.shape

    # 3. Define 5D shapes for block-wise broadcasting
    blocked_slc_shape = (n_bands, coarse_az, full_az // coarse_az, coarse_rg, full_rg // coarse_rg)
    broadcast_phasor_shape = (n_bands, coarse_az, 1, coarse_rg, 1)

    # 4. Apply phase rotation using underlying Dask/NumPy arrays
    correction = slc.data.reshape(blocked_slc_shape) * phasor_interf.reshape(broadcast_phasor_shape).conj()
    mask = np.isnan(correction)
    correction[mask] = 0
    # Apply fading memory filter
    new_reference = (memory_param * current_reference) + correction.reshape(slc.shape)

    return new_reference.compute().chunk(chunks)


def get_current_reference(
    cstack: xr.DataArray,
    last_reference: xr.DataArray | None,
    stable_reference: xr.DataArray | None,
):
    date = np.min(cstack.acquisition_date.values.astype("datetime64"))
    date_l = np.max(last_reference.acquisition_date.values.astype("datetime64")) if last_reference is not None else None
    date_s = (
        np.max(stable_reference.acquisition_date.values.astype("datetime64")) if stable_reference is not None else None
    )
    current_reference = cstack.sel(acquisition_date=date.astype(str))

    if (date_l is not None) and (date_s is not None):
        if date_l > date_s:
            if date_l < date and isinstance(last_reference, xr.DataArray):
                current_reference = last_reference.sel(acquisition_date=date_l.astype(str))
            else:
                raise Exception(f"Last current reference {date_l} is more recent than cstack {date}")
        else:
            if date_s < date and isinstance(stable_reference, xr.DataArray):
                current_reference = stable_reference.sel(acquisition_date=date_s.astype(str))
            else:
                raise Exception(f"Last stable reference {date_s} is more recent than cstack {date}")
    else:
        if date_l is not None and isinstance(last_reference, xr.DataArray):
            if date_l < date:
                current_reference = last_reference.sel(acquisition_date=date_l.astype(str))
            else:
                raise Exception(f"Last current reference {date_l} is more recent than cstack {date}")
        if date_s is not None and isinstance(stable_reference, xr.DataArray):
            if date_s < date:
                current_reference = stable_reference.sel(acquisition_date=date_s.astype(str))
            else:
                raise Exception(f"Last stable reference {date_s} is more recent than cstack {date}")

    date = current_reference.acquisition_date.values
    current_reference = current_reference.drop_vars({k for k in current_reference.coords.keys() if k not in SHORT_DIMS})
    return current_reference, date


@timeit
def _update_stable_reference(
    current_reference: xr.DataArray,
    date: np.datetime64,
    stable_reference_dates: np.ndarray,
    data_path: str,
    pad: dict,
    chunks: dict,
    slices: dict,
    initial_stable_reference: xr.DataArray | None,
) -> xr.DataArray:
    """
    Updates the stable reference store with the current reference state
    and returns the updated stable stack history.

    Parameters
    ----------
    current_reference : xr.DataArray
        The current running reference to be saved.
    date : np.datetime64
        The current acquisition date (must be present in stable_reference_dates).
    stable_reference_dates : np.ndarray
        Array of all pre-calculated stable reference dates.
    data_path : str
        Base path for data storage.
    pad : dict
        Padding to restore original dimensions.
    chunks : dict
        Chunk sizes for dask.
    dims : tuple
        Dimension order (e.g., polarization, acquisition_date, ...).
    slices : dict
        Slicing to remove padding when reading back.
    initial_stable_reference : xr.DataArray | None
        The stable reference passed at the start of the function (historical data).

    Returns
    -------
    xr.DataArray
        The updated stable stack containing history up to the current date.
    """
    logger.info(f"Updating stable reference : {date}")

    # 1. Find the write index
    idx = np.searchsorted(stable_reference_dates, date)

    # 2. Prepare the dataset (Pad -> Chunk -> Expand -> Transpose -> Dataset)
    ds = current_reference.pad(pad, constant_values=np.nan).chunk(chunks).to_dataset(name="cstack")
    # 3. Write to the specific slot
    ds = write(
        ds,
        f"{data_path}/stable_reference/{date}",
        mode="r+",
        skip_read=True,
        consolidated=False,
        region={k: slice(None) for k in SHORT_DIMS},
    )
    # 4. Reload the updated history from disk
    # We read up to the current index (idx + 1) to get the 'new' history
    # We apply the 'slices' to remove the padding we just added (if any)
    new_stable_cstack = [
        read(f"{data_path}/stable_reference/{date}", consolidate_metadata=False).cstack.isel({**slices})
        for date in stable_reference_dates[: idx + 1]
    ]
    # 5. Concatenate with historical reference (if it exists)
    if initial_stable_reference is not None:
        new_stable_cstack = xr.concat([initial_stable_reference, *new_stable_cstack], dim="acquisition_date")
    else:
        new_stable_cstack = xr.concat(new_stable_cstack, dim="acquisition_date")
    return new_stable_cstack


@timeit
def _correct_drift(
    stable_cstack: xr.DataArray,
    current_reference: xr.DataArray,
    multilook: dict,
    coherence_threshold: float,
    chunks: dict,
    shape: tuple,
    output_path: str,
) -> xr.DataArray:
    """
    Estimates and applies long-term drift correction using historical stable references.

    Parameters
    ----------
    stable_cstack : xr.DataArray
        Stack of past stable references (sorted by date).
    current_reference : xr.DataArray
        The current running reference to be corrected.
    multilook : dict
        Multilook factors.
    coherence_threshold : float
        Minimum coherence required to consider a pixel reliable for drift estimation.
    y : np.ndarray
        Indices for upsampling azimuth.
    x : np.ndarray
        Indices for upsampling range.
    chunks : dict
        Chunk sizes for the output Dask array.

    Returns
    -------
    xr.DataArray
        The drift-corrected current reference.
    """

    # 1. Initialize Drift Accumulator
    shape_coh = (shape[0], stable_cstack.acquisition_date.size, shape[1], shape[2])
    drift_interf = np.full(shape, np.nan, dtype="complex64")
    long_term_coh = np.empty(shape_coh, dtype="float32")
    # 2. Iteratively fill drift_interf with coherent pixels from history
    for iteration, stable_date in enumerate(stable_cstack.acquisition_date.values):
        stable_ref_slice = stable_cstack.sel(acquisition_date=stable_date)

        # Calculate interference between history and current
        normalized_vals = cop_ard_app.utils.get_normalized_interf(current_reference, stable_ref_slice, multilook).values
        coherence = np.abs(normalized_vals)
        is_empty = np.isnan(drift_interf)
        # Check if we are fully filled
        if is_empty.any():
            # Create Masks
            is_coherent = coherence >= coherence_threshold
            mask = is_coherent * is_empty
            # Fill holes
            drift_interf[mask] = normalized_vals[mask]

        long_term_coh[:, iteration, ...] = coherence

    data = {}
    data["long_term_coh"] = (LONG_DIMS, long_term_coh)
    ds = xr.Dataset(data)
    # 3. Write to Disk
    write(
        ds,
        output_path,
        mode="r+",
        skip_read=True,
        consolidated=False,
    )

    # 3. Normalize to Unit Phasor
    drift_phasor = drift_interf / np.abs(drift_interf)

    mask = np.isnan(drift_phasor) + (drift_phasor == 0)
    drift_phasor[mask] = 1.0

    n_az, n_rg = drift_phasor.shape[1:]
    shape = current_reference.shape
    shape = (shape[0], n_az, shape[1] // n_az, n_rg, shape[2] // n_rg)
    shape_ph = (shape[0], n_az, 1, n_rg, 1)

    # 5. Apply Correction
    # Rotate the current reference by the conjugate of the drift
    corrected_reference = (current_reference.data.reshape(shape) * drift_phasor.reshape(shape_ph).conj()).reshape(
        current_reference.shape
    )
    corrected_reference = (
        xr.DataArray(
            corrected_reference,
            dims=current_reference.dims,
            coords=current_reference.coords,
        )
        .compute()
        .chunk(chunks)
    )
    return corrected_reference
