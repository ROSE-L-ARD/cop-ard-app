from typing import cast

import numpy as np
import xarray as xr
from cop_ard_core.cmlp import calculate_stable_dates, get_current_reference, update_drift, update_recursive_reference
from cop_ard_core.logger import timeit
from cop_ard_core.utils import get_normalized_interf

from cop_ard_app.computation import map
from cop_ard_app.io import Path, get_storage_options, normpath, read, save_empty_dataset, write
from cop_ard_app.utils import slice_and_drop_coordinates

SHORT_DIMS = ["polarization", "azimuth_time", "slant_range_time"]
DIMS = ["polarization", "acquisition_date", "azimuth_time", "slant_range_time"]
LONG_DIMS = ["polarization", "past_reference_date", "azimuth_time", "slant_range_time"]


@timeit
def stack_cmlp(
    slcs: xr.DataTree,
    multilook: dict,
    working_directory: str,
    revisit_cycle: np.timedelta64,
    memory_param: float = 0.95,
    coherence_threshold: float = 0.1,
    stable_reference_step: int = 10,
    max_past_stable_history: int = 100,
    reference_database_path: Path | None = None,
    enable_dask: bool = True,
    **kwargs,
):
    multiburst = "bursts" in slcs
    args = []
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        slc = v.coreg.measurements.cstack
        a = (k, slc)
        args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    results = cast(
        list,
        map(
            single_cmlp,
            args,
            kwargs=dict(
                reference_database_path=reference_database_path,
                multilook=multilook,
                multiburst=multiburst,
                working_directory=working_directory,
                revisit_cycle=revisit_cycle,
                memory_param=memory_param,
                coherence_threshold=coherence_threshold,
                stable_reference_step=stable_reference_step,
                max_past_stable_history=max_past_stable_history,
                **kwargs,
            ),
            enable_dask=enable_dask,
        ),
    )

    # --- Assemble Results ---
    cmlp_tree = xr.DataTree()
    current_reference_tree = xr.DataTree()

    stable_references_tree = xr.DataTree()

    # Reconstruct the DataTree from the list of results returned by the workers
    for burst_id, cmlp, c_ref, s_ref in results:
        multiburst_layer = f"bursts/{burst_id}/" if multiburst else ""
        # Unpack dates and datasets
        for date, ds in cmlp:
            cmlp_tree[f"{multiburst_layer}cmlp/{date}"] = ds

        current_reference_tree[f"{multiburst_layer}measurements/cstack"] = c_ref

        if s_ref is not None:
            stable_references_tree[f"{multiburst_layer}measurements/cstack"] = s_ref

    stable_references_tree = stable_references_tree if stable_references_tree.children else None
    return (cmlp_tree, current_reference_tree, stable_references_tree)


def single_cmlp(
    burst_id: str,
    cstack: xr.DataArray,
    reference_database_path: Path,
    multilook: dict,
    multiburst: bool,
    working_directory: str,
    revisit_cycle: np.timedelta64,
    memory_param: float = 0.95,
    coherence_threshold: float = 0.1,
    stable_reference_step: int = 10,
    max_past_stable_history: int = 100,
    **kwargs,
):
    # 1. Setup paths and validation
    expected = set(DIMS[2:])
    if set(multilook) != expected:
        raise ValueError(f"Wrong keys: {set(multilook)}. Expected {expected}")

    path = f"bursts/{burst_id}" if multiburst else burst_id
    last_reference_path = reference_database_path.append(f"{path}/current_reference")
    stable_reference_path = reference_database_path.append(f"{path}/stable_reference")
    data_path = normpath(f"{working_directory}/{path}")

    # 2. Prepare Data (Slicing, Padding, Chunking)
    cstack_sel, slices, pad, chunks = _prepare_cstack(cstack, multilook)
    acquisition_dates = cstack_sel.acquisition_date.values
    # Handle inputs for resumption (Apply same slicing to inputs)
    last_reference = None
    starting_date = None
    if last_reference_path.exists():
        try:
            ds = read(
                last_reference_path.path, as_type="xr.Dataset", storage_options=last_reference_path.storage_options
            )["cstack"]
            last_reference = slice_and_drop_coordinates(ds, slices, dimension_coordinates_to_keep=DIMS)
            starting_date = last_reference.attrs["starting_date"]
        except Exception as e:
            raise ValueError(f"Failed to load current reference at {last_reference_path.path}: {e}")
    stable_reference = None
    past_reference_dates = None
    if stable_reference_path.exists():
        try:
            # Use a list comprehension for cleaner data loading
            datasets = [
                read(f.path, as_type="xr.Dataset", storage_options=f.storage_options)["cstack"]
                for f in stable_reference_path.glob("*")
            ]

            if datasets:
                concat_ds = xr.concat(datasets, dim="acquisition_date")
                stable_reference = slice_and_drop_coordinates(concat_ds, slices, dimension_coordinates_to_keep=DIMS)
                past_reference_dates = stable_reference["acquisition_date"].values
            else:
                raise ValueError(f"No stable reference files found in {stable_reference_path.path}")
        except Exception as e:
            raise ValueError(f"Failed to load stable references from {stable_reference_path.path}: {e}")
    # 3. Calculate Date Milestones
    stable_reference_dates = calculate_stable_dates(
        acquisition_dates, revisit_cycle, stable_reference_step, stable_reference, starting_date
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
    current_reference = current_reference.drop_vars({k for k in current_reference.coords.keys() if k not in SHORT_DIMS})
    for date_idx, date in enumerate(cstack_sel.acquisition_date.values):
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
        current_reference = (
            update_recursive_reference(
                slc=slc,
                current_reference=current_reference,
                normalized_interf=normalized_interf,
                memory_param=memory_param,
            )
            .compute()
            .chunk(chunks)
        )

        if date in stable_reference_dates:
            stable_reference = _update_stable_reference(
                current_reference=current_reference,
                date=date,
                data_path=data_path,
                pad=pad,
                chunks=chunks,
                slices=slices,
                initial_stable_reference=stable_reference,
            )

        if stable_reference is not None:
            stable_reference = stable_reference.sortby("acquisition_date").isel(
                acquisition_date=slice(-max_past_stable_history, None)
            )

            current_reference = (
                _correct_drift(
                    stable_reference=stable_reference,
                    current_reference=current_reference,
                    multilook=multilook,
                    coherence_threshold=coherence_threshold,
                    shape=normalized_interf.shape,
                    output_path=f"{data_path}/cmlp/{date}",
                )
                .compute()
                .chunk(chunks)
            )
    current_reference = current_reference.assign_coords({"acquisition_date": date})
    current_reference = write(
        current_reference, f"{data_path}/current_reference", storage_options=get_storage_options(kwargs)
    )
    cmlp = [
        (date, read(f"{data_path}/cmlp/{date}", as_type="xr.Dataset", storage_options=get_storage_options(kwargs)))
        for date in cstack_sel.acquisition_date.values
    ]

    if (stable_reference is not None) and (past_reference_dates is not None):
        # 1. Filter out only the dates that actually exist in the dataset
        valid_dates = [d for d in past_reference_dates if d in stable_reference.acquisition_date.values]

        # 2. Check if we are dropping EVERYTHING
        if len(valid_dates) == len(stable_reference.acquisition_date):
            stable_reference = None

        # 3. Otherwise, check if we have a partial list of valid dates to drop
        elif valid_dates:
            stable_reference = stable_reference.drop_sel(acquisition_date=valid_dates)

    return (
        burst_id,
        cmlp,
        current_reference,
        stable_reference,
    )


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
        d = np.datetime64(date, "ns")
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


def _compute_and_save_interferogram(
    target_slc: xr.DataArray,
    reference_slc: xr.DataArray,
    multilook: dict,
    output_path: str,
    compute_phase: bool,
    name: str,
) -> xr.DataArray:
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
    normalized_interf = get_normalized_interf(target_slc, reference_slc, multilook)

    # 2. Extract Statistics
    data = {}
    data[name] = (SHORT_DIMS, abs(normalized_interf))
    if compute_phase:
        data["phase"] = (SHORT_DIMS, xr.ufuncs.angle(normalized_interf))
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


def _update_stable_reference(
    current_reference: xr.DataArray,
    date: np.datetime64,
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
    # 2. Prepare the dataset (Pad -> Chunk -> Expand -> Transpose -> Dataset)
    ds = current_reference.pad(pad, constant_values=np.nan).chunk(chunks).to_dataset(name="cstack")
    # 3. Write to the specific slot
    new_stable_reference = (
        cast(
            xr.Dataset,
            write(
                ds,
                f"{data_path}/stable_reference/{date}",
                mode="r+",
                skip_read=False,
                consolidated=False,
                region={k: slice(None) for k in SHORT_DIMS},
            ),
        )
        .cstack.isel({**slices})
        .expand_dims("acquisition_date")
    )
    # 4. Reload the updated history from disk
    # We read up to the current index (idx + 1) to get the 'new' history
    # We apply the 'slices' to remove the padding we just added (if any)
    # 5. Concatenate with historical reference (if it exists)
    if initial_stable_reference is not None:
        new_stable_reference = xr.concat([initial_stable_reference, new_stable_reference], dim="acquisition_date")

    return new_stable_reference


def _correct_drift(
    stable_reference: xr.DataArray,
    current_reference: xr.DataArray,
    multilook: dict,
    coherence_threshold: float,
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
    shape_coh = (shape[0], stable_reference.acquisition_date.size, shape[1], shape[2])
    drift_interf = np.full(shape, np.nan, dtype="complex64")
    long_term_coh = np.empty(shape_coh, dtype="float32")
    # 2. Iteratively fill drift_interf with coherent pixels from history
    for iteration, stable_date in enumerate(stable_reference.acquisition_date.values):
        stable_ref_slice = stable_reference.sel(acquisition_date=stable_date)
        coherence = update_drift(
            drift_interf=drift_interf,
            reference=current_reference,
            stable_reference=stable_ref_slice,
            multilook=multilook,
            coherence_threshold=coherence_threshold,
        )
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
    corrected_reference = xr.DataArray(
        corrected_reference,
        dims=current_reference.dims,
        coords=current_reference.coords,
    )
    corrected_reference.attrs = current_reference.attrs
    return corrected_reference
