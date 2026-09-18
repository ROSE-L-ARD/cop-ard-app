import inspect
import os
import pathlib
import posixpath
import secrets
import shutil
import warnings
from collections.abc import Mapping
from datetime import datetime
from hashlib import md5
from json import dumps
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast, overload
from unittest import mock

import dask.array as da
import fsspec
import numpy as np
import rasterio
import s3fs
import sarpy
import sarpy.io.general.base
import xarray as xr
import zarr
import zarr.storage
from cop_ard_core.logger import logger


class SentinelSarpyObj:
    """
    A wrapper class to handle Sentinel-1 metadata extraction using Sarpy,
    especially for data stored on S3 where partial file reads are needed.
    """

    def __init__(self, in_path, storage_options=None):
        storage_options = storage_options or {}
        # Check if the input is already a DataTree.
        self._is_dt = isinstance(in_path, xr.DataTree)
        if not self._is_dt:
            # For file paths, set up original path and a temporary working directory.
            self._input = Path(in_path, storage_options=storage_options)
        else:
            # For DataTree input, use it directly.
            self._input = in_path
        # Import the metadata upon initialization.
        self._import_metadata()

    def _import_metadata(self):
        """
        Import metadata from the Sentinel-1 product.
        """
        import sarpy.io.complex.sentinel

        if self._is_dt:
            # If input is a DataTree, Sarpy can read its metadata directly.
            self._sentinel_details = sarpy.io.complex.sentinel.SentinelDetails(self._input)
        else:
            # For file paths (especially S3), copy essential metadata files locally.
            for j in ["manifest.safe", "annotation"]:
                self._input.append(j).ensure_local()
            # Open the local manifest file with Sarpy.
            manifest_path = self._input.append("manifest.safe").replace_dir(self._input._temp_dir.name)._path  # ty:ignore[unresolved-attribute]
            self._sentinel_details = sarpy.io.complex.sentinel.SentinelDetails(manifest_path)

        # Extract the collection of SICD (Sensor Independent Complex Data) metadata objects.
        try:
            self.sicd_meta = [j for k, v in self._sentinel_details.get_sicd_collection() for j in v]
        except Exception as e:  # noqa: BLE001
            logger.info(f"{e.__class__.__name__}: No bursts have been found.")
            self.sicd_meta = [v for k, v in self._sentinel_details.get_sicd_collection()]


def sarpy_open(in_path: str | xr.DataTree, allowed_partial_open: bool = False, storage_options: dict | None = None):
    """
    A robust wrapper to open various SAR data formats using the Sarpy library.

    This function attempts to open SAR data, handling different input types like
    local file paths, S3 paths, and in-memory xarray.DataTree objects. It includes
    fallbacks for S3 access and a special handler for partially reading Sentinel-1
    metadata without downloading the full dataset.

    Args:
        in_path: The path or object to open.
        allowed_partial_open: If True, allows a special handling mode for
        Sentinel-1 on S3 to read only metadata.

    Returns:
        A Sarpy reader object.
    """
    storage_options = storage_options or {}
    try:
        # Handle in-memory DataTree by computing it first.
        if isinstance(in_path, xr.DataTree):
            with mock.patch("sarpy.io.complex.converter.is_file_like", lambda _: True):
                ds = sarpy.io.open(in_path)  # type: ignore
        # Handle standard file paths.
        else:
            ds = sarpy.io.open(in_path)
    except sarpy.io.general.base.SarpyIOError:
        # If standard open fails, it might be an S3 path that Sarpy can't handle directly.
        if not isinstance(in_path, str):
            raise TypeError(f"Expected str, got {type(in_path)}")
        try:
            # Try opening the path with fsspec first.
            ds = sarpy.io.open(open(in_path))
        except Exception:
            # As a last resort, use a custom handler for partial reads of Sentinel-1 on S3.
            if allowed_partial_open:
                if is_s3_path(in_path) and (in_path.endswith(".SAFE")):
                    # This custom object downloads just the metadata files.
                    ds = SentinelSarpyObj(in_path, storage_options)
                else:
                    raise
            else:
                raise
    return ds


def read_slc_static(
    burst_id: str,
    acquisition_ids: list[str],
    working_directory: str,
    reference_id: str | None = None,
    **kwargs,
):
    """
    Check the static directory for a reference SLC.

    This function attempts to read a pre-processed reference SLC for a specific
    burst from the static working directory and validates its presence in the
    current DataTree.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst.
    slc : xr.DataTree
        The DataTree for the specific burst which contains the reference
        SLC in its `orig` group.
    working_directory : str
        The root path of the static working directory.
    **kwargs : dict
        Additional keyword arguments for storage options.

    Returns
    -------
    burst_id : str
        The original identifier of the burst.
    reference_date : str or None
        The extracted reference date as a string, or None if the reference
        data is missing.
    ds : int or xr.Dataset
        Returns `0` if the reference SLC is missing. Returns `1` if it is present
        and already loaded in `slc.orig`. Otherwise, returns the loaded
        `xr.Datatree` itself.
    """
    if (burst_id == "data") and (reference_id is None):
        raise ValueError("In single burst case, the reference_id is mandatory")
    path = f"bursts/{burst_id}" if burst_id != "data" else f"data/{reference_id}"
    path_ok = burst_id if burst_id != "data" else f"data/{reference_id}"
    storage_opts = get_storage_options(kwargs)
    slc_path = f"{working_directory}/processed_data/{path_ok}/slc.ok"
    acquisition_id = None

    if not exists(slc_path, storage_options=storage_opts):
        logger.info(f"Missing reference SLC marker for burst {burst_id}.")
        ds: int | xr.DataTree = 0
    else:
        ds = cast(
            xr.DataTree,
            read(
                normpath(f"{working_directory}/{path}/reference"),
                storage_options=storage_opts,
            ),
        )
        annot = ds.measurements.slc.attrs
        annot = annot[next(iter(annot))]
        collector_name = str(annot["CollectionInfo"]["CollectorName"])
        if burst_id != "data":
            absolute_number = str(annot["CollectionInfo"]["Parameters"]["ABSOLUTE_ORBIT"])
        else:
            absolute_number = str(annot["Timeline"]["CollectStart"]).replace("Z", "")

        acquisition_id = f"{collector_name}_{absolute_number}"
        # Extract date (casting to string to ensure matching with slc.orig keys)
        logger.info(f"Extracted acquisition_id: {acquisition_id} for burst {burst_id}.")

        if acquisition_id in acquisition_ids:
            logger.info(f"Reference date {acquisition_id} already in slc.orig.")
            ds = 1

    return (burst_id, acquisition_id, ds)


def save_slc_static(burst_id: str, slc: xr.DataTree, working_directory: str, reference_id: str | None = None, **kwargs):
    """
    Save a single reference SLC DataTree to the static working directory.

    This function is intended to be used with a map operation (like the one
    in `finalize_import`) to save the reference SLC for a specific burst. It
    also writes an empty `slc.ok` marker file to indicate success.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst.
    slc : xr.DataTree
        The DataTree for the specific burst which contains the
        reference SLC in its `orig` group.
    working_directory : str
        The root path of the static working directory. The function will
        construct the full output path from this.
    **kwargs : dict
        Additional keyword arguments, primarily for passing
        `storage_options` to the `write` and `io` functions.

    Returns
    -------
    None
    """
    path = f"bursts/{burst_id}" if burst_id != "data" else f"{burst_id}/{reference_id}"
    path_ok = burst_id if burst_id != "data" else f"{burst_id}/{reference_id}"
    # Select the specific reference date DataArray from the burst's 'orig' group
    dirname = normpath(f"{working_directory}/processed_data/{path_ok}")
    slc_path = f"{dirname}/slc.ok"
    # Write the DataArray to the static working directory in overwrite mode ('w')
    write(
        slc,
        f"{working_directory}/{path}/reference",
        mode="w",
        skip_read=True,  # Optimization: don't read data back after writing
        storage_options=get_storage_options(kwargs),
    )
    # No explicit return value is needed
    makedirs(
        dirname,
        exist_ok=True,
        **kwargs,
    )
    touch(slc_path, **kwargs)


def open_copdem30m(minx: float, miny: float, maxx: float, maxy: float) -> xr.Dataset:
    """
    Open a subset of the Copernicus GLO-30 DEM from a cloud-based Zarr store.

    Parameters
    ----------
    minx
        The minimum longitude of the bounding box.
    miny
        The minimum latitude of the bounding box.
    maxx
        The maximum longitude of the bounding box.
    maxy
        The maximum latitude of the bounding box.

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset containing the clipped DEM data.
    """
    # Open the remote Copernicus DEM Zarr store.
    dem = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
        decode_coords="all",
        mask_and_scale=False,
    )
    if not isinstance(dem.rio.crs, rasterio.crs.CRS):
        raise TypeError(f"Expected rasterio.crs.CRS, got {type(dem.rio.crs)}")
    # Clip the global DEM to the specified bounding box.
    dem = dem.rio.clip_box(minx, miny, maxx, maxy, crs="EPSG:4326")
    # Rename coordinates to the standard 'y' and 'x'.
    dem = dem.rename({"lat": "y", "lon": "x"})
    return dem


# Standard encoding keys recognized by xarray/Zarr
VALID_ZARR_ENCODING_KEYS = {
    "chunks",
    "compressor",
    "filters",
    "_FillValue",
    "dtype",
    "scale_factor",
    "add_offset",
    "dimension_separator",
}


def generate_uuid4():
    """
    Generate a random UUID (version 4) without using the uuid library.

    A UUID4 is a universally unique identifier generated using random numbers.
    This function creates a UUID4 in the standard 8-4-4-4-12 hexadecimal
    format, ensuring compliance with RFC 4122 specifications.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A randomly generated UUID4 string (e.g., "123e4567-e89b-42d3-a456-556642440000").
    """

    # Helper function to generate a random hexadecimal string of a given length.
    def random_hex(length):
        # `secrets.token_hex` generates a string twice the length of the byte input.
        return "".join(secrets.token_hex(length // 2))

    # --- Generate each segment of the UUID4 ---
    # Part 1: 8 hex digits.
    part1 = random_hex(8)
    # Part 2: 4 hex digits.
    part2 = random_hex(4)
    # Part 3: Starts with '4' to indicate version 4, followed by 3 random hex digits.
    part3 = "4" + random_hex(3)
    # Part 4: Starts with a variant bit ('8', '9', 'a', or 'b'), followed by 3 random hex digits.
    part4 = secrets.choice("89ab") + random_hex(3)
    # Part 5: 12 random hex digits.
    part5 = random_hex(12)

    # Combine all parts into the standard UUID format with hyphens.
    return f"{part1}-{part2}-{part3}-{part4}-{part5}"


def assign_attrs(_data: xr.Dataset | xr.DataTree, _data_attrs: dict) -> xr.Dataset | xr.DataTree:
    """
    Assign a dictionary of attributes to an xarray Dataset or DataTree object.

    Parameters
    ----------
    _data : xr.Dataset | xr.DataTree
        The Dataset or DataTree to which
        attributes will be assigned.
    _data_attrs : dict
        A dictionary containing the attributes to assign.

    Returns
    -------
    xr.Dataset | xr.DataTree
        The original object with the new attributes added.
    """
    # Iterate through the dictionary and assign each key-value pair as an attribute.
    for k, v in _data_attrs.items():
        _data.attrs[k] = v
    return _data


def check_write_arguments(
    data: xr.Dataset | xr.DataArray | xr.DataTree,
    data_path: str,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"] = "w",
    region: Mapping[str, slice] | None = None,
) -> tuple[Literal["w", "w-", "a", "a-", "r+", "r"], str, xr.Dataset | xr.DataTree, str]:
    """
    Validate and prepare arguments for the main `write` function.
    """
    _data_type = f"xr.{type(data).__name__}"
    if not isinstance(data, (xr.DataArray, xr.Dataset, xr.DataTree)):
        raise TypeError(f"Data type must be xr.Dataset, xr.DataArray, or xr.DataTree. Got {_data_type}")

    # Region writes strictly require "r+" mode
    if region is not None:
        mode = "r+"

    _file_path = normpath(data_path)

    # Standardize DataArrays to Datasets
    _data = data
    if isinstance(data, xr.DataArray):
        # Use existing name if present, otherwise default to "v"
        _data = data.to_dataset(name=data.name or "v")

    if not isinstance(_data, (xr.Dataset, xr.DataTree)):
        raise TypeError(f"Expected xr.Dataset or xr.DataTree, got {type(_data)}")
    return mode, _file_path, _data, _data_type


def validate_encoding(data: xr.Dataset | xr.DataTree, encoding: dict | None) -> dict | None:
    """
    Validates a user-provided encoding dictionary against the data object.
    Strips out invalid variables and warns on unknown encoding parameters.
    """
    if encoding is None:
        return None

    # Get a flat list of all variables in the Dataset or DataTree
    if isinstance(data, xr.DataTree):
        valid_vars: set[str] = set()
        for node in data.descendants:
            valid_vars.update([str(k) for k in node.ds.variables])
    else:
        valid_vars = {str(k) for k in data.variables}

    validated_encoding = {}

    for var_name, rules in encoding.items():
        if var_name not in valid_vars:
            logger.warning(f"Encoding provided for '{var_name}', but it doesn't exist in the data. Ignoring.")
            continue

        # Check if the rules are valid Zarr arguments
        clean_rules = {}
        for rule_key, rule_val in rules.items():
            if rule_key not in VALID_ZARR_ENCODING_KEYS:
                logger.warning(f"Unknown encoding parameter '{rule_key}' for variable '{var_name}'. Ignoring.")
            else:
                clean_rules[rule_key] = rule_val

        if clean_rules:
            validated_encoding[var_name] = clean_rules

    return validated_encoding


def is_local(path, storage_options):  # TODO rename
    value = "://" not in path or path.startswith("file://")
    so = storage_options.copy() if isinstance(storage_options, dict) else None
    if value:
        so = None
    else:
        if so is not None:
            so["use_listings_cache"] = False
    return so


def save_sparse_array(
    data: np.ndarray,
    col: np.ndarray,
    row: np.ndarray,
    shape: np.ndarray,
    format: str,
    chunks: int,
    data_path: str,
    skip_read: bool = True,
    attrs: dict | None = None,
    **kwargs,
):
    """
    Save a sparse matrix in COO or CSR format to a Zarr store.

    This function takes the components of a sparse matrix (data, indices, etc.),
    packages them into an xarray.Dataset, and writes them to a Zarr store.
    This allows for efficient storage of sparse data.

    Parameters
    ----------
    data : np.ndarray
        The non-zero values of the sparse matrix.
    col : np.ndarray
        For "COO", the column indices for each value. For "CSR",
        this corresponds to the `indices` array.
    row : np.ndarray
        For "COO", the row indices for each value. For "CSR",
        this corresponds to the `indptr` array.
    shape : np.ndarray
        The shape of the equivalent dense matrix.
    format : str
        The sparse format, which must be either "COO" or "CSR".
    chunks : int
        The chunk size to be used for the Dask arrays in the dataset.
    data_path : str
        The output path for the Zarr store.
    skip_read : bool, optional
        If False, the function will read the data back
        from disk and return it. Defaults to True.
    attrs : dict, optional
        Additional attributes to save with the dataset.

    Returns
    -------
    xr.Dataset or None
        The saved dataset if `skip_read` is False, otherwise None.

    Raises
    ------
    Exception
        If the specified format is not "COO" or "CSR".
    """
    attrs = attrs or {}
    # Validate that the specified sparse format is supported.
    if format not in ["COO", "CSR"]:
        raise ValueError(f"Expected COO or CSR format, got {format}")
    # Initialize a dictionary to hold the data variables for the xarray Dataset.
    data_vars = {
        "data": ("point", data),
        "nnz": data.shape[0],
        "shape": shape,
    }
    # Add format-specific variables.
    if format == "COO":
        # For Coordinate format, store column and row index arrays.
        data_vars["col"] = ("point", col)
        data_vars["row"] = ("point", row)
    elif format == "CSR":
        # For Compressed Sparse Row format, store indices and index pointer arrays.
        data_vars["indices"] = ("point", col)
        data_vars["indptr"] = ("indptr", row)

    # Create the xarray Dataset from the prepared data variables and attributes.
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={"format": format, **attrs},
    ).chunk(chunks)

    # Write the dataset to a Zarr store using the main `write` function.
    write(
        ds,
        data_path=normpath(data_path),
        skip_read=True,
        mode="w",
        storage_options=get_storage_options(kwargs),
    )
    # Optionally, read the data back from disk and return it.
    return None if skip_read else ds


def generate_empty_dataset(
    data_vars: list[Any],
    dims: dict[Any, list[str]],
    shape: dict[Any, list[int]],
    chunks: dict[Any, int],
    dtype: dict[Any, str],
    crs: str | Any | None = None,
    coords: dict[Any, Any] | None = None,
    attrs: dict[Any, Any] | None = None,
    attrs_data_vars: dict[Any, Any] | None = None,
) -> xr.Dataset:
    """
    Create an empty in-memory xarray Dataset with a predefined structure backed by Dask.
    """
    # --- Create the Empty Dataset Structure ---
    empty_dataset = xr.Dataset(
        data_vars={
            k: (
                dims[k],
                da.empty(
                    shape=shape[k],
                    dtype=dtype[k],
                    chunks=[chunks[kk] for kk in dims[k]],
                ),
            )
            for k in data_vars
        },
        coords=coords,
        attrs=attrs,
    ).chunk(chunks)

    # --- Assign Metadata ---
    if attrs_data_vars is not None:
        for k, v in attrs_data_vars.items():
            empty_dataset[k].attrs = v

    if crs is not None:
        empty_dataset.rio.write_crs(crs, inplace=True)

    return empty_dataset


def save_empty_dataset(
    data_vars: list[Any],
    dims: dict[Any, list[str]],
    shape: dict[Any, list[int]],
    chunks: dict[Any, int],
    dtype: dict[Any, str],
    data_path: str,
    crs: str | Any | None = None,
    coords: dict[Any, Any] | None = None,
    attrs: dict[Any, Any] | None = None,
    attrs_data_vars: dict[Any, Any] | None = None,
    skip_read: bool = False,
    compute: bool = False,
    **kwargs,
):
    """
    Create and save an empty xarray Dataset with a predefined structure to a Zarr store.

    This utility initializes the full structure of an output Zarr store on disk
    with empty Dask arrays, allowing parallel workers to write safely.
    """

    # --- Generate the Empty Dataset ---
    empty_dataset = generate_empty_dataset(
        data_vars=data_vars,
        dims=dims,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        crs=crs,
        coords=coords,
        attrs=attrs,
        attrs_data_vars=attrs_data_vars,
    )

    # --- Write to Disk ---
    result = write(
        empty_dataset,
        data_path=normpath(data_path),
        compute=compute,
        skip_read=skip_read,
        storage_options=get_storage_options(kwargs),
    )
    return result


def update_attrs(
    attrs: Mapping[str, Any],
    data_path: str,
    group: str | None = None,
    storage_options: dict | None = None,
) -> None:
    """
    Update the attributes of a Zarr V3 store or a group within it.
    Modifies the metadata in place on disk without touching the data chunks.
    """
    so = is_local(data_path, storage_options)

    # Open the root Zarr group in read/write mode
    root_group = zarr.open_group(data_path, mode="r+", storage_options=so)

    # Navigate to the specific group if requested
    target = root_group[group] if group is not None else root_group

    # Update the attributes (this automatically overwrites existing keys)
    target.attrs.update(attrs)
    store = zarr.storage.FsspecStore.from_url(url=data_path, storage_options=so)
    zarr.consolidate_metadata(store)


class Path:
    """
    A custom string subclass for enhanced file path manipulation.

    This class extends the built-in `str` to provide convenient methods and
    properties for handling file paths, including those on S3. It automatically
    parses path components and provides wrappers for common file system operations.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        temp_dir: TemporaryDirectory | str | None = None,
        storage_options: dict | None = None,
    ):
        """
        Initialize the Path object.

        This constructor normalizes the input path, determines if it is an S3 path,
        and parses its various components like name, directory, bucket, and key.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            The input file path as a string or
            `pathlib.Path` object.
        temp_dir : tempfile.TemporaryDirectory | None, optional
            An optional
            temporary directory associated with this path. Defaults to None.

        Returns
        -------
        None
        """
        # If the input is a pathlib.Path, convert it to a string.
        if isinstance(path, pathlib.Path):
            path = path.as_posix()
        # Initialize attributes.
        self._key = None
        self._bucket = None
        self._is_s3_path = False
        self._storage_options = None
        self._temp_dir = temp_dir if temp_dir is not None else TemporaryDirectory()

        # Normalize the path to a consistent format (e.g., handling s3://).
        self._path = normpath(path)
        # Parse common path components.
        self._name = os.path.basename(self._path)
        self._stem = pathlib.Path(self._path).stem
        self._directory = os.path.dirname(self._path)

        # If the path is an S3 path, parse the bucket and key.
        if is_s3_path(path):
            self._is_s3_path = True
            self._bucket = path.split("/")[2]
            self._key = "/".join(path.split("/")[3:])
            self._storage_options = storage_options

    def __str__(self) -> str:
        """Return the string representation of the path."""
        return self._path

    def __repr__(self) -> str:
        """Return the official string representation of the path."""
        return self._path.__repr__()

    @classmethod
    def search(cls, pattern: str, sort: bool = True, storage_options: dict | None = None) -> list["Path"]:
        """Class method: Search from scratch using an absolute pattern."""
        raw_paths = glob(pattern, sort=sort, storage_options=storage_options)
        return [cls(p, storage_options=storage_options) for p in raw_paths]

    def glob(self, pattern: str, sort: bool = True) -> list["Path"]:
        """Instance method: Search relative to this specific path."""
        base = self._path.rstrip("/")
        pat = pattern.lstrip("/")
        full_pattern = f"{base}/{pat}"

        raw_paths = glob(full_pattern, sort=sort, storage_options=self._storage_options)
        return [self.__class__(p, temp_dir=self._temp_dir, storage_options=self._storage_options) for p in raw_paths]

    @property
    def path(self):
        """Retrieve the raw path string."""
        return self._path

    @property
    def name(self):
        """The final component of the path (file or directory name)."""
        return self._name

    @property
    def key(self):
        """The S3 object key, or None if not an S3 path."""
        return self._key

    @property
    def storage_options(self):
        """The dictionary of storage options, or None."""
        return self._storage_options

    @property
    def bucket(self):
        """The S3 bucket name, or None if not an S3 path."""
        return self._bucket

    @property
    def stem(self):
        """The final path component, without its suffix."""
        return self._stem

    @property
    def directory(self):
        """The directory part of the path."""
        return self._directory

    def append(self, suffix) -> "Path":
        """
        Append a suffix to the path.

        Parameters
        ----------
        suffix : str
            The string to append to the path.

        Returns
        -------
        Path
            A new `Path` object with the suffix joined.
        """
        return Path(os.path.join(self._path, suffix), temp_dir=self._temp_dir, storage_options=self._storage_options)

    def as_gdal(self) -> str:
        """
        Return the path formatted for use with GDAL's virtual file systems.

        For S3 paths, this prepends '/vsis3/'. For other paths, it returns
        the path unchanged.

        Parameters
        ----------
        None.

        Returns
        -------
        str
            The GDAL-compatible path string.
        """
        # Check if it's an S3 path and format accordingly.
        return f"/vsis3/{self._path[5:]}" if is_s3_path(self._path) else self._path

    def replace_dir(self, dir) -> "Path":
        """
        Create a new path with the same filename but in a new directory.

        This method does not move or copy any files; it only manipulates the
        path string.

        Parameters
        ----------
        dir : str
            The new directory path.

        Returns
        -------
        Path
            A new `Path` object with the updated directory.
        """
        return Path(os.path.join(dir, self._name))

    def ensure_local(self, dest_dir: str | None = None) -> "Path":
        """
        Ensure the file is available on the local file system.

        If the path is on S3, it downloads the file to `dest_dir` (or a
        temporary directory if none is provided) and returns a new local Path.
        If the path is already local, it simply returns itself without taking action.

        Parameters
        ----------
        dest_dir : str | None, optional
            The local directory to download the file to. If None, uses the
            instance's temporary directory.

        Returns
        -------
        Path
            A Path object pointing to the local file.
        """

        if not self._is_s3_path:
            return self

        logger.info(f"Creating local copy of {self._name}...")
        # Determine the local destination directory
        if dest_dir is None:
            if self._temp_dir is None:
                self._temp_dir = TemporaryDirectory()

            if isinstance(self._temp_dir, TemporaryDirectory):
                dest_dir = str(self._temp_dir.name)
            elif isinstance(self._temp_dir, str):
                dest_dir = self._temp_dir
            else:
                raise ValueError(f"Expected str or Temporary Directory, got {type(self._temp_dir)}")
        # Construct the full local destination path
        local_dst = os.path.join(dest_dir, self._name)

        # Use your existing copy method to handle the S3 -> Local transfer
        copy_path = self.copy(local_dst, recursive=True)

        logger.info(f"Completed local copy at {copy_path._path}")
        return copy_path

    def copy(self, dst, recursive=False, only_if_different_fs=False) -> "Path":
        """
        Copy the file or directory to a destination.

        This is a wrapper around a file system copy utility. It intelligently
        handles cases where the destination is a directory.

        Parameters
        ----------
        dst : str
            The destination path or directory.
        recursive : bool, optional
            If True, perform a recursive copy. Defaults to False.
        only_if_different_fs : bool, optional
            If True, only copy if the source
            and destination are on different file systems. Defaults to False.

        Returns
        -------
        Path
            A new `Path` object representing the final destination path.
        """
        # If the destination is a directory, construct the full destination path.
        _dst = self.replace_dir(dst)._path if isdir(dst) or dst.endswith("/") else dst
        # Call the underlying copy function and return the result as a new Path object.
        return Path(
            copy(
                self._path,
                _dst,
                recursive=recursive,
                only_if_different_fs=only_if_different_fs,
                storage_options=self._storage_options,
            ),
            temp_dir=self._temp_dir,
        )

    def move(self, dst: str, recursive=False, only_if_different_fs=False) -> "Path":
        """
        Move the file or directory to a destination.

        This is a wrapper around a file system move utility.

        Parameters
        ----------
        dst : str
            The destination path or directory.
        recursive : bool, optional
            If True, perform a recursive move. Defaults to False.
        only_if_different_fs : bool, optional
            If True, only move if the source
            and destination are on different file systems. Defaults to False.

        Returns
        -------
        Path
            A new `Path` object representing the final destination path.
        """
        # If the destination is a directory, construct the full destination path.
        _dst = self.replace_dir(dst)._path if isdir(dst) or dst.endswith("/") else dst
        # Call the underlying move function and return the result as a new Path object.
        return Path(
            move(
                self._path,
                _dst,
                recursive=recursive,
                only_if_different_fs=only_if_different_fs,
            )
        )

    def exists(self):
        """Check if the path exists."""
        return exists(self._path)

    def isdir(self):
        """Check if the path is a directory."""
        return isdir(self._path)

    def isfile(self):
        """Check if the path is a file."""
        return isfile(self._path)


def decorator(func):
    """
    A decorator to wrap filesystem functions with `fsspec`.

    This decorator intercepts calls to standard filesystem operations (like `exists`,
    `delete`, `glob`, etc.) and redirects them to the appropriate `fsspec`
    filesystem implementation based on the path provided. It handles S3 cache
    invalidation and provides specific pre- and post-processing for certain
    functions to ensure consistent behavior across filesystems.

    Parameters
    ----------
    func : Callable
        The function to be decorated. Its name must match a
        method on an `fsspec` filesystem object (e.g., 'exists', 'glob').

    Returns
    -------
    Callable
        The wrapped function that executes the corresponding `fsspec`
        method.
    """

    def wrapper(path: list | tuple | str, **kwargs):
        # 1. TRICK: Call the original function just to trigger coverage.
        # Since it is a 'pass' function, this does nothing but run the line.
        func(path, **kwargs)

        storage_options = get_storage_options(kwargs)
        so = is_local(path, storage_options)
        if not so:
            so = {}
        # --- Filesystem and Cache Handling ---
        # Determine the correct fsspec filesystem (e.g., 'file', 's3') from the path.
        _path = path[0] if isinstance(path, (list, tuple)) else path
        fs, _ = fsspec.core.url_to_fs(_path, **so)

        # For S3, invalidate the cache to ensure the latest directory listings are fetched.
        if hasattr(fs, "invalidate_cache"):
            fs.invalidate_cache()

        # --- Function-Specific Pre-processing ---
        # Special handling for `delete` to avoid errors on non-existent files when allowed.
        if func.__name__ == "delete" and kwargs.get("allow_non_existent"):
            # Ensure the path is a list and filter out any paths that do not exist.
            path = (path,) if isinstance(path, str) else path
            path = [e for e in path if exists(e, **kwargs)]
            # If no paths are left after filtering, there's nothing to do.
            if len(path) == 0:
                return

        # --- Core Function Execution ---
        # Get the actual filesystem method corresponding to the decorated function's name.
        _func = getattr(fs, func.__name__)
        # Filter kwargs to pass only those that are valid for the underlying fsspec method.
        _kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(_func).parameters}
        # Call the fsspec method with the path and filtered arguments.
        res = _func(path, **_kwargs)

        # --- Function-Specific Post-processing ---
        # Special handling for `glob` to ensure consistent output format.
        if func.__name__ == "glob":
            if not isinstance(path, str):
                raise TypeError(f"Expected str, got {type(path)}")
            # Add the 's3://' prefix back to results from S3, as fsspec removes it.
            if path.startswith("s3://"):
                res = [f"s3://{e}" for e in res]
            # Optionally sort the results.
            if kwargs.get("sort", False):
                res = sorted(res)
        return res

    # Copy the docstring and name from the underlying fsspec method to the wrapper for clarity.
    wrapper.__doc__ = getattr(fsspec.filesystem("file"), func.__name__).__doc__
    wrapper.__name__ = func.__name__
    return wrapper


# The following functions are placeholders. Their actual implementation is
# provided by the `decorator` which wraps the corresponding `fsspec` method.


@decorator
def exists(pathname: str, **kwargs):
    """Check if a path exists."""


@decorator
def isdir(pathname: str, **kwargs):
    """Check if a path is a directory."""


@decorator
def isfile(pathname: str, **kwargs):
    """Check if a path is a file."""


@decorator
def touch(pathname: str, **kwargs):
    """Create an empty file at the given path."""


@decorator
def delete(pathname: str, recursive=False, maxdepth=None, allow_non_existent=False, **kwargs):
    """Delete files or directories."""


@decorator
def makedirs(pathname: str, exist_ok: bool = False, **kwargs):
    """Create a directory and any necessary parent directories."""


@decorator
def glob(pattern: str, sort: bool = False, **kwargs):
    """Find files matching a pattern."""


@decorator
def open(pathname: str, mode: str = "rb", **kwargs):
    """Open a file for reading or writing."""


def normpath(pathname: str, **kwargs) -> str:
    """Normalize a pathname, eliminating double slashes, etc., while preserving schema prefixes."""

    # Check for known URL/cloud prefixes
    for prefix in ("s3://", "https://", "http://"):
        if pathname.startswith(prefix):
            # Remove the prefix safely (Python 3.9+)
            stripped_path = pathname.removeprefix(prefix)

            # Use posixpath for web/cloud paths to guarantee forward slashes '/'
            return f"{prefix}{posixpath.normpath(stripped_path)}"

    # Fallback for standard local file paths
    return os.path.normpath(pathname)


def dirname(pathname: str, **kwargs) -> str:
    """

    Returns the path dirname

    """
    return os.path.dirname(normpath(pathname))


def is_s3_path(pathname: str, **kwargs) -> bool:
    """
    Check if a given path string corresponds to an S3 filesystem.

    This function uses `fsspec` to automatically infer the filesystem type from
    a path string (e.g., "s3://...", "file://...", or a local path) and then
    checks if the inferred filesystem is an `s3fs.S3FileSystem`.

    Parameters
    ----------
    pathname : str
        The path string to check.

    Returns
    -------
    bool
        True if the path is an S3 path, False otherwise.
    """
    # Use fsspec to get the appropriate filesystem object for the given path.
    fs, _ = fsspec.core.url_to_fs(pathname)
    # Check if the filesystem object is an instance of the S3FileSystem.
    return isinstance(fs, s3fs.core.S3FileSystem)


def _fop(
    function_name: Literal["move", "copy"],
    src: str,
    dst: str,
    recursive: bool = False,
    only_if_different_fs: bool = False,
    storage_options: dict | None = None,
) -> str:
    """
    A private helper function for file operations (move or copy).

    This function abstracts the logic for moving or copying files and directories
    between different filesystems (local to S3, S3 to local, or within the same
    filesystem) by dispatching to the appropriate `fsspec` method.

    Parameters
    ----------
    function_name : Literal["move", "copy"]
        The operation to perform.
    src : str
        The source path.
    dst : str
        The destination path.
    recursive : bool, optional
        If True, perform the operation recursively
        on directories. Defaults to False.
    only_if_different_fs : bool, optional
        If True, the operation is skipped
        if the source and destination are on the same filesystem. This can be
        used to avoid unnecessary local-to-local copies. Defaults to False.

    Returns
    -------
    str
        The final destination path.
    """
    storage_options = storage_options or {}
    # Check if the source and destination paths are on S3.
    src_flag, dst_flag = [is_s3_path(e) for e in [src, dst]]

    # Case 1: Source and destination are on the same filesystem (e.g., local-to-local or s3-to-s3).
    if (src_flag == dst_flag) and (not only_if_different_fs):
        # Use the native `move` or `copy` method of that filesystem for efficiency.
        fs, _ = fsspec.core.url_to_fs(src, **storage_options)
        f = getattr(fs, function_name)
        f(src, dst, recursive=recursive)
    # Case 2: Source is on S3, destination is local (i.e., a download operation).
    elif src_flag and (not dst_flag):
        fs = fsspec.filesystem("s3", **storage_options)
        fs.download(src, dst, recursive=recursive)
    # Case 3: Source is local, destination is on S3 (i.e., an upload operation).
    elif (not src_flag) and (dst_flag):
        fs = fsspec.filesystem("s3", **storage_options)
        fs.upload(src, dst, recursive=recursive)
    # Case 4: No operation is performed (e.g., if filesystems are the same but `only_if_different_fs` is True).
    else:
        # In this case, the destination is effectively the same as the source.
        dst = src
    # Return the destination path.
    return dst


def copy(
    src: str,
    dst: str,
    recursive: bool = False,
    only_if_different_fs: bool = False,
    storage_options: dict | None = None,
) -> str:
    """Copy src to dst src and dst can be both local and remote (S3) data.

    Parameters
    ----------
    src : str
        soruce data path
    dst : str
        destination data path
    recursive : bool, optional
        when True if src is a directory
        all the content will be trasferred. Defaults to False.
    only_if_different_fs : bool, optional
        when True src data will be copied only if src and
        dst are not on the same filesystem. Defaults to False.

    Returns
    -------
    str
    """
    storage_options = storage_options or {}
    return _fop(
        "copy",
        src,
        dst,
        recursive,
        only_if_different_fs,
        storage_options=storage_options,
    )


def move(
    src: str,
    dst: str,
    recursive: bool = False,
    only_if_different_fs: bool = False,
    storage_options: dict | None = None,
) -> str:
    """Move src to dst src and dst can be both local and remote (S3) data.
    when src or dst is a remote (S3) path the src file will be copied

    Parameters
    ----------
    src : str
        soruce data path
    dst : str
        destination data path
    recursive : bool, optional
        when True if src is a directory
        all the content will be trasferred. Defaults to False.
    only_if_different_fs : bool, optional
        when True src data will be moved only if src
        and dst are not on the same filesystem. Defaults to False.

    Returns
    -------
    str
    """
    storage_options = storage_options or {}
    result = _fop(
        "move",
        src,
        dst,
        recursive,
        only_if_different_fs,
        storage_options=storage_options,
    )
    if not is_s3_path(src) and exists(src):
        shutil.rmtree(src) if isdir(src) else os.remove(src)
    return result


def path_to_dict(data: Mapping, path: str, pattern: str = ".") -> Any | None:
    """
    Access a nested value in a dictionary using a delimited string path.

    This utility function navigates a nested dictionary structure using a simple
    string path (e.g., "key1.key2.key3") to retrieve the final value.

    Parameters
    ----------
    data : Mapping
        The nested dictionary or other mapping to access.
    path : str
        The string path to the desired value, with keys separated
        by the `pattern`.
    pattern : str, optional
        The delimiter used in the `path` string.
        Defaults to ".".

    Returns
    -------
    Any
        The value found at the end of the nested path.

    Examples
    --------
    >>> my_dict = {'a': {'b': {'c': 42}}}
    >>> path_to_dict(my_dict, "a.b.c")
    42
    """
    # Split the path string into a list of keys using the specified delimiter.
    keys = path.split(pattern)
    result: Any = data
    for key in keys:
        if isinstance(result, Mapping):
            result = result.get(key)
        else:
            logger.warning(f"Pattern {pattern} not found")
            return None
    return result


def get_storage_options(
    kwargs: dict,
    key: str = "storage_options",
    default_values: dict | None = None,
):
    storage_options = kwargs.get(key, default_values)
    return storage_options


def get_crc(attrs):
    crc = md5(dumps(attrs, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[0:3].upper()
    return crc


def get_polarization_code(polarizations: list, product_level: bool = True) -> str:
    """
    Maps a list of polarizations to the standard Sentinel-1 code.

    Args:
        polarizations (list): A list of strings, e.g., ['VV', 'VH']
        product_level (bool): If True, single pols map to 'SH'/'SV' (product metadata).
                              If False, they map to 'HH'/'VV' (extracted components).
    """
    # Convert to an uppercase set to make it order-independent and case-insensitive
    pols = {p.upper() for p in polarizations}

    if pols == {"VV", "VH"}:
        return "DV"
    elif pols == {"HH", "HV"}:
        return "DH"
    elif pols == {"HH"}:
        return "SH" if product_level else "HH"
    elif pols == {"VV"}:
        return "SV" if product_level else "VV"
    elif pols == {"HV"}:
        return "HV"
    elif pols == {"VH"}:
        return "VH"
    else:
        raise ValueError(f"Unrecognized polarization combination: {polarizations}")


def get_duration(attrs):

    start_time = attrs["properties"]["start_datetime"]
    end_time = attrs["properties"]["end_datetime"]
    start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    duration_seconds = str(round((end - start).total_seconds())).rjust(4, "0")
    return duration_seconds


def get_id_product(stac, burst_id: str | None = None, tile_name: str | None = None):
    import copy

    attrs = copy.deepcopy(stac)
    attrs["properties"].pop("created", None)
    attrs["properties"].pop("processing:datetime", None)
    product_type = attrs["properties"]["product:type"]
    start_time = attrs["properties"]["start_datetime"].split(".")[0].replace("-", "").replace(":", "")
    duration = get_duration(attrs)
    platform = attrs["properties"]["platform"][-1]
    relative_orbit = str(attrs["properties"]["sat:relative_orbit"]).rjust(3, "0")
    aux = "S"
    crc = get_crc(attrs)
    try:
        datatake_id_int = int(attrs["properties"]["eopf:datatake_id"])
        hexadecimal = hex(datatake_id_int)[2:].upper().rjust(6, "0")
    except (ValueError, TypeError, KeyError):
        hexadecimal = "000000"
    polarization = get_polarization_code(attrs["properties"]["sar:polarizations"], True)
    id = f"{product_type}_{start_time}_{duration}_{platform}{relative_orbit}_{aux}{crc}_{hexadecimal}_{polarization}"
    if (burst_id is not None) and (burst_id != "data"):
        swath, bid = burst_id.split("_")
        swath = swath[2].rjust(2, "0")
        bid = bid.rjust(6, "0")
        id = f"{id}_{swath}_{bid}"
    if tile_name is not None:
        id = f"{id}_{tile_name}"
    return id.upper()


@overload
def read(
    data_path: str | Path,
    group: str | None = None,
    as_type: Literal["xr.DataArray"] = ...,
    consolidated: bool = True,
    consolidate_metadata: bool = False,
    storage_options: dict | None = None,
    **kwargs: Any,
) -> xr.DataArray: ...


@overload
def read(
    data_path: str | Path,
    group: str | None = None,
    as_type: Literal["xr.Dataset"] = ...,
    consolidated: bool = True,
    consolidate_metadata: bool = False,
    storage_options: dict | None = None,
    **kwargs: Any,
) -> xr.Dataset: ...


@overload
def read(
    data_path: str | Path,
    group: str | None = None,
    as_type: None = None,
    consolidated: bool = True,
    consolidate_metadata: bool = False,
    storage_options: dict | None = None,
    **kwargs: Any,
) -> xr.DataTree: ...


@overload
def read(
    data_path: str | Path,
    group: str | None = None,
    as_type: str | None = None,
    consolidated: bool = True,
    consolidate_metadata: bool = False,
    storage_options: dict | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset | xr.DataTree: ...


def read(
    data_path: str | Path,
    group: str | None = None,
    as_type: str | None = None,
    consolidated: bool = True,
    consolidate_metadata: bool = False,
    storage_options: dict | None = None,
    **kwargs,
) -> xr.DataArray | xr.Dataset | xr.DataTree:
    """
    Read an xarray object from a Zarr V3 store.
    """
    if isinstance(data_path, Path):
        storage_options = data_path.storage_options
        data_path = str(data_path)
    # bypass s3fs cache
    so = is_local(data_path, storage_options)
    with zarr.storage.FsspecStore.from_url(url=data_path, storage_options=so) as store, warnings.catch_warnings():
        # Ignore the specific consolidated metadata warning
        warnings.filterwarnings("ignore", message=".*Consolidated metadata is currently not part.*")
        if consolidate_metadata:
            # Consolidate metadata for faster access, especially on cloud storage.
            zarr.consolidate_metadata(store)

    # --- Read Dataset or DataArray ---
    # If we know we just want a flat Dataset or DataArray, open it directly.
    # This is faster than building a full DataTree just to extract one node.
    if as_type in ["xr.Dataset", "xr.DataArray"]:
        ds = xr.open_dataset(
            data_path, engine="zarr", consolidated=consolidated, group=group, chunks={}, storage_options=so, **kwargs
        )
        ds.attrs["store_path"] = data_path
        if as_type == "xr.DataArray":
            if len(ds.data_vars) == 1:
                # Unpack the single variable back into a DataArray
                var = next(iter(ds.data_vars))
                dr = ds[var]
                dr.attrs["store_path"] = f"{data_path}/{var}"
                return dr
            raise ValueError("Cannot cast to DataArray: Store contains multiple variables.")
        return ds

    # --- Read DataTree (Default) ---
    # If no specific type is forced, read the entire structure as a DataTree.
    dt = xr.open_datatree(data_path, engine="zarr", consolidated=consolidated, chunks={}, storage_options=so, **kwargs)

    # Recursively inject the specific store_path into EVERY node
    for node in dt.subtree:  # .subtree includes the root and all descendants
        # node.path returns strings like "/", "/coreg", "/coreg/quality"
        sub_path = node.path.strip("/")

        if sub_path:
            # Safely combine the base path with the node's relative path
            node_store_path = f"{data_path.rstrip('/')}/{sub_path}"
        else:
            node_store_path = data_path

        # Update the attributes for this specific node
        node.attrs["store_path"] = node_store_path
    # If the user requested a specific group from the tree, return just that subtree
    if group:
        group_path = group.strip("/")
        return dt[group_path]

    return dt


def drop_encodings(data):
    """Recursively clears encoding dictionaries across DataTree, Dataset, and DataArray objects."""
    if isinstance(data, xr.DataTree):
        for node in data.subtree:  # includes root
            if node.ds is not None:
                drop_encodings(node.ds)

    elif isinstance(data, xr.Dataset):
        data.encoding.clear()
        for v in data.variables.values():
            v.encoding.clear()

    elif isinstance(data, xr.DataArray):
        data.encoding.clear()
        data.variable.encoding.clear()
        for coord in data.coords.values():
            coord.encoding.clear()
            if hasattr(coord, "variable"):
                coord.variable.encoding.clear()


def write(
    data: xr.Dataset | xr.DataArray | xr.DataTree,
    data_path: str | Path,
    region: Mapping[str, slice] | None = None,
    consolidated: bool = True,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"] = "w",
    compute: bool = True,
    skip_read: bool = False,
    storage_options: dict | None = None,
    encoding: dict | None = None,
) -> xr.DataTree | xr.Dataset | xr.DataArray | None:
    if isinstance(data_path, Path):
        storage_options = data_path.storage_options
        data_path = str(data_path)
    mode, _file_path, _data, _data_type = check_write_arguments(data, data_path, mode, region)

    # 2. DROP inherited encoding
    drop_encodings(_data)

    # 3. VALIDATE user-provided encoding
    safe_encoding = validate_encoding(_data, encoding)

    # 4. Handle region coordinates logic
    if region is not None:
        _data = _data.drop_vars(set(_data.dims).intersection(set(_data.coords)))

    so = is_local(_file_path, storage_options)

    # 5. WRITE
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Consolidated metadata is currently not part.*")
        warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
        _data.to_zarr(  # type: ignore
            store=_file_path,
            mode=mode,
            region=region,
            consolidated=consolidated,
            zarr_format=3,
            storage_options=so,
            encoding=safe_encoding,
            compute=compute,
        )

    if not skip_read:
        return read(_file_path, as_type=_data_type, storage_options=so)

    return None
