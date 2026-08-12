from typing import Any, Union, cast

import numpy as np
import xarray as xr
from cop_ard_core.logger import logger, timeit

from cop_ard_app.computation import map
from cop_ard_app.io import Path, get_storage_options, normpath, read, write
from cop_ard_app.orbit import update_orbit


@timeit
def import_ETAD(
    slcs: xr.DataTree,
    acquisition_dates: list[str],
    etad_files: list[Path],
    wdir_path: str,
    multiburst: bool = False,
    enable_dask: bool = True,
    **kwargs,
):
    """
    Find and import relevant ETAD and precise orbit files for a set of acquisitions.

    This function serves as a high-level wrapper. It filters a list of all available
    ETAD files to find the ones that match the acquisition dates of interest. It then
    calls `import_etad_stack` to perform the import and updates the precise orbit
    information in the `slcs` DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all the SLC data.
    acquisition_dates : list[str]
        A list of acquisition date strings to process.
    etad_files : list[str]
        A list of all available ETAD file paths to search through.
    wdir_path : str
        The root working directory.
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree, now containing the imported ETAD and precise
        orbit information.
    """
    is_valid = all(isinstance(x, Path) for x in etad_files)
    if not is_valid:
        raise TypeError("Invalid input: Expected a list of Path objects for ETAD files.")
    acquisition_date: Union[str, np.datetime64]
    etad_paths = set()
    # --- Find Matching ETAD Files ---
    for acquisition_date in acquisition_dates:
        acquisition_date = np.datetime64(acquisition_date)
        # Format the date as YYYYMMDD to match typical ETAD filenames.
        date = acquisition_date.astype("M8[D]").astype(object).strftime("%Y%m%d")
        # Find all files in the `etad_files` list that contain this date string.
        files = {file for file in etad_files if date in file.name}
        if files:
            etad_paths.update(files)
    used_etad: list[str] = []
    # --- Import ETAD and Orbits ---
    if etad_paths:
        # Call the main import function with the filtered list of ETAD paths.
        slcs, orbit_paths = import_etad_stack(
            etad_file_list=etad_paths,
            slcs=slcs,
            multiburst=multiburst,
            wdir_path=f"{wdir_path}/ETAD",
            enable_dask=enable_dask,
            **kwargs,
        )
        # Check if orbit files were found for all necessary acquisitions.
        if any(e is None for e in orbit_paths):
            raise RuntimeError("The input slcs is not completely convered by the input ETAD file(s)")
        # Update the SLC DataTree with the new precise orbit information.
        update_orbit(
            slcs=slcs,
            multiburst=multiburst,
            orbit_paths=orbit_paths,
            enable_dask=enable_dask,
            wdir_path=wdir_path,
            **kwargs,
        )
        used_etad = [i.path.split(".SAFE")[0] + ".SAFE" for i in orbit_paths]
    return slcs, used_etad


def discover_etaddate_slcsdate(slcs: xr.DataTree, etad: xr.DataTree, multiburst: bool) -> set:
    """
    Discover SLC acquisition dates that fall within the time range of an ETAD product.

    This function checks which Single Look Complex (SLC) acquisitions have an azimuth
    time range that is fully contained within the valid time range of a given
    Extended Timing Annotation Data (ETAD) product.

    Parameters
    ----------
    slcs : xarray.DataTree
        A DataTree object containing the SLCs data, organized by bursts
        if `multiburst` is True.
    etad : xarray.DataTree
        A DataTree object for a single ETAD product, containing its
        global time range in the attributes.
    multiburst : bool
        A boolean flag indicating if the `slcs` DataTree is in
        multiburst format.

    Returns
    -------
    set
        A set of unique acquisition date strings (e.g., "2022-01-01") for all
        SLCs that correspond to the provided ETAD data.
    """
    # Extract the start and end azimuth times of the ETAD correction's validity from its attributes.
    tstart = np.datetime64(etad.attrs["azimuthTimeMin"], "ns")
    tend = np.datetime64(etad.attrs["azimuthTimeMax"], "ns")

    # Initialize a set to store the unique matching acquisition date IDs.
    dateid = set()

    # Iterate over all SLCs (either in bursts or as a single scene).
    for _, i in (slcs.bursts if multiburst else {"data": slcs}).items():
        for name, j in i.orig.items():
            # Check if the SLC's entire azimuth time span is within the ETAD's time span.
            if (j.measurements[0] >= tstart) & (j.measurements[-1] <= tend):
                # If it matches, add the acquisition date to the set.
                dateid.add(name)

    # Return the set of unique SLC acquisition dates covered by this ETAD.
    return dateid


def save_etad(
    etad_burst: xr.DataTree,
    min_acquisition_azimuth_time: np.datetime64,
    min_acquisition_range_time: np.float64,
    etadf: str,
    out_path: str,
    **kwargs,
):
    """
    Save the ETAD correction grids for a single burst to a Zarr store.

    This function extracts the azimuth and range correction grids from an ETAD
    burst, reconstructs their time-based coordinates, and saves them as an
    xarray.Dataset in a specified Zarr store.

    Parameters
    ----------
    etad : xr.DataTree
        The full ETAD DataTree, used for global attributes like min/max times.
    etad_burst : xr.DataTree
        A DataTree node representing a specific burst within the ETAD data.
    etadf : str
        The path to the original source ETAD file, stored as an attribute.
    out_path : str
        The output path for the new Zarr store.

    Returns
    -------
    None
        None. The function writes the data directly to disk.
    """
    # --- Reconstruct Coordinate Grids ---
    # Calculate the absolute start time of the ETAD grid for this burst in azimuth.
    t0az = min_acquisition_azimuth_time + np.timedelta64(
        np.round(etad_burst.attrs["gridStartAzimuthTime"] * 1e9).astype(int), "ns"
    )
    # Calculate the absolute start time of the ETAD grid in range.
    t0rg = min_acquisition_range_time + etad_burst.attrs["gridStartRangeTime"]

    # Extract the grid spacing (sampling interval) in azimuth and range.
    dtaz = etad_burst.attrs["gridSamplingAzimuth"]
    dtrg = etad_burst.attrs["gridSamplingRange"]

    # Create the full coordinate arrays in time units.
    coords = {
        "azimuth_time": np.round(etad_burst.azimuthExtent.values * dtaz * 1e9).astype(int).astype("timedelta64[ns]")
        + t0az,
        "slant_range_time": etad_burst.rangeExtent.values * dtrg + t0rg,
    }

    # --- Create and Save Dataset ---
    # Create the xarray Dataset containing the azimuth and range correction grids.
    etad_ds = xr.Dataset(
        data_vars={
            "azcorr": (["azimuth_time", "slant_range_time"], etad_burst.sumOfCorrectionsAz.values),
            "rgcorr": (["azimuth_time", "slant_range_time"], etad_burst.sumOfCorrectionsRg.values),
        },
        coords=coords,
        attrs={"etad_file": etadf},
    ).chunk(1024)  # Apply chunking for efficient access.

    # Save the dataset to a Zarr store.
    write(etad_ds, out_path, storage_options=get_storage_options(kwargs), skip_read=True)
    return


def update_etad(
    etad_burst: xr.DataTree,
    sw_name: str,
    acquisition_date: str,
    etad_nc_path: str,
    min_acquisition_azimuth_time: np.datetime64,
    min_acquisition_range_time: np.float64,
    wdir_path: str,
    **kwargs,
):
    """
    A worker function to save ETAD data for a burst and return the result.

    This function is designed to be used in a parallel map operation. It constructs
    the appropriate output path, calls `save_etad` to write the data to disk,
    and then reads the data back to return it to the main process.

    Parameters
    ----------
    acquisition_date : str
        The acquisition date string (e.g., "2022-01-01").
    sw_name : str
        The name of the swath (e.g., "IW1").
    etad_nc : str
        The path to the source ETAD NetCDF file.
    etad : xr.DataTree
        The full ETAD DataTree.
    etad_burst : xr.DataTree
        The DataTree node for the specific ETAD burst to process.
    wdir_path : str
        The root working directory for the output.

    Returns
    -------
    tuple
        A tuple containing the acquisition date, the burst ID, and the newly
        created ETAD xarray.Dataset read from disk.
    """
    # Construct the unique burst ID from the swath name and burst number.
    bid = f"{sw_name.lower()}_{etad_burst.attrs['burstId']}"
    # Define the final output path for this burst's ETAD data within the SLC DataTree structure.
    data_path = normpath(f"{wdir_path}/{bid}/orig/{acquisition_date}/conditions/ETAD")
    # Save the ETAD correction grids to the specified path.
    save_etad(etad_burst, min_acquisition_azimuth_time, min_acquisition_range_time, etad_nc_path, data_path, **kwargs)
    # Read the newly created dataset back from disk and return it.
    return (
        acquisition_date,
        bid,
        read(data_path, storage_options=get_storage_options(kwargs)),
    )


def import_etad_stack(
    etad_file_list: set,
    slcs: xr.DataTree,
    multiburst: bool,
    wdir_path: str,
    enable_dask: bool,
    **kwargs,
):

    slcs_az = xr.DataTree()
    for burst_id, slc in slcs.bursts.items():
        for date in slc.orig:
            measurements = slc.orig[date].measurements
            az_min, az_max = measurements.azimuth_time.min().values, measurements.azimuth_time.max().values
            slcs_az[f"bursts/{burst_id}/orig/{date}/measurements"] = xr.DataArray([az_min, az_max], dims=["point"])

    orbit_paths = []
    flat_burst_args = []
    # 1. Iterate over all files to collect bursts and metadata
    for etad_file in etad_file_list:
        nc_files = etad_file.glob("measurement/*.nc")
        if not nc_files:
            raise ValueError(f"No .nc file found at {etad_file.path}/measurement")
        etad_nc_path = nc_files[0].path
        open_kwargs: dict[str, Any] = {"chunks": {}, "cache": False}
        local_file = "://" not in etad_nc_path or etad_nc_path.startswith("file://")
        if not local_file:
            open_kwargs["storage_options"] = etad_file.storage_options
        etad_nc = xr.open_datatree(etad_nc_path, engine="h5netcdf", **open_kwargs)
        # Discover dates and times
        t0az = np.datetime64(etad_nc.attrs["azimuthTimeMin"], "ns")
        t0rg = etad_nc.attrs["rangeTimeMin"]
        etad_acquisition_date = discover_etaddate_slcsdate(slcs_az, etad_nc, multiburst).pop()
        if not etad_acquisition_date:
            logger.info(f"ETAD file without correspondence: {etad_nc_path}")
            continue

        # Save orbit path
        orbit_paths.append(etad_file.glob("annotation/*.EOF")[0])

        # Traverse the DataTree immediately to extract bursts
        for sw_name, etad_sw in etad_nc.items():
            for et_name, etad_burst in etad_sw.items():
                burst_key = f"{sw_name.lower()}_{etad_burst.attrs['burstId']}"

                # Check if burst exists in input SLC data
                if burst_key in (slcs.bursts if multiburst else {"data": slcs}):
                    flat_burst_args.append((etad_burst, sw_name, etad_acquisition_date, etad_nc_path, t0az, t0rg))

    if not flat_burst_args:
        return slcs, orbit_paths

    # Transpose the list of tuples
    mapped_args = list(zip(*flat_burst_args))
    flat_results = cast(
        list,
        map(
            update_etad,
            mapped_args,
            kwargs=dict(wdir_path=f"{wdir_path}/bursts" if multiburst else wdir_path, **kwargs),
            combine=False,
            enable_dask=enable_dask,
        ),
    )

    # 3. Integrate all results back into the DataTree
    for ad, bid, et in flat_results:
        if multiburst:
            slcs[f"bursts/{bid}/orig/{ad}/conditions/ETAD"] = et
        else:
            slcs[f"orig/{ad}/conditions/ETAD"] = et

    return slcs, orbit_paths
