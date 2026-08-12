import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Union, cast
from xml.etree.ElementTree import Element, ElementTree

import dateutil
import numpy as np
import xarray as xr
from cop_ard_core.logger import logger, timeit
from cop_ard_core.orbit import build_orbit_dataset, refit_orbit_coefficients

from cop_ard_app.computation import map
from cop_ard_app.io import Path
from cop_ard_app.slc import overwrite_orbit_coefficients, update_slcs


def read_orbit_file(orbit_path: Path) -> ElementTree:
    local_orbit_path = orbit_path.ensure_local()
    return ET.parse(local_orbit_path.path)


def import_orbit(
    orbit_paths: list[Path] | None = None,
    start_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
    **kwargs,
) -> Union[xr.Dataset | None]:
    """
    Import and filter orbit data from specified files for a given time interval.

    This function reads orbit data from one or more files and then filters the
    resulting dataset to a specific time range defined by `start_time` and
    `end_time`. It is decorated with a retry mechanism to handle transient
    network or file access issues.

    Parameters
    ----------
    orbit_paths : list[cop_ard_core.computing.io.Path] | None, optional
        A list of paths to the orbit files to import.
    start_time : np.datetime64 | None, optional
        The start of the desired time interval for the orbit data.
    end_time : np.datetime64 | None, optional
        The end of the desired time interval for the orbit data.

    Returns
    -------
    xr.Dataset | None
        An xarray.Dataset containing the filtered orbit data, or None if no
        `orbit_paths` are provided.
    """
    # Proceed only if orbit paths are provided.
    if (orbit_paths is not None) and all(e is not None for e in orbit_paths):
        logger.info(f"For the time interval {start_time} - {end_time} these orbit files will be used: {orbit_paths}")
        # Read the raw orbit data from the specified file(s).
        orbit = build_orbit_dataset([read_orbit_file(orbit_path) for orbit_path in orbit_paths])
    else:
        # If no paths are given, log the information and return None.
        logger.info(f"For the time interval {start_time} - {end_time} no external orbits have been found")
        return None

    # If start_time is not provided, default to the first timestamp in the orbit data.
    if start_time is None:
        start_time = orbit.azimuth_time[0]
    # If end_time is not provided, default to the last timestamp.
    if end_time is None:
        end_time = orbit.azimuth_time[-1]

    # Filter the orbit dataset to the specified time interval and add the source filenames as an attribute.
    return orbit.where((orbit.azimuth_time >= start_time) & (orbit.azimuth_time <= end_time), drop=True).assign(
        orbit_file=(["acquisition_date"], np.array([",".join([e.name for e in orbit_paths])], dtype=object))
    )


def update_orbit(
    slcs: xr.DataTree,
    wdir_path: str,
    orbit_paths: list[Path] | None = None,
    multiburst: bool = True,
    poly_degree: int = 10,
    border: np.timedelta64 = np.timedelta64(100, "s"),
    enable_dask: bool = True,
    **kwargs,
):
    """
    Orchestrate the update of orbit polynomials for an entire SLC stack.

    This function identifies the required time intervals for each acquisition,
    imports the corresponding precise orbit data, and then maps the
    `overwrite_coeffs` worker function in parallel to refit the polynomials
    and update the metadata for every SLC in the stack.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all SLC data to be updated.
    orbit_paths : list[cop_ard_core.computing.io.Path] | None, optional
        A list of paths to precise orbit files. If None, it will try to
        find paths from ETAD metadata.
    multiburst : bool, optional
        A boolean flag for multiburst mode. Defaults to True.
    poly_degree : int, optional
        The degree for the new polynomial fit. Defaults to 10.
    border : np.timedelta64, optional
        A time border to add to the acquisition interval when fetching
        orbit data, to ensure sufficient data for fitting.
        Defaults to np.timedelta64(100, "s").

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree with new orbit information.
    """

    def get_reference_time_from_attrs(attrs: dict):
        return attrs[list(attrs)[0]]["Timeline"]["CollectStart"]

    slcs_info: dict[str, Any] = {}
    # --- Step 1: Collect SLC Time Intervals and Metadata ---
    # Iterate through all original SLCs in the DataTree.
    for i in (slcs.bursts if multiburst else {"data": slcs}).values():
        for k, v in i.orig.items():
            # Retrieve the orbit source file path if available (e.g., from ETAD metadata).
            if orbit_paths is None:
                try:
                    orb_file = v.conditions.ETAD.attrs.get("orbit_file", None)
                except Exception as e:
                    logger.warning(f"type is: {e.__class__.__name__}")
                    logger.warning("Orbit file not found")
                    orb_file = None
            else:
                orb_file = None
            # Extract metadata and the start/end azimuth times.
            annot = v.measurements.slc.attrs[list(v.measurements.slc.attrs.keys())[0]]
            info = [
                [
                    annot["CollectionInfo"]["CollectorName"],
                    orb_file,
                    v.measurements.slc.azimuth_time[0].values,
                    v.measurements.slc.azimuth_time[-1].values,
                ]
            ]
            # Group the information by acquisition date.
            if k in slcs_info:
                slcs_info[k] += info
            else:
                slcs_info[k] = info

    # Consolidate the information for each date to find the overall time range.
    slcs_info = {k: list(zip(*v)) for k, v in slcs_info.items()}
    slcs_info = {k: [v[0][:2], v[1], np.min(v[2]) - border, np.max(v[3]) + border] for k, v in slcs_info.items()}

    # --- Step 2: Match Orbit Files to Time Intervals ---
    if orbit_paths is not None:
        # Parse the validity time from the orbit filenames.
        orbit_intervals = np.array(
            [
                [dateutil.parser.parse(j).isoformat() for j in i.name.split("_V")[1][:-4].split("_")]
                for i in orbit_paths
            ],
            dtype="datetime64[ns]",
        )
        # For each acquisition, find the orbit files that cover its time interval.
        for k, v in slcs_info.items():
            (i_valid,) = np.where((v[2] <= orbit_intervals[:, 1]) & (v[3] >= orbit_intervals[:, 0]))
            if i_valid.size:
                slcs_info[k][1] = np.array(orbit_paths)[i_valid].tolist()

    # --- Step 3: Import Orbit Data in Parallel ---
    dates = list(slcs_info)
    args = [[slcs_info[e][i] for e in dates] for i in range(1, 4)]
    orbits = map(import_orbit, args=args, kwargs=kwargs, combine=False, enable_dask=False)
    if not isinstance(orbits, list):
        raise Exception(f"Expected list, got {type(orbits)}")
    orbits = {d: o for d, o in zip(dates, orbits)}

    # --- Step 4: Refit Polynomials and Update Metadata in Parallel ---
    # Prepare arguments for the `refit_orbit_coefficients` worker function.
    args = [
        (
            k,  # kept in order to make downstream coefficient saving easier to maintain
            d,
            orbits[d],
            get_reference_time_from_attrs(v.measurements.slc.attrs)[:-1],
            get_reference_time_from_attrs(v.measurements.slc.attrs),
        )
        for k, s in (slcs.bursts if multiburst else {"data": slcs}).items()
        for d, v in s.orig.items()
    ]
    refit_orbits = cast(
        list[xr.Dataset | None],
        map(
            refit_orbit_coefficients,
            list(zip(*args)),
            kwargs={"poly_degree": poly_degree},
            combine=False,
            enable_dask=enable_dask,
        ),
    )

    # Prepare arguments for the `overwrite_orbit_coefficients` worker function.
    args = [(arg[0], arg[1], res) for arg, res in zip(args, refit_orbits)]
    refit_orbits = cast(
        list,
        map(
            overwrite_orbit_coefficients,
            list(zip(*args)),
            kwargs={"wdir_path": wdir_path, "multiburst": multiburst, **kwargs},
            combine=False,
            enable_dask=enable_dask,
        ),
    )
    # --- Step 5: Integrate Results ---
    # Update the main DataTree with the new orbit condition datasets.
    slcs = update_slcs(slcs, multiburst, refit_orbits)
    return slcs


@timeit
def import_ORBIT(
    slcs: xr.DataTree,
    acquisition_dates: list[np.datetime64],
    orbit_files: list[Path],
    wdir_path: str,
    multiburst: bool = False,
    enable_dask: bool = True,
    **kwargs,
):
    """
    High-level wrapper to find and import precise orbits for a stack.

    This function finds the relevant precise orbit files from a given folder that
    correspond to a list of acquisition dates, and then triggers the main
    `update_orbit` process to refit and update the orbit information in the `slcs`
    DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all SLC data.
    acquisition_dates : list[np.datetime64]
        A list of acquisition dates to find orbits for.
    orbit_folder : str
        The path to a folder containing orbit files (.EOF).
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.
        Defaults to False.

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree containing the new precise
        orbit information.
    """
    orbit_paths = []
    # --- Find Relevant Orbit Files ---
    # For each acquisition date, determine the total time span covered by all its bursts.
    for acquisition_date in acquisition_dates:
        for counter, (k, slc) in enumerate((slcs.bursts if multiburst else {"data": slcs}).items()):
            if acquisition_date in list(slc.orig.keys()):
                az = slc.orig[acquisition_date].measurements.azimuth_time
                s, e = az.min().values, az.max().values
                if counter == 0:
                    start_time = s
                    end_time = e
                else:
                    start_time = min(start_time, s)
                    end_time = max(end_time, e)
        # Find the orbit files whose validity period covers this time span.
        for o in orbit_files:
            # Parse the start and end validity times from the orbit filename.
            pattern = r"V(\d{8}T\d{6})_(\d{8}T\d{6})"
            match = re.search(pattern, o.name)
            if not match:
                raise ValueError(f"No validity dates found in filename: {o.name}")

            # Convert extracted strings directly to numpy datetime objects
            valid_start = np.datetime64(datetime.strptime(match.group(1), "%Y%m%dT%H%M%S"))
            valid_stop = np.datetime64(datetime.strptime(match.group(2), "%Y%m%dT%H%M%S"))
            # Add the orbit file if it completely covers the acquisition time window
            if valid_start <= start_time and end_time <= valid_stop:
                orbit_paths.append(o)

    # --- Trigger Orbit Update ---
    if orbit_paths:
        # Call the main update function with the filtered list of orbit paths.
        update_orbit(
            slcs, multiburst=multiburst, orbit_paths=orbit_paths, wdir_path=wdir_path, enable_dask=enable_dask, **kwargs
        )
    orbit_paths = [o.path for o in orbit_paths]
    return slcs, orbit_paths
