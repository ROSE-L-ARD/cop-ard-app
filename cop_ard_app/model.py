import copy
import os
import re
from datetime import datetime, timezone
from typing import Any, Sequence, Union, cast

import cop_ard_core
import numpy as np
import sarpy.io.complex.sentinel
import xarray as xr
from cop_ard_core.dem import extract_vertical_crs
from cop_ard_core.logger import logger
from pyproj import CRS
from scipy.constants import speed_of_light

import cop_ard_app
from cop_ard_app.io import get_id_product, get_storage_options, path_to_dict, read

DEL_ATTRS = [
    "store_path",
    "obj_type",
    "_FillValue",
    "etad_file",
    "coregistration_reference_date",
]
DEL_ENCODING = ["coordinates"]
ATTRS_GEO = {
    "phase_model": {
        "description": "Projected topographic phase model, in rad, removed from the original interferometric phase."
        "The interferometric phase is evaluated wrt the reference orbit, used for co-registration process.",
        "long_name": "Interferometric phase model",
        "units": "rad",
    },
    "polarization": {
        "description": "acquired polarizations",
        "long_name": "Polarization",
        "units": "n/a",
    },
    "acquisition_date": {
        "description": "Image acquisition date, espressed as YYYY-MM-DD, data type is string",
        "long_name": "Acquisition date",
        "units": "n/a",
    },
    "y": {
        "description": "UTM y coordinates",
        "long_name": "North coordinates",
        "units": "m",
    },
    "x": {
        "description": "UTM x coordinates",
        "long_name": "East coordinates",
        "units": "m",
    },
    "gslc": {
        "description": "Projected Co-registered SLC, with module equal to gamma0 or beta0 coefficients and\
                            phase equal to the projected CSLC phase",
        "long_name": "Geocoded Single Look Complex",
        "units": "n/a",
    },
    "anf": {
        "description": "Optional area normalization factor, used to obtain the gamma0 coefficient",
        "long_name": "terrain simulated radar backscattering",
        "units": "m^2",
    },
    "local_incidence_angle": {
        "description": "DEM-based local Incidence angle image",
        "long_name": "local incidence angle",
        "units": "deg",
    },
    "mask": {
        "description": "layover=1, shadow=2, river=3, lake=4, ocean=5 and valid_point=0",
        "long_name": "mask of valid points",
        "units": "n/a",
    },
    "nlooks": {
        "description": "Number of averaged radar cells",
        "long_name": "Number of looks",
        "units": "n/a",
    },
}


beam_ids = {
    "IW": ["IW1", "IW2", "IW3"],  # Interferometric Wide (3 sub-swaths)
    "EW": ["EW1", "EW2", "EW3", "EW4", "EW5"],  # Extra Wide (5 sub-swaths)
    "SM": ["S1", "S2", "S3", "S4", "S5", "S6"],  # Stripmap (6 overlapping beams)
}


ATTRS_RADAR = {
    "phase_model": {
        "description": "Topographic phase model, in rad, removed from the original interferometric phase."
        "The interferometric phase is evaluated wrt the reference orbit, used for co-registration process.",
        "long_name": "Interferometric phase model",
        "units": "rad",
    },
    "polarization": {
        "description": "acquired polarizations",
        "long_name": "Polarization",
        "units": "n/a",
    },
    "acquisition_date": {
        "description": "Image acquisition date, espressed as YYYY-MM-DD, data type is string",
        "long_name": "Acquisition date",
        "units": "n/a",
    },
    "slant_range_time": {
        "description": "Time across satellite track direction expressed as seconds",
        "long_name": "Slant range time",
        "units": "s",
    },
    "azimuth_time": {
        "description": "Time along satellite track direction expressed as a datetime64[ns]",
        "long_name": "Azimuth time",
    },
    "axis": {
        "description": "Index corresponding to the axis",
        "long_name": "Axis index",
        "units": "n/a",
    },
    "degree": {
        "description": "Polynomial coefficient degree",
        "long_name": "Polynomial coefficient degree",
        "units": "n/a",
    },
    "azcorr": {
        "description": "Extended Timining Annotation Data: azimuth time correction resampled on the common grid of\
                            the reference geometry",
        "long_name": "Azimuth timing correction",
        "units": "s",
    },
    "rgcorr": {
        "description": "Extended Timining Annotation Data: slant range time corection resampled on the common grid\
                            of the reference geometry",
        "long_name": "Slant range timing correction",
        "units": "s",
    },
    "etad_file": {
        "description": "ETAD file used for each of the images belonging to the cslc stack",
        "long_name": "Extended Timing Annotation Data file",
        "units": "n/a",
    },
    "orbit_files": {
        "description": "Orbit files used for each of the images belonging to the cslc stack",
        "long_name": "Orbit files",
        "units": "n/a",
    },
    "cslc": {
        "description": "Co-registred single look complex data",
        "long_name": "Co-registered SLC",
        "units": "n/a",
    },
    "beta0": {
        "description": "LUT of the beta0 coefficients obtained from input SLC annotations.",
        "long_name": "Beta nought",
        "units": "m^2 / m^2",
    },
    "gamma0": {
        "description": "LUT of the gamma0 coefficients obtained from input SLC annotations.",
        "long_name": "Gamma nought",
        "units": "m^2 / m^2",
    },
    "noise": {
        "description": "LUT of thermal noise correction obtained from input SLC annotations.",
        "long_name": "Thermal noise correction",
        "units": "n/a",
    },
    "RCS": {
        "description": "LUT of the Radar Cross Section (RCS) coefficients obtained from input SLC annotations.",
        "long_name": "Radar cross section",
        "units": "m^2 / m^2",
    },
    "sigma0": {
        "description": "LUT of the sigma0 coefficients obtained from input SLC annotations.",
        "long_name": "Sigma nought",
        "units": "m^2 / m^2",
    },
    "crg_accuracy": {
        "description": "Estimated co-registration accuracty in correspondence of each image of the cslc stack.",
        "long_name": "Co-registration accuracy",
        "units": "m",
    },
    "phase_demod": {
        "description": "Phase to be applied to demodulate each image belonging to the cslc stack.",
        "long_name": "Demodulation phase",
        "units": "rad",
    },
    "coeffs": {
        "description": "Polynomial coefficients describing the orbit associated to each input image belonging to\
                            the cslc stack.",
        "long_name": "Orbit coefficients",
        "units": "n/a",
    },
    "reference_time": {
        "description": "Reference time associated to the polynomial coefficients for each input image belonging to\
                            the cslc stack.",
        "long_name": "Reference polynomial coefficient time",
        "units": "s",
    },
    "type": {
        "description": "Type of orbital data used for each of the images belonging to the cslc stack:\
                            ETAD, POE, POD, RES.",
        "long_name": "Orbital data accuracy",
        "units": "n/a",
    },
}


def extract_orbits(
    slcs: xr.DataTree,
    acquisition_id: list[str] | None = None,
    multiburst: bool = False,
    degrees: int = 11,
) -> xr.DataArray:
    """
    Extract and consolidate satellite orbit polynomial coefficients from metadata.
    """
    orbits_datatree = xr.DataTree()
    orbits, reference_times = {}, {}

    # --- Collect Orbit Data ---
    # Iterate through all scenes (bursts or single images) to collect orbit polynomials.
    for burst_id, i in (slcs.bursts if multiburst else {"data": slcs}).items():
        for data_id, j in i.orig.items():
            annot = j.measurements.slc.attrs[list(j.measurements.slc.attrs.keys())[0]]
            orbits.update({data_id: annot["Position"]["ARPPoly"]})
            reference_times.update(
                {data_id: annot["Timeline"]["CollectStart"]},
            )
            # Prioritize pre-computed orbit data if available.
            if "orbit" in j.conditions:
                orbits_datatree[data_id] = j.conditions.orbit.coeffs.squeeze("acquisition_id")

    # Filter by specific acquisition dates if provided.
    if acquisition_id is not None:
        orbits = {k: v for k, v in orbits.items() if k in acquisition_id}
        reference_times = {k: v for k, v in reference_times.items() if k in acquisition_id}

    # --- Consolidate Coefficients ---
    # Initialize an array to hold the coefficients.
    coeffs = np.zeros((len(orbits), degrees, 3))
    for counter, (k, v) in enumerate(orbits.items()):
        # Use pre-computed orbit data if available.
        if k in orbits_datatree:
            data = orbits_datatree[k].values
        # Otherwise, extract from the raw metadata.
        else:
            data = np.array([v[e]["Coefs"] for e in ["X", "Y", "Z"]]).T
        if not isinstance(data, np.ndarray):
            raise Exception(f"Expected np.ndarray, got {type(data)}")
        shape = data.shape
        # Place the coefficients into the consolidated array.
        coeffs[counter][: shape[0], : shape[1]] += data

    # --- Create Final DataArray ---
    orbits = xr.DataArray(
        data=coeffs,
        dims=["acquisition_id", "degree", "axis"],
        coords={
            "acquisition_id": list(orbits),
            "degree": np.arange(0, degrees, 1, dtype=int),
            "axis": np.arange(0, 3, 1, dtype=int),
        },
    ).assign_attrs({"reference_times": reference_times})

    return orbits


def extract_acquisition_dates(
    slcs: xr.DataTree,
    multiburst: bool = False,
) -> list[str]:
    """
    Extract a unique, sorted list of acquisition dates from the DataTree.
    """
    acquisition_dates = set()

    # Iterate through all original data nodes.
    for i in (slcs.bursts if multiburst else {"data": slcs}).values():
        for j in i.orig.values():
            # Get the acquisition date from the coordinates.
            k = j.measurements.slc.azimuth_time.values[0]
            # Use a set to automatically handle uniqueness.
            acquisition_dates.add(k)

    return list(acquisition_dates)


def extract_acquisition_id(
    slcs: xr.DataTree,
    multiburst: bool = False,
) -> set:
    """
    Extract a unique, sorted list of acquisition dates from the DataTree.
    """
    ids = set()

    # Iterate through all original data nodes.
    for i in (slcs.bursts if multiburst else {"data": slcs}).values():
        for j in i.orig:
            # Get the acquisition date from the coordinates.
            ids.add(j)

    return ids


def get_az_rg_times(annot):
    """
    Compute azimuth and slant range time coordinate arrays from SICD metadata.
    """
    import pandas as pd

    try:
        # Access metadata fields directly from the Sarpy object.
        num_cols = annot.ImageData.FullImage.NumCols
        num_rows = annot.ImageData.FullImage.NumRows
        scp_col = annot.ImageData.SCPPixel.Col
        scp_row = annot.ImageData.SCPPixel.Row
        time_ca_poly = annot.RMA.INCA.TimeCAPoly.Coefs
        collect_start = annot.Timeline.CollectStart
        col_ss = annot.Grid.Col.SS
        row_ss = annot.Grid.Row.SS
        r_ca_scp = annot.RMA.INCA.R_CA_SCP
    except AttributeError:
        # Fallback to dictionary-based access if the input is a dict.
        num_cols = path_to_dict(annot, "ImageData.FullImage.NumCols")
        num_rows = path_to_dict(annot, "ImageData.FullImage.NumRows")
        scp_col = path_to_dict(annot, "ImageData.SCPPixel.Col")
        scp_row = path_to_dict(annot, "ImageData.SCPPixel.Row")
        time_ca_poly = path_to_dict(annot, "RMA.INCA.TimeCAPoly.Coefs")
        raw_time = path_to_dict(annot, "Timeline.CollectStart")
        collect_start = pd.to_datetime(raw_time, utc=True).tz_localize(None).to_numpy()  # ty:ignore[no-matching-overload]
        col_ss = path_to_dict(annot, "Grid.Col.SS")
        row_ss = path_to_dict(annot, "Grid.Row.SS")
        r_ca_scp = path_to_dict(annot, "RMA.INCA.R_CA_SCP")
    except Exception as e:
        logger.critical(f"Caught an unexpected error: {type(e).__name__} - {e}")
        azimuth_time, slant_range_time = None, None
        raise
    finally:
        # Calculate azimuth time based on the TimeCAPoly polynomial.
        if time_ca_poly is None:
            raise ValueError
        dtaz = time_ca_poly[1] * col_ss
        azimuth_time_float = (np.arange(cast(int, num_cols), dtype="float64") - scp_col) * dtaz + time_ca_poly[0]
        azimuth_time = (np.round(azimuth_time_float * 1e9)).astype("timedelta64[ns]") + collect_start
        # Calculate slant range time based on the distance from the scene center.
        slant_range_time = (
            2 * ((np.arange(cast(int, num_rows)) - cast(int, scp_row)) * row_ss + r_ca_scp) / speed_of_light
        )
    return azimuth_time, slant_range_time


def get_az_rg_meters(annot):
    """
    Compute azimuth and slant range distance arrays (in meters) from SICD metadata.
    """

    try:
        # Access metadata fields directly from the Sarpy object.
        daz = annot.Grid.Col.SS
        azimuth_m = (np.arange(annot.ImageData.FullImage.NumCols) - annot.ImageData.SCPPixel.Col) * daz
        slant_range_m = (
            np.arange(annot.ImageData.FullImage.NumRows) - annot.ImageData.SCPPixel.Row
        ) * annot.Grid.Row.SS
    except AttributeError:
        # Fallback to dictionary-based access.
        daz = path_to_dict(annot, "Grid.Col.SS")
        drg = path_to_dict(annot, "Grid.Row.SS")
        num_cols = cast(int, path_to_dict(annot, "ImageData.FullImage.NumCols"))
        scp_col = cast(int, path_to_dict(annot, "ImageData.SCPPixel.Col"))
        num_rows = cast(int, path_to_dict(annot, "ImageData.FullImage.NumRows"))
        scp_row = cast(int, path_to_dict(annot, "ImageData.SCPPixel.Row"))
        azimuth_m = (np.arange(num_cols) - scp_col) * daz
        slant_range_m = (np.arange(num_rows) - scp_row) * drg
    except Exception as e:
        logger.critical(f"Caught an unexpected error: {type(e).__name__} - {e}")
        azimuth_m, slant_range_m = None, None
        raise
    return azimuth_m, slant_range_m


def map_index_to_group(dt, swath, burstid, pol):
    """
    Find the corresponding group in a DataTree based on swath, burst ID, and polarization.
    """
    # Iterate through the DataTree nodes.
    for k, v in dt.items():
        # Check if the node's metadata matches the search criteria.
        if swath is not None and burstid is not None:
            cond = (
                (
                    v.other_metadata["general_annotation"]["downlink_information_list"][0]["swath"].upper()
                    == swath.upper()
                )
                and (v.other_metadata["burst_id"] == str(burstid))
                and (v.attrs["stac_discovery"]["properties"]["sar:polarization"] == pol)
            )
        else:
            cond = v.attrs["stac_discovery"]["properties"]["sar:polarization"] == pol

        if cond:
            return k
    # Raise an error if no match is found.
    raise Exception(f"No correspondence found for swath={swath} burst_id={burstid} pol={pol}")


def get_footprint(ds, index: int):
    """
    Return the footprint coordinates of an image.

    The footprint of the indexed element (e.g., a single burst) will be returned.

    Parameters
    ----------
    ds : sarpy.io.general.base.SICDTypeReader
        The input data opened by sarpy.io.
    index : int
        Index of the image element whose footprint has to be returned.

    Returns
    -------
    list
        The footprint coordinates.
    """

    # Extract the corner coordinates for all images from the metadata.
    corners = {
        i: [
            (
                ds.sicd_meta[i].GeoData.ImageCorners[j % 4].Lon,
                ds.sicd_meta[i].GeoData.ImageCorners[j % 4].Lat,
            )
            for j in range(5)
        ]
        for i in range(len(ds.sicd_meta))
    }

    #  return the footprint of that specific image.
    return corners[index]


def get_burst_id(annot) -> tuple[str, int]:
    """
    Computes the swath ID and burst absolute ID based on Sentinel-1 annotation and metadata.

    Parameters:
        annot: An annotation object containing metadata about the collection.

    Returns:
        tuple[str, int]: A tuple containing the swath ID (lowercase) and the burst absolute ID.
    """
    # Get the burst number and swath identifier from the metadata parameters.
    burst_id = int(annot.CollectionInfo.Parameters.get("BURST"))
    swath = annot.CollectionInfo.Parameters.get("SWATH").lower()
    return (swath, burst_id)


def get_burst_id_safe(file_sets: Sequence[dict], annot, relative_orbit_number: int) -> tuple[str, int]:
    """
    Computes the swath ID and burst absolute ID based on Sentinel-1 annotation and metadata.

    Parameters:
        file_sets (Sequence[dict]): A sequence of dictionaries representing file sets with metadata.
        annot: An annotation object containing metadata about the collection.
        relative_orbit_number (int): The relative orbit number for the current observation.

    Returns:
        tuple[str, int]: A tuple containing the swath ID (lowercase) and the burst absolute ID.
    """

    # Identify the corresponding product file
    for file_set in file_sets:
        burst = int(annot.CollectionInfo.Parameters.get("BURST")) - 1
        swath = annot.CollectionInfo.Parameters.get("SWATH").lower()
        polarization = annot.get_processed_polarization().lower().replace(":", "")

        product_file_name = file_set["product"]
        if f"-{swath}-" in os.path.basename(product_file_name) and f"-{polarization}-" in os.path.basename(
            product_file_name,
        ):
            break
    else:
        # If no matching product file is found
        product_file_name = None
    mode = swath[:-1]
    # Extract ANX (Ascending Node Crossing) time from the XML file
    root_node = sarpy.io.complex.sentinel._parse_xml(cast(str, product_file_name), without_ns=True)
    anx_time = float(root_node.findall("./swathTiming/burstList/burst")[burst].find("./azimuthAnxTime").text)  # type: ignore
    # Constants for timing calculations
    TBEAM = {"iw": 2.758273, "ew": 3.0388819516}  # Time per beam in seconds
    TORB = 12 * 86400 / 175  # Orbital period in seconds
    DELTA = {
        "iw1": -0.5469335,
        "iw2": 0.2852695,
        "iw3": 1.3633065,
        "ew1": -0.20158940051821037,
        "ew2": 0.4289565994970726,
        "ew3": 0.9952805994812225,
        "ew4": 1.5820385994932618,
        "ew5": 2.192150599480655,
    }  # Swath-specific time offsets
    t = (relative_orbit_number - 1) * TORB + anx_time - DELTA[swath]
    burst_id = int(np.round(t / TBEAM[mode]))
    if mode == "ew":
        burst_id += 11

    return (swath, burst_id)


def update_container(dt: xr.DataTree, dt_name, cslc_id):
    attrs = copy.deepcopy(next(iter(dt.children.values())).attrs)
    stac = attrs["stac_discovery"]
    processing_history = attrs["processing_history"]
    stac.pop("id", None)
    start = np.datetime64(stac["properties"]["start_datetime"])
    end = np.datetime64(stac["properties"]["end_datetime"])
    bbox = stac["bbox"]
    coords = stac["geometry"]["coordinates"][0][:-1]
    assets = {}
    for k, v in dt.items():
        stac_product = v.attrs["stac_discovery"]
        s = np.datetime64(stac_product["properties"]["start_datetime"])
        e = np.datetime64(stac_product["properties"]["end_datetime"])
        b = stac_product["bbox"]
        c = stac_product["geometry"]["coordinates"][0][:-1]
        bbox = [
            min(b[0], bbox[0]),
            min(b[1], bbox[1]),
            max(b[2], bbox[2]),
            max(b[3], bbox[3]),
        ]
        start = np.minimum(s, start)
        end = np.maximum(e, end)
        for i in range(4):
            i1, i2 = i // 2, i % 2
            lon_f = min if i1 == 0 else max
            lat_f = min if i2 == 0 else max
            coords[i] = [lon_f(coords[i][0], c[i][0]), lat_f(coords[i][1], c[i][1])]
        assets[k] = {"href": f"./{k}", "type": "application/vnd+zarr"}
    coords.append(coords[0])

    datetime = start + round((end.astype(int) - start.astype(int)) / 2)
    start = str(start) + "Z"
    end = str(end) + "Z"
    datetime = str(datetime) + "Z"
    stac["properties"].pop("proj:epsg", None)
    stac["properties"].pop("proj:wkt2", None)
    stac["properties"].pop("proj:shape", None)
    stac["properties"].pop("proj:transform", None)
    stac["properties"]["datetime"] = datetime
    stac["properties"]["start_datetime"] = start
    stac["properties"]["end_datetime"] = end
    stac["bbox"] = bbox
    stac["geometry"]["coordinates"] = [coords]
    instrument = stac["properties"]["sar:instrument_mode"]
    stac["properties"]["sar:beam_ids"] = beam_ids.get(instrument, [instrument])
    stac["assets"] = assets
    stac["stac_extensions"] = [i for i in stac["stac_extensions"] if i.split("/")[-3] not in ["view", "projection"]]
    id = get_id_product(stac)
    stac["id"] = id
    processing_history[f"COP ARD {dt_name} Product"][-1]["outputs"][0] = id

    return stac, processing_history


def update_gmlp(gmlp: xr.DataTree, slcs: xr.DataTree, external_metadata: dict = {}):
    """
    Finalize the GSLC product by adding product-level metadata.

    This function iterates through the processed bursts (or single scene),
    restructures the data into the final product format (e.g., creating
    `measurements`, `conditions`, `quality` groups), and assigns a comprehensive
    set of attributes to conform to the GSLC product specification.

    Parameters
    ----------
    gslcs : xr.DataTree
        The DataTree containing all the processed geocoded bursts.
    external_metadata : dict, optional
        A dictionary of external metadata to be included in
        the final product attributes (e.g., processing center, history).

    Returns
    -------
    xr.DataTree
        A new DataTree, restructured and fully populated with final
        GSLC product metadata.
    """
    processing_start_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    bursts = xr.DataTree()
    multiburst = True if "bursts" in gmlp else False
    # --- Iterate Over Each Burst/Scene ---
    burst_items = gmlp["bursts"].children.items() if multiburst else [("data", gmlp)]

    for burst_id, burst_node in burst_items:
        slc = slcs.bursts[burst_id] if multiburst else slcs
        measurements = slc.coreg.measurements
        dates_ids = {k: v for k, v in zip(measurements.acquisition_date.values, measurements.acquisition_id.values)}
        dates = list(burst_node.children)
        logger.debug(f"Burst '{burst_id}' dates: {dates}")

        if len(dates) != 1:
            raise ValueError(
                f"Expected exactly 1 acquisition date for burst '{burst_id}', but found {len(dates)}: {dates}"
            )
        acquisition_date = dates[0]
        acquisition_id = dates_ids[acquisition_date]
        gslc = burst_node[acquisition_date]
        stac, processing_history = get_stac_and_history_gslc(
            burst_id,
            gslc.measurements.to_dataset(),
            slcs,
            processing_start_time=processing_start_time,
            external_metadata=external_metadata,
        )
        other_metadata = get_other_metadata_gslc(external_metadata)
        bursts[stac["id"]] = clean_gslc(gslc, slc, acquisition_id)
        bursts[stac["id"]].attrs = {
            "stac_discovery": stac,
            "processing_history": processing_history,
            "other_metadata": other_metadata,
        }
    stac_group, history_group = update_container(bursts, "GMLP", slcs.attrs["stac_discovery"]["id"])
    # Apply any final, global attribute fixes.
    fix_attributes(bursts, ATTRS_GEO)
    bursts.attrs = {"stac_discovery": stac_group, "processing_history": history_group}
    return bursts


def fix_attributes(data: Union[xr.DataTree, xr.DataArray], attrs):
    """
    Recursively clean up and standardize attributes in a DataTree or DataArray.

    This utility function applies a standard set of attributes to dimension
    coordinates, removes unwanted attributes and encoding keys from datasets,
    and standardizes the attributes of data variables based on a provided template.
    It operates recursively on DataTree objects.

    Parameters
    ----------
    data : Union[xr.DataTree, xr.DataArray]
        The xarray.DataTree or xarray.DataArray to be cleaned.
    attrsfix_attributes
        A dictionary template where keys are dimension or variable names
        and values are the standard attributes to be assigned.

    Returns
    -------
    xr.DataTree or xr.DataArray
        The modified `data` object with cleaned and standardized attributes.
    """
    # --- Standardize Dimension Attributes ---
    # Iterate through all dimensions in the data object.
    for k in data.dims:
        if not isinstance(k, str):
            raise Exception(f"Expected str, got {type(k)}")
        # If the dimension name is in the attribute template, apply the standard attributes.
        if k in attrs:
            data[k].attrs = attrs[k]
            data[k].attrs.update({"short_name": k})

    # --- Clean Top-Level Attributes and Encoding ---
    # Remove any unwanted attributes from the top level of the object.
    for k in DEL_ATTRS:
        if k in data.attrs:
            del data.attrs[k]
    # Remove any unwanted keys from the object's encoding.
    for k in DEL_ENCODING:
        if k in data.encoding:
            del data.encoding[k]

    # --- Standardize Data Variable Attributes (if data is DataArray) ---
    if isinstance(data, xr.DataArray) and data.name in attrs:
        if not isinstance(data.name, str):
            raise Exception(f"Expected str, got {type(data.name)}")
        data.attrs.update({**attrs[data.name], "short_name": data.name})

    # --- Recurse for DataTree ---
    if isinstance(data, xr.DataTree):
        # 1. Recurse over DataVariables within the current node
        for var_name, var_array in data.data_vars.items():
            fix_attributes(var_array, attrs)

        # 2. Recurse over Child Nodes (Groups) in the DataTree hierarchy
        for child_name, child_node in data.children.items():
            fix_attributes(child_node, attrs)

    return data


def clean_gslc(gslc, slc, acquisition_id):
    # --- Restructure the DataTree for Final Product Format ---
    gslc_fix = xr.DataTree()
    orbit = slc.coreg.conditions.orbit.to_dataset().drop_vars(
        ["orbit_files", "reference_time", "type"], errors="ignore"
    )
    aq_size = gslc.measurements.acquisition_date.size

    mask = gslc.measurements.mask.to_dataset(name="distortion_mask").drop_vars(["acquisition_date", "acquisition_id"])
    look_vector = gslc.conditions.look_vector
    acquisition_dates = gslc.measurements.acquisition_date.values
    measurements = (
        gslc.measurements.to_dataset()
        .drop_vars([k for k in gslc.measurements.data_vars if "phase" not in k])
        .expand_dims(acquisition_id=1)
        .transpose("polarization", "acquisition_id", "y", "x")
    )
    measurements = measurements.assign_coords(
        acquisition_date=("acquisition_id", np.atleast_1d(measurements.acquisition_date.values))
    )
    coherence = (
        gslc.measurements.to_dataset()
        .drop_vars([k for k in gslc.measurements.data_vars if "coh" not in k])
        .expand_dims(acquisition_id=1)
        .transpose("polarization", "acquisition_id", "y", "x")
    )
    coherence = coherence.assign_coords(
        acquisition_date=("acquisition_id", np.atleast_1d(measurements.acquisition_date.values))
    )

    if "etad" in gslc.conditions:
        etad = gslc.conditions.etad.to_dataset()
        etad = add_acquisition_date(etad, measurements)
        gslc_fix["conditions/ETAD"] = etad

    gslc_fix["conditions/look_vector"] = look_vector
    gslc_fix["conditions/orbit"] = add_acquisition_date(orbit, measurements)
    gslc_fix["measurements"] = measurements
    gslc_fix["quality/distortion_mask"] = mask
    gslc_fix["quality/coherence"] = coherence

    return gslc_fix


def get_other_metadata_gslc(external_metadata):
    attrs = {
        "product_data_access": external_metadata[-1]["execution_parameters"]["product_data_access"],
        "product_geometry": "projected",
        "product_pixel_coordinate_convention": external_metadata[-1]["execution_parameters"][
            "product_pixel_coordinate_convention"
        ],
        "noise_correction_flag": external_metadata[0]["execution_parameters"]["noise_removal"],
        "gmlp_status": external_metadata[-1]["execution_parameters"]["gmlp_status"],
        "product_family_specification": external_metadata[-1]["execution_parameters"]["product_family_specification"],
        "measurement_type": external_metadata[0]["execution_parameters"]["measurement_type"],
        "product_resampling_method": "sinc interpolator",
        # "product_filtering": external_metadata[-1]["execution_parameters"]["product_filtering"],
        # "eastern_geometric_accuracy": external_metadata[-1]["execution_parameters"]["eastern_geometric_accuracy"],
        # "northern_geometric_accuracy": external_metadata[-1]["execution_parameters"]["northern_geometric_accuracy"],
        # "geometric_accuracy_radial_rmse": external_metadata[-1]["execution_parameters"][
        #     "geometric_accuracy_radial_rmse"
        # ],
    }
    return attrs


def get_stac_and_history_gslc(burst_id, gslc, slcs, processing_start_time, external_metadata):
    multiburst = True if "bursts" in slcs else False
    attrs = copy.deepcopy((slcs.bursts[burst_id] if multiburst else slcs)).attrs
    stac = attrs["stac_discovery"]
    crs = CRS(gslc.rio.crs).to_3d()
    horizontal_crs = str(crs.to_2d().to_epsg())
    vertical_crs = extract_vertical_crs(crs)
    vertical_crs = f"+{vertical_crs}" if crs.is_vertical else ""
    product_wkt = crs.to_wkt()
    product_epsg = f"{horizontal_crs}{vertical_crs}" if vertical_crs != "" else horizontal_crs
    lines_number, pixels_number = gslc.y.size, gslc.x.size
    pixel_spacing, line_spacing = gslc.rio.resolution()
    gsd = max(pixel_spacing, line_spacing)
    bbox = [round(i, 6) for i in gslc.rio.transform_bounds("EPSG:4326")]
    coordinates = [
        [bbox[0], bbox[1]],
        [bbox[0], bbox[3]],
        [bbox[2], bbox[3]],
        [bbox[2], bbox[1]],
        [bbox[0], bbox[1]],
    ]
    stac.pop("id")
    stac["properties"].pop("sat:anx_datetime", None)
    stac["properties"].pop("sar:pixel_spacing_azimuth", None)
    stac["properties"].pop("sar:pixel_spacing_range", None)
    stac["properties"].pop("view:incidence_angle", None)
    mode = stac["properties"]["sar:instrument_mode"]
    stac["stac_extensions"].append("https://stac-extensions.github.io/projection/v1.1.0/schema.json")
    stac["bbox"] = bbox
    stac["geometry"]["coordinates"] = [coordinates]
    stac["properties"]["title"] = "COP ARD GMLP Product"
    stac["properties"]["created"] = processing_start_time
    stac["properties"]["processing:software"] = {
        "COP ARD CORE": f"{cop_ard_core.__version__}",
        "COP ARD APP": f"{cop_ard_app.__version__}",
    }
    stac["properties"]["processing:version"] = f"{cop_ard_app.__version__}"
    stac["properties"]["processing:datetime"] = processing_start_time
    stac["properties"]["product:type"] = f"S01S{mode}GMP"
    stac["properties"]["gsd"] = gsd
    stac["properties"]["proj:epsg"] = product_epsg
    stac["properties"]["proj:wkt2"] = product_wkt
    stac["properties"]["proj:shape"] = [pixels_number, lines_number]
    stac["properties"]["proj:transform"] = list(gslc.rio.transform())[:6]

    stac["links"] = [{"href": "./zarr.json", "rel": "self", "type": "application/json"}]
    stac["assets"] = {
        "measurements": {
            "href": "./measurements",
            "title": "GSLC Measurements",
            "description": "Zarr group containing the geocoded SAR image data",
            "type": "application/vnd+zarr",
            "roles": ["data"],
        },
        "conditions": {
            "href": "./conditions",
            "title": "GSLC Conditions",
            "description": "Zarr group containing measurement conditions",
            "type": "application/vnd+zarr",
            "roles": ["metadata"],
        },
        "quality": {
            "href": "./quality",
            "title": "GSLC Quality Metrics",
            "description": "Zarr group containing quality assessment and accuracy metrics",
            "type": "application/vnd+zarr",
            "roles": ["qa", "metadata"],
        },
    }
    id = get_id_product(stac, burst_id=burst_id)
    stac["id"] = id

    processing_history = {"COP ARD GMLP Product": external_metadata}
    processing_history["COP ARD GMLP Product"][-1]["outputs"][0] = id

    return stac, processing_history


def stac_discovery_template(burst_id, orig, coreg, external_metadata, stac, processing_start_time):
    sicd = orig.attrs[list(orig.attrs)[0]]
    sicd_coreg = coreg.attrs[list(coreg.attrs)[0]]
    tx_frequency = sicd["RadarCollection"]["TxFrequency"]
    platform, constellation, instruments = extract_stac_sar_props(sicd)

    band = get_rf_band(tx_frequency["Min"], tx_frequency["Max"])
    coordinates = []
    for v in sicd["GeoData"]["ImageCorners"]:
        coordinates.append([round(v["Lon"], 6), round(v["Lat"], 6)])
    coordinates.append(coordinates[0])
    bbox = np.array(coordinates)
    bbox = [
        float(min(bbox[:, 0])),
        float(min(bbox[:, 1])),
        float(max(bbox[:, 0])),
        float(max(bbox[:, 1])),
    ]

    s = orig.azimuth_time.min().values.astype("datetime64[us]").item().isoformat() + "Z"
    e = orig.azimuth_time.max().values.astype("datetime64[us]").item().isoformat() + "Z"
    m = orig.azimuth_time[orig.azimuth_time.size // 2].values.astype("datetime64[us]").item().isoformat() + "Z"
    mode = sicd["CollectionInfo"]["RadarMode"]["ModeID"]
    pol = ["".join(i.split(":")) for i in orig.polarization.values]
    if "properties" in stac:
        datatake_id = stac["properties"]["eopf:datatake_id"]
        absolute_orbit = int(stac["properties"]["sat:absolute_orbit"])
        relative_orbit = int(stac["properties"]["sat:relative_orbit"])
        orbit_state = stac["properties"]["sat:orbit_state"]
    else:
        datatake_id = sicd["CollectionInfo"]["CoreName"][10:16]
        absolute_orbit = int(sicd["CollectionInfo"]["Parameters"].get("ABSOLUTE_ORBIT", 0))
        relative_orbit = int(sicd["CollectionInfo"]["Parameters"].get("RELATIVE_ORBIT", 0))
        orbit_state = "ascending" if sicd["SCPCOA"]["ARPVel"]["Z"] > 0 else "descending"

    stac_payload = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/eopf/v1.2.0/schema.json",
            "https://stac-extensions.github.io/product/v0.1.0/schema.json",
            "https://stac-extensions.github.io/sat/v1.1.0/schema.json",
            "https://stac-extensions.github.io/view/v1.0.0/schema.json",
            "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
            "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
            "https://stac-extensions.github.io/sar/v1.2.0/schema.json",
        ],
        "bbox": bbox,
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        "properties": {
            "title": "COP-ARD CSLC Product",
            "datetime": m,
            "start_datetime": s,
            "end_datetime": e,
            "created": processing_start_time,
            "constellation": constellation,
            "instruments": instruments,
            "platform": platform,
            "eopf:datatake_id": datatake_id,
            "eopf:instrument_mode": mode,
            "processing:lineage": "systematic",
            "processing:software": {
                "COP-ARD-CORE": f"{cop_ard_core.__version__}",
                "COP-ARD-APP": f"{cop_ard_app.__version__}",
            },
            "processing:version": f"{cop_ard_app.__version__}",
            "processing:datetime": processing_start_time,
            "processing:facility": external_metadata[-1]["execution_parameters"]["facility"],
            "product:timeliness": external_metadata[-1]["execution_parameters"]["product_timeliness"],
            "product:timeliness_category": external_metadata[-1]["execution_parameters"]["product_timeliness_category"],
            "product:type": f"S01S{mode}CSL",
            "providers": external_metadata[-1]["execution_parameters"]["providers"],
            "sar:beam_ids": [sicd["CollectionInfo"]["Parameters"]["SWATH"].upper()],
            "sar:center_frequency": sicd["RMA"]["INCA"]["FreqZero"],
            "sar:frequency_band": band,
            "sar:instrument_mode": mode,
            "sar:pixel_spacing_azimuth": round(sicd_coreg["Grid"]["Col"]["SS"], 5),
            "sar:pixel_spacing_range": round(sicd_coreg["Grid"]["Row"]["SS"], 5),
            "sar:polarizations": pol,
            "sat:absolute_orbit": absolute_orbit,
            "sat:orbit_state": orbit_state,
            "sat:relative_orbit": relative_orbit,
            "view:incidence_angle": sicd["SCPCOA"]["IncidenceAng"],
            "sci:citation": external_metadata[-1]["execution_parameters"]["atbd_link"],
        },
        "links": [{"href": "./zarr.json", "rel": "self", "type": "application/json"}],
        "assets": {
            "measurements": {
                "href": "./measurements",
                "title": "CSLC Measurements",
                "description": "Zarr group containing the complex SAR image data",
                "type": "application/vnd+zarr",
                "roles": ["data"],
            },
            "conditions": {
                "href": "./conditions",
                "title": "CSLC Conditions",
                "description": "Zarr group containing measurement conditions",
                "type": "application/vnd+zarr",
                "roles": ["metadata"],
            },
            "quality": {
                "href": "./quality",
                "title": "CSLC Quality Metrics",
                "description": "Zarr group containing quality assessment and accuracy metrics",
                "type": "application/vnd+zarr",
                "roles": ["qa", "metadata"],
            },
        },
    }

    id = get_id_product(stac_payload, burst_id)
    stac_payload["id"] = id
    return stac_payload


def get_stac_cslc(slc: xr.DataTree, burst_id: str, external_metadata, processing_start_time):
    reference_id = slc.coreg.measurements.attrs["coregistration_reference_id"]
    acquisition_dates = slc.coreg.measurements.cstack.acquisition_date.values
    acquisition_ids = slc.coreg.measurements.cstack.acquisition_id.values
    date = acquisition_ids[0] if acquisition_ids.size == 1 else reference_id
    orig = slc.orig[date].measurements.slc
    coreg = slc.coreg.measurements.cstack
    stac = slc.orig[date].conditions.attrs.get("stac_discovery", {})
    stac_discovery = stac_discovery_template(burst_id, orig, coreg, external_metadata, stac, processing_start_time)
    return stac_discovery


def get_processing_history_cslc(slc: xr.DataTree, stac_discovery, external_metadata, processing_start_time):
    reference_id = slc.coreg.measurements.attrs["coregistration_reference_id"]
    acquisition_ids = slc.coreg.measurements.cstack.acquisition_id.values
    date = acquisition_ids[0] if acquisition_ids.size == 1 else reference_id
    processing_history = slc.orig[date].conditions.attrs.get("processing_history", {})
    stac = slc.orig[date].conditions.attrs.get("stac_discovery", {})
    inputs = stac["id"] if "id" in stac else "n/a"
    if "Level-1 Product" in processing_history:
        if processing_history["Level-1 Product"][0]["outputs"]:
            inputs = processing_history["Level-1 Product"][0]["outputs"]

    processing_history["COP ARD CSLC Product"] = external_metadata
    return processing_history


def get_other_metadata_cslc(slc: xr.DataTree, external_metadata):
    reference_id = slc.coreg.measurements.attrs["coregistration_reference_id"]
    acquisition_ids = slc.coreg.measurements.cstack.acquisition_id.values
    date = acquisition_ids[0] if acquisition_ids.size == 1 else reference_id

    other_metadata = {}
    other_metadata["SICD"] = slc.orig[date].measurements.slc.attrs
    other_metadata["product_geometry"] = "slant-range"
    other_metadata["product_pixel_coordinate_convention"] = external_metadata[-1]["execution_parameters"][
        "product_pixel_coordinate_convention"
    ]
    other_metadata["cslc_status"] = external_metadata[-1]["execution_parameters"]["cslc_status"]
    other_metadata["product_family_specification"] = external_metadata[-1]["execution_parameters"][
        "product_family_specification"
    ]
    other_metadata["measurement_type"] = external_metadata[0]["execution_parameters"]["measurement_type"]
    other_metadata["noise_correction_flag"] = external_metadata[0]["execution_parameters"]["noise_removal"]
    return other_metadata


def update_coreg(
    slcs: xr.DataTree,
    wdir_path_static: str,
    reference_id: str,
    external_metadata: dict = {},
    **kwargs,
):
    """
    Finalize the coregistered DataTree by adding product-level metadata.

    This function iterates through the processed bursts, restructures the data
    (e.g., renaming 'cstack' to 'cslc'), and assigns a comprehensive set of
    attributes to conform to the final product specification.

    Parameters
    ----------
    slcs
        The DataTree containing all the processed coregistered bursts.
    external_metadata : dict
        A dictionary of external metadata to be included in
        the final product attributes.

    Returns
    -------
    DataTree
        A new DataTree, restructured and fully populated with final metadata.
    """
    processing_start_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    bursts = xr.DataTree()
    multiburst = True if "bursts" in slcs else False
    # --- Iterate Over Each Burst/Scene ---
    for k, slc in (slcs.bursts if multiburst else {"data": slcs}).items():
        name = k if multiburst else "data"
        path = f"bursts/{k}" if multiburst else f"data/{reference_id}"
        static_data = read(
            f"{wdir_path_static}/{path}/radar_grid",
            as_type="xr.Dataset",
            storage_options=get_storage_options(kwargs),
        )
        mask = static_data.layover.astype(np.int8) + 2 * static_data.shadow.astype(np.int8)

        stac_discovery = get_stac_cslc(
            slc,
            name,
            external_metadata=external_metadata,
            processing_start_time=processing_start_time,
        )
        processing_history = get_processing_history_cslc(
            slc,
            stac_discovery,
            external_metadata=external_metadata,
            processing_start_time=processing_start_time,
        )
        other_metadata = get_other_metadata_cslc(slc, external_metadata)

        cleaned_item = clean_cslc(slc, mask)
        cleaned_item.attrs.update(
            {
                "stac_discovery": stac_discovery.copy(),
                "other_metadata": other_metadata.copy(),
                "processing_history": processing_history.copy(),
            }
        )
        bursts[stac_discovery["id"]] = cleaned_item

        slc.attrs.update(
            {
                "stac_discovery": stac_discovery.copy(),
                "other_metadata": other_metadata.copy(),
                "processing_history": processing_history.copy(),
            }
        )

    stac_group, history_group = update_cslc_container(bursts)

    # Apply any final attribute fixes.
    fix_attributes(bursts, ATTRS_RADAR)

    bursts.attrs.update(
        {
            "stac_discovery": stac_group,
            "processing_history": history_group,
        }
    )
    slcs.attrs.update(
        {
            "stac_discovery": stac_group,
            "processing_history": history_group,
        }
    )

    return bursts, slcs


def get_coefficients(path: str, annot: dict) -> np.ndarray | None:
    """
    Utility function to extract polynomial coefficients from the annotation dictionary.

    Args:
        annot: The annotation dictionary containing the metadata.
        path: The dot-separated path to the desired coefficients (e.g., "Radiometric.SigmaZeroSFPoly.Coefs").
    Returns:
        A numpy array of coefficients if found, otherwise None.
    """
    try:
        coeffs = path_to_dict(annot, path)
        if coeffs is not None:
            return np.array(coeffs)
    except KeyError:
        return None
    return None


def get_carrier_frequency(slcs: xr.DataTree, path: str, multiburst: bool = True) -> float:
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
    for burst in (slcs.bursts if multiburst else {"data": slcs}).values():
        _, annot = next(iter(burst[path].attrs.items()))
        f0.append((annot["RadarCollection"]["TxFrequency"]["Max"] + annot["RadarCollection"]["TxFrequency"]["Min"]) / 2)

    if len(f0) <= 0:
        raise ValueError("Null slc in input")

    return np.mean(f0)


def clean_cslc(slc: xr.DataTree, mask: xr.DataArray):
    # --- Restructure the DataTree ---
    # Rename 'cstack' to 'cslc' and drop unnecessary coordinates.
    cslc = xr.DataTree()

    measurements = (
        slc.coreg.measurements.to_dataset().rename({"cstack": "cslc"}).drop_vars(["slant_range_m", "azimuth_m"])
    ).drop_attrs()
    quality = slc.coreg.quality.to_dataset()
    renames = {"RCS": "rcs", "crg_accuracy": "coregistration_accuracy"}
    quality = quality.rename({k: v for k, v in renames.items() if k in quality.variables})
    orbit = slc.coreg.conditions.orbit.to_dataset()
    phase_demod = slc.coreg.conditions.phase_demod.to_dataset()
    look_vector = slc.coreg.conditions.look_vector.to_dataset()

    quality = add_acquisition_date(quality, measurements)
    phase_demod = add_acquisition_date(phase_demod, measurements)
    orbit = orbit.drop_vars(["orbit_files", "reference_time", "type"], errors="ignore")
    orbit = add_acquisition_date(orbit, measurements)
    if "ETAD" in slc.coreg.conditions:
        etad = slc.coreg.conditions.ETAD.to_dataset()
        etad = add_acquisition_date(etad, measurements)
        cslc["conditions/ETAD"] = etad
    if "side_band" in slc.coreg.conditions:
        side_band = slc.coreg.conditions.side_band.to_dataset()
        side_band = add_acquisition_date(side_band, measurements)
        cslc["conditions/side_band"] = side_band

    cslc["measurements"] = measurements.drop_vars(["daz", "drg"])
    cslc["conditions/orbit"] = orbit
    cslc["conditions/look_vector"] = look_vector
    cslc["conditions/phase_demodulation"] = phase_demod
    cslc["quality/radar_qlt"] = quality
    cslc["quality/distortion_mask"] = mask.to_dataset(name="distortion_mask")

    return cslc


def add_acquisition_date(ds: xr.Dataset | None, ref_measurements: xr.Dataset) -> xr.Dataset | None:
    """
    Attach 'acquisition_date' as a non-dimension coordinate dependent on 'acquisition_id'.
    """
    if ds is None or "acquisition_id" not in ds.dims:
        return ds

    dates = ref_measurements.acquisition_date.sel(acquisition_id=ds.acquisition_id.values)

    return ds.assign_coords(acquisition_date=("acquisition_id", dates.values))


def update_cslc_container(bursts: xr.DataTree):
    attrs = copy.deepcopy(next(iter(bursts.children.values())).attrs)
    stac = attrs["stac_discovery"]
    processing_history = attrs["processing_history"]
    stac["properties"].pop("sat:anx_datetime", None)
    stac["properties"].pop("sar:pixel_spacing_azimuth", None)
    stac["properties"].pop("sar:pixel_spacing_range", None)
    stac["properties"].pop("view:incidence_angle", None)
    stac.pop("id", None)
    start = np.datetime64(stac["properties"]["start_datetime"])
    end = np.datetime64(stac["properties"]["end_datetime"])
    bbox = stac["bbox"]
    coords = stac["geometry"]["coordinates"][0][:-1]
    assets = {}
    for k, v in bursts.items():
        stac_product = v.attrs["stac_discovery"]
        s = np.datetime64(stac_product["properties"]["start_datetime"])
        e = np.datetime64(stac_product["properties"]["end_datetime"])
        b = stac_product["bbox"]
        c = stac_product["geometry"]["coordinates"][0][:-1]
        bbox = [
            min(b[0], bbox[0]),
            min(b[1], bbox[1]),
            max(b[2], bbox[2]),
            max(b[3], bbox[3]),
        ]
        start = np.minimum(s, start)
        end = np.maximum(e, end)
        for i in range(4):
            i1, i2 = i // 2, i % 2
            lon_f = min if i1 == 0 else max
            lat_f = min if i2 == 0 else max
            coords[i] = [lon_f(coords[i][0], c[i][0]), lat_f(coords[i][1], c[i][1])]
        assets[k] = {"href": f"./{k}", "type": "application/vnd+zarr"}
    coords.append(coords[0])

    datetime = start + round((end.astype(int) - start.astype(int)) / 2)
    start = str(start) + "Z"
    end = str(end) + "Z"
    datetime = str(datetime) + "Z"
    stac["properties"]["datetime"] = datetime
    stac["properties"]["start_datetime"] = start
    stac["properties"]["end_datetime"] = end
    stac["bbox"] = bbox
    stac["geometry"]["coordinates"] = [coords]
    try:
        instrument_mode = stac["properties"]["sar:instrument_mode"]
        logger.critical(f"instrument mode: {instrument_mode}")
        logger.critical(f"beam_ids: {beam_ids[instrument_mode]}")
        stac["properties"]["sar:beam_ids"] = beam_ids[instrument_mode]
    except (KeyError, TypeError) as err:
        logger.warning(f"Failed to set 'sar:beam_ids': missing key or invalid mapping ({err})")
        stac["properties"]["sar:beam_ids"] = []
    stac["assets"] = assets
    id = get_id_product(stac)
    stac["id"] = id
    processing_history["COP ARD CSLC Product"][-1]["outputs"][0] = id

    return stac, processing_history


def get_rf_band(tx_frequency_min, tx_frequency_max):
    f_min = float(tx_frequency_min)  # Hz
    f_max = float(tx_frequency_max)  # Hz

    freq_hz = (f_min + f_max) / 2
    freq_ghz = freq_hz / 1e9
    if 0.3 <= freq_ghz < 1.0:
        return "P"
    elif 1.0 <= freq_ghz < 2.0:
        return "L"
    elif 2.0 <= freq_ghz < 4.0:
        return "S"
    elif 4.0 <= freq_ghz < 8.0:
        return "C"
    elif 8.0 <= freq_ghz < 12.0:
        return "X"
    return "Unknown"


FAMILIES_MAP = {
    "S1": ("sentinel-1", "sentinel-1"),
    "SENTINEL-1": ("sentinel-1", "sentinel-1"),
    "SAOCOM": ("saocom-1", "saocom"),
    "ROSE": ("rose-l", "rose-l"),
    "CSK": ("cosmo-skymed", "csk"),
    "CSG": ("cosmo-skymed", "csg"),
    "TSX": ("terrasar-x", "tsx"),
    "TDX": ("terrasar-x", "tdx"),
    "PAZ": ("paz", "paz"),
    "BIOMASS": ("biomass", "biomass"),
}


def extract_stac_sar_props(sicd):
    raw_collector = sicd["CollectionInfo"]["CollectorName"].strip().upper()

    # 1. PLATFORM & CONSTELLATION
    constellation = None
    platform = None

    for prefix, (const_name, plat_prefix) in FAMILIES_MAP.items():
        if raw_collector.startswith(prefix):
            constellation = const_name
            suffix = raw_collector[len(prefix) :].replace("_", "").replace("-", "").lower()
            platform = f"{plat_prefix}{suffix}" if suffix else plat_prefix
            break

    if not constellation:
        platform = re.sub(r"[\s_]+", "-", raw_collector.lower())
        constellation = re.sub(r"[-_][a-z0-9]$", "", platform)

    instruments = ["sar"]
    try:
        tx_freq = sicd["RadarCollection"]["TxFrequency"]
        f_center_ghz = (float(tx_freq["Min"]) + float(tx_freq["Max"])) / 2 / 1e9

        if 0.3 <= f_center_ghz < 1.0:
            instruments = ["p-sar"]
        elif 1.0 <= f_center_ghz < 2.0:
            instruments = ["l-sar"]
        elif 2.0 <= f_center_ghz < 4.0:
            instruments = ["s-sar"]
        elif 4.0 <= f_center_ghz < 8.0:
            instruments = ["c-sar"]
        elif 8.0 <= f_center_ghz < 12.0:
            instruments = ["x-sar"]
    except (KeyError, TypeError, ValueError):
        pass

    return platform, constellation, instruments
