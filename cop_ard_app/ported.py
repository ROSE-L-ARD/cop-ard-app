from cop_ard_core.interf import extract_interf_single
from cop_ard_core.split_spectrum import remove_iono_surface_single_burst, split_spectrum, fit_ionosphere
from functools import reduce
from cop_ard_core.computing.kernels import update_dense_array
import pyproj
import rasterio
import copy
from xml.etree.ElementTree import Element, ElementTree
import defusedxml.ElementTree as ET
import itertools
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence, Union, cast
from unittest import mock

import botocore.exceptions
import dask.array as da
import dask.distributed
import dateutil
import numpy as np
import sarpy.io
import sarpy.io.general.base
import xarray as xr
from pyproj import CRS, Transformer, datadir
from pyproj.transformer import TransformerGroup
import pyproj.network
from scipy.constants import speed_of_light

import cop_ard_core
import cop_ard_core.utils
import cop_ard_core.esd
from cop_ard_core.computing.area_projection import (
    compute_geocoded_mask,
    compute_rasterization,
    compute_shadow,
    compute_rtc_ap,
)
from cop_ard_core.computing.computation import map, map_blocks
from cop_ard_core.computing.dem import (
    check_available_transformation,
    compute_shape_and_off,
    extract_vertical_crs,
    get_default_epsg,
    get_resolution_in_meters,
    interp_dem,
)
from cop_ard_core.computing.griddata import init_interpolate
from cop_ard_core.computing.io import (
    Path,
    delete,
    exists,
    generate_empty_dataset,
    get_id_product,
    get_storage_options,
    glob,
    is_s3_path,
    makedirs,
    normpath,
    path_to_dict,
    read,
    save_empty_dataset,
    touch,
    write,
)
from cop_ard_core.computing.logger import logger, timeit
from cop_ard_core.computing.orbit import build_orbit_dataset, refit_orbit_coefficients
from cop_ard_core.computing.simul import (
    SPATIAL_DIMS_,
    compute_simulation,
    extract_simul_on_radar_chunk,
    get_geo_from_radar_bbox,
    get_radar_bbox,
    init_simulation,
    extract_simulation_ml,
)
from cop_ard_core.computing.slc import (
    amplitude_calibration,
    apply_invalid_mask,
    get_relative_orbit_number,
    image_deramp,
    intersect,
    merge_dict,
    noise_removal,
)
from cop_ard_core.computing.warp import coreg_qlt, sinc, warp_cslc, warp_dazdrg, warp_gslc, getLUT, GRID


def encode_poly(v_poly: xr.Dataset, name: str, stack: xr.Dataset) -> xr.Dataset:
    """
    Return a version of input v_poly object so that it can be saved.

    Merges separate variables in v_poly into a single variable aligned
    with stack.acquisition_date.
    """

    def _safe_encode(obj):
        """Recursively encode attributes for serialization."""
        if isinstance(obj, (list, tuple)):
            return [_safe_encode(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _safe_encode(v) for k, v in obj.items()}
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()  # Convert numpy scalars to python scalars
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        else:
            return obj

    # 1. Extract attributes BEFORE transforming the dataset
    # We capture attributes from variables that match the stack's dates
    poly_attrs = {}

    # We convert stack dates to the same type as v_poly keys (usually strings) for checking
    stack_date_lookup = set(stack.acquisition_date.astype(str).values)

    for var_name, data_array in v_poly.data_vars.items():
        # Check if this variable name corresponds to a date in the stack
        if str(var_name) in stack_date_lookup:
            poly_attrs[str(var_name)] = _safe_encode(data_array.attrs)

    # 2. Transform: Convert variables (dates) into a dimension
    da_poly = v_poly.to_array(dim="acquisition_date", name=name)

    # 3. Ensure Type Consistency (Crucial step)
    # to_array usually results in string/object coords. Cast to datetime to match stack.
    try:
        da_poly["acquisition_date"] = da_poly.acquisition_date.astype("datetime64[ns]")
    except ValueError:
        # Fallback if keys weren't date strings
        pass

    # 4. Filter and Assign Attributes
    da_poly = (
        da_poly.sel(acquisition_date=stack.acquisition_date)  # Filter dates
        .assign_attrs(poly_attrs)  # Add the attrs we extracted earlier
        .assign_attrs(_safe_encode(stack.attrs))  # Add stack global attrs
    )

    # 5. Convert back to Dataset
    ds_out = da_poly.to_dataset()

    # 6. Chunking
    # We use .get() to avoid KeyErrors if dims are missing or not chunked
    chunk_plan = {"acquisition_date": stack.chunksizes.get("acquisition_date", "auto"), "i": 1, "j": 1}

    # Only chunk dimensions that actually exist in the output
    final_chunks = {k: v for k, v in chunk_plan.items() if k in ds_out.dims}

    return ds_out.chunk(final_chunks)


def decode_poly(v_poly: xr.Dataset, name: str, stack: xr.Dataset) -> xr.Dataset:
    """
    Read and reformat x_ or y_poly from the slcs structure.

    Splits the 'acquisition_date' dimension back into separate variables.
    """

    def _safe_decode(obj):
        """Recursively decode attributes, handling strings to datetime/timedelta."""
        if isinstance(obj, dict):
            return {k: _safe_decode(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_safe_decode(x) for x in obj]
        elif isinstance(obj, str):
            # 1. Try Timedelta (specific format from encode)
            if "nanoseconds" in obj:
                try:
                    # Extracts "12345" from "12345 nanoseconds"
                    val = float(obj.split()[0])
                    return np.timedelta64(int(val), "ns")
                except (ValueError, IndexError):
                    pass

            # 2. Try Datetime
            # Only attempt conversion if it looks like a date (ISO-ish) to avoid false positives
            if "-" in obj and ":" in obj:
                try:
                    return np.datetime64(obj)
                except ValueError:
                    pass

            # Return original string if no conversion applied
            return obj

        # Return numbers/arrays as is
        return obj

    # Dictionary to hold the new variables
    data_vars = {}

    # We iterate over the dates in the stack
    # We assume stack["acquisition_date"] exists
    dates = stack["acquisition_date"].values

    for date in dates:
        # Convert date to string to find the matching attribute key
        # (Assuming encode_poly saved keys as strings)
        date_key = str(date)

        # 1. Select data lazily
        # drop=True removes 'acquisition_date' so it doesn't persist as a scalar coord
        da_slice = v_poly[name].sel(acquisition_date=date, drop=True)

        # 2. Decode specific attributes for this date
        # We use .get() in case the attribute is missing
        raw_attrs = da_slice.attrs.get(date_key, {})
        da_slice.attrs = _safe_decode(raw_attrs)

        # 3. Add to dictionary
        # We use the string representation of the date as the variable name
        data_vars[date_key] = da_slice

    # Create the final dataset
    decoded_ds = xr.Dataset(data_vars)

    # Preserve the specific 'type' attribute from the input
    decoded_ds = decoded_ds.assign_attrs({"type": v_poly.attrs.get("type", "times")})

    return decoded_ds


def remove_iono_surface(
    slcs: xr.DataTree, coeffs: xr.DataArray, H_iono: float, H_orb: float, outfolder: Path, degree: int = 1
):

    # args = [slc for burst_id,slc in slcs.bursts.items() if burst_id.startswith('iw1_14619') ]
    args = [(burst_id, slc) for burst_id, slc in slcs.bursts.items()]  # if burst_id.startswith('iw1_14619') ]
    args = list(zip(*args))

    # init out datatree
    outdatatree = xr.DataTree()
    for burst_id in slcs.bursts.keys():
        outdatatree[f"bursts/{burst_id}"] = xr.DataTree()
    logger.debug(outfolder)
    write(outdatatree, data_path=outfolder, skip_read=True)

    # update all bursts
    results = cast(
        list[tuple[str, xr.DataTree]],
        map(
        remove_iono_surface_single_burst, 
        args=args,
        kwargs={
            'coeffs': coeffs, 
            'H_iono': H_iono,
            'H_orb': H_orb, 
            'degree': degree,
            'outfolder': outfolder
        }, 
        enable_dask = True, combine=False)
    )

    updated_slcs = xr.DataTree()

    for k,v in results:
        updated_slcs[f"bursts/{k}"] = v

    return updated_slcs


def split_spectrum_flow(
    slcs: xr.DataTree,
    reference_date: str,
    multilook: dict,
    frequency_offset: float,
    bandwidth: float,
    cohe_th: float,
    H_iono: float,
    H_orb: float,
    degree: int = 1,
    norm: int = 1,
    wdir_path: Path = Path("/tmp"),
):

    # evaluate bounding box
    bbox = cop_ard_core.utils.get_bounding_box(slcs.bursts, "coreg/measurements")

    # estimate ESD at full-band
    logger.debug("estimate ESD at full-band")
    esd, y_poly = cop_ard_core.esd.estimate(slcs, wdir_path, reference_date, poly_degree=2, enable_dask=True)

    # separate dispersive and non-dispersive components
    logger.debug("separate dispersive and non-dispersive components")
    phase_iono, phase_nd = split_spectrum(
        slcs,
        reference_date,
        frequency_offset=frequency_offset,
        bandwidth=bandwidth,
        cohe_th=cohe_th,
        bbox=bbox,
        y_poly=y_poly,
        multilook=multilook,
        wdir_path=wdir_path,
    )

    phase_iono_path = wdir_path.append('phase_iono')
    logger.debug(f"save phase_iono to file {str(phase_iono_path)}")
    write(phase_iono, data_path=phase_iono_path, mode="w", skip_read=True)
    phase_iono = read(data_path=phase_iono_path, as_type="xr.Dataset")

    # fit surface at ionospheric height
    logger.debug("fit surface at ionospheric height")
    coeffs = fit_ionosphere(phase_iono.isel(acquisition_date=1), H_iono, H_orb, degree=degree, norm=norm)

    # remove ionosphere from cslc and update structure
    logger.debug("remove ionosphere from cslc and update structure")
    slcs_filt = remove_iono_surface(
        slcs, coeffs, H_iono, H_orb, degree=degree, outfolder=Path(os.path.join(str(wdir_path), "slc_filt"))
    )

    return slcs_filt


def extract_interf_multi(slcs: xr.DataTree, reference_date: str, enable_dask: bool = True, **kwargs):
    name = list(slcs.bursts)[0]
    available_pols = {str(i) for i in slcs.bursts[name].coreg.measurements.polarization.values}
    priorities = ["VV", "HH", "VH", "HV"]
    pol = next((p for p in priorities if p in available_pols), None)
    if pol is None:
        raise Exception(f"No polarization has been found {available_pols}")
    args = [(burst_id, json.dumps({k: v.measurements.slc.attrs[pol] for k,v in slc.orig.items()}), slc.coreg.measurements.ds.unify_chunks()) for burst_id,slc in slcs.bursts.items() ]
    args = list(zip(*args))

    results = cop_ard_core.computing.computation.map(
        extract_interf_single,
        args=args,
        kwargs={"reference_date": reference_date, **kwargs},
        enable_dask=enable_dask,
        combine=False,
    )

    return results


def mosaic_interf(results: tuple) -> xr.Dataset:

    interf = xr.DataTree()
    for id, (burst_id, ds) in enumerate(results):
        ds = ds.assign(
            {"id": xr.full_like(ds.phase.isel(polarization=0, acquisition_date=0), fill_value=id, dtype="int8")}
        )
        interf[f"bursts/{burst_id}"] = ds

    mosaic = interf.bursts[list(interf.bursts.children)[0]].to_dataset().isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8))
    mosaic = mosaic.swap_dims({'azimuth_time': 'y', 'slant_range_time': 'x'})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])

    for k, layer in interf.bursts.items():
        layer_sel = layer.isel(azimuth_time=slice(3, -3), slant_range_time=slice(3, -8)).to_dataset()
        layer_sel = layer_sel.swap_dims({'azimuth_time': 'y', 'slant_range_time': 'x'})
        layer_sel = layer_sel.reset_coords([e for e in layer_sel.coords if e not in layer_sel.dims]).compute()

        mosaic = mosaic.combine_first(layer_sel).compute()

    mosaic = mosaic.swap_dims({"y": "azimuth_time", "x": "slant_range_time"})
    mosaic = mosaic.reset_coords([e for e in mosaic.coords if e not in mosaic.dims])

    return mosaic


def get_mosaic_interf(slcs: xr.DataTree, reference_date: str, **kwargs) -> xr.Dataset:

    # compute interferograms in correspondence of each burst
    results = extract_interf_multi(slcs, reference_date, **kwargs)

    # mosaic all interferograms
    mosaic = mosaic_interf(results)

    return mosaic