import os
from collections import defaultdict
from typing import Any, Hashable, Sequence, cast

import numpy as np
import sarpy.io.complex.sentinel
import xarray as xr
from cop_ard_core.logger import logger, timeit
from cop_ard_core.slc import (
    amplitude_calibration,
    apply_invalid_mask,
    get_relative_orbit_number,
    image_deramp,
    intersect,
    merge_dict,
    noise_removal,
)
from cop_ard_core.utils import sync_sicd_metadata
from cop_ard_core.griddata import interpolate

import cop_ard_app.readers
from cop_ard_app.computation import init_zarr_groups, map, map_blocks
from cop_ard_app.io import (
    Path,
    exists,
    generate_empty_dataset,
    get_storage_options,
    is_s3_path,
    normpath,
    path_to_dict,
    read,
    read_slc_static,
    sarpy_open,
    save_slc_static,
    write,
)
from cop_ard_app.model import (
    extract_acquisition_id,
    extract_orbits,
    get_az_rg_meters,
    get_az_rg_times,
    get_burst_id,
    get_burst_id_safe,
    get_coefficients,
    get_footprint,
)
from cop_ard_core.griddata import interp_dataset_scaled

cop_ard_app.readers.__file__


def import_slice(
    slc: xr.DataArray,
    in_path: str,
    index: int,
    processed_pol: str,
    calibration_flag: bool = False,
    calibration_polynomial_coeffs: np.ndarray | None = None,
    noise_removal_flag: str | None = None,
    noise_poly_coeffs: np.ndarray | None = None,
    apply_invalid_mask_flag: bool = False,
    valid_data_y_idxs: np.ndarray | None = None,
    valid_data_x_idxs: np.ndarray | None = None,
    deramp_flag: bool = False,
    dkcoa_coeffs: np.ndarray | None = None,
    chunk={},
):
    """
    Read, process, and save a single chunk of an SLC image.

    This function is the lowest-level worker in the pipeline. It reads a specific
    chunk of data from the source file, applies a series of processing steps
    (noise removal, calibration, deramping, masking), and writes the result
    to the corresponding region in the output Zarr store.

    Args:
        slc: The target dataset slice, used for coordinates and metadata.
        in_path: The path to the source data file or the xr.DataArray of the input data.
        index: The index of the image within the source file.
        calibration_flag: If True, apply radiometric calibration.
        noise_removal_flag: If True, apply thermal noise removal.
        apply_invalid_mask_flag: If True, apply an invalid data mask.
        deramp_flag: If True, apply phase deramping.
        chunk: A dictionary describing the specific slice (region) to process.
    """
    # --- Read Data Chunk ---
    # Skip if the polarization doesn't match the target.
    if processed_pol not in slc.polarization.values:
        return
    # Handle data that's already in memory (as an xarray.DataArray).
    ds = sarpy_open(in_path)
    # Read the specific binary data chunk from the file.
    arr = ds.read(slice(None, None), chunk["azimuth_time"], index=index).astype("complex64").T
    slc.data = arr.reshape(slc.shape)
    # --- Processing Steps ---
    # Noise must be removed before calibration as it's a function of raw DN.
    if noise_removal_flag:
        slc = noise_removal(slc, noise_poly_coeffs)
    # Apply amplitude calibration if specified.
    if calibration_flag:
        slc = amplitude_calibration(slc, calibration_polynomial_coeffs)
    # Perform deramping if specified.
    if deramp_flag:
        slc = image_deramp(slc, dkcoa_coeffs)
    # Apply a mask to zero out invalid data regions if specified.
    if apply_invalid_mask_flag:
        first_idx_y = chunk["azimuth_time"].start
        first_idx_x = chunk["slant_range_time"].start
        if valid_data_y_idxs is not None and valid_data_x_idxs is not None:
            slc = apply_invalid_mask(slc, valid_data_y_idxs, valid_data_x_idxs, first_idx_y, first_idx_x)
    slc = slc.astype("complex64")
    return slc.to_dataset(name="slc")


def import_image(
    group: str,
    outer_dict: dict,
    in_path: str,
    out_path: str,
    calib_type: str | None = None,
    noise_removal_flag: bool | None = None,
    deramp: bool = False,
    apply_invalid_mask_flag: bool = False,
    do_import: bool = True,
    chunks: dict[Hashable, int] = {
        "azimuth_time": 2048,
        "slant_range_time": 2048,
        "acquisition_id": 1,
        "polarization": 1,
    },
    compute_on_host: bool = True,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.DataTree]:
    """
    Import, process, and save a single SLC image (or burst).

    This is the core worker function that handles a single image. It computes
    coordinates, initializes the output Zarr storage for the image, and then
    uses `map_blocks` to process the image data slice by slice.

    Args:
        group: The target group path within the Zarr store (e.g., 'bursts/iw1_123').
        outer_dict: A dictionary containing polarization and metadata information.
        in_path: The path to the input data file.
        out_path: The root path of the output Zarr store.
        calib_type: The radiometric calibration type.
        noise_removal_flag: If True, apply thermal noise removal.
        deramp: If True, apply phase deramping.
        apply_invalid_mask_flag: If True, apply an invalid data mask.
        chunks: Dask chunking configuration.
        reread: If True, reread the data from disk and return it.

    Returns:
        An xarray.DataTree of the processed image if `reread` is True, otherwise None.
    """
    multiburst = kwargs["multiburst"]
    # --- Metadata and Coordinate Extraction ---
    pols = []
    annot = {}
    data = {}
    indexes = {}
    # Extract polarization and metadata for each layer.
    for k, v in outer_dict.items():
        pols.append(k)
        indexes[k] = v[0]
        data[k] = v[1]
        annot[k] = sarpy_open(v[1]).sicd_meta[v[0]].to_dict()
        for el in ["Row", "Col"]:
            if "WgtFunct" in annot[k]["Grid"][el]:
                del annot[k]["Grid"][el]["WgtFunct"]
    annot = sync_sicd_metadata(annot)

    # Compute the coordinate arrays (azimuth/range in meters and time).
    azimuth_m, slant_range_m = get_az_rg_meters(annot[pols[0]])
    azimuth_time, slant_range_time = get_az_rg_times(annot[pols[0]])

    # Get the acquisition date from the metadata.
    collector_name = str(path_to_dict(annot[pols[0]], "CollectionInfo.CollectorName"))
    acquisition_date = str(path_to_dict(annot[pols[0]], "Timeline.CollectStart")).replace("Z", "")
    if multiburst:
        absolute_number = str(path_to_dict(annot[pols[0]], "CollectionInfo.Parameters.ABSOLUTE_ORBIT"))
    else:
        absolute_number = acquisition_date

    acquisition_id = f"{collector_name}_{absolute_number}"
    # Define the final output path for this specific image group.
    data_path = os.path.join(out_path, group)
    # --- Initialize Zarr Storage ---
    dt = xr.DataTree.from_dict({k: xr.Dataset() for k in ["measurements", "conditions", "quality"]})
    dt["conditions"].attrs["stac_discovery"] = {}
    dt["conditions"].attrs["processing_history"] = {}
    # --- Create and Save Empty SLC Dataset ---
    # This creates the structure on disk with the correct dimensions, coordinates, and metadata.
    coords = {
        "acquisition_id": ("acquisition_id", [acquisition_id]),
        "acquisition_date": ("acquisition_id", [acquisition_date]),
        "azimuth_time": ("azimuth_time", azimuth_time),
        "slant_range_time": ("slant_range_time", slant_range_time),
        "azimuth_m": ("azimuth_time", azimuth_m),
        "slant_range_m": ("slant_range_time", slant_range_m),
    }

    chunks_ = chunks.copy()
    if do_import:
        dims = ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"]
        shape = [len(pols), 1, annot[pols[0]]["ImageData"]["NumCols"], annot[pols[0]]["ImageData"]["NumRows"]]
        coords["polarization"] = ("polarization", pols)
    else:
        dims = ["azimuth_time", "slant_range_time"]
        shape = [annot[pols[0]]["ImageData"]["NumCols"], annot[pols[0]]["ImageData"]["NumRows"]]
        chunks_.pop("polarization", None)
        chunks_.pop("acquisition_id", None)

    image = generate_empty_dataset(
        data_vars=["slc"],
        dims={"slc": dims},
        shape={"slc": shape},
        chunks=chunks_,
        dtype={"slc": "complex64"},
        coords=coords,
        attrs_data_vars={"slc": annot},
    )
    dt["measurements"] = image
    write(dt, data_path, mode="w", compute=True, storage_options=get_storage_options(kwargs))

    if do_import:
        # Define the path to the measurements group.
        measurements_path = os.path.join(data_path, "measurements")
        save_options = {
            "data_path": measurements_path,
            "save_options": {
                "skip_read": True,
                "consolidated": False,
            },
        }
        # --- Process Data Slice by Slice ---
        ds = sarpy_open(in_path)
        for pol in pols:
            index = indexes[pol]
            slc = data[pol]
            args = []
            obj = [image.slc]
            # Define the chunking strategy for `map_blocks`.
            chunksizes = {k: v for k, v in chunks.items() if k in image.dims}
            chunksizes["slant_range_time"] = slant_range_time.size
            chunksizes["azimuth_time"] = min([azimuth_time.size, chunksizes["azimuth_time"]])
            annot = ds.sicd_meta[index]
            processed_pol = annot.get_processed_polarization().replace(":", "")
            annot = annot.to_dict()
            args.append(slc)
            calibration_flag = calib_type is not None
            noise_poly_coeffs = get_coefficients("Radiometric.NoiseLevel.NoisePoly.Coefs", annot)
            calibration_polynomial_coeffs = get_coefficients(f"Radiometric.{calib_type}SFPoly.Coefs", annot)
            dkcoa_polynomial_coeffs = get_coefficients("Grid.Col.DeltaKCOAPoly.Coefs", annot)
            valid_mask = annot.get("ImageData", {}).get("ValidData")
            valid_y, valid_x = None, None
            if valid_mask is not None:
                valid_y, valid_x = np.array([(e["Col"], e["Row"]) for e in valid_mask]).T
            args.extend(
                [
                    index,
                    processed_pol,
                    calibration_flag,
                    calibration_polynomial_coeffs,
                    noise_removal_flag,
                    noise_poly_coeffs,
                    apply_invalid_mask_flag,
                    valid_y,
                    valid_x,
                    deramp,
                    dkcoa_polynomial_coeffs,
                ]
            )
            map_blocks(
                import_slice,
                obj=obj,
                args=args,
                enable_dask=enable_dask,
                chunksizes=chunksizes,
                combine=False,
                compute_on_host=compute_on_host,
                store_mapping={0: save_options},
            )

    image = read(data_path, storage_options=get_storage_options(kwargs))
    if not isinstance(image, xr.DataTree):
        raise ValueError(f"Expected xr.DataTree, got {type(image)}")
    return (group, image)


def group_discover(
    in_path: str | xr.DataTree,
    aoi_wkt: str | None = None,
    multiburst: bool = False,
    indices: Sequence[int] | None = None,
    polarizations: Sequence[str] | None = None,
    **kwargs,
):
    """
    Discover and group image metadata from a SAR data file.

    This function reads the metadata from a SAR product, filters it based on
    the area of interest (AOI), polarization, and other criteria, and then
    groups the images into a nested dictionary structure. For multiburst data,
    it organizes by burst ID; otherwise, it organizes by acquisition date.

    Args:
        in_path: Path to the input SAR data or a DataTree.
        aoi_wkt: Area of Interest in Well-Known Text format.
        multiburst: Flag to indicate multiburst mode.
        indices: Specific indices of the images to process.
        polarizations: List of polarizations of interest.

    Returns:
        A nested dictionary grouping image metadata by group ID and polarization.
    """
    storage_options = get_storage_options(kwargs, key="storage_options_slc")
    # Open the data with Sarpy, allowing partial reads for metadata discovery.
    ds = sarpy_open(in_path, allowed_partial_open=True, storage_options=storage_options)

    # If no specific indices are provided, process all images in the file.
    if indices is None:
        indices = range(len(ds.sicd_meta))

    # Use a defaultdict for convenient grouping.
    mydict: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for index in indices:
        annot = ds.sicd_meta[index]
        pol = annot.get_processed_polarization().replace(":", "")
        # Filter by polarization if specified.
        if polarizations:
            if pol not in polarizations:
                continue

        # --- Define the output group path ---
        if multiburst:
            # Get acquisition date and polarization from metadata.
            Parameters = annot.CollectionInfo.Parameters.to_dict()
            absolute_number = str(Parameters["ABSOLUTE_ORBIT"])
            collector_name = str(annot.CollectionInfo.CollectorName)
            acquisition_id = f"{collector_name}_{absolute_number}"

            # For multiburst, filter by geographic footprint intersection with AOI.
            footprint = get_footprint(ds, index)
            if (aoi_wkt is not None) and (not intersect(footprint, aoi_wkt)):
                continue
            # Generate a unique group ID based on swath and burst ID.
            if not is_s3_path(in_path):
                swath, burstid = get_burst_id(annot)
            else:
                storage_options = get_storage_options(kwargs)
                manifest_path = Path(f"{in_path}/manifest.safe", storage_options=storage_options)
                local_manifest_path = manifest_path.ensure_local()._path
                ns, root_node = sarpy.io.complex.sentinel._parse_xml(local_manifest_path)
                swath, burstid = get_burst_id_safe(
                    ds._sentinel_details._get_file_sets(),
                    annot,
                    get_relative_orbit_number(root_node, ns),
                )
            gg = os.path.join("bursts", f"{swath}_{burstid}", "orig", acquisition_id)
        else:
            Parameters = annot.CollectionInfo.Parameters.to_dict()
            collector_name = str(annot.CollectionInfo.CollectorName)
            acquisition_date = str(annot.Timeline.CollectStart).replace("Z", "")
            acquisition_id = f"{collector_name}_{acquisition_date}"
            # For single burst or stripmap, the group is just 'orig/date'.
            gg = os.path.join("orig", acquisition_id)
        # Append the relevant metadata/data to the corresponding group.
        mydict[gg][pol].extend([index, in_path])

    return mydict


def import_slc(
    in_path,
    out_path,
    aoi_wkt: str | None,
    multiburst: bool,
    calib_type: str | None = None,
    noise_removal_flag: bool = False,
    deramp: bool = False,
    indices: Sequence[int] | None = None,
    polarizations: list[str] | None = None,
    apply_invalid_mask_flag: bool = False,
    do_import: bool = True,
    chunks={"acquisition_id": 1, "azimuth_time": 2048, "slant_range_time": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> list[xr.DataTree]:
    """
    Import and process all the images contained within a single input file.

    This function acts as a dispatcher. It first discovers the relevant image groups
    within a single SAFE file (or other input format), then maps the `import_image`
    function to process each discovered group in parallel.

    Args:
        in_path: The path to the input data file (e.g., a .SAFE directory).
        out_path: The root path of the output Zarr store.
        aoi_wkt: Area of Interest in WKT format.
        multiburst: If True, handle data in multiburst mode.
        calib_type: The radiometric calibration type to apply.
        noise_removal_flag: If True, apply thermal noise removal.
        deramp: If True, apply phase deramping.
        indices: Specific image indices to process within the file.
        polarizations: Specific polarizations to process.
        reread: If True, reread the data from disk after processing and return it.
        apply_invalid_mask_flag: If True, apply a mask for invalid data.
        chunks: Dask chunking configuration.

    Returns:
        An xarray.DataTree of the processed data if `reread` is True, otherwise None.
    """
    storage_options = get_storage_options(kwargs, key="storage_options_slc")
    # Use a temporary directory for local processing, especially for S3 inputs.
    in_local_path = None
    if isinstance(in_path, str):
        path = Path(in_path, storage_options=storage_options).ensure_local()
        in_local_path = path._path
    # Discover all the image groups (e.g., bursts) within the local path.
    group_dict = group_discover(
        in_path if in_local_path is None else in_local_path, aoi_wkt, multiburst, indices, polarizations
    )
    # Prepare the arguments for the `import_image` map operation.
    args = [[index for index in group_dict.keys()], [group for group in group_dict.values()]]

    # Map the `import_image` function over the discovered groups.
    map_results = map(
        import_image,
        args,
        kwargs=dict(
            in_path=in_local_path,
            out_path=out_path,
            calib_type=calib_type,
            noise_removal_flag=noise_removal_flag,
            deramp=deramp,
            apply_invalid_mask_flag=apply_invalid_mask_flag,
            chunks=chunks,
            do_import=do_import,
            enable_dask=enable_dask,
            multiburst=multiburst,
            **kwargs,
        ),
        compute_on_host=True,  # Ensure data reading happens on the host.
        combine=False,
        enable_dask=enable_dask,
    )
    # If requested, open and return the DataTree from the output path.
    if not isinstance(map_results, list):
        raise ValueError(f"Expected list of xr.DataTree, got {type(map_results)}")
    return map_results


@timeit
def import_slc_stack(
    or_f: list[str],
    out_path,
    aoi_wkt: str | None = None,
    multiburst: bool = False,
    calib_type: str | None = None,
    noise_removal_flag: str | None = None,
    deramp: bool = True,
    indices: list[int] | None = None,
    polarizations: list[str] | None = None,
    apply_invalid_mask_flag: bool = False,
    do_import: bool = True,
    chunks={"polarization": 1, "acquisition_id": 1, "azimuth_time": 2048, "slant_range_time": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> xr.DataTree:
    """
    Import, process, and save a stack of SLC images into a unified DataTree structure.

    This function is the main entry point for the import pipeline. It first discovers all
    relevant image groups (e.g., bursts) from the input files, initializes the
    output Zarr storage, and then maps the `import_slc` function over each input
    file to perform the actual data processing and writing in parallel. Finally, it
    rereads the processed data from the output path and returns it as a DataTree.

    Args:
        or_f: A list of input file paths or pre-loaded DataTrees.
        out_path: The root path for the output Zarr store.
        aoi_wkt: An Area of Interest in WKT format to filter the data.
        multiburst: If True, handle the data as a collection of bursts that
        need to be organized by swath and burst ID.
        calib_type: The type of radiometric calibration to apply (e.g., "SigmaZero").
        noise_removal_flag: If True, apply thermal noise removal.
        deramp: If True, apply a phase deramping correction.
        indices: A list of specific image indices to process from the input files.
        polarizations: A string specifying the polarizations to process (e.g., "VV VH").
        apply_invalid_mask_flag: If True, apply a mask to zero out invalid data regions.
        chunks: A dictionary defining the chunk sizes for Dask processing.

    Returns:
        An xarray.DataTree containing the fully processed and consolidated SLC stack.
    """

    if not or_f:
        raise ValueError("The input list cannot be empty.")

    if exists(out_path, storage_options=get_storage_options(kwargs)):
        raise ValueError(f"Output path {out_path} already exists.")

    if not do_import:
        logger.info("Importing SLC image structure only (data import skipped).")
    # --- Step 1: Discover Image Groups ---
    # In parallel, discover all the relevant image groups (bursts, polarizations, etc.) from the input files.
    res = map(
        group_discover,
        or_f,
        kwargs=dict(aoi_wkt=aoi_wkt, multiburst=multiburst, indices=indices, polarizations=polarizations, **kwargs),
        enable_dask=enable_dask,
        combine=False,
    )

    if not isinstance(res, list):
        raise Exception(f"Expected list, got {type(res)}")
    # Merge the list of dictionaries from all workers into a single dictionary of groups.
    groups = merge_dict(res)
    # --- Step 2: Initialize Output Storage ---
    logger.info(f"Initializing groups: {groups.keys()}")
    init_zarr_groups(out_path, groups.keys(), multiburst, **kwargs)
    # --- Step 3: Import and Process Data ---
    # Map the `import_slc` function over each input file. This will process the data
    # in parallel and write the results directly to the initialized Zarr store.
    slcs = xr.DataTree()
    map_results = map(
        import_slc,
        or_f,
        kwargs=dict(
            out_path=out_path,
            aoi_wkt=aoi_wkt,
            multiburst=multiburst,
            calib_type=calib_type,
            noise_removal_flag=noise_removal_flag,
            deramp=deramp,
            indices=indices,
            polarizations=polarizations,
            apply_invalid_mask_flag=apply_invalid_mask_flag,
            chunks=chunks,
            do_import=do_import,
            enable_dask=enable_dask,
            **kwargs,
        ),
        map_equally=True,  # Distribute work evenly.
        combine=False,
        enable_dask=enable_dask,
    )
    # --- Step 4: Reread and Return Result ---
    # After all workers have finished writing, open the consolidated Zarr store.
    if not isinstance(map_results, list):
        raise ValueError(f"Expected list of xr.DataTree, got {type(map_results)}")
    for result in map_results:
        for group, slc in result:
            slcs[group] = slc

    return slcs


@timeit
def finalize_import(
    slcs: xr.DataTree,
    working_directory: str,
    recompute: bool = False,
    acquisition_id: str | None = None,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[xr.DataTree, str]:
    """
    Finalize the import process by ensuring the reference date SLC is
    available in the DataTree.

    This function handles different scenarios based on whether the data is
    organized by bursts and whether the user wants to recompute (overwrite)
    the static reference data in the `working_directory`.

    Scenarios handled:

    1.  **Bursts, No Recompute, Ref Date Missing:** Tries to load the reference
        date data from the `working_directory`.
    2.  **Bursts, Recompute, Ref Date Present:** The reference date data from
        the input `slcs` stack is saved (overwritten) to the `working_directory`.
    3.  **Bursts, Ref Date Partially Present:** Remove bursts with no reference
        date in an inconsistent state.
    4.  **No Bursts, Ref Date Missing:** Raises an error.

    Parameters
    ----------

    slcs : xr.DataTree
        The DataTree containing the stack of SLCs.
    working_directory : str
        The path to the static working directory where reference data
        is stored or will be saved.
    recompute : bool, optional
        If True, indicates that the static reference data in the
        `working_directory` should be overwritten if the reference date
        is present in the `slcs` input. Defaults to False.
    reference_date : str or None, optional
        The date string (e.g., 'YYYY-MM-DD') of the reference (master) SLC.
        Defaults to None.
    enable_dask : bool, optional
        Whether to enable parallel computation via Dask. Defaults to True.
    **kwargs : dict
        Additional keyword arguments, often including `storage_options`
        passed to read/write functions.

    Returns
    -------
    tuple[xr.DataTree, str]
        A tuple containing:

        - The `slcs` DataTree, potentially modified to include the reference
          date data loaded from the `working_directory`.
        - The reference date string.

    Raises
    ------
    ValueError
        If the reference image cannot be found/loaded, if there are multiple
        acquisition dates when one is expected, or if `reference_date` is missing.
    TypeError
        If the output of the static loading function or the reference date
        is of an unexpected type.
    """
    multiburst = "bursts" in slcs

    if multiburst:
        # Check if the DataTree is organized by bursts and needs a reference date
        acquisition_ids = extract_acquisition_id(slcs, multiburst=True)
        if recompute:
            if len(acquisition_ids) != 1:
                raise ValueError(f"Expected exactly one acquisition id, got {len(acquisition_ids)}")

            acquisition_id = next(iter(acquisition_ids))
            logger.info("Saving reference SLC structure to static working directory.")

            # Unzip dictionary items into two lists for mapping: (burst_ids, slc_objects)
            args = [(burst_id, slc.orig[acquisition_id]) for burst_id, slc in slcs.bursts.items()]
            args = list(zip(*args))
            # Use a parallel map to save each burst's reference SLC
            map(
                save_slc_static,
                args=args,
                kwargs=dict(working_directory=working_directory, **kwargs),
                enable_dask=enable_dask,
            )
        else:
            args = [(burst_id, list(slc.orig)) for burst_id, slc in slcs.bursts.items()]
            args = list(zip(*args))
            static_slcs = map(
                read_slc_static,
                args=args,
                kwargs=dict(working_directory=working_directory, **kwargs),
                enable_dask=enable_dask,
            )
            if not isinstance(static_slcs, list):
                raise TypeError(f"Expected a list from computation.map, got {type(static_slcs).__name__}")

            for burst_id, acq_id, status_or_data in static_slcs:
                if acq_id is not None:
                    acquisition_id = acq_id
                if isinstance(status_or_data, int):
                    if status_or_data == 0:
                        logger.warning(f"No reference image found for burst {burst_id}. Removing from datatree.")
                        del slcs.bursts[burst_id]
                    elif status_or_data == 1:
                        logger.info(f"Reference image already in stack for burst {burst_id}.")
                else:
                    logger.info(f"Adding reference data for burst {burst_id}.")
                    slcs[f"bursts/{burst_id}/orig/{acq_id}"] = status_or_data

            if len(slcs.bursts) == 0:
                raise RuntimeError("No reference image found for any bursts. DataTree is empty.")
    else:
        if recompute:
            acquisition_ids = list(slcs.orig.keys())
            if len(acquisition_ids) != 1:
                raise ValueError(f"Expected exactly one acquisition id when recompute=True, got {len(acquisition_ids)}")

            acquisition_id = acquisition_ids[0]
            logger.info(f"Saving reference SLC {acquisition_id} to static working directory (non-multiburst).")
            save_slc_static(
                "data",
                slcs.orig[acquisition_id],
                working_directory=working_directory,
                reference_id=acquisition_id,
                **kwargs,
            )
        else:
            if acquisition_id is None:
                raise ValueError(
                    "`acquisition_id` must be provided when data is not organized by bursts and recompute=False."
                )
            if acquisition_id not in slcs.orig:
                logger.info(f"Reading reference data for {acquisition_id} from static directory.")
                current_ids = list(slcs.orig.keys())
                _, acq_id, status_or_data = read_slc_static(
                    "data",
                    current_ids,
                    working_directory=working_directory,
                    reference_id=acquisition_id,
                    **kwargs,
                )

                if isinstance(status_or_data, int):
                    if status_or_data == 0:
                        raise ValueError(
                            f"Static reference folder/data for {acquisition_id} does not exist in {working_directory}."
                        )
                    elif status_or_data == 1:
                        logger.info("Reference image already in stack.")
                else:
                    logger.info(f"Successfully added reference data for {acquisition_id} to DataTree.")
                    slcs.orig[acquisition_id] = status_or_data

    if not isinstance(acquisition_id, str):
        raise TypeError(f"Expected `acquisition_id` to be a string, got {type(acquisition_id).__name__}")

    return slcs, acquisition_id


def overwrite_orbit_coefficients(
    burst_id: str,
    acquisition_id: str,
    orbit: xr.Dataset | None,
    wdir_path: str,
    multiburst: bool,
    **kwargs,
) -> tuple[str, str, xr.Dataset | None]:
    """
    Refit orbit polynomials using precise orbit data and update SLC metadata.

    This worker function takes precise orbit state vectors, fits new polynomials
    of a specified degree to them, and then overwrites the original orbit
    polynomial coefficients in the SLC metadata.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst being processed.
    acquisition_id : str
        The acquisition date of the SLC.
    slc : xr.DataTree
        The DataTree node for the specific burst to be updated.
    orbit : xr.Dataset | None
        The precise orbit data as an xarray Dataset. If None, no update is performed.
    poly_degree : int
        The degree of the polynomial to fit to the orbit data.

    Returns
    -------
    tuple[str, str, xr.Dataset | None]
        A tuple containing the burst_id, acquisition_id, and a new Dataset
        with the refitted orbit coefficients, or None if no update was performed.
    """
    path = f"bursts/{burst_id}" if multiburst else ""
    data_path = normpath(f"{wdir_path}/{path}/orig/{acquisition_id}/conditions/orbit")

    # Proceed only if precise orbit data is available.
    if orbit is None:
        # If no orbit data was provided, log and return None.
        logger.info(f"No orbit update for the {burst_id} image: {acquisition_id}")
        return (burst_id, acquisition_id, None)

    # Write the new orbit dataset to the 'conditions/orbit' group in the Zarr store.
    orbit = cast(
        xr.Dataset,
        write(
            orbit,
            data_path,
            mode="w",
            storage_options=get_storage_options(kwargs),
        ),
    )
    return (burst_id, acquisition_id, orbit)


def update_slcs(slcs: xr.DataTree, multiburst: bool, conditions: list):
    """
    Update the main DataTree with newly computed orbit conditions.

    This function takes the results from the `refit_orbit_coefficients` parallel map
    operation and places each new orbit dataset into the correct location
    within the `slcs` DataTree structure.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree to be updated.
    multiburst : bool
        A boolean flag for multiburst mode, which determines the
        path structure.
    conditions : list
        The list of results from `overwrite_coeffs`, where each item is
        a tuple (burst_id, acquisition_date, orbit_dataset).

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree.
    """
    # Iterate through the list of results.
    for burst_id, acquisition_id, orbit in conditions:
        if orbit is None:
            continue

        path_prefix = f"bursts/{burst_id}" if multiburst else ""
        base_node = f"{path_prefix}/orig/{acquisition_id}"

        # 2. Assign the new orbit dataset to its place in the DataTree
        slcs[f"{base_node}/conditions/orbit"] = orbit

        # 3. Pre-extract coefficients
        xyz_coeffs = orbit.coeffs.squeeze(dim="acquisition_id")

        # 4. Update the metadata attributes
        slc_path = f"{base_node}/measurements/slc"
        attrs = slcs[slc_path].attrs

        for pol in attrs:
            for axis_idx, tag in enumerate(["X", "Y", "Z"]):
                # .tolist() safely converts numpy arrays to standard Python lists
                attrs[pol]["Position"]["ARPPoly"][tag]["Coefs"] = xyz_coeffs.isel(axis=axis_idx).values.tolist()

        # Reassign the updated attributes back to the node
        slcs[slc_path].attrs = attrs

    return slcs


def updateOrbits(slc: xr.DataTree) -> xr.Dataset:
    """
    Consolidate orbit information from all acquisitions into a single Dataset.

    This function extracts orbit polynomial coefficients from the metadata of
    each original SLC image and combines them with any externally provided
    precise orbit data. The result is a clean, unified Dataset of orbit
    information for the entire stack.

    Parameters
    ----------
    slc
        The DataTree for a single burst, containing the `orig` nodes with
        source metadata.

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset containing the consolidated orbit coefficients,
        reference times, and types for all acquisitions.
    """
    # --- Extract Orbits from Annotations ---
    # This provides a baseline set of orbit data from the original product files.
    orbits = extract_orbits(slc, multiburst=False)
    orbits = orbits.to_dataset(name="coeffs")
    num_dates = orbits.sizes["acquisition_id"]
    orbits["reference_time"] = (
        "acquisition_id",
        [orbits.coeffs.reference_times[date] for date in orbits.acquisition_id.values],
    )
    orbits["type"] = ("acquisition_id", ["AUX_ANNOTATED"] * num_dates)
    orbits["orbit_files"] = ("acquisition_id", [""] * num_dates)
    for date_key, v in slc.orig.items():
        if ("conditions" in v) and ("orbit" in v.conditions):
            precise_orbit = v.conditions.orbit
            # Safely locate the specific date across the DataArray
            loc_dict = {"acquisition_id": date_key}
            # Overwrite the coefficients
            orbits["coeffs"].loc[loc_dict] = precise_orbit.coeffs.loc[loc_dict]
            # Overwrite the metadata to reflect the precise orbit
            orbits["type"].loc[loc_dict] = precise_orbit.attrs.get("orbit_type", "AUX_POEORB")
            # Extract the orbit file name (accounting for lists or strings in the attributes)
            if "orbit_files" in precise_orbit.attrs:
                file_attr = precise_orbit.attrs["orbit_files"]
                file_name = file_attr[0] if isinstance(file_attr, list) else file_attr
                orbits["orbit_files"].loc[loc_dict] = file_name

            # Overwrite reference time if the precise orbit has a different one
            if "reference_time" in precise_orbit.attrs:
                orbits["reference_time"].loc[loc_dict] = precise_orbit.attrs["reference_time"]

    orbits = orbits.chunk(axis=-1, degree=-1, acquisition_id=1)

    return orbits


def updateETAD(
    slc: xr.DataTree,
    measurements: xr.Dataset,
    grid_size: Sequence[int] = (200, 200),
) -> xr.Dataset | None:
    """
    Consolidate Extended Timing Annotation Data (ETAD) from all acquisitions.

    Gathers ETAD correction grids from each acquisition's `conditions` group within
    the DataTree, resamples them onto a coregistered target grid using
    scaled spline interpolation, converts time delays to interferometric phase,
    and concatenates them along the `acquisition_id` dimension.

    Parameters
    ----------
    slc : xr.DataTree
        The DataTree structure for a single burst containing the `orig` nodes.
    measurements : xr.Dataset
        Dataset containing coregistration offsets (`daz`, `drg`) and reference coordinates.
    grid_size : Sequence[int], optional
        Dimensions (azimuth, range) for sampling the interpolation grid, by default (200, 200).

    Returns
    -------
    xr.Dataset | None
        An xarray.Dataset containing consolidated ETAD corrections in radians and seconds,
        or None if no ETAD metadata is found.
    """
    etads = []

    # Extract reference coordinate vectors
    azimuth_time = measurements.azimuth_time.values
    slant_range_time = measurements.slant_range_time.values

    # Generate equispaced indices for spatial sampling
    idx_az = np.linspace(0, len(azimuth_time) - 1, grid_size[0], dtype=int)
    idx_rg = np.linspace(0, len(slant_range_time) - 1, grid_size[1], dtype=int)
    az_sampled = azimuth_time[idx_az]
    rg_sampled = slant_range_time[idx_rg]

    # Iterate through each acquisition in the DataTree
    for k, v in slc.orig.items():
        if ("conditions" in v) and ("ETAD" in v.conditions):
            etad = v.conditions.ETAD.to_dataset()
            annot = v.measurements.slc.attrs
            annot = annot[list(annot)[0]]
            center_frequency = annot["RMA"]["INCA"]["FreqZero"]
            # Select coregistration shifts for the current acquisition
            daz = measurements.daz.sel(acquisition_id=k)
            drg = measurements.drg.sel(acquisition_id=k)
            daz_sel = daz.sel(azimuth_time=az_sampled, slant_range_time=rg_sampled)
            drg_sel = drg.sel(azimuth_time=az_sampled, slant_range_time=rg_sampled)

            # Build deformation-compensated interpolation meshgrid (xi)
            az_grid, rg_grid = np.meshgrid(az_sampled, rg_sampled, indexing="ij")
            az_grid = (az_grid + daz_sel).values
            rg_grid = (rg_grid + drg_sel).values
            xi = (az_grid, rg_grid)

            target_coords = {
                "azimuth_time": az_sampled,
                "slant_range_time": rg_sampled,
            }

            # Perform scaled spline interpolation to avoid numerical underflow
            etad_interp = interp_dataset_scaled(
                data=etad,
                xi=xi,
                common_dims=("azimuth_time", "slant_range_time"),
                target_coords=target_coords,
                method="cubic",
            )

            # Assign acquisition coordinate and stack dimension
            etad_interp = etad_interp.assign_coords(acquisition_id=k).expand_dims("acquisition_id")
            etads.append(etad_interp)

    if etads:
        # Concatenate along acquisition dimension and re-chunk for optimal I/O
        consolidated_etad = xr.concat(etads, dim="acquisition_id").compute()
        consolidated_etad = consolidated_etad.chunk(
            {
                "acquisition_id": 1,
                "azimuth_time": -1,
                "slant_range_time": -1,
            }
        )
        return consolidated_etad

    return None


def updateLookVector(
    measurements: xr.Dataset,
    simulation: xr.Dataset,
    grid_size: Sequence[int] = (200, 200),
) -> xr.Dataset | None:

    # Optional subsampling of the simulation dataset for performance
    step_y = max(1, simulation.sizes["y"] // (grid_size[0] * 2))
    step_x = max(1, simulation.sizes["x"] // (grid_size[1] * 2))
    simulation = simulation.squeeze().isel(y=slice(None, None, step_y), x=slice(None, None, step_x))

    # 1. Calculate the look vector (maintaining float64)
    dem_distance = simulation.dem_distance.squeeze()
    norm = np.linalg.norm(dem_distance.values, ord=2, axis=-1, keepdims=True)
    look_vector_arr = dem_distance.values / norm
    look_vector_arr = np.moveaxis(look_vector_arr, -1, 0)  # shape: (3, y, x)

    azimuth_time = measurements.azimuth_time
    slant_range_time = measurements.slant_range_time

    idx_az = np.linspace(0, len(azimuth_time) - 1, grid_size[0], dtype=int)
    idx_rg = np.linspace(0, len(slant_range_time) - 1, grid_size[1], dtype=int)
    az_sampled = azimuth_time[idx_az]
    rg_sampled = slant_range_time[idx_rg]

    target_coords = {"azimuth_time": az_sampled, "slant_range_time": rg_sampled}

    dt_az_ns = (az_sampled.max() - az_sampled.min()).astype(float).values / (az_sampled.size - 1)
    dt_rg_s = (rg_sampled.max() - rg_sampled.min()).values / (rg_sampled.size - 1)
    az_ref, rg_ref = azimuth_time[0].values, slant_range_time[0].values

    # Build points mapping
    points = np.array(
        [
            (((simulation.azimuth_time - az_ref).astype(float)) / dt_az_ns).values,
            ((simulation.slant_range_time - rg_ref).astype(float) / dt_rg_s).values,
        ]
    ).transpose(2, 1, 0)

    # Prepare values to be interpolated and define output grid bounding box
    z = look_vector_arr.transpose(0, 2, 1)
    bbox = np.array([0, 0, rg_sampled.size, az_sampled.size])

    # Interpolate using custom griddata interpolate function
    m = interpolate(points, z, bbox)
    m = m / np.linalg.norm(m, axis=0)

    # Build the resulting xarray Dataset
    ds_interp = xr.Dataset(
        data_vars={"look_vector": (("axis", "azimuth_time", "slant_range_time"), m)},
        coords={
            "axis": [0, 1, 2],
            "azimuth_time": az_sampled,
            "slant_range_time": rg_sampled,
        },
    )

    return ds_interp
