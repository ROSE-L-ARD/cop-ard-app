# Copyright 2025, B-Open Solutions srl. All rights reserved.
# This source code is protected under international copyright law.  All rights
# reserved and protected by the copyright holders.
# This file is confidential and only available to authorized individuals with the
# permission of the copyright holders.  If you encounter this file and do not have
# permission, please contact the copyright holders and delete this file.
# This file must not be copied, modified, or distributed without explicit
# written permission from the copyright holder.
from typing import Literal, Sequence

import numpy as np
import structlog
import xarray as xr
from cop_ard_core.matching import compute_pearson_coefficient, initialize_patches, polyfit
from cop_ard_core.utils import get_best_pol
from scipy.constants import speed_of_light

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.model import extract_acquisition_id

logger = structlog.get_logger(__name__)


def chunks_optimization(
    stack: xr.DataArray,
    patches: xr.DataArray | xr.Dataset,
):
    """

    Compute the optimal chunks to minimize reading

    Returns the computed chunks and a changed version of patched to be used
    for the submission of the pearson correlation computation via map_blocks

    """

    # Removing of all the patches not completely inside stack
    patches = patches.sel(
        patch=(
            (patches.x_start >= 0)
            & (patches.y_start >= 0)
            & (patches.x_end < stack.x.size)
            & (patches.y_end < stack.y.size)
        )
    )

    stack_chunk = {k: v[0] for k, v in stack.chunksizes.items() if k in ["x", "y"]}
    y_mean = (patches.y_start + patches.y_end) / 2.0
    x_mean = (patches.x_start + patches.x_end) / 2.0
    row = np.floor(y_mean / stack_chunk["y"]).astype(int).values
    col = np.floor(x_mean / stack_chunk["x"]).astype(int).values

    index = row * len(stack.chunksizes["x"]) + col
    # Index that allows to group patches belonging to the same chunk
    patches = patches.assign_coords({"index": (["patch"], index)})
    # Now considering all the patches belonging to each chunk the optimized chunksizes can be computed
    g = patches.groupby("index")
    chunks = g.reduce(np.min)[["x_start", "y_start"]].assign(g.reduce(np.max)[["x_end", "y_end"]])
    chunks = {
        "x": np.vstack((chunks.x_start.values, chunks.x_end.values + 1)).T.tolist(),
        "y": np.vstack((chunks.y_start.values, chunks.y_end.values + 1)).T.tolist(),
    }

    # Preparing patches dataset to be passed to the processing function
    mask = np.zeros(patches.index.size + 1, dtype="int32")
    mask[patches.index] = 1
    mask[mask == 1] = np.arange(np.count_nonzero(mask))
    patches = patches.assign_coords({"job_id": (["patch"], mask[patches.index])})
    patches = patches.drop_vars("index").swap_dims(patch="job_id")
    chunks["z"] = [[0, stack.z.size] for k in range(len(chunks["x"]))]
    return chunks, patches


# (disparity, slcs.coreg.measurements.drg, slcs.coreg.measurements.daz),
# args=(annot),
# kwargs={
#     "deg": options.get("poly_deg", 3),
#     "cohe_th": options.get("poly_cohe_th", 0.3),
#     "norm": options.get("poly_norm", 1),
# },


def estimate_poly_matching(
    slcs: xr.DataTree,
    disparity_map: xr.DataTree,
    deg: int = 3,
    cohe_th: float = 0.3,
    norm: int = 1,
    kernel_shape: list = [15, 15],
    enable_dask: bool = True,
):
    multiburst = True if "bursts" in slcs else False
    acquisition_ids = extract_acquisition_id(slcs, multiburst)
    args = []

    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        measurements = v.coreg.measurements
        for id in measurements.acquisition_date.values:
            m = measurements.sel(acquisition_date=id)
            d = (disparity_map.bursts[k].sel() if multiburst else disparity_map).sel(acquisition_date=id)
            a = (k, id, d, m.daz, m.drg)
            args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    results = map(
        polyfit,
        args,
        kwargs={"deg": deg, "cohe_th": cohe_th, "kernel_shape": kernel_shape},
        enable_dask=enable_dask,
    )

    from cop_ard_app.io import write

    for burst_id, acquisition_id, m1_x, m1_y, dx, dy in results:
        path = f"/data/tmp/polyfit/{burst_id}/{acquisition_id}"
        for n, m in zip(["m1_x", "m1_y", "dx", "dy"], [m1_x, m1_y, dx, dy]):
            write(m.to_dataset(name="A"), f"{path}/{n}")


def stack_disparity(
    slcs: xr.DataTree,
    reference_date: str,
    wsize: Sequence[int] = [256, 256],
    wstep: Sequence[int] = [256, 256],
    wsearch: Sequence[int] = [5, 5],
    dim: Sequence[int] = [1, 1],
    block_size: int = 10,
    enable_dask: bool = True,
) -> xr.DataTree:
    multiburst = True if "bursts" in slcs else False
    args = []
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        a = (k, v.coreg.measurements.cstack)
        args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Coregistration in Parallel ---
    # Map the `coreg_single_burst` function over the list of arguments.
    results = map(
        compute_stack_pearson_coefficient,
        args,
        kwargs=dict(
            reference_date=reference_date,
            wsize=wsize,
            wstep=wstep,
            wsearch=wsearch,
            block_size=block_size,
            dim=dim,
            enable_dask=enable_dask,
        ),
        enable_dask=enable_dask,
        combine=False,
    )
    disparity_map = xr.DataTree()
    if multiburst:
        for k, v in results:
            disparity_map[f"bursts/{k}"] = v
    else:
        for k, v in results:
            disparity_map = xr.DataTree(v)

    return disparity_map


def compute_stack_pearson_coefficient(
    burst_id: str,
    stack: xr.DataArray,
    reference_date: str,
    wsize: Sequence[int],
    wstep: Sequence[int],
    wsearch: Sequence[int],
    block_size: int,
    dim: Sequence[int],
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.DataArray]:
    """

    Compute the maximum of the Pearson correlation coefficient of a stack of images w.r.t a reference image

    The reference image is one of the images of the stack. The maximization is performed considering a window

    :param wsize: patches size (y, x)
    :param wstep: distance (y, x) between two consecutive patches
    :param wsearch: search window size (y, x)

    """
    # Check if the master date is in the input stack
    if reference_date not in stack.acquisition_date:
        raise Exception("Unable to find the master date in stack")

    best_pol = get_best_pol(stack.polarization.values)
    # x and y coordinates must be added and used here as dimensions replacing slant_range_time, azimuth_time
    annot = stack.attrs[best_pol]
    timeca_polynomial_coeffs = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"]
    azimuth_grid_spacing = annot["Grid"]["Col"]["SS"]
    range_grid_spacing = annot["Grid"]["Row"]["SS"]
    xyz_stack = (
        stack.sel(polarization=best_pol)
        .squeeze()
        .assign_coords(
            {k: ([j], range(stack[j].size)) for k, j in zip(["y", "x"], ["azimuth_time", "slant_range_time"])}
        )
        .swap_dims(azimuth_time="y", slant_range_time="x")
        .rename(acquisition_date="z")
        .transpose("y", "x", "z")
    )
    # Patches to be used to cross-correlate initialization
    cc_patches = initialize_patches(
        xyz_stack.y.size,
        wsize[0],
        wstep[0],
        xyz_stack.x.size,
        wsize[1],
        wstep[1],
    )

    # Considering patches and stack chunking the optimal chunks can be computed
    chunks_opt, cc_patches = chunks_optimization(xyz_stack, cc_patches)
    args = (
        cc_patches,
        block_size,
        wsearch,
        timeca_polynomial_coeffs,
        azimuth_grid_spacing,
        range_grid_spacing,
        dim,
    )
    # Cross-Correlation computation via map_blocks
    results = map_blocks(
        blockwise_pearson_coefficient,
        (xyz_stack, xyz_stack.sel(z=reference_date)),
        args=args,
        chunks=chunks_opt,
        combine=False,
        enable_dask=enable_dask,
    )

    disparity = xr.concat(results, dim="point").drop_vars("point")
    return (burst_id, disparity)


def blockwise_pearson_coefficient(
    stack: xr.DataArray,
    reference: xr.DataArray,
    patches: xr.Dataset,
    block_size: int,
    wsearch: Sequence[int],
    timeca_polynomial_coeffs: np.ndarray,
    azimuth_grid_spacing: float,
    range_grid_spacing: float,
    dim: Sequence[int],
    chunk: dict = {},
    job_id: int = 0,
):
    """

    Compute the Pearson coefficient for all the patches belong to the current stack chunk

    """
    # Results initialization
    results = []
    # Identification of patches belong to the current chunk
    j_patches = patches.sel(job_id=job_id)

    # Patch center and spacing expressed in times
    pixel_to_time = np.array(
        [
            [timeca_polynomial_coeffs[1] * azimuth_grid_spacing * 1e9],
            [2 * range_grid_spacing / speed_of_light],
        ],
        dtype="float64",
    )
    y_dim, x_dim = "azimuth_time", "slant_range_time"

    # Reference image initialize
    reference = abs(reference.compute())

    # Loop over chunks along the acquisition_date (z) dimension to keep controlled the memory usage
    for i in range(0, stack.z.size, block_size):
        i_stack = stack.isel(z=slice(i, i + block_size)).compute()
        i_stack = abs(i_stack)

        for j in range(j_patches.patch.size):
            j_patch = j_patches.isel(job_id=j)  # patch is expected to be indexed over job_id
            # Identification of region for this patch
            x_slice = slice(j_patch.x_start.values, j_patch.x_end.values)
            y_slice = slice(j_patch.y_start.values, j_patch.y_end.values)

            # Extraction from stack and reference of the required region
            j_stack = i_stack.sel(x=x_slice, y=y_slice)
            j_reference = reference.sel(x=x_slice, y=y_slice)

            results.append(
                compute_pearson_coefficient(
                    patch=j_patch,
                    stack=j_stack,
                    reference=j_reference,
                    pixel_to_time=pixel_to_time,
                    peak_window_size=wsearch,
                    y_dim=y_dim,
                    x_dim=x_dim,
                    correlation_kernel_size=dim,
                )
            )
    return xr.concat(results, dim="point")
