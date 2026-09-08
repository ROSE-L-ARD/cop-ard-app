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
import xarray as xr
from cop_ard_core.matching import compute_pearson_coefficient, initialize_patches, fix_disparity
from cop_ard_core.detrend import fit_2d_poly, to_polygrid2d
from scipy.constants import speed_of_light

from cop_ard_app.computation import map, map_blocks
from cop_ard_app.model import extract_acquisition_id
from cop_ard_app.utils import get_best_pol


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
    degree: int = 2,
    cohe_th: float = 0.3,
    kernel_shape: list = [15, 15],
    enable_dask: bool = True,
) -> xr.DataTree:
    """Starting from the disparity maps evaluated ion cotrrespondence of each burst / slice the global polynomial correction function is evaluated.
    Corrections are stored in a datatree. At first level the nodes corresponds to the acquisition_id.
    At the second level the two polynomial corrections x_poly and y_poly are provided.
    Each single polynomial is stored in a dataarray with the following structure:
    coordinates: i, j
    coeffs: coefficients to be used by the function numpy.polynomial.polyval2d
    attrs:
    - deg: degree of the polynomial
    - offset: offset to be applied to the azimuth_time and slant_range_time coordinates
    - scale_fact: scaling factor to be appied to the azimuth_time and slant_range_time coordinates
    Example of usage:
    from numpy.polynomial.polynomial import polyval2d
    x = (slant_range_time - offset[0]) / scale_fact[0]
    y = (azimuth_time - offset[1]) / scale_fact[1]

    surf = polyval2d(coeffs, x, y)
    """
    multiburst = True if "bursts" in slcs else False
    acquisition_ids = extract_acquisition_id(slcs, multiburst)
    args = []

    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        measurements = v.coreg.measurements
        for id in measurements.acquisition_id.values:
            m = measurements.sel(acquisition_id=id)
            d = (disparity_map.bursts[k].to_dataset() if multiburst else disparity_map["data"]).sel(acquisition_id=id).to_dataset()
            a = (k, id, d, m.drg, m.daz)
            args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    results = map(
        fix_disparity,
        args,
        kwargs={"cohe_th": cohe_th, "kernel_shape": kernel_shape},
        enable_dask=enable_dask,
    )

    x_poly = get_polynomial_correction(acquisition_ids, results, "dx", degree=degree)
    y_poly = get_polynomial_correction(acquisition_ids, results, "dy", degree=degree)

    return x_poly, y_poly


def get_polynomial_correction(acquisition_ids, results, dim: str, degree: int, norm: int = 1):
    poly_corr = {}
    for id in acquisition_ids:
        to_merge = [ds.drop_vars("point", errors='ignore') for _, acquisition_id, ds in results if (acquisition_id == id) and (ds is not None) and (dim in ds.data_vars)]
        ds_tot = xr.concat(to_merge, dim="point")
        off_y = ds_tot.m1_y.min().values

        (
            coeffs,
            x_sc,
            x_0,
        ) = fit_2d_poly(
            ds_tot.m1_x.values,
            (ds_tot.m1_y - off_y).astype(float).values,
            ds_tot[dim].values,
            degree=degree,
            norm=norm,
            weights=ds_tot.coherence.values,
        )
        attrs = {"deg": degree, "offset": x_0, "scale_fact": x_sc, "off_y": str(off_y)}

        # costruct dataarray with fit coefficients and attributes for rescaling and centering the coordinates
        poly_corr[id] = xr.DataArray(
            data=to_polygrid2d(coeffs.x, degree),
            dims=["i", "j"],
            name=id,
            coords={key: np.arange(degree + 1) for key in ["i", "j"]},
            attrs=attrs,
        )

    return xr.Dataset(poly_corr)


def stack_disparity(
    slcs: xr.DataTree,
    reference_id: str,
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
            reference_id=reference_id,
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

    # Construct disparity datatree for all bursts or for the single slice
    disparity_map = xr.DataTree()
    if multiburst:
        disparity_map['bursts'] = xr.DataTree()
        for k, v in results:
            if k in disparity_map.bursts:
                raise ValueError(f"Duplicate burst ID {k} found in results. Each burst ID should be unique.")
            disparity_map[f"bursts/{k}"] = v.to_dataset(dim="variable")
    else:
        for k, v in results:
            disparity_map["data"] = v.to_dataset(dim="variable")

    return disparity_map


def compute_stack_pearson_coefficient(
    burst_id: str,
    stack: xr.DataArray,
    reference_id: str,
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
    if reference_id not in stack.acquisition_id:
        raise Exception(f"Unable to find the master date {reference_id} in stack {stack.acquisition_id.values}")

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
        .rename(acquisition_id="z")
        .transpose("y", "x", "z")
        .chunk({'y': stack.chunksizes['azimuth_time'][0],
             'x': stack.chunksizes['slant_range_time'][0],
             'z': stack.chunksizes['acquisition_id'][0]})
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
        (xyz_stack, xyz_stack.sel(z=reference_id)),
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

    # Loop over chunks along the acquisition_id (z) dimension to keep controlled the memory usage
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
