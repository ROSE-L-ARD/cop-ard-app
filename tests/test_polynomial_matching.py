# Tests that a secondary image shifted from a reference image by a
# spatially-varying polynomial (degree 1 and 2) is correctly identified by the
# image-matching + polynomial-fitting pipeline.
#
# Pipeline under test (the same building blocks the app composes in
# ``stack_disparity`` / ``estimate_poly_matching``):
#
#   1. *matching* -- ``cop_ard_core.matching.compute_pearson_coefficient``
#      finds the sub-pixel offset that best aligns each image patch with the
#      reference patch. The offset it returns is the *disparity*, i.e. the
#      correction needed to bring the secondary back onto the reference.
#   2. *polyfit* -- ``cop_ard_core.detrend.fit_2d_poly``
#      fits a 2-D polynomial of the requested degree to the measured
#      disparities.
#
# Ground truth: the secondary is created by displacing the reference by a known
# polynomial *shift* field, so the expected disparity is the *negative* of that
# shift. The test asserts that the fitted polynomial recovers the expected
# disparity within a tolerance margin.

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import map_coordinates

from cop_ard_core.detrend import fit_2d_poly, get_model_matrix
from cop_ard_core.matching import compute_pearson_coefficient

SIZE = 256
PATCH_HALF = 32  # patch side is 2 * PATCH_HALF = 64 pixels
WSEARCH = 6  # search half-window (pixels) for the cross-correlation peak
EVAL = slice(48, 208)  # interior region over which the fitted surface is checked

# Tolerances: Measured pointwise errors are of
# order 0.1 px, so these bounds give a comfortable margin while still rejecting
# a grossly wrong (or degenerate) fit.
ABS_TOL_PX = 0.2  # max pointwise deviation between recovered and true disparity
CORR_TOL = 0.9  # correlation between the recovered and true disparity pattern


def _reference_image(size=SIZE, seed=0):
    """Deterministic textured amplitude image with rich multi-scale content."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = (
        np.sin(2 * np.pi * xx / 37.0) * np.cos(2 * np.pi * yy / 53.0)
        + 0.7 * np.sin(2 * np.pi * (xx + yy) / 71.0)
        + 0.5 * np.cos(2 * np.pi * xx / 13.0)
        + 0.3 * np.sin(2 * np.pi * yy / 19.0)
    )
    return np.abs(base + 0.2 * rng.standard_normal(base.shape))


def _shift_fields(degree, size=SIZE):
    """Ground-truth shift field (in pixels) by which the secondary is displaced
    from the reference. Returns ``(shift_x, shift_y)`` as 2-D arrays."""
    yy, xx = np.mgrid[0:size, 0:size]
    xn = (xx / size - 0.5) * 2  # normalised coordinates in [-1, 1]
    yn = (yy / size - 0.5) * 2
    if degree == 1:
        shift_x = 0.15 + 0.30 * xn + 0.10 * yn
        shift_y = -0.10 + 0.20 * xn + 0.18 * yn
    else:  # degree == 2
        shift_x = 0.10 + 0.20 * xn + 0.10 * yn + 0.15 * xn**2 + 0.05 * xn * yn + 0.12 * yn**2
        shift_y = -0.05 + 0.15 * xn + 0.08 * yn + 0.10 * xn**2 - 0.04 * xn * yn + 0.14 * yn**2
    return shift_x, shift_y


def _shifted_secondary(reference, shift_x, shift_y, order=5):
    """Displace ``reference`` by ``(shift_x, shift_y)`` with sub-pixel sampling."""
    yy, xx = np.mgrid[0 : reference.shape[0], 0 : reference.shape[1]]
    return map_coordinates(reference, [yy - shift_y, xx - shift_x], order=order, mode="nearest")


def _patch_centers():
    """Regular grid of patch centre coordinates, kept clear of the image border."""
    centers = [40, 80, 128, 176, 216]
    cx, cy = np.meshgrid(centers, centers)
    return cx.ravel().astype(int), cy.ravel().astype(int)


def _measure_disparity(reference, secondary):
    """Run sub-pixel matching on every patch. Returns ``(xs, ys, dxs, dys)``:
    the patch-centre coordinates and the measured disparity, all in pixels."""
    h = PATCH_HALF
    stack = xr.DataArray(
        np.stack([reference, secondary], axis=-1),
        dims=["y", "x", "z"],
        coords={
            "y": np.arange(reference.shape[0]),
            "x": np.arange(reference.shape[1]),
            "z": ["ref", "sec"],
        },
    )
    ref_da = xr.DataArray(
        reference,
        dims=["y", "x"],
        coords={"y": np.arange(reference.shape[0]), "x": np.arange(reference.shape[1])},
    )
    xs, ys, dxs, dys = [], [], [], []
    for x, y in zip(*_patch_centers()):
        js = stack.sel(x=slice(x - h, x + h + 1), y=slice(y - h, y + h + 1))
        jr = ref_da.sel(x=slice(x - h, x + h + 1), y=slice(y - h, y + h + 1))
        r = compute_pearson_coefficient(
            patch=xr.Dataset(coords={"patch": 0}),
            stack=js,
            reference=jr,
            pixel_to_time=np.array([[1.0], [1.0]], dtype="float64"),
            peak_window_size=[WSEARCH, WSEARCH],
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=[1, 1],
        )
        dxs.append(r.sel(acquisition_id="sec", variable="dx").values[0])
        dys.append(r.sel(acquisition_id="sec", variable="dy").values[0])
        xs.append(float(js["x"].mean()))
        ys.append(float(js["y"].mean()))
    return np.array(xs), np.array(ys), np.array(dxs), np.array(dys)


def _fit_surface(xs, ys, data, degree, yy, xx):
    """Fit a ``degree``-order 2-D polynomial to ``data`` sampled at ``(xs, ys)``
    and evaluate it on the pixel grid given by the ``(yy, xx)`` mesh."""
    coeffs, x_sc, x_0 = fit_2d_poly(xs, ys, data, degree=degree, norm=1)
    y_norm = (yy - x_0[0]) / x_sc[0]
    x_norm = (xx - x_0[1]) / x_sc[1]
    model = get_model_matrix(np.vstack([y_norm.ravel(), x_norm.ravel()]).T, degree=degree)
    return (model @ coeffs.x).reshape(yy.shape)


class TestPolynomialMatching():

    @pytest.mark.unit
    @pytest.mark.parametrize("degree", [1, 2])
    def test_secondary_shifted_by_polynomial_is_recovered(self, degree):
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        reference = _reference_image()
        shift_x, shift_y = _shift_fields(degree)

        secondary = _shifted_secondary(reference, shift_x, shift_y)

        # --- matching: measure the disparity (correction) at every patch centre
        xs, ys, dxs, dys = _measure_disparity(reference, secondary)

        # The expected disparity is the correction back to the reference, i.e. the
        # negative of the shift field used to build the secondary.
        expected = {"x": -shift_x, "y": -shift_y}

        # --- polyfit: fit a `degree`-order polynomial to the disparities
        recovered = {
            "x": _fit_surface(xs, ys, dxs, degree, yy, xx),
            "y": _fit_surface(xs, ys, dys, degree, yy, xx),
        }

        for axis in ("x", "y"):
            err = (recovered[axis] - expected[axis])[EVAL, :][:, EVAL]
            rec = recovered[axis][EVAL, :][:, EVAL].ravel()
            tru = expected[axis][EVAL, :][:, EVAL].ravel()

            # (1) the polynomial is identified pointwise within a tolerance margin
            max_err = float(np.abs(err).max())
            assert max_err <= ABS_TOL_PX, (
                f"degree {degree}, axis {axis}: max disparity error "
                f"{max_err:.3f} px exceeds tolerance {ABS_TOL_PX} px"
            )
            # (2) the spatial pattern of the polynomial is captured (not degenerate)
            corr = float(np.corrcoef(rec, tru)[0, 1])
            assert corr > CORR_TOL, (
                f"degree {degree}, axis {axis}: recovered pattern correlation "
                f"{corr:.3f} is below tolerance {CORR_TOL}"
            )
