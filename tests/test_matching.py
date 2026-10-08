import numpy as np
import pytest
import xarray as xr
from cop_ard_core.matching import initialize_patches
from scipy.constants import speed_of_light

from cop_ard_app.matching import (
    blockwise_pearson_coefficient,
    chunks_optimization,
    compute_stack_pearson_coefficient,
    estimate_poly_matching,
    get_polynomial_correction,
    stack_disparity,
)

N = 16
BURST = "iw1_254"
MASTER = "2020-01-01"
SECOND = "2020-01-02"
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1, "ns")
POL = "VV"
TOL = 1e-6

WSIZE = [8, 8]
WSTEP = [4, 4]
WSEARCH = [2, 2]
BLOCK_SIZE = 10
DIM = [1, 1]

# set the pixel -> time conversion factors to exactly 1 ns:
# azimuth: TimeCAPoly[1] * Col.SS * 1e9 ; range: 2 * Row.SS / speed_of_light
AZ_SS = 1e-9
RG_SS = speed_of_light / 2

rng = np.random.default_rng(42)
base = rng.standard_normal((N, N))


def _annotations():
    """TimeCAPoly/range-gait annotations read by the matching pipeline."""

    return {
        POL: {
            "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0]}}},
            "Grid": {"Col": {"SS": AZ_SS}, "Row": {"SS": RG_SS}},
        }
    }


def _coreg():
    """coreg/measurements Dataset with one cstack over two acquisitions.

    The secondary image is the master one rolled by one pixel along
    slant_range_time, so the azimuth/range disparities are exactly -1/0 ns
    and the coherence is exactly 1.
    """

    data = np.stack([base, np.roll(base, 1, axis=1)])[None]
    drg = np.zeros((N, N))
    daz = np.zeros((N, N))
    return xr.Dataset(
        {
            "cstack": (
                ("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
                data.astype(np.complex64),
                _annotations(),
            ),
            "drg": (("azimuth_time", "slant_range_time"), drg),
            "daz": (("azimuth_time", "slant_range_time"), daz),
        },
        coords={
            "polarization": [POL],
            "acquisition_id": [MASTER, SECOND],
            "azimuth_time": T0 + np.arange(N) * AZ_DT,
            "slant_range_time": np.arange(N, dtype=np.float64),
        },
    )


CHUNKS = {"polarization": 1, "acquisition_id": N, "azimuth_time": N, "slant_range_time": 8}


@pytest.fixture
def slcs():
    return _build_tree(_coreg().chunk(CHUNKS))


def _build_tree(measurements):
    """Scene with a single burst and its two acquisitions."""

    slcs = xr.DataTree()
    slcs[f"bursts/{BURST}/coreg/measurements"] = measurements
    for acq_id in (MASTER, SECOND):
        slcs[f"bursts/{BURST}/orig/{acq_id}/measurements"] = xr.Dataset()
    return slcs


def _xyz():
    """Stack in pixel coordinates, exactly as fed to the blockwise computation.

    Mirrors the x/y dimension swap of compute_stack_pearson_coefficient.
    """

    stack = _coreg().chunk(CHUNKS).cstack
    return (
        stack.sel(polarization=POL)
        .squeeze()
        .assign_coords(
            {k: ([j], range(stack[j].size)) for k, j in zip(["y", "x"], ["azimuth_time", "slant_range_time"])}
        )
        .swap_dims(azimuth_time="y", slant_range_time="x")
        .rename(acquisition_id="z")
        .transpose("y", "x", "z")
        .chunk({"y": N, "x": 8, "z": N})
    )


def _patches():
    """Patch set covering the stack, clipped and grouped by job_id."""

    xyz = _xyz()
    cc = initialize_patches(xyz.y.size, WSIZE[0], WSTEP[0], xyz.x.size, WSIZE[1], WSTEP[1])
    return xyz, chunks_optimization(xyz, cc)


def _results(dx, dy, m1x, m1y):
    """Fixed results tuples as produced by fix_disparity."""

    n = m1x.size
    ds = xr.Dataset(
        {
            "m1_x": ("point", m1x),
            "m1_y": ("point", m1y),
            "dx": ("point", np.full(n, dx)),
            "dy": ("point", np.full(n, dy)),
            "coherence": ("point", np.ones(n)),
        },
        coords={"point": np.arange(n)},
    )
    return [(BURST, MASTER, ds)]


class TestChunksOptimization:
    @pytest.mark.unit
    def test_optimize(self):
        """Chunks are the bounding boxes of the patches sharing a job_id."""

        xyz, (chunks, patches) = _patches()
        assert chunks == {"x": [[0, 12], [8, 16]], "y": [[0, 16], [0, 16]], "z": [[0, 2], [0, 2]]}
        # patches sticking out of the stack are dropped and the others relabelled
        assert patches.sizes == {"job_id": 9}
        np.testing.assert_array_equal(
            patches[["x_start", "x_end", "y_start", "y_end"]].to_array().to_numpy().T.tolist(),
            [
                [0, 7, 0, 7],
                [4, 11, 0, 7],
                [8, 15, 0, 7],
                [0, 7, 4, 11],
                [4, 11, 4, 11],
                [8, 15, 4, 11],
                [0, 7, 8, 15],
                [4, 11, 8, 15],
                [8, 15, 8, 15],
            ],
        )
        np.testing.assert_array_equal(patches.job_id.values, [0, 0, 1, 0, 0, 1, 0, 0, 1])
        assert (xyz.x.size, xyz.y.size) == (N, N)


class TestBlockwisePearsonCoefficient:
    @pytest.mark.unit
    def test_compute(self):
        """Identical images give zero disparity and coherence 1, shifted ones -1 ns."""

        xyz, (_, patches) = _patches()
        disparity = blockwise_pearson_coefficient(
            xyz,
            xyz.sel(z=MASTER),
            patches,
            BLOCK_SIZE,
            WSEARCH,
            np.array([0.0, 1.0]),
            AZ_SS,
            RG_SS,
            DIM,
        )
        assert disparity.sizes == {"acquisition_id": 2, "point": 6, "variable": 5}
        np.testing.assert_array_equal(disparity.coords["variable"].values, ["dy", "dx", "coherence", "s_y", "s_x"])
        # the first acquisition is the reference itself: zero disparity
        np.testing.assert_array_equal(disparity.isel(variable=0).values[0], np.zeros(6))
        np.testing.assert_array_equal(disparity.isel(variable=1).values[0], np.zeros(6))
        # a one pixel shift along x is exactly -1 ns, as the conversion factors are 1
        np.testing.assert_array_equal(disparity.isel(variable=0).values[1], np.zeros(6))
        np.testing.assert_array_equal(disparity.isel(variable=1).values[1], -np.ones(6))
        np.testing.assert_allclose(disparity.isel(variable=2).values, np.ones((2, 6)), atol=TOL)
        # patch centres in pixel and time coordinates
        np.testing.assert_array_equal(disparity.coords["x"].values, np.tile([3.5, 7.5], 3))
        np.testing.assert_array_equal(disparity.coords["y"].values, T0 + np.array([3, 3, 7, 7, 11, 11]) * AZ_DT)


class TestGetPolynomialCorrection:
    @pytest.mark.unit
    def test_fit_constant(self):
        """A constant disparity is recovered by the degree 0 polynomial."""

        m1x = np.arange(4, dtype=np.float64)
        m1y = np.arange(4, dtype=np.float64)
        poly = get_polynomial_correction([MASTER], _results(5.0, 0.0, m1x, m1y), "dx", degree=0)
        assert list(poly.data_vars) == [MASTER]
        np.testing.assert_array_equal(poly[MASTER].to_numpy(), [[5.0]])
        # the fit coordinates are centred and scaled by the points extent
        assert poly[MASTER].attrs["deg"] == 0
        np.testing.assert_array_equal(poly[MASTER].attrs["offset"], [1.5, 1.5])
        np.testing.assert_array_equal(poly[MASTER].attrs["scale_fact"], [3.0, 3.0])
        assert poly[MASTER].attrs["off_y"] == "0.0"

    @pytest.mark.unit
    def test_fit_linear(self):
        """A constant disparity is still recovered by the higher degree polynomial."""

        m1x = np.arange(4, dtype=np.float64)
        m1y = np.arange(4, dtype=np.float64)
        poly = get_polynomial_correction([MASTER], _results(-1.0, 0.0, m1x, m1y), "dx", degree=1)
        np.testing.assert_array_equal(poly[MASTER].to_numpy(), [[-1.0, 0.0], [0.0, 0.0]])
        # evaluating the polynomial over the rescaled fit points gives back the disparity
        x = (m1x - poly[MASTER].attrs["offset"][1]) / poly[MASTER].attrs["scale_fact"][1]
        y = (m1y - 0.0 - poly[MASTER].attrs["offset"][0]) / poly[MASTER].attrs["scale_fact"][0]
        xx, yy = np.meshgrid(x, y)
        np.testing.assert_array_equal(
            np.polynomial.polynomial.polyval2d(xx, yy, poly[MASTER].to_numpy()),
            np.full((4, 4), -1.0),
        )


class TestEstimatePolyMatching:
    @pytest.mark.unit
    def test_fit(self, slcs):
        """One-pixel shifted acquisitions yield constant one-ns corrections."""

        disparity_map = stack_disparity(slcs, MASTER, WSIZE, WSTEP, WSEARCH, DIM, BLOCK_SIZE, enable_dask=False)
        x_poly, y_poly = estimate_poly_matching(
            slcs,
            disparity_map,
            degree=1,
            cohe_th=0.3,
            kernel_shape=[3, 3],
            enable_dask=False,
        )
        assert list(x_poly.data_vars) == list(y_poly.data_vars)
        # the sign is flipped w.r.t. the disparity when the drg/daz grids are constant
        np.testing.assert_array_equal(x_poly[SECOND].data, [[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(x_poly[MASTER].data, np.zeros((2, 2)))
        np.testing.assert_array_equal(y_poly[SECOND].data, np.zeros((2, 2)))
        np.testing.assert_array_equal(y_poly[MASTER].data, np.zeros((2, 2)))
        attrs = x_poly[SECOND].attrs
        assert attrs["deg"] == 1
        np.testing.assert_array_equal(attrs["offset"], [4.0, 6.5])
        np.testing.assert_array_equal(attrs["scale_fact"], [8.0, 8.0])
        assert attrs["off_y"] == str(T0 + 3 * AZ_DT)


class TestStackDisparity:
    @pytest.mark.unit
    def test_map(self, slcs):
        """One disparity map per burst, one variable per correlation output."""

        result = stack_disparity(slcs, MASTER, WSIZE, WSTEP, WSEARCH, DIM, BLOCK_SIZE, enable_dask=False)
        assert result.groups == ("/", "/bursts", f"/bursts/{BURST}")
        assert list(result.bursts[BURST]) == ["dy", "dx", "coherence", "s_y", "s_x"]
        assert result.bursts[BURST].dy.sizes == {"acquisition_id": 2, "point": 9}

    @pytest.mark.unit
    def test_compute(self, slcs):
        """Master/secondary stacks give 0/-1 ns range disparities."""

        result = stack_disparity(slcs, MASTER, WSIZE, WSTEP, WSEARCH, DIM, BLOCK_SIZE, enable_dask=False)
        dx = result.bursts[BURST].dx
        np.testing.assert_array_equal(dx.values[0], np.zeros(9))
        np.testing.assert_array_equal(dx.values[1], -np.ones(9))
        np.testing.assert_array_equal(result.bursts[BURST].dy.values, np.zeros((2, 9)))
        np.testing.assert_allclose(result.bursts[BURST].coherence.values, np.ones((2, 9)), atol=TOL)
        np.testing.assert_array_equal(
            dx.coords["x"].values,
            [3.5, 7.5, 3.5, 7.5, 3.5, 7.5, 11.5, 11.5, 11.5],
        )
        np.testing.assert_array_equal(
            dx.coords["y"].values,
            T0 + np.array([3, 3, 7, 7, 11, 11, 3, 7, 11]) * AZ_DT,
        )


class TestComputeStackPearsonCoefficient:
    @pytest.mark.unit
    def test_compute(self):
        """The reference acquisition has zero disparity w.r.t. itself."""

        stack = _coreg().chunk(CHUNKS).cstack
        burst_id, disparity = compute_stack_pearson_coefficient(
            BURST,
            stack,
            MASTER,
            WSIZE,
            WSTEP,
            WSEARCH,
            BLOCK_SIZE,
            DIM,
            enable_dask=False,
        )
        assert burst_id == BURST
        assert disparity.sizes == {"acquisition_id": 2, "point": 9, "variable": 5}
        np.testing.assert_array_equal(disparity.isel(variable=0).values, np.zeros((2, 9)))
        np.testing.assert_array_equal(disparity.isel(variable=1).values[0], np.zeros(9))
        np.testing.assert_array_equal(disparity.isel(variable=1).values[1], -np.ones(9))
        np.testing.assert_allclose(disparity.isel(variable=2).values, np.ones((2, 9)), atol=TOL)
        np.testing.assert_array_equal(
            disparity.coords["x"].values,
            [3.5, 7.5, 3.5, 7.5, 3.5, 7.5, 11.5, 11.5, 11.5],
        )
        np.testing.assert_array_equal(
            disparity.coords["y"].values,
            T0 + np.array([3, 3, 7, 7, 11, 11, 3, 7, 11]) * AZ_DT,
        )

    @pytest.mark.unit
    def test_missing_reference(self):
        """The reference acquisition must belong to the stack."""

        stack = _coreg().chunk(CHUNKS).cstack
        with pytest.raises(Exception, match="Unable to find the master date 2020-01-03"):
            compute_stack_pearson_coefficient(
                BURST,
                stack,
                "2020-01-03",
                WSIZE,
                WSTEP,
                WSEARCH,
                BLOCK_SIZE,
                DIM,
                enable_dask=False,
            )
