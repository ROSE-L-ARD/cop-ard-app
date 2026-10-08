import numpy as np
import pytest
import xarray as xr

from cop_ard_app.utils import (
    build_coreg_poly_dict,
    compute_chunks,
    get_best_pol,
    get_margin,
    get_poly,
    get_spacing,
    sanitize_coordinates,
    slice_and_drop_coordinates,
)

N_AZ = 8
N_RG = 10
BURSTS = ["iw1_254", "iw1_255"]
MASTER = "2020-01-01"
SECONDARIES = ["2020-01-12"]
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
RG_DT = 1.0
POL = "VV"
SPEED_OF_LIGHT = 299_792_458.0
TOL = 1e-9

# chunk fixtures for compute_chunks
CHUNK_Y = (8, 8, 4)
CHUNK_X = (16,)

# TimeCAPoly fixture for get_margin: only the slope (coefficients[1]) is used
TIMECA = [0.0, 1.0]
AZ_SPACING = 1e-6
RG_SPACING = 1.0


def _measurements(dates, az_dt=AZ_DT, rg_dt=RG_DT, az0=0, rg0=0.0):
    """Build a coreg/measurements Dataset for one burst, with deterministic data."""
    data = np.ones((1, len(dates), N_AZ, N_RG), dtype=np.complex64)
    return xr.Dataset(
        {
            "cstack": (
                ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
                data,
                {POL: {"SigmaZeroSFPoly": {"Coefs": [[1.0]]}}},
            )
        },
        coords={
            "polarization": [POL],
            "acquisition_id": list(dates),
            "acquisition_date": ("acquisition_id", list(dates)),
            "azimuth_time": T0 + (np.arange(N_AZ) + az0) * az_dt,
            "slant_range_time": (np.arange(N_RG) + rg0) * rg_dt,
        },
    )


def _slc_tree(bursts, dates):
    """Multiburst SLC DataTree, as consumed by get_spacing."""
    tree = xr.DataTree()
    for burst in bursts:
        tree[f"bursts/{burst}/coreg/measurements"] = _measurements(dates)
    return tree


def _poly_attrs():
    """ISARIC-like polynomial attributes: a top level offset plus per acquisition ones."""
    return {
        "offset": [None, np.array([0.0, 0.0])],
        SECONDARIES[0]: {
            "offset": ["2020-01-12T00:00:04", np.array([1.5, 2.5])],
            "scale_fact": ["1000 meters", np.array([3.0, 4.0])],
        },
    }


def _poly_dataset(attrs):
    """A polynomial Dataset carrying only attributes."""
    return xr.Dataset({"p": xr.DataArray(np.ones((2, 2)), dims=["i", "j"])}, attrs=attrs)


def _correction_poly(coefficients, off_y=None, offset=(0.0, 0.0), scale_fact=(1.0, 1.0)):
    """One coregistration polynomial for a single acquisition, as built by the esd module."""
    attrs = {"deg": len(coefficients) - 1, "offset": offset, "scale_fact": scale_fact}
    if off_y is not None:
        attrs["off_y"] = off_y
    return xr.Dataset({SECONDARIES[0]: xr.DataArray(np.asarray(coefficients), dims=["i"], attrs=attrs)})


class TestGetMargin:
    """Unit test for cop_ard_app.utils.get_margin."""

    @pytest.mark.unit
    def test_margin_scales_the_selection_step(self):
        margin = get_margin(np.array(TIMECA), AZ_SPACING, RG_SPACING, border=10)
        assert set(margin) == {"azimuth_time", "slant_range_time"}
        # the azimuth step is the TimeCAPoly slope times the azimuth spacing, in integer nanoseconds
        assert margin["azimuth_time"] == np.timedelta64(10 * round(TIMECA[1] * AZ_SPACING * 1e9), "ns")
        # the range margin is twice the light travel time over border pixels
        assert margin["slant_range_time"] == pytest.approx(2 * RG_SPACING * 10 / SPEED_OF_LIGHT, rel=TOL)

    @pytest.mark.unit
    def test_margin_ignores_the_time_intercept(self):
        shifted = np.array([123.0, TIMECA[1]])
        assert get_margin(shifted, AZ_SPACING, RG_SPACING) == get_margin(np.array(TIMECA), AZ_SPACING, RG_SPACING)

    @pytest.mark.unit
    def test_margin_is_linear_in_the_border(self):
        narrow = get_margin(np.array(TIMECA), AZ_SPACING, RG_SPACING, border=1)
        wide = get_margin(np.array(TIMECA), AZ_SPACING, RG_SPACING, border=4)
        assert wide["azimuth_time"] == 4 * narrow["azimuth_time"]
        assert wide["slant_range_time"] == pytest.approx(4 * narrow["slant_range_time"], rel=TOL)


class TestComputeChunks:
    """Unit test for cop_ard_app.utils.compute_chunks."""

    @pytest.mark.unit
    def test_compute_chunks_expands_chunk_bounds(self):
        out = compute_chunks({"y": CHUNK_Y, "x": CHUNK_X}, overlap={"y": 2, "x": 1})
        # every y window is extended by the overlap, and the last one starts overlap pixels early
        assert out["y"] == [[0, 10], [8, 18], [16, 20]]
        # a single chunk along x is not overlapped, and is repeated for every y window
        assert out["x"] == [[0, 16], [0, 16], [0, 16]]
        assert len(out["y"]) == len(out["x"]) == 3

    @pytest.mark.unit
    def test_compute_chunks_leaves_unoverlapped_dimensions_plain(self):
        out = compute_chunks({"y": CHUNK_Y, "x": CHUNK_X, "z": (4, 4)}, overlap={"y": 2, "x": 1})
        # z is not requested to overlap, so it keeps plain [start, end] bounds
        assert out["z"] == [[0, 4], [4, 8], [0, 4], [4, 8], [0, 4], [4, 8]]
        assert out["y"] == [[0, 10], [0, 10], [8, 18], [8, 18], [16, 20], [16, 20]]
        assert out["x"] == [[0, 16]] * 6

    @pytest.mark.unit
    def test_compute_chunks_single_chunk_has_plain_bounds(self):
        out = compute_chunks({"y": (8,), "x": (16,)}, overlap={"y": 1, "x": 1})
        assert out == {"y": [[0, 8]], "x": [[0, 16]]}

    @pytest.mark.unit
    def test_compute_chunks_rejects_overlap_not_smaller_than_chunk(self):
        with pytest.raises(Exception, match="Overlap 8 is bigger than chunk length 8"):
            compute_chunks({"y": (8, 8), "x": CHUNK_X}, overlap={"y": 8, "x": 1})
        with pytest.raises(Exception, match="Overlap 16 is bigger than chunk length 16"):
            compute_chunks({"y": CHUNK_Y, "x": (16,)}, overlap={"y": 1, "x": 16})


class TestGetSpacing:
    """Unit test for cop_ard_app.utils.get_spacing.

    The function iterates over the values of the mapping it is given, so it is called
    with the `.bursts` mapping of an SLC DataTree.
    """

    @pytest.mark.unit
    def test_get_spacing_averages_the_burst_spacings(self):
        slcs = xr.DataTree()
        slcs["bursts/b1/coreg/measurements"] = _measurements([MASTER], az_dt=AZ_DT, rg_dt=RG_DT)
        slcs["bursts/b2/coreg/measurements"] = _measurements([MASTER], az_dt=2 * AZ_DT, rg_dt=2 * RG_DT)
        dt_az, dt_rg = get_spacing(slcs.bursts, "coreg/measurements")
        # azimuth spacing is returned in nanoseconds
        assert dt_az == pytest.approx(1.5e9, rel=TOL)
        assert dt_rg == pytest.approx(1.5 * RG_DT, rel=TOL)

    @pytest.mark.unit
    def test_get_spacing_is_the_spacing_of_a_single_burst(self):
        slcs = _slc_tree(BURSTS[:1], [MASTER, SECONDARIES[0]])
        dt_az, dt_rg = get_spacing(slcs.bursts, "coreg/measurements")
        assert dt_az == pytest.approx(AZ_DT.astype(np.int64), rel=TOL)
        assert dt_rg == pytest.approx(RG_DT, rel=TOL)

    @pytest.mark.unit
    def test_get_spacing_rejects_empty_coordinates(self):
        slcs = xr.DataTree()
        slcs["bursts/b1/coreg/measurements"] = xr.Dataset(
            {"cstack": ("slant_range_time", np.zeros(N_RG, dtype=np.complex64))},
            coords={
                "azimuth_time": np.array([], dtype="datetime64[ns]"),
                "slant_range_time": np.arange(N_RG, dtype=np.float64),
            },
        )
        with pytest.raises(ValueError, match="Null slc in input"):
            get_spacing(slcs.bursts, "coreg/measurements")


class TestGetPoly:
    """Unit test for cop_ard_app.utils.get_poly."""

    @pytest.mark.unit
    def test_get_poly_normalizes_the_poly_attribute(self):
        poly = _poly_dataset(_poly_attrs())
        out = get_poly(poly, "azimuth", SECONDARIES[0])
        # the top level azimuth offset is promoted to a nanosecond datetime
        assert out.attrs["offset"][0] == np.datetime64("2020-01-12T00:00:04", "ns")
        # the per acquisition offsets and scale factors become arrays
        assert np.array_equal(out.attrs[SECONDARIES[0]]["offset"][1], np.array([1.5, 2.5]))
        assert out.attrs[SECONDARIES[0]]["scale_fact"][0] == np.array(1000)
        assert np.array_equal(out.attrs[SECONDARIES[0]]["scale_fact"][1], np.array([3.0, 4.0]))

    @pytest.mark.unit
    def test_get_poly_leaves_the_rest_of_the_attributes_alone(self):
        out = get_poly(_poly_dataset(_poly_attrs()), "range", SECONDARIES[0])
        # only the first element of the top level offset is rewritten
        assert np.array_equal(out.attrs["offset"][1], np.array([0.0, 0.0]))
        assert out.attrs[SECONDARIES[0]]["offset"][0] == "2020-01-12T00:00:04"
        assert np.array_equal(out.attrs[SECONDARIES[0]]["scale_fact"][1], np.array([3.0, 4.0]))
        assert np.array_equal(out["p"].values, np.ones((2, 2)))


class TestSliceAndDropCoordinates:
    """Unit test for cop_ard_app.utils.slice_and_drop_coordinates."""

    @pytest.mark.unit
    def test_slice_and_drop_coordinates_keeps_only_listed_coordinates(self):
        da = xr.DataArray(
            np.arange(20.0).reshape(4, 5),
            dims=["a", "b"],
            coords={"a": np.arange(4), "b": np.arange(5), "c": ("a", np.arange(4))},
        )
        out = slice_and_drop_coordinates(da, ["a"], slices={"b": slice(0, 2)})
        assert out.shape == (4, 2)
        assert list(out.coords) == ["a"]
        assert np.array_equal(out.isel(b=0).values, np.arange(20.0).reshape(4, 5)[:, 0])

    @pytest.mark.unit
    def test_slice_and_drop_coordinates_without_slices_only_drops(self):
        da = xr.DataArray(
            np.arange(20.0).reshape(4, 5),
            dims=["a", "b"],
            coords={"a": np.arange(4), "b": np.arange(5), "c": ("a", np.arange(4))},
        )
        out = slice_and_drop_coordinates(da, ["a", "b"])
        assert out.shape == (4, 5)
        assert list(out.coords) == ["a", "b"]
        assert np.array_equal(out.values, da.values)


class TestSanitizeCoordinates:
    """Unit test for cop_ard_app.utils.sanitize_coordinates.

    Non dimension coordinates are re-attached with their own dimensions and values, so
    well formed objects are returned unchanged and the operation is idempotent.
    """

    @pytest.mark.unit
    def test_sanitize_coordinates_preserves_every_coordinate_value(self):
        ds = xr.Dataset(
            {"v": ("x", np.arange(3.0))},
            coords={
                "x": np.arange(3),
                "w": ("x", np.arange(3.0)),
                "y": ("x2", np.arange(2.0)),
                "z": 5.0,
            },
        )
        out = sanitize_coordinates(ds)
        assert list(out.coords) == list(ds.coords)
        for name in ds.coords:
            assert out.coords[name].dims == ds.coords[name].dims
            assert np.array_equal(np.asarray(out.coords[name].values), np.asarray(ds.coords[name].values))
        assert np.array_equal(out["v"].values, ds["v"].values)

    @pytest.mark.unit
    def test_sanitize_coordinates_accepts_a_dataarray(self):
        da = xr.DataArray(
            np.arange(3.0),
            dims=["x"],
            coords={"x": np.arange(3), "w": ("x", np.arange(3.0)), "z": 5.0},
        )
        out = sanitize_coordinates(da)
        assert out.dims == da.dims
        assert list(out.coords) == list(da.coords)
        assert np.array_equal(out.values, da.values)
        assert np.array_equal(out.coords["w"].values, da.coords["w"].values)

    @pytest.mark.unit
    def test_sanitize_coordinates_is_idempotent(self):
        ds = xr.Dataset({"v": ("x", np.arange(3.0))}, coords={"x": np.arange(3), "z": 5.0})
        once = sanitize_coordinates(ds)
        twice = sanitize_coordinates(once)
        assert list(twice.coords) == list(ds.coords)
        assert np.array_equal(twice["v"].values, ds["v"].values)
        assert np.array_equal(twice.coords["z"].values, ds.coords["z"].values)


class TestGetBestPol:
    """Unit test for cop_ard_app.utils.get_best_pol."""

    @pytest.mark.unit
    def test_get_best_pol_follows_the_priority_list(self):
        assert get_best_pol(["HV", "VH", "HH", "VV"]) == "VV"
        assert get_best_pol(["HV", "VH", "HH"]) == "HH"
        assert get_best_pol(["HV", "VH"]) == "VH"
        assert get_best_pol(["HV"]) == "HV"

    @pytest.mark.unit
    def test_get_best_pol_coerces_the_available_polarizations(self):
        assert get_best_pol([None, 42, "VV"]) == "VV"

    @pytest.mark.unit
    def test_get_best_pol_rejects_unknown_polarizations(self):
        with pytest.raises(Exception, match="No polarization has been found"):
            get_best_pol(["XX", "YY"])


class TestBuildCoregPolyDict:
    """Unit test for cop_ard_app.utils.build_coreg_poly_dict."""

    @pytest.mark.unit
    def test_build_coreg_poly_dict_serializes_every_direction(self):
        y_poly = _correction_poly(
            [0.0, 1.0, 2.0, 3.0], off_y="2020-01-12T00:00:04", offset=(1.0, 2.0), scale_fact=(3.0, 4.0)
        )
        x_poly = _correction_poly([1.0, 1.0], offset=(5.0, 6.0), scale_fact=(7.0, 8.0))
        iono_poly = _correction_poly([9.0])
        out = build_coreg_poly_dict("coregistration", y_poly=y_poly, x_poly=x_poly, iono_poly=iono_poly)
        assert out["description"] == "coregistration"
        assert list(out["acquisitions"]) == [SECONDARIES[0]]
        az = out["acquisitions"][SECONDARIES[0]]["azimuth"]
        assert az["coefficients"] == [0.0, 1.0, 2.0, 3.0]
        assert az["shape"] == [4]
        assert (az["azimuth_offset"], az["range_offset"]) == (1.0, 2.0)
        assert (az["azimuth_scale"], az["range_scale"]) == (3.0, 4.0)
        assert az["azimuth_time_reference_ns"] == "2020-01-12T00:00:04"
        rg = out["acquisitions"][SECONDARIES[0]]["range"]
        assert rg["coefficients"] == [1.0, 1.0]
        assert (rg["azimuth_offset"], rg["range_offset"]) == (5.0, 6.0)
        assert (rg["azimuth_scale"], rg["range_scale"]) == (7.0, 8.0)
        # a polynomial without the off_y attribute gets an empty time reference
        assert rg["azimuth_time_reference_ns"] == ""
        assert out["acquisitions"][SECONDARIES[0]]["ionosphere"]["coefficients"] == [9.0]
        assert out["acquisitions"][SECONDARIES[0]]["ionosphere"]["shape"] == [1]

    @pytest.mark.unit
    def test_build_coreg_poly_dict_defaults_are_used_when_absent(self):
        out = build_coreg_poly_dict("no correction", y_poly=_correction_poly([0.0]))
        az = out["acquisitions"][SECONDARIES[0]]["azimuth"]
        assert (az["azimuth_offset"], az["range_offset"]) == (0.0, 0.0)
        assert (az["azimuth_scale"], az["range_scale"]) == (1.0, 1.0)
        assert az["azimuth_time_reference_ns"] == ""

    @pytest.mark.unit
    def test_build_coreg_poly_dict_without_polynomials_is_empty(self):
        assert build_coreg_poly_dict("nothing applied") == {"description": "nothing applied", "acquisitions": {}}
