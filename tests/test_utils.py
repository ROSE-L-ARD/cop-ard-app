import numpy as np
import pytest
import xarray as xr

from cop_ard_app.model import get_az_rg_times
from cop_ard_app.utils import (
    build_coreg_poly_dict,
    compute_chunks,
    compute_iono_and_orb_height,
    get_bbox,
    get_best_pol,
    get_margin,
    get_poly,
    get_spacing,
    get_stack_phramp_diff,
    sanitize_coordinates,
    save_single_burst,
    slice_and_drop_coordinates,
    sparse_multilook,
    sparse_multilook_core,
    update_reference_database,
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

# scene metadata fixtures for get_stack_phramp_diff
SCP_COL = N_AZ // 2
SCP_ROW = N_RG // 2
COL_SS = 1.0
ROW_SS = 1.0
DKCOA = [[0.05, 0.0], [0.0, 0.0]]
COLLECT_START = "2020-01-01T00:00:00"
R_CA_SCP = 800_000.0
Y_POLY_OFF_Y = str(T0 + np.timedelta64(4, "s"))

# scene corners for get_bbox: (lat, lon) pairs per acquisition date, and the expected
# bounding boxes rounded outward to one decimal place
CORNERS = {
    MASTER: [(41.234, 12.345), (41.234, 12.999), (41.9, 12.999), (41.9, 12.345)],
    SECONDARIES[0]: [(40.0, 11.0), (40.0, 13.5), (43.0, 13.5), (43.0, 11.0)],
}
BBOX_MASTER = [12.3, 41.2, 13.0, 41.9]
BBOX_BOTH = [11.0, 40.0, 13.5, 43.0]

# ARPPos on the geoid sphere 700 km above the equator, so h_orb is known in closed form
ARPPOS = {"X": 6_378_137.0 + 700_000.0, "Y": 0.0, "Z": 0.0}
H_ORB = 700_000.0
DEFAULT_IONO = 350_000.0


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


def _annotation(dkcoa=DKCOA):
    """Minimal SAR metadata dictionary as consumed by get_az_rg_times and get_coefficients."""
    return {
        "ImageData": {
            "FullImage": {"NumCols": N_AZ, "NumRows": N_RG},
            "SCPPixel": {"Col": SCP_COL, "Row": SCP_ROW},
        },
        "RMA": {"INCA": {"TimeCAPoly": {"Coefs": list(TIMECA)}, "R_CA_SCP": R_CA_SCP}},
        "Timeline": {"CollectStart": COLLECT_START},
        "Grid": {
            "Col": {
                "SS": COL_SS,
                "DeltaKCOAPoly": {"Coefs": [list(r) for r in dkcoa]} if dkcoa is not None else {},
            },
            "Row": {"SS": ROW_SS},
        },
    }


def _stack_times():
    """Azimuth/slant-range coordinate arrays consistent with _annotation()."""
    azimuth_time, slant_range_time = get_az_rg_times(_annotation())
    return (
        xr.DataArray(azimuth_time, dims=["azimuth_time"]),
        xr.DataArray(slant_range_time, dims=["slant_range_time"]),
    )


def _shift_stack(value: float):
    """daz (timedelta) and drg (float) stacks for a single acquisition."""
    dims = ["acquisition_id", "azimuth_time", "slant_range_time"]
    coords = {
        "acquisition_id": [SECONDARIES[0]],
        "azimuth_time": T0 + np.arange(N_AZ) * AZ_DT,
        "slant_range_time": np.arange(N_RG, dtype=np.float64) * RG_DT,
    }
    daz = np.zeros((1, N_AZ, N_RG)).astype("timedelta64[ns]") + np.timedelta64(int(value * 1e9), "ns")
    drg = np.zeros((1, N_AZ, N_RG), dtype=np.float64)
    return xr.DataArray(daz, dims=dims, coords=coords), xr.DataArray(drg, dims=dims, coords=coords)


def _scene_poly(shift: float):
    """Constant azimuth correction polynomial with the attributes get_phramp_diff expects."""
    return xr.Dataset(
        {
            SECONDARIES[0]: xr.DataArray(
                np.array([[shift]], dtype=np.float64),
                dims=["i", "j"],
                attrs={"deg": 0, "offset": (0.0, 0.0), "scale_fact": (1.0, 1.0), "off_y": Y_POLY_OFF_Y},
            )
        }
    )


def _coherence_dataset(seed: int = 3):
    """A measurements-like Dataset with the coherence/value variables sparse_multilook aggregates."""
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "coherence": (["azimuth_time", "slant_range_time"], rng.random((N_AZ, N_RG))),
            "value": (["azimuth_time", "slant_range_time"], rng.standard_normal((N_AZ, N_RG))),
        },
        coords={
            "azimuth_time": T0 + np.arange(N_AZ) * AZ_DT,
            "slant_range_time": np.arange(N_RG, dtype=np.float64),
        },
    )


def _scene_tree(bursts, dates, corners=None, arppos=None, multiburst=True):
    """SLC DataTree with coreg/measurements and orig nodes carrying scene metadata."""
    tree = xr.DataTree()
    for burst in bursts:
        prefix = f"bursts/{burst}/" if multiburst else ""
        tree[f"{prefix}coreg/measurements"] = _measurements(dates)
        for date in dates:
            path = f"{prefix}orig/{date}/measurements/slc"
            slc = xr.Dataset(
                {"x": ("azimuth_time", np.arange(N_AZ, dtype=np.float64))},
                coords={"azimuth_time": T0 + np.arange(N_AZ) * AZ_DT},
            )
            scene = {}
            if corners is not None:
                scene["GeoData"] = {"ImageCorners": [{"Lat": la, "Lon": lo} for la, lo in corners[date]]}
            if arppos is not None:
                scene["SCPCOA"] = {"ARPPos": arppos}
            slc.attrs = {POL: scene}
            tree[path] = slc
    return tree


def _reference_tree(prefix, dates):
    """A current/stable reference tree, as consumed by update_reference_database."""
    tree = xr.DataTree()
    tree[f"{prefix}measurements"] = xr.Dataset({"cstack": _measurements(dates)["cstack"]})
    return tree


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


class TestGetStackPhrampDiff:
    """Unit test for cop_ard_app.utils.get_stack_phramp_diff.

    With a zero shift stack, a constant DeltaKCOAPoly of DKCOA and a constant azimuth
    polynomial of `shift` pixels moves the modulation phase ramp by 2 * pi * DKCOA * shift
    radians at every pixel of the acquisition grid.
    """

    @pytest.mark.unit
    def test_zero_polynomial_is_a_zero_correction(self):
        azimuth_time, slant_range_time = _stack_times()
        daz, drg = _shift_stack(0.0)
        res = get_stack_phramp_diff(
            xr.DataArray([SECONDARIES[0]], dims=["acquisition_id"]),
            azimuth_time,
            slant_range_time,
            drg,
            daz,
            {SECONDARIES[0]: _annotation()},
            _scene_poly(0.0),
        )
        assert res.dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert res.sizes == {"acquisition_id": 1, "azimuth_time": N_AZ, "slant_range_time": N_RG}
        assert np.all(res.values == 0.0)

    @pytest.mark.unit
    def test_recovers_a_constant_azimuth_shift(self):
        azimuth_time, slant_range_time = _stack_times()
        daz, drg = _shift_stack(0.0)
        shift = 2.0
        res = get_stack_phramp_diff(
            xr.DataArray([SECONDARIES[0]], dims=["acquisition_id"]),
            azimuth_time,
            slant_range_time,
            drg,
            daz,
            {SECONDARIES[0]: _annotation()},
            _scene_poly(shift),
        )
        expected = 2 * np.pi * DKCOA[0][0] * shift
        assert np.allclose(res.values, expected, atol=1e-9)
        assert np.all(np.isfinite(res.values))

    @pytest.mark.unit
    def test_multilook_coarsens_the_shift_stack_before_the_ramp(self):
        azimuth_time, slant_range_time = _stack_times()
        daz, drg = _shift_stack(0.0)
        # the ramp is requested on a grid half the resolution of the shift stack
        coarse_az = azimuth_time.isel(azimuth_time=slice(None, None, 2))
        coarse_rg = slant_range_time.isel(slant_range_time=slice(None, None, 2))
        res = get_stack_phramp_diff(
            xr.DataArray([SECONDARIES[0]], dims=["acquisition_id"]),
            coarse_az,
            coarse_rg,
            drg,
            daz,
            {SECONDARIES[0]: _annotation()},
            _scene_poly(2.0),
            multilook={"azimuth_time": 2, "slant_range_time": 2},
        )
        # coarsening a constant shift stack leaves it constant, so the ramp is unchanged
        assert res.sizes == {"acquisition_id": 1, "azimuth_time": N_AZ // 2, "slant_range_time": N_RG // 2}
        assert np.allclose(res.values, 2 * np.pi * DKCOA[0][0] * 2.0, atol=1e-9)

    @pytest.mark.unit
    def test_rejects_an_annotation_without_delta_kcoa_coefficients(self):
        azimuth_time, slant_range_time = _stack_times()
        daz, drg = _shift_stack(0.0)
        with pytest.raises(ValueError, match="Missing DeltaKCOA polynomial coefficients"):
            get_stack_phramp_diff(
                xr.DataArray([SECONDARIES[0]], dims=["acquisition_id"]),
                azimuth_time,
                slant_range_time,
                drg,
                daz,
                {SECONDARIES[0]: _annotation(dkcoa=None)},
                _scene_poly(0.0),
            )


class TestSparseMultilook:
    """Unit test for cop_ard_app.utils.sparse_multilook and sparse_multilook_core.

    The aggregation (coordinate change, coherence masking, block indexing and grouped median)
    runs to completion; the pinned xarray release then rejects the final `xr.Dataset(<Dataset>)`
    reassembly, so the public entry point raises a TypeError once it is done.
    """

    @pytest.mark.unit
    def test_sparse_multilook_rejects_unknown_multilook_dimensions(self):
        with pytest.raises(ValueError, match=r"Not all \['azimuth_time', 'nope'\] in"):
            sparse_multilook(_coherence_dataset(), cohe_th=0.5, multilook={"azimuth_time": 2, "nope": 2})

    @pytest.mark.unit
    def test_sparse_multilook_aggregates_before_the_final_reassembly(self):
        with pytest.raises(TypeError, match="Passing a Dataset as"):
            sparse_multilook(_coherence_dataset(), cohe_th=0.5, multilook={"azimuth_time": 2, "slant_range_time": 2})

    @pytest.mark.unit
    def test_sparse_multilook_declares_the_multilook_on_the_input_dims(self):
        # a grid already linearized for sparse_multilook_core is not a valid input here
        with pytest.raises(ValueError, match=r"Not all \['azimuth_time', 'slant_range_time'\] in"):
            sparse_multilook(
                _coherence_dataset()
                .assign_coords(
                    {
                        "y_": ("azimuth_time", np.arange(N_AZ, dtype=float)),
                        "x_": ("slant_range_time", np.arange(N_RG, dtype=float)),
                    }
                )
                .swap_dims({"azimuth_time": "y_", "slant_range_time": "x_"}),
                cohe_th=0.5,
                multilook={"azimuth_time": 2, "slant_range_time": 2},
            )

    @pytest.mark.unit
    def test_sparse_multilook_core_groups_the_median_by_block(self):
        data = (
            _coherence_dataset()
            .assign_coords(
                {
                    "y_": ("azimuth_time", np.arange(N_AZ, dtype=float)),
                    "x_": ("slant_range_time", np.arange(N_RG, dtype=float)),
                }
            )
            .swap_dims({"azimuth_time": "y_", "slant_range_time": "x_"})
        )
        with pytest.raises(TypeError, match="Passing a Dataset as"):
            sparse_multilook_core(data, cohe_th=0.5, multilook={"y_": 2, "x_": 2})


class TestGetBbox:
    """Unit test for cop_ard_app.utils.get_bbox."""

    @pytest.mark.unit
    def test_get_bbox_encompasses_every_scene_corner(self):
        slcs = _scene_tree(BURSTS, [MASTER, SECONDARIES[0]], corners=CORNERS)
        assert get_bbox(slcs) == BBOX_BOTH

    @pytest.mark.unit
    def test_get_bbox_rounds_the_scene_corners_outward(self):
        slcs = _scene_tree(BURSTS[:1], [MASTER, SECONDARIES[0]], corners=CORNERS)
        # restricting to the reference date keeps only its corners in the bounding box
        assert get_bbox(slcs, reference_date=MASTER) == BBOX_MASTER

    @pytest.mark.unit
    def test_get_bbox_accepts_a_single_burst_layout(self):
        slcs = _scene_tree(["b1"], [MASTER], corners=CORNERS, multiburst=False)
        assert get_bbox(slcs) == BBOX_MASTER


class TestSaveSingleBurst:
    """Unit test for cop_ard_app.utils.save_single_burst (writes to temporary zarr stores)."""

    @pytest.mark.unit
    def test_save_single_burst_writes_current_and_stable_references(self, tmp_path):
        save_single_burst(
            "b1", _measurements([MASTER])["cstack"], _measurements(list(SECONDARIES))["cstack"], str(tmp_path), True
        )
        assert (tmp_path / "bursts/b1/current_reference").is_dir()
        assert (tmp_path / "bursts/b1/stable_reference" / SECONDARIES[0]).is_dir()
        assert not (tmp_path / "bursts/b1/stable_reference" / MASTER).exists()

    @pytest.mark.unit
    def test_save_single_burst_without_stable_reference_skips_it(self, tmp_path):
        save_single_burst("b1", _measurements([MASTER])["cstack"], None, str(tmp_path), True)
        assert (tmp_path / "bursts/b1/current_reference").is_dir()
        assert not (tmp_path / "bursts/b1/stable_reference").exists()

    @pytest.mark.unit
    def test_save_single_burst_single_burst_requires_reference_id(self, tmp_path):
        with pytest.raises(ValueError, match="the reference_id is mandatory"):
            save_single_burst("b1", _measurements([MASTER])["cstack"], None, str(tmp_path), False)
        save_single_burst("b1", _measurements([MASTER])["cstack"], None, str(tmp_path), False, reference_id="r1")
        assert (tmp_path / "data/r1/current_reference").is_dir()
        assert not (tmp_path / "data/r1/stable_reference").exists()
        assert not (tmp_path / "bursts").exists()


class TestUpdateReferenceDatabase:
    """Unit test for cop_ard_app.utils.update_reference_database.

    The dask dispatch is replaced by a recorder: the unit under test is the argument
    assembly (burst ids, current and stable references) handed over to `save_single_burst`.
    """

    @pytest.mark.unit
    def test_update_reference_database_transposes_the_burst_arguments(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr("cop_ard_app.utils.map", lambda func, args, **kw: calls.append((func, args, kw)))
        current = xr.DataTree()
        stable = xr.DataTree()
        for burst in BURSTS:
            current[f"bursts/{burst}/measurements"] = xr.Dataset({"cstack": _measurements([MASTER])["cstack"]})
            stable[f"bursts/{burst}/measurements"] = xr.Dataset({"cstack": _measurements(list(SECONDARIES))["cstack"]})
        update_reference_database(current, stable, str(tmp_path))

        ((func, args, dispatch),) = calls
        assert func is save_single_burst
        expected = {"reference_db_path": str(tmp_path), "multiburst": True, "reference_id": None}
        assert dispatch == {"kwargs": expected, "enable_dask": True}
        burst_ids, current_args, stable_args = args
        assert burst_ids == tuple(BURSTS)
        assert [c.sizes["acquisition_id"] for c in current_args] == [1, 1]
        assert [s.sizes["acquisition_id"] for s in stable_args] == [1, 1]

    @pytest.mark.unit
    def test_update_reference_database_drops_a_dummy_stable_tree(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr("cop_ard_app.utils.map", lambda func, args, **kw: calls.append((func, args, kw)))
        current = _reference_tree(f"bursts/{BURSTS[0]}/", [MASTER])
        dummy = xr.DataTree(dataset=xr.Dataset({"status": xr.DataArray(0).assign_attrs(is_empty=True)}))
        update_reference_database(current, dummy, str(tmp_path))

        ((_, args, dispatch),) = calls
        assert dispatch["kwargs"]["multiburst"] is True
        assert args[0] == (BURSTS[0],)
        assert args[1][0].sizes["acquisition_id"] == 1
        assert args[2] == (None,)

    @pytest.mark.unit
    def test_update_reference_database_accepts_a_single_burst_layout(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr("cop_ard_app.utils.map", lambda func, args, **kw: calls.append((func, args, kw)))
        current = _reference_tree("", [MASTER])
        stable = _reference_tree("", list(SECONDARIES))
        update_reference_database(current, stable, str(tmp_path), reference_id="r1")
        ((func, args, dispatch),) = calls
        assert func is save_single_burst
        expected = {"reference_db_path": str(tmp_path), "multiburst": False, "reference_id": "r1"}
        assert dispatch == {"kwargs": expected, "enable_dask": True}
        burst_ids, current_args, stable_args = args
        assert burst_ids == ("data",)
        assert current_args[0].sizes["acquisition_id"] == 1
        assert stable_args[0].sizes["acquisition_id"] == 1

    @pytest.mark.unit
    def test_update_reference_database_tolerates_a_missing_stable_node(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr("cop_ard_app.utils.map", lambda func, args, **kw: calls.append((func, args, kw)))
        current = xr.DataTree()
        stable = xr.DataTree()
        for burst in BURSTS:
            current[f"bursts/{burst}/measurements"] = xr.Dataset({"cstack": _measurements([MASTER])["cstack"]})
        stable[f"bursts/{BURSTS[0]}/measurements"] = xr.Dataset({"cstack": _measurements(list(SECONDARIES))["cstack"]})
        update_reference_database(current, stable, str(tmp_path))

        ((_, args, _),) = calls
        assert args[0] == tuple(BURSTS)
        assert args[2][0] is not None
        assert args[2][1] is None


class TestComputeIonoAndOrbHeight:
    """Unit test for cop_ard_app.utils.compute_iono_and_orb_height.

    ARPPOS lies on the geoid sphere H_ORB metres above the equator, so the orbital height
    is known in closed form. The ionospheric lookup hits the network, so it is stubbed.
    """

    @pytest.mark.unit
    def test_recovers_the_orbital_height_of_the_reference_scene(self, tmp_path):
        reference = _scene_tree(BURSTS[:1], [MASTER], arppos=ARPPOS)
        secondary = _scene_tree(BURSTS[:1], [SECONDARIES[0]], arppos=ARPPOS)
        h_orb, h_iono = compute_iono_and_orb_height(
            reference, secondary, str(tmp_path), MASTER, iono_height_input=250_000.0
        )
        assert h_orb == pytest.approx(H_ORB, rel=1e-6)
        assert h_iono == 250_000.0

    @pytest.mark.unit
    def test_accepts_a_single_burst_layout(self, tmp_path):
        reference = _scene_tree(["b1"], [MASTER], arppos=ARPPOS, multiburst=False)
        secondary = _scene_tree(["b1"], [SECONDARIES[0]], arppos=ARPPOS, multiburst=False)
        h_orb, h_iono = compute_iono_and_orb_height(reference, secondary, str(tmp_path), MASTER, iono_height_input=0)
        assert h_orb == pytest.approx(H_ORB, rel=1e-6)
        assert h_iono == 0.0

    @pytest.mark.unit
    def test_asks_the_ionosphere_model_when_no_height_is_given(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cop_ard_app.utils.get_iono_height", lambda *args: 260_000.0)
        reference = _scene_tree(BURSTS[:1], [MASTER], arppos=ARPPOS)
        secondary = _scene_tree(BURSTS[:1], [SECONDARIES[0]], arppos=ARPPOS)
        h_orb, h_iono = compute_iono_and_orb_height(reference, secondary, str(tmp_path), MASTER)
        assert h_orb == pytest.approx(H_ORB, rel=1e-6)
        assert h_iono == 260_000.0

    @pytest.mark.unit
    def test_falls_back_to_the_default_height_when_the_lookup_fails(self, tmp_path, monkeypatch):
        def no_space_weather(*args):
            raise RuntimeError("no space weather database available")

        monkeypatch.setattr("cop_ard_app.utils.get_iono_height", no_space_weather)
        reference = _scene_tree(BURSTS[:1], [MASTER], arppos=ARPPOS)
        secondary = _scene_tree(BURSTS[:1], [SECONDARIES[0]], arppos=ARPPOS)
        h_orb, h_iono = compute_iono_and_orb_height(
            reference, secondary, str(tmp_path), MASTER, default_iono_height=DEFAULT_IONO
        )
        assert h_orb == pytest.approx(H_ORB, rel=1e-6)
        assert h_iono == DEFAULT_IONO

    @pytest.mark.unit
    def test_requires_exactly_one_secondary_acquisition(self, tmp_path):
        reference = _scene_tree(BURSTS[:1], [MASTER], arppos=ARPPOS)
        secondary = _scene_tree(BURSTS[:1], [MASTER] + SECONDARIES, arppos=ARPPOS)
        with pytest.raises(ValueError, match="Expected exactly 1 acquisition date"):
            compute_iono_and_orb_height(reference, secondary, str(tmp_path), MASTER)

    @pytest.mark.unit
    def test_rejects_a_non_numeric_ionospheric_height(self, tmp_path):
        reference = _scene_tree(BURSTS[:1], [MASTER], arppos=ARPPOS)
        secondary = _scene_tree(BURSTS[:1], [SECONDARIES[0]], arppos=ARPPOS)
        with pytest.raises(TypeError, match="Invalid iono_height_input type"):
            compute_iono_and_orb_height(reference, secondary, str(tmp_path), MASTER, iono_height_input="auto")
