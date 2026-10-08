from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.constants import speed_of_light

from cop_ard_app.model import (
    DEL_ATTRS,
    DEL_ENCODING,
    add_acquisition_date,
    clean_cslc,
    clean_gslc,
    extract_acquisition_dates,
    extract_acquisition_id,
    extract_orbits,
    fix_attributes,
    get_az_rg_meters,
    get_az_rg_times,
    get_burst_id,
    get_coefficients,
    get_footprint,
    get_other_metadata_gslc,
    get_rf_band,
    map_index_to_group,
    update_container,
    update_coreg,
    update_cslc_container,
)

N_COLS = 5
N_ROWS = 3
SCP_COL = 2
SCP_ROW = 1
COL_SS = 1.0
ROW_SS = 2.0
R_CA_SCP = 700000.0
TIME_CA_POLY = [0.0, 1.0]
COLLECT_START = "2020-01-01T00:00:00"
T0 = np.datetime64(COLLECT_START, "ns")
TOL = 1e-9

ACQ1 = "2020-01-01"
ACQ2 = "2020-01-12"
BURST = "iw1_254"
ORBIT_DEGREES = 11
ANNOT_KEY = "annotation"
POL = "VV"

# Analytic orbit polynomials: two acquisitions with degree-1 X/Y/Z coefficients.
ORBIT_ARPPOLY = {
    ACQ1: {"X": {"Coefs": [1000.0, 11.0]}, "Y": {"Coefs": [2000.0, 22.0]}, "Z": {"Coefs": [3000.0, 33.0]}},
    ACQ2: {"X": {"Coefs": [-1000.0, -11.0]}, "Y": {"Coefs": [-2000.0, -22.0]}, "Z": {"Coefs": [-3000.0, -33.0]}},
}
ORBIT_REF_TIMES = {ACQ1: COLLECT_START, ACQ2: "2020-01-12T00:00:00"}
PRECOMPUTED_COEFS = np.array([[100.0, 200.0, 300.0], [1.0, 2.0, 3.0]])

DATE_A = np.datetime64("2020-01-01T00:00:00", "ns")
DATE_B = np.datetime64("2020-01-12T00:00:00", "ns")

# SICD-style metadata dictionary (dict fallback branch of get_az_rg_*).
SICD_ANNOT = {
    "ImageData": {
        "FullImage": {"NumCols": N_COLS, "NumRows": N_ROWS},
        "SCPPixel": {"Col": SCP_COL, "Row": SCP_ROW},
    },
    "Grid": {"Col": {"SS": COL_SS}, "Row": {"SS": ROW_SS}},
    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": list(TIME_CA_POLY)}, "R_CA_SCP": R_CA_SCP}},
    "Timeline": {"CollectStart": COLLECT_START},
}

# Annotation used for a group carrying a stac_discovery and other_metadata entries.
STAC_GROUPS = {
    "b0_123": {"swath": "A", "burst_id": "123", "pol": POL},
    "b1_456": {"swath": "C", "burst_id": "456", "pol": "HH"},
}

RADIOMETRIC_COEFS = [0.05, 0.15]
ANNOT_COEFFS = {"Radiometric": {"SigmaZeroSFPoly": {"Coefs": list(RADIOMETRIC_COEFS)}}}

# (tx_frequency_min, tx_frequency_max, expected band); midpoints are exact in binary.
RF_BAND_CASES = [
    (5.0e8, 5.0e8, "P"),
    (1.0e9, 1.0e9, "L"),
    (3.0e9, 5.0e9, "C"),
    (2.0e9, 2.0e9, "S"),
    (4.0e9, 4.0e9, "C"),
    (8.0e9, 8.0e9, "X"),
    (8.5e9, 11.5e9, "X"),
    (12.0e9, 12.0e9, "Unknown"),
    (0.0, 0.0, "Unknown"),
]


def _orbit_annotation(acq: str) -> dict:
    return {
        "Position": {"ARPPoly": ORBIT_ARPPOLY[acq]},
        "Timeline": {"CollectStart": ORBIT_REF_TIMES[acq]},
    }


def _slc_dataarray(dates: list, annotation: dict | None = None) -> xr.DataArray:
    if dates:
        dims = ("azimuth_time", "x")
        coords = {"azimuth_time": list(dates)}
        shape = (len(dates), N_COLS)
    else:
        dims = ("y", "x")
        coords = {}
        shape = (1, N_COLS)
    attrs = {ANNOT_KEY: annotation} if annotation is not None else {}
    return xr.DataArray(np.zeros(shape), dims=dims, coords=coords, attrs=attrs)


def _make_orbits_slcs(multiburst: bool = False, precomputed: tuple = ()) -> xr.DataTree:
    slcs = xr.DataTree()
    for acq in (ACQ1, ACQ2):
        base = f"bursts/{BURST}" if multiburst else ""
        slcs[f"{base}/orig/{acq}/measurements"] = xr.Dataset({"slc": _slc_dataarray([], _orbit_annotation(acq))})
        # the collection loop requires a conditions group on every acquisition
        slcs[f"{base}/orig/{acq}/conditions"] = xr.Dataset()
        if acq in precomputed:
            coeffs = xr.DataArray(
                PRECOMPUTED_COEFS[..., None],
                dims=("degree", "axis", "acquisition_id"),
                coords={
                    "degree": np.arange(PRECOMPUTED_COEFS.shape[0]),
                    "axis": np.arange(3),
                    "acquisition_id": [acq],
                },
            )
            slcs[f"{base}/orig/{acq}/conditions/orbit"] = xr.Dataset({"coeffs": coeffs})
    return slcs


def _expected_orbit_row(acq: str) -> np.ndarray:
    coefs = np.array([ORBIT_ARPPOLY[acq][axis]["Coefs"] for axis in ("X", "Y", "Z")])
    row = np.zeros((ORBIT_DEGREES, 3))
    row[: coefs.shape[1], : coefs.shape[0]] = coefs.T
    return row


def _dated_slcs(dates_per_id: dict[str, list]) -> xr.DataTree:
    slcs = xr.DataTree()
    for data_id, dates in dates_per_id.items():
        slcs[f"orig/{data_id}/measurements"] = xr.Dataset({"slc": _slc_dataarray(dates)})
    return slcs


def _group_tree() -> xr.DataTree:
    dt = xr.DataTree()
    for name, spec in STAC_GROUPS.items():
        dt[name] = xr.DataTree()
        dt[name].attrs = {
            "stac_discovery": {"type": "Feature", "id": name, "properties": {"sar:polarization": spec["pol"]}},
            "other_metadata": {
                "burst_id": spec["burst_id"],
                "general_annotation": {"downlink_information_list": [{"swath": spec["swath"]}]},
            },
        }
    return dt


def _image_meta(lon: float, lat: float) -> SimpleNamespace:
    corners = [SimpleNamespace(Lon=lon + i, Lat=lat + i) for i in range(4)]
    return SimpleNamespace(GeoData=SimpleNamespace(ImageCorners=corners))


FOOTPRINT_DS = SimpleNamespace(sicd_meta=[_image_meta(10.0, 20.0), _image_meta(-30.0, 40.0)])

BURST_PARAMS = {"BURST": "1234", "SWATH": "AB"}
BURST_ANNOT = SimpleNamespace(CollectionInfo=SimpleNamespace(Parameters=dict(BURST_PARAMS)))


class TestExtractOrbits:
    """Unit tests for cop_ard_app.model.extract_orbits."""

    @pytest.mark.unit
    def test_extract_orbits_consolidates_arppoly(self):
        slcs = _make_orbits_slcs()
        out = extract_orbits(slcs, degrees=ORBIT_DEGREES)

        assert out.dims == ("acquisition_id", "degree", "axis")
        assert list(out.coords["acquisition_id"].values) == [ACQ1, ACQ2]
        assert out.shape == (2, ORBIT_DEGREES, 3)
        assert out.attrs["reference_times"] == ORBIT_REF_TIMES

        expected = np.stack([_expected_orbit_row(acq) for acq in (ACQ1, ACQ2)])
        np.testing.assert_array_equal(out.values, expected)
        # degrees beyond the polynomial order stay zero
        np.testing.assert_array_equal(out.values[:, 2:, :], np.zeros((2, ORBIT_DEGREES - 2, 3)))

    @pytest.mark.unit
    def test_extract_orbits_filters_by_acquisition_id(self):
        slcs = _make_orbits_slcs()
        out = extract_orbits(slcs, acquisition_id=[ACQ2], degrees=ORBIT_DEGREES)

        assert out.shape == (1, ORBIT_DEGREES, 3)
        assert list(out.coords["acquisition_id"].values) == [ACQ2]
        assert list(out.attrs["reference_times"].keys()) == [ACQ2]
        np.testing.assert_array_equal(out.values[0], _expected_orbit_row(ACQ2))

    @pytest.mark.unit
    def test_extract_orbits_multiburst_tree(self):
        slcs = _make_orbits_slcs(multiburst=True)
        out = extract_orbits(slcs, multiburst=True, degrees=ORBIT_DEGREES)

        assert list(out.coords["acquisition_id"].values) == [ACQ1, ACQ2]
        expected = np.stack([_expected_orbit_row(acq) for acq in (ACQ1, ACQ2)])
        np.testing.assert_array_equal(out.values, expected)

    @pytest.mark.unit
    def test_extract_orbits_prefers_precomputed_orbit(self):
        slcs = _make_orbits_slcs(precomputed=(ACQ1,))
        out = extract_orbits(slcs, degrees=ORBIT_DEGREES)

        # ACQ1 has a pre-computed orbit in its conditions group: it takes priority
        np.testing.assert_array_equal(out.values[0, : PRECOMPUTED_COEFS.shape[0], :], PRECOMPUTED_COEFS)
        np.testing.assert_array_equal(out.values[1], _expected_orbit_row(ACQ2))


class TestExtractAcquisitionDates:
    """Unit tests for cop_ard_app.model.extract_acquisition_dates."""

    @pytest.mark.unit
    def test_extract_acquisition_dates_unique(self):
        # two acquisitions on the same date must collapse to a single entry
        slcs = _dated_slcs({ACQ1: [DATE_A], "MA002": [DATE_A], ACQ2: [DATE_B]})
        dates = extract_acquisition_dates(slcs)

        assert len(dates) == 2
        assert set(dates) == {DATE_A, DATE_B}

    @pytest.mark.unit
    def test_extract_acquisition_dates_multiburst_tree(self):
        slcs = xr.DataTree()
        for data_id, dates in ((ACQ1, [DATE_A]), (ACQ2, [DATE_B])):
            slcs[f"bursts/{BURST}/orig/{data_id}/measurements"] = xr.Dataset({"slc": _slc_dataarray(dates)})
        dates = extract_acquisition_dates(slcs, multiburst=True)

        assert set(dates) == {DATE_A, DATE_B}


class TestExtractAcquisitionId:
    """Unit tests for cop_ard_app.model.extract_acquisition_id."""

    @pytest.mark.unit
    def test_extract_acquisition_id_returns_group_names(self):
        slcs = _dated_slcs({ACQ1: [DATE_A], ACQ2: [DATE_B]})
        ids = extract_acquisition_id(slcs)

        assert ids == {ACQ1, ACQ2}

    @pytest.mark.unit
    def test_extract_acquisition_id_multiburst_tree(self):
        slcs = xr.DataTree()
        for data_id, dates in ((ACQ1, [DATE_A]), (ACQ2, [DATE_B])):
            slcs[f"bursts/{BURST}/orig/{data_id}/measurements"] = xr.Dataset({"slc": _slc_dataarray(dates)})
        ids = extract_acquisition_id(slcs, multiburst=True)

        assert ids == {ACQ1, ACQ2}


class TestGetAzRgTimes:
    """Unit tests for cop_ard_app.model.get_az_rg_times."""

    @pytest.mark.unit
    def test_get_az_rg_times_exact_polynomial_times(self):
        azimuth_time, slant_range_time = get_az_rg_times(SICD_ANNOT)

        assert azimuth_time.shape == (N_COLS,)
        assert slant_range_time.shape == (N_ROWS,)

        # TimeCAPoly = [0, 1] and Col.SS = 1 -> integer-second offsets around the SCP
        expected_az = T0 + (np.arange(N_COLS) - SCP_COL) * np.timedelta64(1, "s")
        np.testing.assert_array_equal(azimuth_time, expected_az)

        expected_rg = 2.0 * ((np.arange(N_ROWS) - SCP_ROW) * ROW_SS + R_CA_SCP) / speed_of_light
        assert np.allclose(slant_range_time, expected_rg, atol=TOL)
        assert slant_range_time[SCP_ROW] == pytest.approx(2.0 * R_CA_SCP / speed_of_light, abs=TOL)


class TestGetAzRgMeters:
    """Unit tests for cop_ard_app.model.get_az_rg_meters."""

    @pytest.mark.unit
    def test_get_az_rg_meters_exact_spacing(self):
        azimuth_m, slant_range_m = get_az_rg_meters(SICD_ANNOT)

        assert azimuth_m.shape == (N_COLS,)
        assert slant_range_m.shape == (N_ROWS,)

        expected_az = (np.arange(N_COLS) - SCP_COL) * COL_SS
        expected_rg = (np.arange(N_ROWS) - SCP_ROW) * ROW_SS
        assert np.allclose(azimuth_m, expected_az, atol=TOL)
        assert np.allclose(slant_range_m, expected_rg, atol=TOL)
        # the scene center pixel maps to zero distance
        assert azimuth_m[SCP_COL] == 0.0
        assert slant_range_m[SCP_ROW] == 0.0


class TestMapIndexToGroup:
    """Unit tests for cop_ard_app.model.map_index_to_group."""

    @pytest.mark.unit
    def test_map_index_to_group_matches_swath_burst_polarization(self):
        name = map_index_to_group(_group_tree(), swath="c", burstid=456, pol="HH")
        assert name == "b1_456"

    @pytest.mark.unit
    def test_map_index_to_group_polarization_only(self):
        name = map_index_to_group(_group_tree(), swath=None, burstid=None, pol=POL)
        assert name == "b0_123"

    @pytest.mark.unit
    def test_map_index_to_group_raises_when_unmatched(self):
        with pytest.raises(Exception, match="No correspondence found"):
            map_index_to_group(_group_tree(), swath="c", burstid=456, pol="VH")


class TestGetFootprint:
    """Unit tests for cop_ard_app.model.get_footprint."""

    @pytest.mark.unit
    def test_get_footprint_returns_closed_ring(self):
        footprint = get_footprint(FOOTPRINT_DS, 1)

        assert footprint == [
            (-30.0, 40.0),
            (-29.0, 41.0),
            (-28.0, 42.0),
            (-27.0, 43.0),
            (-30.0, 40.0),
        ]
        # a closed ring repeats the first corner last, without dropping any corner
        assert len(footprint) == 5
        assert footprint[0] == footprint[-1]

        first = get_footprint(FOOTPRINT_DS, 0)
        assert first[0] == (10.0, 20.0)


class TestGetBurstId:
    """Unit tests for cop_ard_app.model.get_burst_id."""

    @pytest.mark.unit
    def test_get_burst_id_lowercase_swath_and_int_burst(self):
        swath, burst_id = get_burst_id(BURST_ANNOT)
        assert (swath, burst_id) == ("ab", 1234)
        assert isinstance(burst_id, int)


class TestGetCoefficients:
    """Unit tests for cop_ard_app.model.get_coefficients."""

    @pytest.mark.unit
    def test_get_coefficients_returns_ndarray(self):
        coeffs = get_coefficients("Radiometric.SigmaZeroSFPoly.Coefs", ANNOT_COEFFS)
        assert coeffs is not None
        assert coeffs.shape == (2,)
        assert np.allclose(coeffs, np.array(RADIOMETRIC_COEFS), atol=TOL)

    @pytest.mark.unit
    def test_get_coefficients_none_when_missing(self):
        assert get_coefficients("RMA.INCA.TimeCAPoly.Coefs", ANNOT_COEFFS) is None
        # descending past a list leaf yields None as well
        assert get_coefficients("Radiometric.SigmaZeroSFPoly.Coefs.0", ANNOT_COEFFS) is None


class TestGetRfBand:
    """Unit tests for cop_ard_app.model.get_rf_band."""

    @pytest.mark.unit
    @pytest.mark.parametrize(("tx_frequency_min", "tx_frequency_max", "band"), RF_BAND_CASES)
    def test_get_rf_band_classifies_midpoint(self, tx_frequency_min, tx_frequency_max, band):
        assert get_rf_band(tx_frequency_min, tx_frequency_max) == band


# --- Fixtures for the coreg restructure and STAC-container functions ----------


def _ring(bbox: list) -> list:
    """Closed corner ring ordered (minlon,minlat),(minlon,maxlat),(maxlon,minlat),(maxlon,maxlat)."""
    minx, miny, maxx, maxy = bbox
    corners = [[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]]
    return corners + [list(corners[0])]


def _product_stac(product_type, start, end, bbox, instrument, polarizations, orbit, datatake_id, preset_id) -> dict:
    return {
        "type": "Feature",
        "stac": "1.0.0",
        "id": preset_id,
        "bbox": list(bbox),
        "geometry": {"type": "Polygon", "coordinates": [_ring(bbox)]},
        "properties": {
            "datetime": start,
            "start_datetime": start,
            "end_datetime": end,
            "created": start,
            "proj:epsg": 4326,
            "proj:wkt2": "POINT (0 0)",
            "proj:shape": [4, 4],
            "proj:transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "sat:anx_datetime": "2020-01-01T00:00:00Z",
            "sar:pixel_spacing_azimuth": 1.0,
            "sar:pixel_spacing_range": 1.0,
            "view:incidence_angle": 38.5,
            "sar:instrument_mode": instrument,
            "sar:polarizations": list(polarizations),
            "product:type": product_type,
            "platform": ["sentinel-1"],
            "sat:relative_orbit": orbit,
            "eopf:datatake_id": datatake_id,
        },
        "stac_extensions": list(PRODUCT_EXTENSIONS),
    }


def _product_tree(children: dict, history_key: str) -> xr.DataTree:
    dt = xr.DataTree()
    for name, stac in children.items():
        dt[name] = xr.DataTree()
        dt[name].attrs = {
            "stac_discovery": stac,
            "processing_history": {history_key: [{"inputs": [f"INPUT-{name}"], "outputs": ["OLD-OUTPUT"]}]},
        }
    return dt


PRODUCT_EXTENSIONS = [
    "https://stac-extensions.github.io/projection/v1.0.0/schema.json",
    "https://stac-extensions.github.io/view/v1.0.0/schema.json",
    "https://stac-extensions.github.io/eopf/v1.0.0/schema.json",
]

GMLP_BBOX = [10.0, 20.0, 30.0, 40.0]
GMLP_BBOX_WIDER = [5.0, 20.0, 30.0, 50.0]
GMLP_START = "2020-01-01T00:00:00Z"
GMLP_END_FIRST = "2020-01-04T00:00:00Z"
GMLP_START_SECOND = "2020-01-03T00:00:00Z"
GMLP_END_SECOND = "2020-01-06T00:00:00Z"
# midpoint of the union of the two products' time spans (exact whole seconds)
GMLP_MERGED_MID = "2020-01-03T12:00:00Z"
GMLP_MERGED_RING = [[5.0, 20.0], [5.0, 50.0], [30.0, 20.0], [30.0, 50.0], [5.0, 20.0]]
GMLP_ID_HEAD = ["GMLP", "20200101T000000Z", "432000", "SENTINEL-1254"]
GMLP_ID_TAIL = ["100000", "SV"]

CSLC_BBOX = [0.0, 0.0, 1.0, 1.0]
CSLC_START = "2020-02-01T00:00:00Z"
CSLC_END = "2020-02-03T00:00:00Z"
CSLC_MID = "2020-02-02T00:00:00Z"
CSLC_ID_HEAD = ["CSLC", "20200201T000000Z", "172800", "SENTINEL-1255"]
CSLC_ID_TAIL = ["0F4240", "SH"]
CSLC_PRESET_ID = "CSLC-OLD-0001"

REF_ACQ_IDS = ["MA001_sec", "MB002_sec"]
REF_DATES = ["2020-01-05", "2020-01-16"]
REF_MEASUREMENTS = xr.Dataset(
    {"cstack": (["acquisition_id", "y", "x"], np.zeros((2, 2, 2)))},
    coords={
        "acquisition_id": REF_ACQ_IDS,
        "acquisition_date": ("acquisition_id", REF_DATES),
        "y": [0, 1],
        "x": [0, 1],
    },
)

MASK_VALUES = np.array([[1, 2], [2, 1]], dtype=np.int8)


def _mask_array() -> xr.DataArray:
    return xr.DataArray(MASK_VALUES, dims=("y", "x"), name="mask")


def _coreg_scene(base: str = "", optional: bool = False) -> xr.DataTree:
    slc = xr.DataTree()
    slc[f"{base}/coreg/measurements"] = xr.Dataset(
        {
            "cstack": xr.DataArray(np.ones((4, 4)), dims=("y", "x")),
            "daz": xr.DataArray(np.array(1.0)),
            "drg": xr.DataArray(np.array(2.0)),
            "azimuth_m": xr.DataArray(np.array(3.0)),
            "slant_range_m": xr.DataArray(np.array(4.0)),
        },
        attrs={"store_path": "/tmp/x", "coregistration_reference_date": "2020-01-01"},
    )
    slc[f"{base}/coreg/quality"] = xr.Dataset(
        {
            "RCS": xr.DataArray(np.arange(4.0), dims=("y",)),
            "crg_accuracy": xr.DataArray(np.arange(4.0), dims=("y",)),
        }
    )
    slc[f"{base}/coreg/conditions/orbit"] = xr.Dataset(
        {
            "coeffs": xr.DataArray(np.zeros((2, 3)), dims=("degree", "axis")),
            "orbit_files": xr.DataArray(np.array("f1")),
            "reference_time": xr.DataArray(np.array("t")),
            "type": xr.DataArray(np.array("g")),
        },
        coords={"degree": [0, 1], "axis": np.arange(3)},
    )
    slc[f"{base}/coreg/conditions/phase_demod"] = xr.Dataset({"phase_demod": xr.DataArray(np.zeros(4), dims=("y",))})
    slc[f"{base}/coreg/conditions/look_vector"] = xr.Dataset({"look": xr.DataArray(np.zeros(4), dims=("y",))})
    if optional:
        slc[f"{base}/coreg/conditions/ETAD"] = xr.Dataset({"elevation": xr.DataArray(np.zeros(4), dims=("y",))})
        slc[f"{base}/coreg/conditions/side_band"] = xr.Dataset({"side_band": xr.DataArray(np.array("R"))})
    return slc


GSLC_DATE = "2020-01-05"
GSLC_MASK_VALUES = np.arange(16, dtype=np.int8).reshape(1, 4, 4)


def _gslc_scene(with_etad: bool = False) -> xr.DataTree:
    dims = ("polarization", "y", "x")
    measurements = xr.Dataset(
        {
            "phase": xr.DataArray(np.ones((1, 4, 4)), dims=dims),
            "speckle_phase": xr.DataArray(np.full((1, 4, 4), 0.5), dims=dims),
            "amplitude": xr.DataArray(np.ones((1, 4, 4)), dims=dims),
            "coherence": xr.DataArray(np.full((1, 4, 4), 0.75), dims=dims),
            "mask": xr.DataArray(GSLC_MASK_VALUES, dims=dims),
        },
        coords={"polarization": ["VV"], "y": np.arange(4), "x": np.arange(4)},
    )
    measurements = measurements.assign_coords(acquisition_date=GSLC_DATE)
    gslc = xr.DataTree()
    gslc["measurements"] = measurements
    gslc["conditions/look_vector"] = xr.Dataset({"look": xr.DataArray(np.zeros(4), dims=("y",))})
    if with_etad:
        gslc["conditions/etad"] = xr.Dataset({"elevation": xr.DataArray(np.zeros(4), dims=("y",))})
    return gslc


EXTERNAL_GSLC = [
    {
        "execution_parameters": {
            "noise_removal": False,
            "measurement_type": "area",
            "product_data_access": "FROM-FIRST",
            "product_pixel_coordinate_convention": "FROM-FIRST",
            "gmlp_status": "FROM-FIRST",
            "product_family_specification": "FROM-FIRST",
        }
    },
    {"execution_parameters": {"unused": True}},
    {
        "execution_parameters": {
            "noise_removal": True,
            "measurement_type": "points",
            "product_data_access": ["external"],
            "product_pixel_coordinate_convention": "row-col",
            "gmlp_status": "gmlp-ok",
            "product_family_specification": "S1-RG",
        }
    },
]
EXPECTED_GSLC_OTHER_METADATA = {
    "product_data_access": ["external"],
    "product_geometry": "projected",
    "product_pixel_coordinate_convention": "row-col",
    "noise_correction_flag": False,
    "gmlp_status": "gmlp-ok",
    "product_family_specification": "S1-RG",
    "measurement_type": "area",
    "product_resampling_method": "sinc interpolator",
}

FIX_TEMPLATE = {
    "y": {"long_name": "row index", "units": "m"},
    "x": {"long_name": "column index", "units": "m"},
    "cslc": {"long_name": "co-registered stack", "units": "s"},
}


def _radar_grid() -> xr.Dataset:
    return xr.Dataset(
        {
            "layover": xr.DataArray(np.array([[1, 0], [0, 1]], dtype=np.int8), dims=("y", "x")),
            "shadow": xr.DataArray(np.array([[0, 1], [1, 0]], dtype=np.int8), dims=("y", "x")),
        }
    )


def _stub_cslc_stac() -> dict:
    return _product_stac("CSLC", CSLC_START, CSLC_END, CSLC_BBOX, "EW", ["HH"], 255, 1000000, CSLC_PRESET_ID)


def _stub_cslc_history() -> dict:
    return {"COP ARD CSLC Product": [{"inputs": ["INPUT-1"], "outputs": ["OLD-OUTPUT"]}]}


def _stub_other_metadata() -> dict:
    return {"SICD": {"meta": "x"}}


EXPECTED_CSLC_GROUPS = {
    "/",
    "/measurements",
    "/conditions",
    "/conditions/orbit",
    "/conditions/look_vector",
    "/conditions/phase_demodulation",
    "/quality",
    "/quality/radar_qlt",
    "/quality/distortion_mask",
}


def _prefixed_groups(prefix: str) -> set:
    """Group names as seen from the containing tree when the product sits at ``prefix``."""
    return {prefix if g == "/" else f"{prefix}{g}" for g in EXPECTED_CSLC_GROUPS}


class TestUpdateContainer:
    """Unit tests for cop_ard_app.model.update_container."""

    @pytest.mark.unit
    def test_update_container_merges_child_products(self):
        stac_a = _product_stac(
            "GMLP", GMLP_START, GMLP_END_FIRST, GMLP_BBOX, "IW", [POL], 254, 1048576, "GMLP-PRESET-A"
        )
        stac_b = _product_stac(
            "GMLP", GMLP_START_SECOND, GMLP_END_SECOND, GMLP_BBOX_WIDER, "EW", ["HH"], 254, 1048576, "GMLP-PRESET-B"
        )
        dt = _product_tree({"c1": stac_a, "c2": stac_b}, "COP ARD GMLP Product")

        merged, history = update_container(dt, "GMLP", "UNUSED-CSLC-ID")

        # bbox, time span and geometry are the union of the two child products
        assert merged["bbox"] == GMLP_BBOX_WIDER
        assert merged["geometry"]["coordinates"] == [GMLP_MERGED_RING]
        assert merged["properties"]["start_datetime"] == GMLP_START
        assert merged["properties"]["end_datetime"] == GMLP_END_SECOND
        assert merged["properties"]["datetime"] == GMLP_MERGED_MID
        assert merged["assets"] == {k: {"href": f"./{k}", "type": "application/vnd+zarr"} for k in ("c1", "c2")}
        # GMLP containers strip projection keys but keep satellite/scene metadata
        for key in ("proj:epsg", "proj:wkt2", "proj:shape", "proj:transform"):
            assert key not in merged["properties"]
        assert merged["properties"]["sat:anx_datetime"] == "2020-01-01T00:00:00Z"
        # sub-swath ids come from the template child's instrument mode
        assert merged["properties"]["sar:beam_ids"] == ["IW1", "IW2", "IW3"]
        # projection/view STAC extensions are filtered out of the container
        assert merged["stac_extensions"] == [PRODUCT_EXTENSIONS[2]]
        # product id rebuilt from the merged STAC (crc field is md5-derived)
        parts = merged["id"].split("_")
        assert len(parts) == 7
        assert parts[:4] == GMLP_ID_HEAD
        assert parts[4].startswith("S") and len(parts[4]) == 4
        assert parts[5:] == GMLP_ID_TAIL
        assert history["COP ARD GMLP Product"][-1]["outputs"] == [merged["id"]]
        # children were deep-copied: the input tree is untouched
        assert dt["c1"].attrs["stac_discovery"]["id"] == "GMLP-PRESET-A"
        assert dt["c1"].attrs["stac_discovery"]["bbox"] == GMLP_BBOX
        assert dt["c1"].attrs["processing_history"]["COP ARD GMLP Product"][-1]["outputs"] == ["OLD-OUTPUT"]


class TestFixAttributes:
    """Unit tests for cop_ard_app.model.fix_attributes."""

    @pytest.mark.unit
    def test_fix_attributes_standardizes_dataarray_attrs(self):
        da = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}, name="cslc")
        da.attrs = {key: "drop-me" for key in DEL_ATTRS} | {"keep_attr": "keep-me"}
        for key in DEL_ENCODING:
            da.encoding[key] = "drop-me"

        out = fix_attributes(da, FIX_TEMPLATE)

        assert out is da
        assert out.coords["y"].attrs == {**FIX_TEMPLATE["y"], "short_name": "y"}
        assert out.coords["x"].attrs == {**FIX_TEMPLATE["x"], "short_name": "x"}
        assert out.attrs == {"keep_attr": "keep-me", **FIX_TEMPLATE["cslc"], "short_name": "cslc"}
        assert out.encoding == {}

    @pytest.mark.unit
    def test_fix_attributes_recurses_through_tree(self):
        tree = xr.DataTree()
        tree.attrs = {key: "drop-me" for key in DEL_ATTRS} | {"keep_attr": "keep-me"}
        tree.encoding["coordinates"] = ["y"]
        tree["a/b"] = xr.Dataset(
            {"cslc": xr.DataArray(np.ones((2, 2)), dims=("y", "x"))},
            coords={"y": [0, 1], "x": [0, 1]},
            attrs={key: "drop-me" for key in DEL_ATTRS},
        )
        tree["a/b/inner"] = xr.Dataset(
            {"cslc": xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))},
            coords={"y": [0, 1], "x": [0, 1]},
        )

        fix_attributes(tree, FIX_TEMPLATE)

        assert tree.attrs == {"keep_attr": "keep-me"}
        assert tree.encoding == {}
        assert tree["a/b"].attrs == {}
        assert tree["a/b"].coords["y"].attrs == {**FIX_TEMPLATE["y"], "short_name": "y"}
        assert tree["a/b/cslc"].attrs == {**FIX_TEMPLATE["cslc"], "short_name": "cslc"}
        # recursion reaches nested groups and their variables
        assert tree["a/b/inner/cslc"].attrs == {**FIX_TEMPLATE["cslc"], "short_name": "cslc"}


class TestCleanGslc:
    """Unit tests for cop_ard_app.model.clean_gslc."""

    @pytest.mark.unit
    def test_clean_gslc_splits_measurements_quality_and_conditions(self):
        out = clean_gslc(_gslc_scene(with_etad=True), _coreg_scene(), [ACQ1])

        assert set(out.groups) == {
            "/",
            "/conditions",
            "/conditions/ETAD",
            "/conditions/look_vector",
            "/conditions/orbit",
            "/measurements",
            "/quality",
            "/quality/coherence",
            "/quality/distortion_mask",
        }
        # only variables whose name contains "phase" stay in measurements
        measurements = out["measurements"].to_dataset()
        assert set(measurements.data_vars) == {"phase", "speckle_phase"}
        assert measurements["phase"].dims == ("polarization", "acquisition_id", "y", "x")
        # the scalar acquisition date is promoted to an acquisition_id coordinate
        assert measurements.coords["acquisition_date"].dims == ("acquisition_id",)
        assert list(measurements.coords["acquisition_date"].values) == [GSLC_DATE]
        # only variables whose name contains "coh" go to quality/coherence
        coherence = out["quality/coherence"].to_dataset()
        assert set(coherence.data_vars) == {"coherence"}
        assert coherence["coherence"].dims == ("polarization", "acquisition_id", "y", "x")
        masked = out["quality/distortion_mask"].to_dataset()
        assert set(masked.data_vars) == {"distortion_mask"}
        np.testing.assert_array_equal(masked["distortion_mask"].values, GSLC_MASK_VALUES)
        # orbit comes from the coreg tree with the non-product variables dropped
        assert set(out["conditions/orbit"].to_dataset().data_vars) == {"coeffs"}

    @pytest.mark.unit
    def test_clean_gslc_without_etad_omits_etad_group(self):
        out = clean_gslc(_gslc_scene(), _coreg_scene(), [])
        assert "/conditions/ETAD" not in out.groups


class TestGetOtherMetadataGslc:
    """Unit tests for cop_ard_app.model.get_other_metadata_gslc."""

    @pytest.mark.unit
    def test_get_other_metadata_gslc_selects_first_and_last_entries(self):
        # the first entry provides noise/measurement, the last one product-level fields
        assert get_other_metadata_gslc(EXTERNAL_GSLC) == EXPECTED_GSLC_OTHER_METADATA


class TestAddAcquisitionDate:
    """Unit tests for cop_ard_app.model.add_acquisition_date."""

    @pytest.mark.unit
    def test_add_acquisition_date_maps_base_ids_to_secondary(self):
        ds = xr.Dataset(
            {"coeffs": (["acquisition_id", "degree"], np.zeros((2, 2)))},
            coords={"acquisition_id": ["MA001", "MB002_sec"], "degree": [0, 1]},
        )

        out = add_acquisition_date(ds, REF_MEASUREMENTS)

        # the base id is remapped to its *_sec counterpart from the reference
        assert list(out.coords["acquisition_id"].values) == list(REF_ACQ_IDS)
        assert out.coords["acquisition_date"].dims == ("acquisition_id",)
        assert list(out.coords["acquisition_date"].values) == list(REF_DATES)
        assert "acquisition_date" not in ds.coords

    @pytest.mark.unit
    def test_add_acquisition_date_passes_through_without_acquisition_id(self):
        assert add_acquisition_date(None, REF_MEASUREMENTS) is None
        plain = xr.Dataset({"v": np.zeros(2)}, coords={"z": [0, 1]})
        assert add_acquisition_date(plain, REF_MEASUREMENTS) is plain


class TestCleanCslc:
    """Unit tests for cop_ard_app.model.clean_cslc."""

    @pytest.mark.unit
    def test_clean_cslc_renames_and_restructures_coreg_tree(self):
        cslc = clean_cslc(_coreg_scene(), _mask_array())

        assert set(cslc.groups) == EXPECTED_CSLC_GROUPS
        # 'cstack' is renamed to 'cslc'; auxiliary range/azimuth variables drop out
        assert set(cslc["measurements"].data_vars) == {"cslc"}
        assert cslc["measurements"].attrs == {}
        assert set(cslc["quality/radar_qlt"].data_vars) == {"rcs", "coregistration_accuracy"}
        assert set(cslc["conditions/orbit"].data_vars) == {"coeffs"}
        masked = cslc["quality/distortion_mask"].to_dataset()
        assert set(masked.data_vars) == {"distortion_mask"}
        np.testing.assert_array_equal(masked["distortion_mask"].values, MASK_VALUES)
        assert masked["distortion_mask"].dtype == np.int8

    @pytest.mark.unit
    def test_clean_cslc_keeps_optional_etad_and_side_band(self):
        cslc = clean_cslc(_coreg_scene(optional=True), _mask_array())
        assert set(cslc.groups) == EXPECTED_CSLC_GROUPS | {"/conditions/ETAD", "/conditions/side_band"}
        assert set(cslc["conditions/side_band"].data_vars) == {"side_band"}


class TestUpdateCslcContainer:
    """Unit tests for cop_ard_app.model.update_cslc_container."""

    @pytest.mark.unit
    def test_update_cslc_container_rebuilds_product_stac(self):
        stac = _product_stac("CSLC", CSLC_START, CSLC_END, CSLC_BBOX, "EW", ["HH"], 255, 1000000, CSLC_PRESET_ID)
        bursts = _product_tree({"c1": stac}, "COP ARD CSLC Product")

        merged, history = update_cslc_container(bursts)

        # a single child is the union itself: bbox and geometry ring are untouched
        assert merged["bbox"] == CSLC_BBOX
        assert merged["geometry"]["coordinates"] == [_ring(CSLC_BBOX)]
        assert merged["assets"] == {"c1": {"href": "./c1", "type": "application/vnd+zarr"}}
        # unlike the GMLP container, satellite-specific keys are stripped here
        for key in ("sat:anx_datetime", "sar:pixel_spacing_azimuth", "sar:pixel_spacing_range", "view:incidence_angle"):
            assert key not in merged["properties"]
        assert merged["properties"]["proj:epsg"] == 4326
        # the datetime is the exact midpoint of the product time span
        assert merged["properties"]["datetime"] == CSLC_MID
        assert merged["properties"]["sar:beam_ids"] == ["EW1", "EW2", "EW3", "EW4", "EW5"]
        parts = merged["id"].split("_")
        assert len(parts) == 7
        assert parts[:4] == CSLC_ID_HEAD
        assert parts[4].startswith("S") and len(parts[4]) == 4
        assert parts[5:] == CSLC_ID_TAIL
        assert history["COP ARD CSLC Product"][-1]["outputs"] == [merged["id"]]

    @pytest.mark.unit
    def test_update_cslc_container_unknown_instrument_yields_no_beams(self):
        stac = _product_stac("CSLC", CSLC_START, CSLC_END, CSLC_BBOX, "UNKNOWN", ["HH"], 255, 1000000, CSLC_PRESET_ID)
        merged, _history = update_cslc_container(_product_tree({"c1": stac}, "COP ARD CSLC Product"))
        assert merged["properties"]["sar:beam_ids"] == []


class TestUpdateCoreg:
    """Unit tests for cop_ard_app.model.update_coreg (IO/STAC helpers stubbed)."""

    @pytest.mark.unit
    def test_update_coreg_single_scene(self, monkeypatch):
        slcs = _coreg_scene()
        calls = []

        def _fake_read(path, **kwargs):
            calls.append(path)
            return _radar_grid()

        monkeypatch.setattr("cop_ard_app.model.read", _fake_read)
        monkeypatch.setattr("cop_ard_app.model.get_stac_cslc", lambda *a, **k: _stub_cslc_stac())
        monkeypatch.setattr("cop_ard_app.model.get_processing_history_cslc", lambda *a, **k: _stub_cslc_history())
        monkeypatch.setattr("cop_ard_app.model.get_other_metadata_cslc", lambda *a, **k: _stub_other_metadata())

        bursts, slcs_out = update_coreg(slcs, "/tmp/wdir", "REF-ID")

        # the static radar grid is read from the non-multiburst path
        assert calls == ["/tmp/wdir/data/REF-ID/radar_grid"]
        assert list(bursts.children) == [CSLC_PRESET_ID]
        product = bursts[CSLC_PRESET_ID]
        assert set(product.groups) == _prefixed_groups(f"/{CSLC_PRESET_ID}")
        assert set(product.attrs) >= {"stac_discovery", "other_metadata", "processing_history"}
        # the distortion mask is exactly layover + 2*shadow
        masked = product["quality/distortion_mask"].to_dataset()
        np.testing.assert_array_equal(masked["distortion_mask"].values, MASK_VALUES)
        assert set(product["measurements"].data_vars) == {"cslc"}
        # the container metadata lands on the returned trees
        merged = slcs_out.attrs["stac_discovery"]
        assert merged["assets"] == {CSLC_PRESET_ID: {"href": f"./{CSLC_PRESET_ID}", "type": "application/vnd+zarr"}}
        assert merged["bbox"] == CSLC_BBOX
        parts = merged["id"].split("_")
        assert len(parts) == 7
        assert parts[:4] == CSLC_ID_HEAD
        assert parts[5:] == CSLC_ID_TAIL
        assert slcs_out.attrs["processing_history"]["COP ARD CSLC Product"][-1]["outputs"] == [merged["id"]]
        assert "stac_discovery" in bursts.attrs

    @pytest.mark.unit
    def test_update_coreg_multiburst_tree(self, monkeypatch):
        slcs = _coreg_scene(base=f"bursts/{BURST}")
        calls = []

        def _fake_read(path, **kwargs):
            calls.append(path)
            return _radar_grid()

        monkeypatch.setattr("cop_ard_app.model.read", _fake_read)
        monkeypatch.setattr("cop_ard_app.model.get_stac_cslc", lambda *a, **k: _stub_cslc_stac())
        monkeypatch.setattr("cop_ard_app.model.get_processing_history_cslc", lambda *a, **k: _stub_cslc_history())
        monkeypatch.setattr("cop_ard_app.model.get_other_metadata_cslc", lambda *a, **k: _stub_other_metadata())

        bursts, slcs_out = update_coreg(slcs, "/tmp/wdir", "REF-ID")

        # with a "bursts" group every scene is read through the bursts path
        assert calls == [f"/tmp/wdir/bursts/{BURST}/radar_grid"]
        assert list(bursts.children) == [CSLC_PRESET_ID]
        assert set(bursts[CSLC_PRESET_ID].groups) == _prefixed_groups(f"/{CSLC_PRESET_ID}")
        assert slcs_out.attrs["stac_discovery"]["id"].split("_")[:4] == CSLC_ID_HEAD
