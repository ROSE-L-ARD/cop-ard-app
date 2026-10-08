"""Unit tests for the module-level helpers of `cop_ard_app.readers.sarpy_rosel`."""

import io
from xml.etree import ElementTree

import numpy as np
import pytest
from sarpy.io.general.base import SarpyIOError

from cop_ard_app.readers.sarpy_rosel import (
    _clean_text,
    _ecf_scp_from_zero_doppler,
    _float,
    _int,
    _parse_xml,
    _polarization,
    _required,
    _vals,
    is_a,
    parse_timestring,
    poly_fit_2d,
)

# Text cleaning.
CLEAN_VALUE = "  A01234  "
CLEAN_SENTINEL = "NOT_AVAILABLE"
CLEAN_SLASH_SENTINEL = "N/A"
CLEAN_BLANK = "   "

# Timestamps.
TIME_UNITS = "us"
ESA_STAMP = "31-DEC-1999 23:59:58.077657806819"
ESA_STAMP_NO_FRACTION = "31-DEC-1999 23:59:58"
ESA_STAMP_LOWER = "31-dec-1999 23:59:58"
ISO_STAMP = "2024-05-27T04:57:02.981885Z"
BAD_STAMP = "not-a-time"
BAD_MONTH = "31-FOO-1999 23:59:58"
ESA_AS_US = np.datetime64("1999-12-31T23:59:58.077657")
ESA_NO_FRACTION_AS_US = np.datetime64("1999-12-31T23:59:58.000000")
ISO_AS_US = np.datetime64("2024-05-27T04:57:02.981885")
ISO_AS_S = np.datetime64("2024-05-27T04:57:02")

# XML fragments written to ``tmp_path`` and parsed by ``_parse_xml``.
XML_SAMPLE = """<?xml version="1.0"?>
<Product xmlns="https://example.com/aresys">
  <Value>2.5e1</Value>
  <Count>7</Count>
  <Empty></Empty>
  <List>
    <Item>1.5</Item>
    <Item>-2.25</Item>
  </List>
</Product>
"""
XML_PREFIXED = """<?xml version="1.0"?>
<Product xmlns:p="https://example.com/other"><p:Thing>1</p:Thing></Product>
"""
XML_MALFORMED = "this is not xml <<<"
XML_FOREIGN = "<Product/>"

# Exact polynomial surfaces for ``poly_fit_2d``.
POLY_X_ORDER = 2
POLY_Y_ORDER = 2
POLY_TREND = np.array([[3.0, 2.0, 0.5], [7.0, 1.25, -0.25], [-0.5, 0.25, 0.125]])
POLY_AXIS_UNIT = np.linspace(0.0, 1.0, 4)
POLY_AXIS_SCALED_X = np.linspace(0.0, 8.0, 5)
POLY_AXIS_SCALED_Y = np.linspace(0.0, 3.0, 5)
POLY_COEFFICIENT_TOLERANCE = 1e-6

# WGS-84 model used by the reader to place the SCP.
WGS84_A = 6378137.0
WGS84_B = 6356752.314245179

# Zero-Doppler geometry: spacecraft above the north pole flying east.
ARP = np.array([0.0, 0.0, 7.0e6])
VELOCITY = np.array([1.0, 0.0, 0.0])
SLANT_RANGE = 1.0e6
RANGE_TOLERANCE = 1e-9
ELLIPSOID_TOLERANCE = 1e-9


def _root(tmp_path, xml):
    """Write an XML fragment under ``tmp_path`` and return the parsed root element."""
    path = tmp_path / "metadata.xml"
    path.write_text(xml)
    return _parse_xml(str(path))


def _polynomial_surface(axis_x, axis_y, trend=POLY_TREND):
    """Sample the exact surface ``sum trend[i, j] * x**i * y**j`` over a grid."""
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    x = grid_x.ravel()
    y = grid_y.ravel()
    z = np.zeros_like(x)
    for index, coefficient in np.ndenumerate(trend):
        z += coefficient * np.power(x, index[0]) * np.power(y, index[1])
    return x, y, z


class TestCleanText:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._clean_text``."""

    @pytest.mark.unit
    def test_returns_none_for_missing_value(self):
        assert _clean_text(None) is None

    @pytest.mark.unit
    def test_trims_surrounding_whitespace(self):
        assert _clean_text(CLEAN_VALUE) == CLEAN_VALUE.strip()

    @pytest.mark.unit
    def test_whitespace_only_is_treated_as_missing(self):
        assert _clean_text(CLEAN_BLANK) is None

    @pytest.mark.unit
    def test_keeps_the_sentinel_when_requested(self):
        assert _clean_text(CLEAN_SENTINEL, keep_not_available=True) == CLEAN_SENTINEL
        assert _clean_text(CLEAN_SENTINEL.casefold(), keep_not_available=True) == CLEAN_SENTINEL.casefold()

    @pytest.mark.unit
    def test_drops_the_sentinel_case_insensitively(self):
        assert _clean_text(CLEAN_SENTINEL, keep_not_available=False) is None
        assert _clean_text(CLEAN_SENTINEL.casefold(), keep_not_available=False) is None

    @pytest.mark.unit
    def test_other_placeholders_are_passed_through(self):
        assert _clean_text(CLEAN_SLASH_SENTINEL, keep_not_available=False) == CLEAN_SLASH_SENTINEL

    @pytest.mark.unit
    def test_case_is_never_rewritten(self):
        assert _clean_text(CLEAN_VALUE.lower()) == CLEAN_VALUE.strip().lower()


class TestPolyFit2D:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel.poly_fit_2d``."""

    @pytest.mark.unit
    def test_recovers_the_trend_of_a_unit_grid(self):
        x, y, z = _polynomial_surface(POLY_AXIS_UNIT, POLY_AXIS_UNIT)
        trend, residuals, rank, _ = poly_fit_2d(x, y, z, POLY_X_ORDER, POLY_Y_ORDER)
        assert trend.shape == (POLY_X_ORDER + 1, POLY_Y_ORDER + 1)
        np.testing.assert_allclose(trend, POLY_TREND, atol=1e-9)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)
        assert rank == (POLY_X_ORDER + 1) * (POLY_Y_ORDER + 1)

    @pytest.mark.unit
    def test_returns_coefficients_in_unnormalized_coordinates(self):
        x, y, z = _polynomial_surface(POLY_AXIS_SCALED_X, POLY_AXIS_SCALED_Y)
        trend, residuals, rank, _ = poly_fit_2d(x, y, z, POLY_X_ORDER, POLY_Y_ORDER)
        np.testing.assert_allclose(trend, POLY_TREND, atol=POLY_COEFFICIENT_TOLERANCE)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)
        assert rank == (POLY_X_ORDER + 1) * (POLY_Y_ORDER + 1)

    @pytest.mark.unit
    def test_rejects_non_array_arguments(self):
        with pytest.raises(Exception, match="must be numpy arrays"):
            poly_fit_2d([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])

    @pytest.mark.unit
    def test_rejects_mismatched_cardinality(self):
        with pytest.raises(Exception, match="same cardinality size"):
            poly_fit_2d(np.zeros(4), np.zeros(4), np.zeros(3))


class TestParseTimestring:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel.parse_timestring``."""

    @pytest.mark.unit
    def test_parses_esa_stamp_to_microseconds(self):
        result = parse_timestring(ESA_STAMP, TIME_UNITS)
        assert result == ESA_AS_US
        assert result.dtype == np.dtype(f"datetime64[{TIME_UNITS}]")

    @pytest.mark.unit
    def test_parses_esa_stamp_without_fraction(self):
        assert parse_timestring(ESA_STAMP_NO_FRACTION) == ESA_NO_FRACTION_AS_US

    @pytest.mark.unit
    def test_month_name_is_case_insensitive(self):
        assert parse_timestring(ESA_STAMP_LOWER) == ESA_NO_FRACTION_AS_US

    @pytest.mark.unit
    def test_falls_back_to_iso_stamps(self):
        assert parse_timestring(ISO_STAMP) == ISO_AS_US

    @pytest.mark.unit
    def test_honours_the_requested_precision(self):
        assert parse_timestring(ISO_STAMP, "s") == ISO_AS_S

    @pytest.mark.unit
    def test_returns_none_for_the_not_available_placeholder(self):
        assert parse_timestring(CLEAN_SENTINEL) is None

    @pytest.mark.unit
    def test_returns_none_for_unparsable_stamps(self):
        assert parse_timestring(BAD_STAMP) is None
        assert parse_timestring("") is None
        assert parse_timestring(None) is None

    @pytest.mark.unit
    def test_rejects_an_unknown_month_abbreviation(self):
        with pytest.raises(ValueError, match="does not match format"):
            parse_timestring(BAD_MONTH)


class TestParseXml:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._parse_xml``."""

    @pytest.mark.unit
    def test_strips_the_default_namespace(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        assert root.tag == "Product"
        assert [child.tag for child in root] == ["Value", "Count", "Empty", "List"]

    @pytest.mark.unit
    def test_keeps_prefixed_namespaces(self, tmp_path):
        root = _root(tmp_path, XML_PREFIXED)
        assert [child.tag for child in root] == ["{https://example.com/other}Thing"]

    @pytest.mark.unit
    def test_rejects_malformed_documents(self, tmp_path):
        path = tmp_path / "metadata.xml"
        path.write_text(XML_MALFORMED)
        with pytest.raises(ElementTree.ParseError):
            _parse_xml(str(path))

    @pytest.mark.unit
    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(OSError, match="metadata.xml"):
            _parse_xml(str(tmp_path / "metadata.xml"))


class TestRequired:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._required``."""

    @pytest.mark.unit
    def test_returns_the_element_of_a_present_path(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        assert _required(root, "./Value").tag == "Value"

    @pytest.mark.unit
    def test_rejects_a_missing_path(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        with pytest.raises(SarpyIOError, match="element is missing"):
            _required(root, "./Absent")

    @pytest.mark.unit
    def test_rejects_an_element_without_content(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        with pytest.raises(SarpyIOError, match="element is missing"):
            _required(root, "./Empty")


class TestFloat:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._float``."""

    @pytest.mark.unit
    def test_parses_a_scientific_notation_value(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        assert _float(root, "./Value") == 25.0

    @pytest.mark.unit
    def test_rejects_a_missing_path(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        with pytest.raises(SarpyIOError, match="element is missing"):
            _float(root, "./Absent")

    @pytest.mark.unit
    def test_rejects_an_empty_element(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        with pytest.raises(SarpyIOError, match="element is missing"):
            _float(root, "./Empty")


class TestInt:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._int``."""

    @pytest.mark.unit
    def test_parses_an_integer_element(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        assert _int(root, "./Count") == 7

    @pytest.mark.unit
    def test_rejects_a_missing_path(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        with pytest.raises(SarpyIOError, match="element is missing"):
            _int(root, "./Absent")


class TestVals:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._vals``."""

    @pytest.mark.unit
    def test_collects_a_repeated_element(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        values = _vals(root, "./List/Item")
        assert values.dtype == np.dtype("float64")
        np.testing.assert_allclose(values, np.array([1.5, -2.25]))

    @pytest.mark.unit
    def test_returns_an_empty_array_for_an_absent_path(self, tmp_path):
        root = _root(tmp_path, XML_SAMPLE)
        values = _vals(root, "./Absent")
        assert values.dtype == np.dtype("float64")
        assert values.shape == (0,)


class TestPolarization:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._polarization``."""

    @pytest.mark.unit
    def test_splits_a_slash_separated_pair(self):
        assert _polarization("VV/VH") == ("VV", "VH", "VV:VH")

    @pytest.mark.unit
    def test_normalizes_a_colon_separated_pair(self):
        assert _polarization("V:H") == ("V", "H", "V:H")

    @pytest.mark.unit
    def test_uppercases_and_trims_the_components(self):
        assert _polarization(" v / h ") == ("V", "H", "V:H")

    @pytest.mark.unit
    def test_rejects_an_unpaired_string(self):
        with pytest.raises(SarpyIOError, match="Unexpected ROSE-L polarization"):
            _polarization("VV")

    @pytest.mark.unit
    def test_rejects_more_than_two_components(self):
        with pytest.raises(SarpyIOError, match="Unexpected ROSE-L polarization"):
            _polarization("V/H/H")


class TestEcfScpFromZeroDoppler:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel._ecf_scp_from_zero_doppler``."""

    @pytest.mark.unit
    def test_returns_a_point_on_the_wgs84_ellipsoid(self):
        point = _ecf_scp_from_zero_doppler(ARP, VELOCITY, SLANT_RANGE, "R")
        assert point.shape == (3,)
        assert point.dtype == np.dtype("float64")
        curvature = (point[0] ** 2 + point[1] ** 2) / WGS84_A**2 + point[2] ** 2 / WGS84_B**2 - 1.0
        np.testing.assert_allclose(curvature, 0.0, atol=ELLIPSOID_TOLERANCE)

    @pytest.mark.unit
    def test_holds_the_requested_slant_range(self):
        point = _ecf_scp_from_zero_doppler(ARP, VELOCITY, SLANT_RANGE, "L")
        np.testing.assert_allclose(np.linalg.norm(point - ARP), SLANT_RANGE, rtol=RANGE_TOLERANCE)

    @pytest.mark.unit
    def test_returns_a_zero_doppler_point(self):
        point = _ecf_scp_from_zero_doppler(ARP, VELOCITY, SLANT_RANGE, "R")
        np.testing.assert_allclose(np.dot(VELOCITY, point - ARP), 0.0, atol=1e-3)

    @pytest.mark.unit
    def test_side_selects_the_track_normal_direction(self):
        radial = ARP / np.linalg.norm(ARP)
        cross = np.cross(VELOCITY / np.linalg.norm(VELOCITY), radial)
        cross = cross / np.linalg.norm(cross)
        right = _ecf_scp_from_zero_doppler(ARP, VELOCITY, SLANT_RANGE, "Right")
        left = _ecf_scp_from_zero_doppler(ARP, VELOCITY, SLANT_RANGE, "left")
        assert np.dot(right - ARP, cross) > 0.0
        assert np.dot(left - ARP, cross) < 0.0
        np.testing.assert_allclose(right[0] - left[0], 0.0, atol=1e-9)
        assert not np.array_equal(right, left)


class TestIsA:
    """Unit tests for ``cop_ard_app.readers.sarpy_rosel.is_a``."""

    @pytest.mark.unit
    def test_returns_none_for_a_file_like(self):
        assert is_a(io.BytesIO(b"<Product/>")) is None

    @pytest.mark.unit
    def test_returns_none_for_an_absent_file(self, tmp_path):
        assert is_a(str(tmp_path / "absent.xml")) is None

    @pytest.mark.unit
    def test_returns_none_for_a_malformed_document(self, tmp_path):
        path = tmp_path / "metadata.xml"
        path.write_text(XML_MALFORMED)
        assert is_a(str(path)) is None

    @pytest.mark.unit
    def test_returns_none_for_a_foreign_document(self, tmp_path):
        path = tmp_path / "metadata.xml"
        path.write_text(XML_FOREIGN)
        assert is_a(str(path)) is None
