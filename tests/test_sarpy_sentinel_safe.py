"""Unit tests for the module-level helpers of ``cop_ard_app.readers.sarpy_sentinel_safe``."""

import io
from xml.etree import ElementTree

import pytest

from cop_ard_app.readers.sarpy_sentinel_safe import (  # type: ignore
    _parse_xml,
    find_string,
    is_a,
)

# XML fragments written to ``tmp_path`` and parsed by ``_parse_xml``.
DEFAULT_NS = "https://sentinel.esa.int/xsd/sentinel-safe"
PREFIXED_NS = "https://example.com/other"
XML_SAMPLE = """<?xml version="1.0"?>
<Product xmlns="https://sentinel.esa.int/xsd/sentinel-safe" xmlns:p="https://example.com/other">
  <Value>2.5e1</Value>
  <Empty></Empty>
</Product>
"""
XML_PLAIN = """<?xml version="1.0"?>
<Root>
  <Value>2.5e1</Value>
  <Empty></Empty>
</Root>
"""
XML_MALFORMED = "this is not xml <<<"
XML_FOREIGN = "<Product/>"


def _metadata(tmp_path, text):
    """Write an XML fragment under ``tmp_path`` and return the file path."""
    path = tmp_path / "metadata.xml"
    path.write_text(text)
    return str(path)


class TestParseXml:
    """Unit tests for ``cop_ard_app.readers.sarpy_sentinel_safe._parse_xml``."""

    @pytest.mark.unit
    def test_default_call_returns_the_namespace_map_and_root(self, tmp_path):
        ns, root = _parse_xml(_metadata(tmp_path, XML_SAMPLE))
        assert ns == {"": DEFAULT_NS, "p": PREFIXED_NS}
        assert root.tag == f"{{{DEFAULT_NS}}}Product"

    @pytest.mark.unit
    def test_without_ns_returns_only_the_root_element(self, tmp_path):
        root = _parse_xml(_metadata(tmp_path, XML_SAMPLE), without_ns=True)
        assert isinstance(root, ElementTree.Element)
        assert root.tag == f"{{{DEFAULT_NS}}}Product"

    @pytest.mark.unit
    def test_collects_no_namespaces_for_a_plain_document(self, tmp_path):
        ns, root = _parse_xml(_metadata(tmp_path, XML_PLAIN))
        assert ns == {}
        assert root.tag == "Root"
        assert [child.tag for child in root] == ["Value", "Empty"]

    @pytest.mark.unit
    def test_rejects_malformed_documents(self, tmp_path):
        with pytest.raises(ElementTree.ParseError):
            _parse_xml(_metadata(tmp_path, XML_MALFORMED))

    @pytest.mark.unit
    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(OSError, match="metadata.xml"):
            _parse_xml(str(tmp_path / "metadata.xml"))


class TestFindString:
    """Unit tests for ``cop_ard_app.readers.sarpy_sentinel_safe.find_string``."""

    @pytest.mark.unit
    def test_returns_the_text_of_a_present_element(self, tmp_path):
        root = _parse_xml(_metadata(tmp_path, XML_PLAIN), without_ns=True)
        assert find_string(root, "./Value") == "2.5e1"

    @pytest.mark.unit
    def test_raises_when_the_path_is_missing(self, tmp_path):
        root = _parse_xml(_metadata(tmp_path, XML_PLAIN), without_ns=True)
        with pytest.raises(Exception, match="Expected Element for"):
            find_string(root, "./Absent")

    @pytest.mark.unit
    def test_raises_when_the_element_has_no_text(self, tmp_path):
        root = _parse_xml(_metadata(tmp_path, XML_PLAIN), without_ns=True)
        with pytest.raises(Exception, match="Expected str for .* got None"):
            find_string(root, "./Empty")


class TestIsA:
    """Unit tests for the negative paths of ``cop_ard_app.readers.sarpy_sentinel_safe.is_a``.

    The positive sniff delegates to the unread ``SentinelDetails`` parser, so a
    synthetic SAFE manifest is not constructed here; only negative paths are covered.
    """

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
