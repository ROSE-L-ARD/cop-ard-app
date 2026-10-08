import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from cop_ard_app.io import Path
from cop_ard_app.orbit import import_ORBIT, import_orbit, read_orbit_file, update_orbit

N = 8
POL = "VV"
BURST = "iw1_254"
SECOND_BURST = "iw1_255"
ACQUISITION = "2020-01-01"
COLLECT_START = "2020-01-01T00:00:00.000000Z"
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")

# the synthetic orbit track is exactly linear: position(t) = OFFSET + SLOPE * t_seconds
ORBIT_T0 = np.datetime64("2020-01-01T00:00:00", "s")
ORBIT_DT = 10
ORBIT_N = 60
OFFSET = [100.0, 200.0, 300.0]
SLOPE = [0.1, 0.2, 0.3]
VELOCITY = [7000.0, 7001.0, 7002.0]
COVERING = "S1A_ORB__IEA_A__V20200101T000000_20200101T010000.EOF"
PARTIAL = "S1A_ORB__IEA_A__V20200101T000000_20200101T001000.EOF"
NON_COVERING = "S1A_ORB__IEA_A__V20190101T000000_20190101T010000.EOF"
NO_VALIDITY = "S1A_ORB__IEA_A.EOF"
TOL = 1e-6

ORBIT_NODE = f"/bursts/{BURST}/orig/{ACQUISITION}/conditions/orbit"


def _annotations():
    """A fresh ISARIC-like annotation dict.

    update_slcs rewrites the ARPPoly coefficients in place, so every tree must own its dict.
    """
    return {
        "CollectionInfo": {"CollectorName": "test-collector"},
        "Timeline": {"CollectStart": COLLECT_START},
        "Position": {"ARPPoly": {axis: {"Coefs": [[0.0]]} for axis in "XYZ"}},
    }


def _measurements(dates, az_start=T0):
    """The measurements node of one original SLC, as consumed by update_orbit."""
    return xr.Dataset(
        {
            "slc": xr.DataArray(
                np.ones((1, len(dates), N, N), dtype=np.complex64),
                dims=["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
                attrs={POL: _annotations()},
            )
        },
        coords={
            "polarization": [POL],
            "acquisition_id": list(dates),
            "azimuth_time": az_start + np.arange(N) * AZ_DT,
            "slant_range_time": np.arange(N, dtype=np.float64),
        },
    )


def _slc_tree(bursts=(BURST,), az_starts=(T0,), multiburst=True):
    """An SLC stack with one acquisition, in multiburst or single-burst layout."""
    tree = xr.DataTree()
    for burst, az in zip(bursts, az_starts):
        path = f"bursts/{burst}/orig/{ACQUISITION}/measurements" if multiburst else f"orig/{ACQUISITION}/measurements"
        tree[path] = _measurements([ACQUISITION], az)
    return tree


def _orbit_times(n=ORBIT_N, dt=ORBIT_DT, start=ORBIT_T0):
    """State vector timestamps, one every dt seconds from start."""
    return [start + int(i * dt) * np.timedelta64(1, "s") for i in range(n)]


def _eof_xml(times, otype="POEORB"):
    """XML text of a minimal Earth_Orbit_File matching the schema build_orbit_dataset expects.

    Each OSV carries its UTC/TAI/UT1 stamps as 'KEY=value' text and the position/velocity
    components of the exact linear track described by OFFSET/SLOPE.
    """
    lines = [
        "<Earth_Orbit_File>",
        "<Header>",
        f"<File_Type>{otype}</File_Type>",
        "</Header>",
        "<Body>",
        f'<List_of_OSVs count="{len(times)}">',
    ]
    for t in times:
        stamp = str(np.datetime64(t, "s"))
        elapsed = int((np.datetime64(t, "s") - ORBIT_T0) / np.timedelta64(1, "s"))
        lines.append("<OSV>")
        lines.append(f"<UTC>TAI={stamp}</UTC>")
        lines.append(f"<TAI>UTC={stamp}</TAI>")
        lines.append(f"<UT1>UTC={stamp}</UT1>")
        for k, tag in enumerate(["X", "Y", "Z"]):
            lines.append(f"<{tag}>{OFFSET[k] + SLOPE[k] * elapsed}</{tag}>")
        for k, tag in enumerate(["VX", "VY", "VZ"]):
            lines.append(f"<{tag}>{VELOCITY[k] + float(elapsed)}</{tag}>")
        lines.append("</OSV>")
    lines += ["</List_of_OSVs>", "</Body>", "</Earth_Orbit_File>"]
    return "\n".join(lines)


def _write_orbit_file(directory, name, times, otype="POEORB"):
    """Write one synthetic orbit file and hand back the cop_ard_app Path to it."""
    file_path = directory / name
    file_path.write_text(_eof_xml(times, otype))
    return Path(str(file_path))


def _expected_coeffs(degree):
    """The polynomial sarpy must recover for the exact linear track (constant term first)."""
    expected = np.zeros((degree + 1, 3))
    expected[0] = OFFSET
    expected[1] = SLOPE
    return expected


class TestReadOrbitFile:
    """Unit test for cop_ard_app.orbit.read_orbit_file."""

    @pytest.mark.unit
    def test_read_orbit_file_returns_the_parsed_tree(self, tmp_path):
        _write_orbit_file(tmp_path, COVERING, _orbit_times(3))
        tree = read_orbit_file(Path(str(tmp_path / COVERING)))
        assert isinstance(tree, ET.ElementTree)
        root = tree.getroot()
        assert root.tag == "Earth_Orbit_File"
        assert root.find(".//File_Type").text == "POEORB"
        osvs = root.find(".//List_of_OSVs")
        assert osvs is not None
        assert len(osvs.findall("OSV")) == 3

    @pytest.mark.unit
    def test_read_orbit_file_does_not_validate_the_orbit_content(self, tmp_path):
        (tmp_path / "header_only.EOF").write_text("<Earth_Orbit_File><Header></Header></Earth_Orbit_File>")
        tree = read_orbit_file(Path(str(tmp_path / "header_only.EOF")))
        # reading is only XML parsing: a file without state vectors is still returned as a tree
        assert tree.getroot().tag == "Earth_Orbit_File"
        assert tree.getroot().find(".//List_of_OSVs") is None


class TestImportOrbit:
    """Unit test for cop_ard_app.orbit.import_orbit."""

    @pytest.mark.unit
    def test_import_orbit_without_paths_returns_none(self):
        assert import_orbit(None) is None
        assert import_orbit([None]) is None

    @pytest.mark.unit
    def test_import_orbit_builds_the_state_vector_dataset(self, tmp_path):
        times = _orbit_times(5, dt=ORBIT_DT)
        orbit = import_orbit([_write_orbit_file(tmp_path, COVERING, times)])
        assert set(orbit.data_vars) == {"position", "velocity", "orbit_file"}
        assert orbit.attrs["type"] == "POEORB"
        assert orbit.position.dims == ("azimuth_time", "axis")
        assert orbit.position.shape == (5, 3)
        assert np.array_equal(orbit.coords["azimuth_time"].values, np.array(times, dtype="datetime64[ns]"))
        elapsed = np.arange(5) * ORBIT_DT
        expected = np.array([[OFFSET[k] + SLOPE[k] * s for k in range(3)] for s in elapsed])
        assert np.allclose(orbit.position.values, expected, atol=TOL)
        expected_velocity = np.array([[VELOCITY[k] + float(s) for k in range(3)] for s in elapsed])
        assert np.allclose(orbit.velocity.values, expected_velocity, atol=TOL)
        # the source file names are attached as the orbit_file variable
        assert str(orbit.orbit_file.values[0]) == COVERING

    @pytest.mark.unit
    def test_import_orbit_filters_to_the_requested_interval(self, tmp_path):
        times = _orbit_times(10, dt=60)
        start = ORBIT_T0 + 120 * np.timedelta64(1, "s")
        end = ORBIT_T0 + 300 * np.timedelta64(1, "s")
        orbit = import_orbit([_write_orbit_file(tmp_path, COVERING, times)], start_time=start, end_time=end)
        # both interval bounds are kept, everything outside is dropped
        assert np.array_equal(orbit.coords["azimuth_time"].values, np.array(times[2:6], dtype="datetime64[ns]"))
        assert orbit.position.shape == (4, 3)

    @pytest.mark.unit
    def test_import_orbit_concatenates_several_files(self, tmp_path):
        first = _write_orbit_file(tmp_path, COVERING, _orbit_times(5, dt=60))
        second_name = "S1A_ORB__IEA_B__V20200101T000500_20200101T000959.EOF"
        second = _write_orbit_file(
            tmp_path,
            second_name,
            _orbit_times(5, dt=60, start=ORBIT_T0 + 300 * np.timedelta64(1, "s")),
        )
        orbit = import_orbit([first, second])
        assert orbit.position.shape == (10, 3)
        merged = [np.datetime64(t, "ns") for t in _orbit_times(5, dt=60)]
        merged += [
            np.datetime64(t, "ns") for t in _orbit_times(5, dt=60, start=ORBIT_T0 + 300 * np.timedelta64(1, "s"))
        ]
        assert np.array_equal(orbit.coords["azimuth_time"].values, np.array(merged, dtype="datetime64[ns]"))
        # every source file is recorded, in the order it was given
        assert str(orbit.orbit_file.values[0]) == f"{COVERING},{second_name}"

    @pytest.mark.unit
    def test_import_orbit_rejects_mixed_orbit_types(self, tmp_path):
        first = _write_orbit_file(tmp_path, COVERING, _orbit_times(2), otype="POEORB")
        second = _write_orbit_file(
            tmp_path,
            "S1A_ORB__IEA_B__V20200101T000500_20200101T000959.EOF",
            _orbit_times(2, dt=60, start=ORBIT_T0 + 300 * np.timedelta64(1, "s")),
            otype="RESORB",
        )
        with pytest.raises(RuntimeError, match="same orbit type"):
            import_orbit([first, second])

    @pytest.mark.unit
    def test_import_orbit_rejects_a_file_without_state_vectors(self, tmp_path):
        (tmp_path / "header_only.EOF").write_text(
            "<Earth_Orbit_File><Header><File_Type>POEORB</File_Type></Header></Earth_Orbit_File>"
        )
        with pytest.raises(ValueError, match="error reading orbit file"):
            import_orbit([Path(str(tmp_path / "header_only.EOF"))])


class TestUpdateOrbit:
    """Unit test for cop_ard_app.orbit.update_orbit."""

    @pytest.mark.unit
    def test_update_orbit_refits_the_orbit_polynomials(self, tmp_path):
        orbit_file = _write_orbit_file(tmp_path, COVERING, _orbit_times())
        slcs = update_orbit(_slc_tree(), str(tmp_path), orbit_paths=[orbit_file], poly_degree=2, enable_dask=False)
        node = slcs[ORBIT_NODE]
        coeffs = np.asarray(node.coeffs.values)
        assert coeffs.shape == (1, 3, 3)
        # the synthetic track is exactly linear, so the fit must reproduce it
        assert np.allclose(coeffs[0], _expected_coeffs(2), atol=1e-9)
        assert node.attrs["orbit_type"] == "POEORB"
        assert node.attrs["reference_time"] == COLLECT_START
        assert list(node.attrs["orbit_files"]) == [COVERING]
        # the SLC metadata is overwritten with the same coefficients
        arppoly = slcs[f"bursts/{BURST}/orig/{ACQUISITION}/measurements/slc"].attrs[POL]["Position"]["ARPPoly"]
        for k, axis in enumerate("XYZ"):
            assert np.allclose(np.asarray(arppoly[axis]["Coefs"], dtype=float), coeffs[0, :, k], atol=TOL)

    @pytest.mark.unit
    def test_update_orbit_handles_a_single_burst_stack(self, tmp_path):
        orbit_file = _write_orbit_file(tmp_path, COVERING, _orbit_times())
        slcs = update_orbit(
            _slc_tree(multiburst=False),
            str(tmp_path),
            orbit_paths=[orbit_file],
            multiburst=False,
            poly_degree=1,
            enable_dask=False,
        )
        node = slcs[f"orig/{ACQUISITION}/conditions/orbit"]
        assert node.coeffs.shape == (1, 2, 3)
        assert np.allclose(np.asarray(node.coeffs.values)[0], _expected_coeffs(1), atol=1e-9)
        assert node.attrs["orbit_type"] == "POEORB"

    @pytest.mark.unit
    def test_update_orbit_leaves_the_stack_untouched_when_no_orbit_covers_it(self, tmp_path):
        orbit_file = _write_orbit_file(tmp_path, NON_COVERING, _orbit_times())
        slcs = update_orbit(_slc_tree(), str(tmp_path), orbit_paths=[orbit_file], poly_degree=1, enable_dask=False)
        # the validity window parsed from the filename does not overlap the acquisition
        assert ORBIT_NODE not in slcs.groups
        arppoly = slcs[f"bursts/{BURST}/orig/{ACQUISITION}/measurements/slc"].attrs[POL]["Position"]["ARPPoly"]
        assert arppoly == {axis: {"Coefs": [[0.0]]} for axis in "XYZ"}

    @pytest.mark.unit
    def test_update_orbit_without_orbit_paths_finds_no_orbit_in_the_etad_metadata(self, tmp_path):
        slcs = update_orbit(_slc_tree(), str(tmp_path), poly_degree=1, enable_dask=False)
        # no ETAD node carries an orbit_file, so the acquisition is skipped
        assert ORBIT_NODE not in slcs.groups
        arppoly = slcs[f"bursts/{BURST}/orig/{ACQUISITION}/measurements/slc"].attrs[POL]["Position"]["ARPPoly"]
        assert arppoly == {axis: {"Coefs": [[0.0]]} for axis in "XYZ"}


class TestImportORBIT:
    """Unit test for cop_ard_app.orbit.import_ORBIT."""

    @pytest.mark.unit
    def test_import_orbit_selects_the_covering_orbit_files(self, tmp_path):
        covering = _write_orbit_file(tmp_path, COVERING, _orbit_times())
        non_covering = _write_orbit_file(tmp_path, NON_COVERING, _orbit_times())
        slcs, paths = import_ORBIT(
            _slc_tree(),
            [ACQUISITION],
            [non_covering, covering],
            str(tmp_path),
            multiburst=True,
            poly_degree=1,
            enable_dask=False,
        )
        # only the file whose validity window contains the acquisition is kept
        assert paths == [str(tmp_path / COVERING)]
        assert ORBIT_NODE in slcs.groups
        coeffs = np.asarray(slcs[ORBIT_NODE].coeffs.values)
        assert np.allclose(coeffs[0], _expected_coeffs(1), atol=1e-9)

    @pytest.mark.unit
    def test_import_orbit_returns_nothing_when_no_file_covers_the_acquisition(self, tmp_path):
        non_covering = _write_orbit_file(tmp_path, NON_COVERING, _orbit_times())
        slcs, paths = import_ORBIT(
            _slc_tree(),
            [ACQUISITION],
            [non_covering],
            str(tmp_path),
            multiburst=True,
            poly_degree=1,
            enable_dask=False,
        )
        assert paths == []
        assert ORBIT_NODE not in slcs.groups

    @pytest.mark.unit
    def test_import_orbit_rejects_a_filename_without_validity_dates(self, tmp_path):
        malformed = _write_orbit_file(tmp_path, NO_VALIDITY, _orbit_times())
        with pytest.raises(ValueError, match="No validity dates found in filename"):
            import_ORBIT(_slc_tree(), [ACQUISITION], [malformed], str(tmp_path), multiburst=True)

    @pytest.mark.unit
    def test_import_orbit_covers_the_whole_multiburst_span(self, tmp_path):
        late = T0 + 3000 * AZ_DT
        covering = _write_orbit_file(tmp_path, COVERING, _orbit_times())
        partial = _write_orbit_file(tmp_path, PARTIAL, _orbit_times())
        slcs, paths = import_ORBIT(
            _slc_tree(bursts=(BURST, SECOND_BURST), az_starts=(T0, late)),
            [ACQUISITION],
            [partial, covering],
            str(tmp_path),
            multiburst=True,
            poly_degree=1,
            enable_dask=False,
        )
        # the second burst ends 50 minutes in, so only the full hour window covers the stack
        assert paths == [str(tmp_path / COVERING)]
        for burst in (BURST, SECOND_BURST):
            assert f"/bursts/{burst}/orig/{ACQUISITION}/conditions/orbit" in slcs.groups
