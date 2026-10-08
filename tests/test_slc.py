"""Unit tests for cop_ard_app.slc."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from sarpy.io.complex.sicd_elements.blocks import LatLonCornerStringType, LatLonHAERestrictionType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType

from cop_ard_app.io import save_slc_static
from cop_ard_app.slc import (
    finalize_import,
    group_discover,
    import_slice,
    overwrite_orbit_coefficients,
    update_slcs,
    updateETAD,
    updateLookVector,
    updateOrbits,
)

NX, NY = 6, 6
SIM_Y, SIM_X = 12, 16
N_ACQ = 2

COLLECTOR = "sentinel-1"
START = "2023-01-07T00:33:11.203626"
ORBIT = 20456
ACQ_MB = f"{COLLECTOR}_{ORBIT}"
ACQ_NB = f"{COLLECTOR}_{START}"

CENTER = (45.0, 10.0)
SPAN = 0.01

AZ_DT = np.timedelta64(36_000_000, "ns")
RG_DT = 1.0
TOL = 1e-9

ETAD_VARS = ["az0", "az_r0", "sfct", "r0", "rg0", "rr", "rfg"]
_AZ_RG = (np.arange(NX, dtype=float), np.arange(NY, dtype=float))
_ETAD_GRID = 1.0 + 0.1 * _AZ_RG[0][:, None] + 0.2 * _AZ_RG[1][None, :]
_ETAD_FIELDS = [(j + 1.0) * _ETAD_GRID for j in range(len(ETAD_VARS))]

PIXEL_VALUE = 5.0 + 1.0j


def _corner(i, j):
    """Corner coordinates on a synthetic image footprint."""
    lat = CENTER[0] + (j - 0.5) * SPAN
    lon = CENTER[1] + (i - 0.5) * SPAN
    return LatLonCornerStringType(Lat=lat, Lon=lon)


def _make_sicd(swath=None, burst=None):
    """Synthetic SICD metadata for the sarpy_open seam fixtures."""
    if swath is None:
        idx = [(0, 0), (1, 0), (1, 1), (0, 1)]
    else:
        idx = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1), (1, 1), (0, 1)]
    corners = [_corner(i, j) for (i, j) in idx]
    params = {"ABSOLUTE_ORBIT": str(ORBIT)}
    if swath is not None:
        params["SWATH"] = swath
        params["BURST"] = str(burst)
    return SICDType(
        CollectionInfo=CollectionInfoType(CollectorName=COLLECTOR, Parameters=params),
        Timeline=TimelineType(CollectStart=np.datetime64(START, "us"), CollectDuration=1.5),
        GeoData=GeoDataType(
            SCP=SCPType(LLH=LatLonHAERestrictionType(Lat=CENTER[0], Lon=CENTER[1], HAE=700.0)),
            ImageCorners=corners,
        ),
    )


class _FakeReader:
    """Minimal stand-in for a sarpy reader exposing sicd_meta."""

    def __init__(self, metas):
        self.sicd_meta = metas


def _open_sicd(*metas):
    """Mock of sarpy_open returning a reader over fixed SICD metadata."""

    def _open(path, **kwargs):
        return _FakeReader(list(metas))

    return _open


def _slc_attrs(collector=COLLECTOR, orbit=ORBIT, start=START):
    """Annotation-style attrs attached to the slc DataArray."""
    return {
        f"{collector}_{start}": {
            "CollectionInfo": {"CollectorName": collector, "Parameters": {"ABSOLUTE_ORBIT": str(orbit)}},
            "Timeline": {"CollectStart": start},
            "Position": {"ARPPoly": {tag: {"Coefs": [0.0] * 11} for tag in ["X", "Y", "Z"]}},
            "RMA": {"INCA": {"FreqZero": 1.4e9}},
        },
    }


def _slc_leaf(az=NX, rg=NY, pol=1, value=PIXEL_VALUE):
    """Zero-filled 4D slc DataArray matching the pipeline layout."""
    return xr.DataArray(
        np.full((pol, az, rg), value, np.complex64),
        dims=("acquisition_id", "azimuth_time", "slant_range_time"),
    )


def _meas_tree(acq_ids=("acq_1",), with_orbit=False, with_etad=False, attrs_key=None, start=None):
    """DataTree with orig nodes carrying slc metadata (and optionally orbit/ETAD nodes)."""
    tree = {}
    for acq in acq_ids:
        key = acq if attrs_key is None else attrs_key
        tree[f"orig/{acq}/measurements/slc"] = _slc_leaf().assign_attrs(_slc_attrs(start=start or key))
        tree[f"orig/{acq}/conditions"] = xr.Dataset()
        if with_orbit:
            coeffs = np.full((1, 11, 3), 7.0)
            tree[f"orig/{acq}/conditions/orbit/coeffs"] = xr.DataArray(
                coeffs,
                dims=("acquisition_id", "degree", "axis"),
                coords={"acquisition_id": [acq]},
            )
        if with_etad:
            for j, name in enumerate(ETAD_VARS):
                tree[f"orig/{acq}/conditions/ETAD/{name}"] = xr.DataArray(
                    _ETAD_FIELDS[j],
                    dims=("azimuth_time", "slant_range_time"),
                    coords={"azimuth_time": _AZ_RG[0], "slant_range_time": _AZ_RG[1]},
                )
    return xr.DataTree.from_dict(tree)


def _meas_tree_bursts(acq="acq_1", burst="b_1", attrs_key=None):
    """Multiburst-shaped DataTree."""
    return xr.DataTree.from_dict(
        {
            f"bursts/{burst}/orig/{acq}/measurements/slc": _slc_leaf().assign_attrs(_slc_attrs(start=attrs_key or acq)),
            f"bursts/{burst}/orig/{acq}/conditions": xr.Dataset(),
        },
    )


def _meas_coreg(n_acq=N_ACQ):
    """Coregistration measurements Dataset consumed by updateETAD."""
    return xr.Dataset(
        data_vars={
            "daz": (
                ("acquisition_id", "azimuth_time", "slant_range_time"),
                np.zeros((n_acq, NX, NY)),
            ),
            "drg": (
                ("acquisition_id", "azimuth_time", "slant_range_time"),
                np.zeros((n_acq, NX, NY)),
            ),
        },
        coords={
            "acquisition_id": [f"acq_{i + 1}" for i in range(n_acq)],
            "azimuth_time": _AZ_RG[0],
            "slant_range_time": _AZ_RG[1],
        },
    )


class TestGroupDiscover:
    """Tests of cop_ard_app.slc.group_discover."""

    @pytest.mark.unit
    def test_group_discover_single(self):
        """SICDs without SWATH/BURST parameters group under orig/ with the collector and date."""
        with patch("cop_ard_app.slc.sarpy_open", _open_sicd(_make_sicd())):
            res = group_discover("dir/sample.xml")
        assert res == {f"orig/{ACQ_NB}": {"UN": [0, "dir/sample.xml"]}}

    @pytest.mark.unit
    def test_group_discover_multiburst(self):
        """SICDs with SWATH/BURST parameters group under bursts/swath_burst/orig/ with the orbit number."""
        metas = [_make_sicd(swath="IW1", burst=254), _make_sicd(swath="IW2", burst=255)]
        with patch("cop_ard_app.slc.sarpy_open", _open_sicd(*metas)):
            res = group_discover("dir/product.SAFE", multiburst=True)
        assert res == {
            "bursts/iw1_254/orig/sentinel-1_20456": {"UN": [0, "dir/product.SAFE"]},
            "bursts/iw2_255/orig/sentinel-1_20456": {"UN": [1, "dir/product.SAFE"]},
        }

    @pytest.mark.unit
    def test_group_discover_aoi_miss(self):
        """Paths whose footprint misses the AOI must be dropped."""
        metas = [_make_sicd(swath="IW1", burst=254)]
        aoi = "POLYGON ((100.0 44.99, 100.01 44.99, 100.01 45.01, 100.0 45.01, 100.0 44.99))"
        with patch("cop_ard_app.slc.sarpy_open", _open_sicd(*metas)):
            res = group_discover("dir/product.SAFE", multiburst=True, aoi_wkt=aoi)
        assert res == {}

    @pytest.mark.unit
    def test_group_discover_polarization_filter(self):
        """Paths whose processed polarization is not requested must be dropped."""
        metas = [_make_sicd(swath="IW1", burst=254)]
        with patch("cop_ard_app.slc.sarpy_open", _open_sicd(*metas)):
            res = group_discover("dir/product.SAFE", multiburst=True, polarizations=["P"])
        assert res == {}

    @pytest.mark.unit
    def test_group_discover_indices(self):
        """Only the requested image indices are processed."""
        metas = [_make_sicd(swath="IW1", burst=254), _make_sicd(swath="IW2", burst=255)]
        with patch("cop_ard_app.slc.sarpy_open", _open_sicd(*metas)):
            res = group_discover("dir/product.SAFE", multiburst=True, indices=[1])
        assert res == {"bursts/iw2_255/orig/sentinel-1_20456": {"UN": [1, "dir/product.SAFE"]}}


class TestFinalizeImport:
    """Tests of cop_ard_app.slc.finalize_import."""

    @pytest.mark.unit
    def test_finalize_import_multiburst_recompute(self, tmp_path):
        """Recompute saves the reference structure to the static directory."""
        slcs = _meas_tree_bursts()
        res, acquisition_id = finalize_import(slcs, str(tmp_path), recompute=True, enable_dask=False)
        assert acquisition_id == "acq_1"
        assert list(res.bursts) == ["b_1"]
        assert (tmp_path / "processed_data/b_1/slc.ok").exists()

    @pytest.mark.unit
    def test_finalize_import_multiburst_recompute_ambiguous(self, tmp_path):
        """Recompute must resolve a single acquisition id, otherwise it is refused."""
        slcs = xr.merge([_meas_tree_bursts(acq="acq_1"), _meas_tree_bursts(acq="acq_2")])
        with pytest.raises(ValueError, match="Expected exactly one acquisition id"):
            finalize_import(slcs, str(tmp_path), recompute=True, enable_dask=False)

    @pytest.mark.unit
    def test_finalize_import_multiburst_missing_static(self, tmp_path):
        """Bursts without a reference are removed; an empty stack raises."""
        slcs = _meas_tree_bursts()
        with pytest.raises(RuntimeError, match="No reference image found"):
            finalize_import(slcs, str(tmp_path / "empty"), recompute=False, enable_dask=False)

    @pytest.mark.unit
    def test_finalize_import_multiburst_already_stacked(self, tmp_path):
        """An already present reference id is reported and left untouched."""
        tree = _meas_tree_bursts(acq=ACQ_MB)
        save_slc_static("b_1", tree[f"bursts/b_1/orig/{ACQ_MB}"], str(tmp_path))
        res, acquisition_id = finalize_import(tree, str(tmp_path), recompute=False, enable_dask=False)
        assert acquisition_id == ACQ_MB
        assert list(res.bursts) == ["b_1"]
        assert list(res["bursts/b_1"].orig) == [ACQ_MB]

    @pytest.mark.unit
    def test_finalize_import_non_multiburst_requires_id(self, tmp_path):
        """Without bursts organization the acquisition id must be provided."""
        slcs = _meas_tree(("acq_1",))
        with pytest.raises(ValueError, match="must be provided when data is not organized by bursts"):
            finalize_import(slcs, str(tmp_path), recompute=False, enable_dask=False)

    @pytest.mark.unit
    def test_finalize_import_non_multiburst_missing_static(self, tmp_path):
        """A non-multiburst stack without static reference data raises."""
        slcs = _meas_tree(("acq_1",))
        with pytest.raises(ValueError, match="Static reference folder/data for"):
            finalize_import(
                slcs,
                str(tmp_path / "empty"),
                recompute=False,
                acquisition_id=ACQ_NB,
                enable_dask=False,
            )

    @pytest.mark.unit
    def test_finalize_import_non_multiburst_static_present(self, tmp_path):
        """The reference id is found in the static directory and already in the stack."""
        tree = _meas_tree((ACQ_NB,))
        save_slc_static("data", tree["orig"][ACQ_NB], str(tmp_path), reference_id=ACQ_NB)
        res, acquisition_id = finalize_import(
            tree,
            str(tmp_path),
            recompute=False,
            acquisition_id=ACQ_NB,
            enable_dask=False,
        )
        assert acquisition_id == ACQ_NB
        assert list(res.orig) == [ACQ_NB]

    @pytest.mark.unit
    def test_finalize_import_non_multiburst_recompute_ambiguous(self, tmp_path):
        """Two acquisition ids with recompute raise."""
        slcs = xr.merge([_meas_tree(("acq_1",)), _meas_tree(("acq_2",))])
        with pytest.raises(ValueError, match="Expected exactly one acquisition id when recompute=True"):
            finalize_import(slcs, str(tmp_path), recompute=True, enable_dask=False)


class TestUpdateSlcs:
    """Tests of cop_ard_app.slc.update_slcs."""

    @pytest.mark.unit
    def test_update_slcs_multiburst(self):
        """The ARPPoly coefficients of the slc attrs are replaced by the refitted orbit."""
        coeffs = np.arange(33, dtype=float).reshape((1, 3, 11))
        slcs = _meas_tree_bursts()
        orbit = xr.Dataset(
            {"coeffs": xr.DataArray(coeffs, dims=("acquisition_id", "axis", "degree"))},
        )
        res = update_slcs(slcs, True, [("b_1", "acq_1", orbit), ("b_2", "acq_2", None)])
        attrs = res["bursts/b_1/orig/acq_1/measurements/slc"].attrs
        pol = next(iter(attrs))
        assert attrs[pol]["Position"]["ARPPoly"]["X"]["Coefs"] == coeffs[0, 0].tolist()
        assert attrs[pol]["Position"]["ARPPoly"]["Y"]["Coefs"] == coeffs[0, 1].tolist()
        assert attrs[pol]["Position"]["ARPPoly"]["Z"]["Coefs"] == coeffs[0, 2].tolist()
        assert list(res["bursts/b_1/orig/acq_1/conditions/orbit"]) == ["coeffs"]
        assert "b_2" not in res.bursts

    @pytest.mark.unit
    def test_update_slcs_non_multiburst(self):
        """Same update for a single acquisition tree."""
        coeffs = np.arange(33, dtype=float).reshape((1, 3, 11)) + 1.0
        coeffs[:, 2] = 7.0
        slcs = _meas_tree()
        orbit = xr.Dataset({"coeffs": xr.DataArray(coeffs, dims=("acquisition_id", "axis", "degree"))})
        res = update_slcs(slcs, False, [(None, "acq_1", orbit)])
        attrs = res["orig/acq_1/measurements/slc"].attrs
        pol = next(iter(attrs))
        assert attrs[pol]["Position"]["ARPPoly"]["X"]["Coefs"] == coeffs[0, 0].tolist()

    @pytest.mark.unit
    def test_update_slcs_skips_missing_orbits(self):
        """A tuple with no orbit data leaves the tree unchanged."""
        slcs = _meas_tree()
        res = update_slcs(slcs, False, [(None, "acq_1", None)])
        attrs = res["orig/acq_1/measurements/slc"].attrs
        pol = next(iter(attrs))
        assert attrs[pol]["Position"]["ARPPoly"]["X"]["Coefs"] == [0.0] * 11
        assert "orbit" not in res["orig/acq_1"].conditions


class TestUpdateOrbits:
    """Tests of cop_ard_app.slc.updateOrbits."""

    @pytest.mark.unit
    def test_update_orbits_from_annotations(self):
        """Coefficients come from the ARPPoly annotations with annotated type."""
        slcs = _meas_tree(start=START)
        res = updateOrbits(slcs)
        assert res.coeffs.shape == (1, 11, 3)
        assert np.allclose(res.coeffs.values, 0.0, atol=TOL)
        assert res.type.values.tolist() == ["AUX_ANNOTATED"]
        assert res.reference_time.values.tolist() == [START]
        assert res.orbit_files.values.tolist() == [""]

    @pytest.mark.unit
    def test_update_orbits_precise_override(self):
        """A precise orbit node overwrites coefficients, type, files and reference time."""
        slcs = _meas_tree(with_orbit=True, start=START)
        slcs["orig/acq_1/conditions/orbit"].attrs.update(
            {
                "orbit_type": "AUX_POEORB",
                "orbit_files": ["EO"],
                "reference_time": "2023-01-07T00:33:10.000000",
            },
        )
        res = updateOrbits(slcs)
        assert np.allclose(res.coeffs.values, 7.0, atol=TOL)
        assert res.type.values.tolist() == ["AUX_POEORB"]
        # numpy keeps the fixed width of the empty annotated field, so the name is truncated
        assert res.orbit_files.values.tolist() == ["E"]
        assert res.reference_time.values.tolist() == ["2023-01-07T00:33:10.000000"]


class TestUpdateETAD:
    """Tests of cop_ard_app.slc.updateETAD."""

    @pytest.mark.unit
    def test_update_etad_identity(self):
        """A target grid identical to the ETAD grid reproduces the correction grids exactly."""
        tree = _meas_tree(with_etad=True)
        res = updateETAD(tree, _meas_coreg(), (NX, NY))
        assert set(res.data_vars) == set(ETAD_VARS)
        for j, name in enumerate(ETAD_VARS):
            assert np.allclose(res[name].values[0], _ETAD_FIELDS[j], atol=1e-9)

    @pytest.mark.unit
    def test_update_etad_concat(self):
        """ETAD nodes from several acquisitions concatenate along the acquisition axis."""
        tree = _meas_tree(acq_ids=("acq_1", "acq_2"), with_etad=True)
        res = updateETAD(tree, _meas_coreg(n_acq=N_ACQ), (NX, NY))
        assert res.acquisition_id.values.tolist() == ["acq_1", "acq_2"]
        assert res.sizes["acquisition_id"] == N_ACQ
        for j, name in enumerate(ETAD_VARS):
            assert np.allclose(res[name].values, _ETAD_FIELDS[j], atol=1e-9)

    @pytest.mark.unit
    def test_update_etad_missing(self):
        """A tree without any ETAD node yields None."""
        tree = _meas_tree()
        assert updateETAD(tree, _meas_coreg(), (NX, NY)) is None


class TestUpdateLookVector:
    """Tests of cop_ard_app.slc.updateLookVector."""

    @pytest.mark.unit
    def test_update_look_vector_constant_field(self):
        """A constant DEM vector field yields exactly that unit vector on the target grid."""
        base = np.datetime64(START, "ns")
        az_meas = base + np.arange(NX) * 2 * AZ_DT
        rg_meas = np.arange(NY, dtype=float) * 2 * RG_DT
        measurements = xr.Dataset(coords={"azimuth_time": az_meas, "slant_range_time": rg_meas})
        sim_az = (
            base + (np.arange(SIM_Y)[:, None] + 0.5) * AZ_DT + (np.arange(SIM_X)[None, :]) * np.timedelta64(1, "ms")
        )
        sim_rg = (np.arange(SIM_Y)[:, None] + 0.5) * RG_DT + (np.arange(SIM_X)[None, :]) * 0.05
        simulation = xr.Dataset(
            data_vars={
                "dem_distance": (
                    ("y", "x", "axis"),
                    np.tile(np.array([1.0, 0.0, 0.0]), (SIM_Y, SIM_X, 1)),
                ),
                "azimuth_time": (("y", "x"), sim_az, {}),
                "slant_range_time": (("y", "x"), sim_rg, {}),
            },
            coords={"axis": [0, 1, 2]},
        )
        res = updateLookVector(measurements, simulation, (4, 5))
        assert res.look_vector.sizes == {"axis": 3, "azimuth_time": 4, "slant_range_time": 5}
        assert np.allclose(res.look_vector.values, np.array([1.0, 0.0, 0.0])[:, None, None], atol=1e-8)


class TestOverwriteOrbitCoefficients:
    """Tests of cop_ard_app.slc.overwrite_orbit_coefficients."""

    @pytest.mark.unit
    def test_overwrite_orbit_coefficients_none(self, tmp_path):
        """Without an orbit Dataset no path is written and None is returned."""
        res = overwrite_orbit_coefficients("b_1", "acq_1", None, str(tmp_path), True)
        assert res == ("b_1", "acq_1", None)
        assert not (tmp_path / "bursts").exists()

    @pytest.mark.unit
    def test_overwrite_orbit_coefficients_multiburst(self, tmp_path):
        """The orbit Dataset is written under bursts/{id}/orig/{id}/conditions/orbit and read back."""
        coeffs = np.arange(33, dtype=float).reshape((1, 3, 11))
        orbit = xr.Dataset(
            {"coeffs": xr.DataArray(coeffs, dims=("acquisition_id", "axis", "degree"))},
        ).assign_attrs({"absolute_orbit": ORBIT})
        burst_id, acq, written = overwrite_orbit_coefficients("b_1", "acq_1", orbit, str(tmp_path), True)
        assert (burst_id, acq) == ("b_1", "acq_1")
        assert (tmp_path / "bursts/b_1/orig/acq_1/conditions/orbit").is_dir()
        assert np.allclose(written.coeffs.values, coeffs, atol=TOL)
        assert written.attrs["absolute_orbit"] == ORBIT

    @pytest.mark.unit
    def test_overwrite_orbit_coefficients_non_multiburst(self, tmp_path):
        """Without multiburst the orbit goes to orig/{id}/conditions/orbit."""
        coeffs = np.arange(33, dtype=float).reshape((1, 3, 11))
        orbit = xr.Dataset(
            {"coeffs": xr.DataArray(coeffs, dims=("acquisition_id", "axis", "degree"))},
        )
        _, _, written = overwrite_orbit_coefficients("b_1", "acq_1", orbit, str(tmp_path), False)
        assert (tmp_path / "orig/acq_1/conditions/orbit").is_dir()
        assert not (tmp_path / "bursts").exists()
        assert np.allclose(written.coeffs.values, coeffs, atol=TOL)


class TestImportSlice:
    """Tests of cop_ard_app.slc.import_slice."""

    def _chunk(self):
        return {"azimuth_time": slice(0, NX), "slant_range_time": slice(0, NY)}

    def _slc(self):
        return xr.DataArray(
            np.zeros((1, 1, NX, NY), np.complex64),
            dims=("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
            coords={
                "azimuth_time": np.arange(NX, dtype=float),
                "slant_range_time": np.arange(NY, dtype=float),
                "azimuth_m": ("azimuth_time", np.arange(NX, dtype=float)),
                "slant_range_m": ("slant_range_time", np.arange(NY, dtype=float)),
                "polarization": ("polarization", ["UN"]),
            },
        )

    def _reader(self):
        data = np.full((NY, NX), PIXEL_VALUE, np.complex64)

        class _R:
            def read(self, slice_, *args, **kwargs):
                return data

        return _R()

    @pytest.mark.unit
    def test_import_slice_identity(self):
        """An unprocessed read returns the source pixels unchanged."""
        with patch("cop_ard_app.slc.sarpy_open", lambda path, **kw: self._reader()):
            res = import_slice(self._slc(), "product.xml", 0, "UN", chunk=self._chunk())
        assert np.all(res["slc"].values == PIXEL_VALUE)

    @pytest.mark.unit
    def test_import_slice_polarization_mismatch(self):
        """A polarization mismatch skips the read and returns None."""
        with patch("cop_ard_app.slc.sarpy_open", lambda path, **kw: self._reader()):
            res = import_slice(self._slc(), "product.xml", 0, "P", chunk=self._chunk())
        assert res is None

    @pytest.mark.unit
    def test_import_slice_noise_removal(self):
        """Constant noise polynomial of 10 dB is subtracted from the pixel intensity."""
        with patch("cop_ard_app.slc.sarpy_open", lambda path, **kw: self._reader()):
            res = import_slice(
                self._slc(),
                "product.xml",
                0,
                "UN",
                noise_removal_flag=True,
                noise_poly_coeffs=np.array([[10.0]]),
                chunk=self._chunk(),
            )
        expected = np.sqrt(abs(PIXEL_VALUE) ** 2 - 10.0)
        assert np.allclose(np.abs(res["slc"].values), expected, atol=1e-5)
        assert np.allclose(np.angle(res["slc"].values), np.angle(PIXEL_VALUE), atol=1e-5)

    @pytest.mark.unit
    def test_import_slice_amplitude_calibration(self):
        """A constant calibration polynomial of 4 scales amplitudes by sqrt(4)."""
        with patch("cop_ard_app.slc.sarpy_open", lambda path, **kw: self._reader()):
            res = import_slice(
                self._slc(),
                "product.xml",
                0,
                "UN",
                calibration_flag=True,
                calibration_polynomial_coeffs=np.array([[4.0]]),
                chunk=self._chunk(),
            )
        assert np.allclose(np.abs(res["slc"].values), np.abs(PIXEL_VALUE) * 2.0, atol=1e-4)

    @pytest.mark.unit
    def test_import_slice_deramp(self):
        """Deramping is a pure rotation: amplitudes are preserved."""
        with patch("cop_ard_app.slc.sarpy_open", lambda path, **kw: self._reader()):
            res = import_slice(
                self._slc(),
                "product.xml",
                0,
                "UN",
                deramp_flag=True,
                dkcoa_coeffs=np.array([[0.25]]),
                chunk=self._chunk(),
            )
        assert np.allclose(np.abs(res["slc"].values), np.abs(PIXEL_VALUE), atol=1e-5)

    @pytest.mark.unit
    def test_import_slice_invalid_mask(self):
        """Invalid borders are zeroed inside the chunk window."""
        with patch("cop_ard_app.slc.sarpy_open", lambda path, **kw: self._reader()):
            res = import_slice(
                self._slc(),
                "product.xml",
                0,
                "UN",
                apply_invalid_mask_flag=True,
                valid_data_y_idxs=np.array([1, NX - 1, NX - 1, 1]),
                valid_data_x_idxs=np.array([2, 2, NY, NY]),
                chunk=self._chunk(),
            )
        data = res["slc"].values[0, 0]
        assert np.all(data[:1, :] == 0)
        assert np.all(data[NX - 1 :, :] == 0)
        assert np.all(data[:, :2] == 0)
        assert np.all(data[1 : NX - 1, 2:NY] == PIXEL_VALUE)
        assert np.all(data[1 : NX - 1, 2:] == PIXEL_VALUE)
