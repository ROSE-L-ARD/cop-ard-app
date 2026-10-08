import numpy as np
import pytest
import xarray as xr

from cop_ard_app.etad import discover_etaddate_slcsdate, import_ETAD, import_etad_stack, save_etad, update_etad
from cop_ard_app.io import Path

N_ACQ = 14
N_AZ = 16
N_RG = 8
SWATH = "IW1"
BURST_ID = "iw1_1"
STACK = "EO-SYNTHETIC.20200104S001000E001S.SAFE"
OTHER_STACK = "EO-SYNTHETIC.20210101S001000E001S.SAFE"
NC_NAME = f"{STACK}.ETAD.nc"
EOF_NAME = f"{STACK}.EOF"
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
ACQ_STEP = np.timedelta64(800, "ms")
HALF_SPAN = np.timedelta64(100, "ms")
ACQ_DATES = [f"2020-01-{1 + i:02d}" for i in range(N_ACQ)]
ACQ_AZ = np.array([T0 + i * ACQ_STEP for i in range(N_ACQ)])
WIDE_START = T0 - np.timedelta64(1, "s")
WIDE_END = T0 + np.timedelta64(4, "s")
IN_WIDE = [d for d, az in zip(ACQ_DATES, ACQ_AZ) if az - HALF_SPAN >= WIDE_START and az + HALF_SPAN <= WIDE_END]
SINGLE_IDX = 3
SINGLE_DATE = ACQ_DATES[SINGLE_IDX]
ETAD_START = ACQ_AZ[SINGLE_IDX] - np.timedelta64(200, "ms")
ETAD_END = ACQ_AZ[SINGLE_IDX] + np.timedelta64(200, "ms")
GRID_START_AZ = -0.5
GRID_START_RG = -1.25
DT_AZ = 0.1
DT_RG = 0.05
CORR_AZ = 0.25
CORR_RG = -0.5
ETAD_VARS = {"azcorr", "rgcorr", "lowrescorr", "tropocorr"}
TOL = 1e-9


def _iso(dt):
    return str(np.datetime64(dt, "ns"))


def _measurements(idx):
    """The measurements record of one acquisition, as the COP SLC layout has it: a
    `measurements` variable with a non-index azimuth_time coordinate along n_az.
    discover indexes it via measurements[0]/[-1]; import_etad_stack reads its azimuth_time."""
    az = np.array([ACQ_AZ[idx] - HALF_SPAN, ACQ_AZ[idx] + HALF_SPAN])
    return xr.Dataset({"measurements": ("n_az", az)}, coords={"azimuth_time": ("n_az", az)})


def _build_slcs(idxs, multiburst=True, burst_id=BURST_ID):
    groups = {}
    for idx in idxs:
        date = ACQ_DATES[idx]
        if multiburst:
            groups[f"/bursts/{burst_id}/orig/{date}"] = _measurements(idx)
        else:
            groups[f"/orig/{date}"] = _measurements(idx)
    return xr.DataTree.from_dict(groups)


def _etad_burst_ds():
    rng = np.random.default_rng(7)
    shape = (N_AZ, N_RG)
    azc = rng.standard_normal(shape) + CORR_AZ
    rgc = rng.standard_normal(shape) + CORR_RG
    dop = rng.standard_normal(shape)
    geo = rng.standard_normal(shape)
    ion = rng.standard_normal(shape)
    ds = xr.Dataset(
        {
            "sumOfCorrectionsAz": (["n_az", "n_rg"], azc),
            "sumOfCorrectionsRg": (["n_az", "n_rg"], rgc),
            "dopplerRangeShiftRg": (["n_az", "n_rg"], dop),
            "geodeticCorrectionRg": (["n_az", "n_rg"], geo),
            "ionosphericCorrectionRg": (["n_az", "n_rg"], ion),
        },
        coords={
            "azimuthExtent": np.arange(N_AZ, dtype=np.float64),
            "rangeExtent": np.arange(N_RG, dtype=np.float64),
        },
        attrs={
            "burstId": 1,
            "gridStartAzimuthTime": GRID_START_AZ,
            "gridStartRangeTime": GRID_START_RG,
            "gridSamplingAzimuth": DT_AZ,
            "gridSamplingRange": DT_RG,
        },
    )
    return ds, azc, rgc, dop, geo, ion


def _build_etad(tstart, tend, z_suffix=True):
    burst, *_ = _etad_burst_ds()
    root = xr.Dataset(
        attrs={
            "azimuthTimeMin": _iso(tstart) + ("Z" if z_suffix else ""),
            "azimuthTimeMax": _iso(tend) + ("Z" if z_suffix else ""),
            "rangeTimeMin": 0.0,
        }
    )
    return xr.DataTree.from_dict({"/": root, f"/{SWATH}": xr.Dataset(), f"/{SWATH}/1": burst})


def _write_etad_stack(base, tree):
    """Lay out an ETAD product directory: <SAFE root>/measurement/*.nc + annotation/*.EOF."""
    stack = base / STACK
    (stack / "measurement").mkdir(parents=True)
    tree.to_netcdf(stack / "measurement" / NC_NAME)
    (stack / "annotation").mkdir(parents=True)
    (stack / "annotation" / EOF_NAME).write_text("stub orbit file\n")


def _expected_grids(min_acq_az, min_acq_rg):
    """Closed-form expected correction-grid coordinates (seconds offsets -> ns)."""
    off_az = np.round((GRID_START_AZ + np.arange(N_AZ, dtype=np.float64) * DT_AZ) * 1e9).astype(np.int64)
    az = min_acq_az + off_az.astype("timedelta64[ns]")
    rg = min_acq_rg + GRID_START_RG + np.arange(N_RG, dtype=np.float64) * DT_RG
    return az, rg


def _attached_ds(tree, path):
    """The Dataset stored at `path` inside a DataTree.

    Walking the tree avoids depending on the detached `tree[path]` view, which does
    not expose datasets assigned by the etad functions themselves."""
    target = f"/{path.lstrip('/')}" if path else "/"
    for node in tree.subtree:
        if node.path == target and node.ds is not None:
            return node.ds
    raise AssertionError(f"no dataset attached at {path!r}")


class TestImportETAD:
    """Unit tests for cop_ard_app.etad.import_ETAD.

    import_ETAD is exercised as a wrapper: date filtering of the ETAD file list, delegation to
    import_etad_stack, and computation of the SAFE root paths. update_orbit is in cop_ard_app.orbit
    and parses ESA orbit files, so its real behaviour is out of scope and stubbed here.
    """

    @pytest.mark.unit
    def test_import_etad_rejects_non_path_entries(self):
        with pytest.raises(TypeError, match="Expected a list of Path objects"):
            import_ETAD(_build_slcs([0]), [ACQ_DATES[0]], ["EO-plain.nc"], "wdir")

    @pytest.mark.unit
    def test_import_etad_skips_unmatched_acquisition_dates(self, tmp_path):
        slcs = _build_slcs(range(N_ACQ))
        stack = Path(str(tmp_path / STACK))
        out, used = import_ETAD(slcs, ["2021-06-01"], [stack], str(tmp_path))
        # nothing matches the requested date: the slcs is returned untouched, no orbit recorded
        assert used == []
        assert "conditions" not in out[f"bursts/{BURST_ID}/orig/{ACQ_DATES[0]}"].groups

    @pytest.mark.unit
    def test_import_etad_filters_and_delegates(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr("cop_ard_app.etad.update_orbit", lambda **kw: calls.append(kw))
        _write_etad_stack(tmp_path, _build_etad(ETAD_START, ETAD_END, z_suffix=False))
        slcs = _build_slcs(range(N_ACQ))
        out, used = import_ETAD(
            slcs,
            [SINGLE_DATE],
            [Path(str(tmp_path / STACK)), Path(str(tmp_path / OTHER_STACK))],
            str(tmp_path),
            multiburst=True,
            enable_dask=False,
        )
        # only the file whose name embeds the acquisition date is kept and reported
        assert used == [str(tmp_path / STACK)]
        assert len(calls) == 1
        assert calls[0]["orbit_paths"][0].path == f"{tmp_path / STACK}/annotation/{EOF_NAME}"
        # the delegated import attached the ETAD to the discovered acquisition
        etad = _attached_ds(out, f"bursts/{BURST_ID}/orig/{SINGLE_DATE}/conditions/ETAD")
        assert set(etad.data_vars) == ETAD_VARS


class TestDiscoverEtaddateSlcsdate:
    """Unit tests for cop_ard_app.etad.discover_etaddate_slcsdate.

    An ETAD whose validity window is [T0-1s, T0+4s] must cover exactly those SLCs whose full
    azimuth span (ACQ_AZ[i] +/- 100ms) falls inside the window: acquisition i has
    i*800ms+100ms <= 4000ms, i.e. i <= 4 (5 of 14 dates)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("multiburst", [True, False])
    def test_discover_selects_contained_acquisitions(self, multiburst):
        slcs = _build_slcs(range(N_ACQ), multiburst=multiburst)
        etad = _build_etad(WIDE_START, WIDE_END)
        assert discover_etaddate_slcsdate(slcs, etad, multiburst) == set(IN_WIDE)

    @pytest.mark.unit
    def test_discover_single_acquisition_window(self):
        slcs = _build_slcs(range(N_ACQ), multiburst=True)
        etad = _build_etad(ETAD_START, ETAD_END)
        # +/-200ms around acquisition 3 admits only that one (neighbours are 800ms away)
        assert discover_etaddate_slcsdate(slcs, etad, True) == {SINGLE_DATE}

    @pytest.mark.unit
    def test_discover_no_overlap(self):
        slcs = _build_slcs(range(N_ACQ), multiburst=True)
        etad = _build_etad(T0 - np.timedelta64(10, "s"), T0 - np.timedelta64(2, "s"))
        # a window entirely before the acquisitions: azimuth spans never satisfy az[0] >= tstart
        assert discover_etaddate_slcsdate(slcs, etad, True) == set()


class TestSaveEtad:
    """Unit tests for cop_ard_app.etad.save_etad.

    The grid start times (offsets in seconds, e.g. -0.5s) must be reconstructed as ns offsets from
    the minimum acquisition times, and all four correction grids must round-trip exactly through the
    Zarr store written by io.write."""

    @pytest.mark.unit
    def test_save_etad_writes_correction_grids(self, tmp_path):
        burst, azc, rgc, dop, geo, ion = _etad_burst_ds()
        out = str(tmp_path / "etad_store")
        save_etad(
            burst,
            ACQ_AZ[0],
            float(len(ACQ_DATES)),
            NC_NAME,
            out,
        )
        ds = xr.open_dataset(out, engine="zarr", chunks={})
        assert set(ds.data_vars) == ETAD_VARS
        assert ds.attrs["etad_file"] == NC_NAME
        exp_az, exp_rg = _expected_grids(ACQ_AZ[0], float(len(ACQ_DATES)))
        np.testing.assert_array_equal(ds["azimuth_time"].values, exp_az)
        np.testing.assert_allclose(ds["slant_range_time"].values, exp_rg, rtol=0, atol=TOL)
        np.testing.assert_allclose(ds["azcorr"].values, azc, rtol=0, atol=TOL)
        np.testing.assert_allclose(ds["rgcorr"].values, rgc, rtol=0, atol=TOL)
        np.testing.assert_allclose(ds["lowrescorr"].values, dop + geo, rtol=0, atol=TOL)
        np.testing.assert_allclose(ds["tropocorr"].values, ion, rtol=0, atol=TOL)


class TestUpdateEtad:
    """Unit tests for cop_ard_app.etad.update_etad.

    Checks the derived burst id and store path, the Zarr store written on disk, and that the value
    read back carries the correction grids and the source .nc path."""

    @pytest.mark.unit
    def test_update_etad_saves_and_returns_burst(self, tmp_path):
        tree = _build_etad(WIDE_START, WIDE_END, z_suffix=False)
        burst = tree[f"{SWATH}/1"]
        nc_path = str(tmp_path / STACK / "measurement" / NC_NAME)
        min_rg = float(len(ACQ_DATES))
        acq_date, bid, et = update_etad(burst, SWATH, ACQ_DATES[0], nc_path, ACQ_AZ[0], min_rg, str(tmp_path))
        assert (acq_date, bid) == (ACQ_DATES[0], BURST_ID)
        store = tmp_path / BURST_ID / "orig" / ACQ_DATES[0] / "conditions" / "ETAD"
        assert (store / "zarr.json").exists()
        node = _attached_ds(et, "")
        assert set(node.data_vars) == ETAD_VARS
        assert node.attrs["etad_file"] == nc_path
        assert node.attrs["store_path"] == str(store)
        _, azc, rgc, dop, geo, ion = _etad_burst_ds()
        exp_az, exp_rg = _expected_grids(ACQ_AZ[0], min_rg)
        np.testing.assert_array_equal(node["azimuth_time"].values, exp_az)
        np.testing.assert_allclose(node["slant_range_time"].values, exp_rg, rtol=0, atol=TOL)
        np.testing.assert_allclose(node["azcorr"].values, azc, rtol=0, atol=TOL)
        np.testing.assert_allclose(node["rgcorr"].values, rgc, rtol=0, atol=TOL)
        np.testing.assert_allclose(node["lowrescorr"].values, dop + geo, rtol=0, atol=TOL)
        np.testing.assert_allclose(node["tropocorr"].values, ion, rtol=0, atol=TOL)


class TestImportEtadStack:
    """Unit tests for cop_ard_app.etad.import_etad_stack.

    Full pipeline over a synthetic ETAD product directory (a NetCDF DataTree with a global validity
    window and one IW1/1 burst, plus a dummy .EOF): the SLCs discovered to fall inside the window
    receive the correction grids under conditions/ETAD, with coordinates rebuilt from the burst
    grid attributes, and the matching orbit file is reported."""

    @pytest.mark.unit
    def test_import_etad_stack_injects_discovered_date(self, tmp_path):
        _write_etad_stack(tmp_path, _build_etad(ETAD_START, ETAD_END, z_suffix=False))
        slcs = _build_slcs(range(N_ACQ))
        out, orbit_paths = import_etad_stack(
            etad_file_list={Path(str(tmp_path / STACK))},
            slcs=slcs,
            multiburst=True,
            wdir_path=str(tmp_path),
            enable_dask=False,
        )
        assert len(orbit_paths) == 1
        assert orbit_paths[0].name == EOF_NAME
        etad = _attached_ds(out, f"bursts/{BURST_ID}/orig/{SINGLE_DATE}/conditions/ETAD")
        assert set(etad.data_vars) == ETAD_VARS
        assert etad.attrs["etad_file"] == f"{tmp_path / STACK}/measurement/{NC_NAME}"
        _, azc, rgc, dop, geo, ion = _etad_burst_ds()
        t0az = np.datetime64(_iso(ETAD_START), "ns")
        exp_az, exp_rg = _expected_grids(t0az, 0.0)
        np.testing.assert_array_equal(etad["azimuth_time"].values, exp_az)
        np.testing.assert_allclose(etad["slant_range_time"].values, exp_rg, rtol=0, atol=TOL)
        np.testing.assert_allclose(etad["azcorr"].values, azc, rtol=0, atol=TOL)
        np.testing.assert_allclose(etad["rgcorr"].values, rgc, rtol=0, atol=TOL)
        np.testing.assert_allclose(etad["lowrescorr"].values, dop + geo, rtol=0, atol=TOL)
        np.testing.assert_allclose(etad["tropocorr"].values, ion, rtol=0, atol=TOL)

    @pytest.mark.unit
    def test_import_etad_stack_without_nc_file(self, tmp_path):
        stack = Path(str(tmp_path / STACK))
        (tmp_path / STACK / "measurement").mkdir(parents=True)
        (tmp_path / STACK / "annotation").mkdir()
        with pytest.raises(ValueError, match="No .nc file found"):
            import_etad_stack(
                etad_file_list={stack},
                slcs=_build_slcs(range(N_ACQ)),
                multiburst=True,
                wdir_path=str(tmp_path),
                enable_dask=False,
            )
