import numpy as np
import pytest
import xarray as xr

from cop_ard_app.dem import dem_reproject, import_dem, single_dem, stack_dem
from cop_ard_app.io import write

N = 12
X0, Y0 = 10.0, 45.0
STEP = 0.0025
CRS_IN = "EPSG:4326"
ELEV = 73.0
CACHED = 5.0
BURSTS = ["iw1_254", "iw1_255"]
BBOX = (10.005, 45.005, 10.015, 45.015)
TOL = 1e-6
STEP_TOL = 1e-9
SPREAD = 1e-4


def _flat_dem():
    """A flat-plane DEM over EPSG:4326: every pixel is exactly ELEV, so any
    reprojection of it must return a constant field (constant offset, zero
    tilt): the analytically exact expected height for all fixtures below."""
    x = np.linspace(X0, X0 + (N - 1) * STEP, N)
    y = np.linspace(Y0, Y0 + (N - 1) * STEP, N)
    dem = xr.DataArray(np.full((N, N), ELEV), dims=("y", "x"), coords={"x": x, "y": y})
    dem.rio.write_crs(CRS_IN, inplace=True)
    return dem


def _image_corners(lon0, lat0):
    """Four ImageCorners of a 0.004 deg x 0.004 deg footprint."""
    return [
        {"Lat": lat0, "Lon": lon0},
        {"Lat": lat0, "Lon": lon0 + 0.004},
        {"Lat": lat0 + 0.004, "Lon": lon0 + 0.004},
        {"Lat": lat0 + 0.004, "Lon": lon0},
    ]


def _slc_attrs(lon0, lat0):
    """The list of per-image slc attrs that single_dem consumes (one burst)."""
    return [{"slc": {"GeoData": {"ImageCorners": _image_corners(lon0, lat0)}}}]


def _slcs_tree(lon0_by_burst):
    """A burst-level slcs DataTree as expected by stack_dem (multiburst mode):
    each burst holds two original acquisitions under bursts/{k}/orig/{acq}."""
    tree = xr.DataTree()
    for k, lon0 in lon0_by_burst.items():
        for i in (0, 1):
            ds = xr.Dataset({"dummy": xr.DataArray([1.0], dims=("i",))})
            ds.attrs = {"VV": {"GeoData": {"ImageCorners": _image_corners(lon0, Y0 + 0.004)}}}
            tree[f"bursts/{k}/orig/{k}_{i}/measurements/slc"] = ds
    return tree


def _flat_slcs_tree(lon0):
    """A single-burst slcs DataTree: acquisitions live directly under /orig."""
    tree = xr.DataTree()
    for i in (0, 1):
        ds = xr.Dataset({"dummy": xr.DataArray([1.0], dims=("i",))})
        ds.attrs = {"VV": {"GeoData": {"ImageCorners": _image_corners(lon0, Y0 + 0.004)}}}
        tree[f"orig/{i}/measurements/slc"] = ds
    return tree


def _write_dem_store(path, crs=CRS_IN, lonlat=True):
    """Hand-build a local zarr DEM store as parsed by import_dem (geographic
    CRS, lat/lon coordinates and a single 'z' variable)."""
    x = np.linspace(X0, X0 + (N - 1) * STEP, N)
    y = np.linspace(Y0, Y0 + (N - 1) * STEP, N)
    dims, coords = (["lat", "lon"], {"lat": y, "lon": x}) if lonlat else (["y", "x"], {"y": y, "x": x})
    ds = xr.Dataset({"z": xr.DataArray(np.full((N, N), ELEV), dims=dims, coords=coords)})
    if crs is not None:
        ds.rio.write_crs(crs, inplace=True)
    ds.to_zarr(str(path), mode="w-")


class TestImportDem:
    """Unit test for cop_ard_app.dem.import_dem.

    Covers the local zarr branch (the only one synthesizable offline): CRS
    validation, lat/lon standardization, bounding-box clipping with border,
    and user-requested chunking. The "Copernicus" branch downloads ESA
    GLO-30 over the network and cannot be synthesized in a unit test; it is
    reported as a coverage gap.
    """

    @pytest.mark.unit
    def test_import_dem_local_store_is_standardized_and_clipped(self, tmp_path):
        store = tmp_path / "dem_store"
        _write_dem_store(store, crs=CRS_IN, lonlat=True)

        dem = import_dem(list(BBOX), dem_type=str(store), chunks={"x": 4, "y": 4}, border=0.0)

        assert isinstance(dem, xr.DataArray)
        assert dem.name == "z"
        assert dem.dims == ("y", "x")
        # geographic lat/lon coordinates were renamed to the x/y convention
        assert "x" in dem.coords and "lon" not in dem.coords
        # the clip keeps exactly the 5 grid points inside the (border-free) bbox
        assert dem.sizes["x"] == 5 and dem.sizes["y"] == 5
        assert float(np.min(dem.x)) >= BBOX[0] and float(np.max(dem.x)) <= BBOX[2]
        assert float(np.min(dem.y)) >= BBOX[1] and float(np.max(dem.y)) <= BBOX[3]
        # the user-requested chunking is applied to the 5x5 clip
        assert dem.chunksizes["x"] == (4, 1) and dem.chunksizes["y"] == (4, 1)
        # clipping is lossless for a flat plane: values stay exactly ELEV
        assert np.array_equal(dem.values, np.full((5, 5), ELEV))

    @pytest.mark.unit
    def test_import_dem_rejects_store_without_projection(self, tmp_path):
        store = tmp_path / "no_crs_store"
        _write_dem_store(store, crs=None)

        with pytest.raises(Exception, match="projection not defined"):
            import_dem(list(BBOX), dem_type=str(store))


class TestDemReproject:
    """Unit tests for cop_ard_app.dem.dem_reproject.

    The flat-plane fixture makes the expected output exactly known: a constant
    field ELEV on a grid aligned to the requested spacing.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("outproj", [4326, CRS_IN])
    def test_reproject_flat_dem_preserves_constant_heights(self, tmp_path, outproj):
        # integer and "EPSG:..." string spellings of the same CRS are equivalent
        ds = dem_reproject(
            _flat_dem(),
            {"x": STEP, "y": -STEP},
            outproj,
            str(tmp_path / "out"),
            chunks={"x": 8, "y": 8},
            enable_dask=False,
            method="linear",
        )

        assert isinstance(ds, xr.Dataset)
        z = ds["z"].values
        assert z.ndim == 2 and z.size >= 4
        assert np.all(np.isfinite(z))
        # a constant input field must reproject to the same constant (zero offset)
        assert np.max(np.abs(z - ELEV)) < TOL
        # the output grid honors the requested spacing (x increasing, y decreasing)
        assert np.allclose(np.diff(ds["x"].values), STEP, atol=STEP_TOL)
        assert np.allclose(np.diff(ds["y"].values), -STEP, atol=STEP_TOL)

    @pytest.mark.unit
    def test_reproject_rejects_missing_spatial_coords(self, tmp_path):
        dem = xr.DataArray(np.full((N, N), ELEV), dims=("row", "col"))
        dem.rio.write_crs(CRS_IN, inplace=True)

        with pytest.raises(Exception, match='must contain "x"'):
            dem_reproject(dem, {"x": STEP, "y": -STEP}, 4326, str(tmp_path / "out"), enable_dask=False)


class TestSingleDem:
    """Unit tests for cop_ard_app.dem.single_dem: AOI computation from image
    corners, clipping, reprojection and the cached-simulation shortcut."""

    @pytest.mark.unit
    def test_single_dem_multiburst_reprojects_aoi(self, tmp_path):
        burst_id, ds = single_dem(
            BURSTS[0],
            _slc_attrs(X0 + 0.004, Y0 + 0.004),
            True,
            _flat_dem(),
            {"x": STEP, "y": -STEP},
            str(tmp_path / "static"),
            str(tmp_path / "tmp"),
            chunks={"x": 8, "y": 8},
            crs=CRS_IN,
            enable_dask=False,
            method="linear",
        )

        assert burst_id == BURSTS[0]
        z = ds["z"].values
        assert z.ndim == 2 and z.size >= 4
        assert np.all(np.isfinite(z))
        # flat plane over the same CRS: exactly constant heights
        assert np.max(np.abs(z - ELEV)) < TOL
        # the reprojected grid was written under the tmp working directory
        assert (tmp_path / "tmp" / "bursts" / BURSTS[0] / "dem_resampled").is_dir()

    @pytest.mark.unit
    def test_single_dem_single_burst_uses_reference_id_and_default_crs(self, tmp_path):
        burst_id, ds = single_dem(
            BURSTS[0],
            _slc_attrs(X0 + 0.004, Y0 + 0.004),
            False,
            _flat_dem(),
            {"x": 300.0, "y": -300.0},
            str(tmp_path / "static"),
            str(tmp_path / "tmp"),
            chunks={"x": 8, "y": 8},
            reference_id="ref-1",
            enable_dask=False,
            method="linear",
        )

        assert burst_id == BURSTS[0]
        # without an explicit CRS the default UTM zone is selected: output
        # coordinates are meters (spacing given in meters as well), not degrees
        assert np.min(np.abs(ds["x"].values)) > 1e4
        assert np.min(ds["y"].values) > 1e6
        assert np.allclose(np.diff(ds["x"].values), 300.0, atol=1e-6)
        z = ds["z"].values
        assert np.all(np.isfinite(z))
        # the flat plane stays flat across the CRS change; the vertical datum
        # shift may move the constant, but only by the (small) geoid amplitude
        assert np.max(z) - np.min(z) < SPREAD
        assert np.all(np.abs(z - ELEV) < 100.0)

    @pytest.mark.unit
    def test_single_dem_requires_reference_id_for_single_burst(self, tmp_path):
        with pytest.raises(ValueError, match="reference_id is mandatory"):
            single_dem(
                BURSTS[0],
                _slc_attrs(X0 + 0.004, Y0 + 0.004),
                False,
                _flat_dem(),
                {"x": STEP, "y": -STEP},
                str(tmp_path / "static"),
                str(tmp_path / "tmp"),
            )

    @pytest.mark.unit
    def test_single_dem_reuses_existing_simulation(self, tmp_path):
        static = tmp_path / "static"
        path = f"bursts/{BURSTS[1]}"
        marker = static / "processed_data" / path / "simulation.ok"
        marker.parent.mkdir(parents=True)
        marker.touch()
        cached = xr.Dataset({"h": xr.DataArray(np.full((3, 3), CACHED), dims=("y", "x"))})
        write(cached, str(static / path / "simulation"))

        burst_id, ds = single_dem(
            BURSTS[1],
            _slc_attrs(X0 + 0.004, Y0 + 0.004),
            True,
            _flat_dem(),
            {"x": STEP, "y": -STEP},
            str(static),
            str(tmp_path / "tmp"),
            recompute=False,
        )

        assert burst_id == BURSTS[1]
        # the cached store was read back unchanged and no reprojection ran
        assert np.array_equal(ds["z"].values, np.full((3, 3), CACHED))
        assert not (tmp_path / "tmp" / "bursts" / BURSTS[1]).exists()


class TestStackDem:
    """Unit tests for cop_ard_app.dem.stack_dem: parallel single_dem over a
    slcs DataTree and reassembly into a DEM DataTree."""

    @pytest.mark.unit
    def test_stack_dem_multiburst_returns_one_dem_per_burst(self, tmp_path):
        tree = stack_dem(
            _flat_dem(),
            _slcs_tree({k: X0 + 0.004 + 0.004 * i for i, k in enumerate(BURSTS)}),
            True,
            {"x": STEP, "y": -STEP},
            str(tmp_path / "static"),
            str(tmp_path / "tmp"),
            chunks={"x": 8, "y": 8},
            crs=CRS_IN,
            enable_dask=False,
            method="linear",
        )

        assert isinstance(tree, xr.DataTree)
        assert set(tree.bursts.keys()) == set(BURSTS)
        for k, group in tree.bursts.items():
            z = group["z"].values
            assert z.ndim == 2 and z.size >= 4
            assert np.all(np.isfinite(z))
            assert np.max(np.abs(z - ELEV)) < TOL
            # each burst's reprojected grid went to its own tmp path
            assert (tmp_path / "tmp" / "bursts" / k / "dem_resampled").is_dir()

    @pytest.mark.unit
    def test_stack_dem_single_mode_returns_root_dataset(self, tmp_path):
        tree = stack_dem(
            _flat_dem(),
            _flat_slcs_tree(X0 + 0.004),
            False,
            {"x": STEP, "y": -STEP},
            str(tmp_path / "static"),
            str(tmp_path / "tmp"),
            chunks={"x": 8, "y": 8},
            crs=CRS_IN,
            reference_id="ref-1",
            enable_dask=False,
            method="linear",
        )

        assert isinstance(tree, xr.DataTree)
        z = tree["z"].values
        assert z.ndim == 2 and z.size >= 4
        assert np.all(np.isfinite(z))
        assert np.max(np.abs(z - ELEV)) < TOL
        assert (tmp_path / "tmp" / "data" / "ref-1" / "dem_resampled").is_dir()
