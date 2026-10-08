import numpy as np
import pytest
import xarray as xr
from pyproj import CRS
from pyproj.transformer import Transformer
from scipy.constants import speed_of_light

from cop_ard_app.io import write
from cop_ard_app.simulation import (
    dem_to_radar,
    init_simulation_dataset,
    initialize_simulation,
    single_simulation,
    single_simulation_ml,
    stack_simulation,
    stack_simulation_ml,
    update_elevation,
)

N = 8
X0, Y0 = 10.0, 45.0
STEP = 0.0025
CRS_IN = "EPSG:4326"
ELEV = 73.0
CACHED = 5.0
BURSTS = ["iw1_254", "iw1_255"]
ACQUISITION = "2020-01-01"
COLLECT_START = "2020-01-01T00:00:00.000000Z"
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
POL = "VV"
CHUNK = 4
SLC_NAZ, SLC_NPRG = 5, 6
SLC_CHUNK = 3
RG0 = 1.0

# the synthetic orbit track is exactly linear: position(t) = OFFSET + SLOPE * t_seconds
# with t scaled by 1e-9 (the scaleTime compute_simulation attaches to the orbit), so the
# range-Doppler condition slope.(dem - position(t)) = 0 has the closed-form solution
# t = slope.(dem - offset) / |slope|^2 and the solver reaches it in a single Newton step.
OFFSET = np.array([3_000_000.0, 2_000_000.0, 5_200_000.0])
SLOPE = np.array([100.0, 200.0, -50.0])
DEGREE = 3
SLOPE_NORM2 = float(SLOPE @ SLOPE)

TOL = 1e-6
TOL_M = 1e-3
TOL_S = 1e-6
TOL_NS = 1e4

# variables initialize_simulation writes into its global_values dataset
GLOBAL_VARS = {
    "radar_grid_pixel_area",
    "chunk_length",
    "chunks_slc",
    "radar_xsize",
    "radar_ysize",
    "radar_xmin",
    "radar_ymin",
    "radar_xmax",
    "geo_xsize",
    "geo_ysize",
    "bbox",
    "x_coords",
    "y_coords",
    "rg_coords",
    "az_coords",
}
SIMUL_VARS = {"azimuth_time", "dem_distance", "h", "slant_range_time"}


def _flat_dem(lon0=X0, lat0=Y0):
    """A flat-plane DEM over EPSG:4326: every pixel is exactly ELEV, so the
    simulation must carry that exact constant as its elevation layer."""
    x = np.linspace(lon0, lon0 + (N - 1) * STEP, N)
    y = np.linspace(lat0, lat0 + (N - 1) * STEP, N)
    dem = xr.DataArray(np.full((N, N), ELEV), dims=("y", "x"), coords={"x": x, "y": y})
    dem.rio.write_crs(CRS_IN, inplace=True)
    return dem


def _orbits():
    """Orbit polynomials as produced by cop_ard_app.model.extract_orbits:
    coefficients (constant term first) per acquisition, with the collection
    start time attached as the reference_times attribute."""
    coeffs = np.zeros((1, DEGREE, 3))
    coeffs[0, 0] = OFFSET
    coeffs[0, 1] = SLOPE
    return xr.DataArray(
        coeffs,
        dims=["acquisition_id", "degree", "axis"],
        coords={
            "acquisition_id": [ACQUISITION],
            "degree": np.arange(DEGREE, dtype=int),
            "axis": np.arange(3, dtype=int),
        },
    ).assign_attrs({"reference_times": {ACQUISITION: COLLECT_START}})


def _expected(dem):
    """The exact simulation output for _flat_dem: (az_ns, slant_range_time, dem_distance).

    The DEM pixels are converted to ECEF with the same pyproj transformation the
    core applies (EPSG:4326.to_3d -> EPSG:4978); for the linear orbit the range-Doppler
    solution, the satellite position at that time, the distance vector and the two-way
    light travel time all follow in closed form.
    """
    x, y = dem.x.values, dem.y.values
    transformer = Transformer.from_crs(CRS(dem.rio.crs).to_3d(), CRS(4978), always_xy=True)
    xg, yg = np.meshgrid(x, y)
    X, Y, Z = transformer.transform(xg, yg, np.full_like(xg, ELEV))
    ground = np.stack([X, Y, Z])
    delta = ground - OFFSET[:, None, None]
    t = np.einsum("k,kij", SLOPE, delta) / SLOPE_NORM2
    sat = OFFSET[:, None, None] + SLOPE[:, None, None] * t[None, :, :]
    dem_distance = ground - sat
    slant_range_time = 2.0 / speed_of_light * np.sqrt((dem_distance**2).sum(axis=0))
    az_ns = np.datetime64(T0, "ns").astype(np.int64) + t * 1e9
    return az_ns, slant_range_time, dem_distance


def _block_extrema(values, chunk, kind):
    """Min/max of a full-resolution field over CHUNK x CHUNK coarsen cells,
    which is exactly what extract_simulation_ml reduces to."""
    blocks = values.reshape(N // chunk, chunk, N // chunk, chunk)
    return blocks.min(axis=(1, 3)) if kind == "min" else blocks.max(axis=(1, 3))


def _annotations():
    """ISARIC annotations for the synthetic radar grid of initialize_simulation:
    Col.SS=1.0 with TimeCAPoly [0, 1] gives one azimuth second per pixel, while
    Row.SS=c/2 gives one slant-range second per pixel (dt_rg = 2*SS/c)."""
    return {
        "Grid": {
            "Col": {"SS": 1.0, "DeltaKCOAPoly": {"Coefs": [[0.0]]}},
            "Row": {"SS": speed_of_light / 2},
        },
        "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0]}}},
    }


def _synthetic_simulation():
    """A simulation dataset with an exactly known radar mapping: azimuth_time
    advances one second per x pixel, slant_range_time one second per y pixel, so
    radar pixel coordinates of every selected point are the grid indices."""
    x = np.linspace(X0, X0 + (N - 1) * STEP, N)
    y = np.linspace(Y0, Y0 + (N - 1) * STEP, N)
    ix, iy = np.meshgrid(np.arange(N), np.arange(N))
    sim = xr.Dataset(
        {
            "azimuth_time": xr.DataArray(T0 + ix[None] * AZ_DT, dims=("acquisition_id", "y", "x")),
            "slant_range_time": xr.DataArray(RG0 + iy[None] * 1.0, dims=("acquisition_id", "y", "x")),
            "dem_distance": xr.DataArray(np.zeros((1, N, N, 3)), dims=("acquisition_id", "y", "x", "axis")),
            "h": xr.DataArray(np.full((N, N), ELEV), dims=("y", "x")),
        },
        coords={
            "acquisition_id": [ACQUISITION],
            "y": y,
            "x": x,
            "axis": np.arange(3, dtype=int),
        },
    )
    sim.rio.write_crs(CRS_IN, inplace=True)
    return x, y, sim


def _synthetic_ml(x, y):
    """A single multilook cell covering the whole synthetic grid: its min/max
    radar times are the corner values of _synthetic_simulation."""
    return {
        "min": xr.Dataset(
            {
                "azimuth_time": xr.DataArray([[T0]], dims=("y", "x")),
                "slant_range_time": xr.DataArray([[RG0]], dims=("y", "x")),
            },
            coords={"y": [y[0]], "x": [x[0]]},
        ),
        "max": xr.Dataset(
            {
                "azimuth_time": xr.DataArray([[T0 + (N - 1) * AZ_DT]], dims=("y", "x")),
                "slant_range_time": xr.DataArray([[RG0 + N - 1]], dims=("y", "x")),
            },
            coords={"y": [y[-1]], "x": [x[-1]]},
        ),
    }


def _synthetic_slc(az_offset=0):
    """A chunked SLC DataArray: get_margin turns its annotations into a 10 pixel
    radar margin, and its footprint selects the simulation grid when az_offset=0."""
    slc = xr.DataArray(
        np.ones((1, SLC_NAZ, SLC_NPRG), dtype=np.complex64),
        dims=("polarization", "azimuth_time", "slant_range_time"),
        coords={
            "polarization": [POL],
            "azimuth_time": T0 + az_offset * AZ_DT + np.arange(SLC_NAZ) * AZ_DT,
            "slant_range_time": RG0 + np.arange(SLC_NPRG, dtype=np.float64),
        },
        name="slc",
    )
    slc.attrs = {POL: _annotations()}
    return slc.chunk({"azimuth_time": SLC_CHUNK, "slant_range_time": SLC_CHUNK})


class TestInitSimulationDataset:
    """Unit test for cop_ard_app.simulation.init_simulation_dataset: the empty
    Zarr scaffold for the simulation, with its variables, dimensions, chunking,
    NaN-marked slant range and preserved CRS."""

    @pytest.mark.unit
    def test_init_simulation_dataset_builds_the_nan_scaffold(self, tmp_path):
        x = np.linspace(X0, X0 + (N - 1) * STEP, N)
        y = np.linspace(Y0, Y0 + (N - 1) * STEP, N)

        ds = init_simulation_dataset(
            np.array([ACQUISITION]),
            CRS(CRS_IN),
            x,
            y,
            {"x": CHUNK, "y": CHUNK},
            str(tmp_path / "simulation"),
        )

        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars) == SIMUL_VARS
        assert dict(ds.sizes) == {"acquisition_id": 1, "y": N, "x": N, "axis": 3}
        assert np.array_equal(ds.x.values, x) and np.array_equal(ds.y.values, y)
        # the scaffold was written on disk with the requested chunking
        assert (tmp_path / "simulation").is_dir()
        assert ds.chunksizes["x"] == (CHUNK, CHUNK)
        # slant range marks every pixel as unprocessed
        assert np.isnan(ds.slant_range_time.values).all()
        # the output grid keeps the requested projection
        assert ds.rio.crs.to_epsg() == 4326


class TestUpdateElevation:
    """Unit test for cop_ard_app.simulation.update_elevation: pasting a flat
    DEM onto the simulation scaffold writes the exact constant elevation."""

    @pytest.mark.unit
    def test_update_elevation_pastes_the_flat_dem(self, tmp_path):
        x = np.linspace(X0, X0 + (N - 1) * STEP, N)
        y = np.linspace(Y0, Y0 + (N - 1) * STEP, N)
        data_path = str(tmp_path / "simulation")
        scaffold = init_simulation_dataset(
            np.array([ACQUISITION]),
            CRS(CRS_IN),
            x,
            y,
            {"x": CHUNK, "y": CHUNK},
            data_path,
        )

        ds = update_elevation(
            scaffold,
            _flat_dem().to_dataset(name="z"),
            {"x": CHUNK, "y": CHUNK},
            data_path,
        )

        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars) >= SIMUL_VARS
        # a flat plane is pasted without any numeric drift (float32 exact)
        assert np.array_equal(ds.h.values, np.full((N, N), ELEV, dtype=np.float32))
        # geometry is not computed by this step: slant range stays NaN
        assert np.isnan(ds.slant_range_time.values).all()
        assert ds.rio.crs.to_epsg() == 4326


class TestDemToRadar:
    """Unit test for cop_ard_app.simulation.dem_to_radar over the full analytic
    fixture: the flat DEM and the linear orbit make elevation, azimuth time,
    slant range time and distance vectors exactly predictable."""

    @pytest.mark.unit
    def test_dem_to_radar_reproduces_the_closed_form_simulation(self, tmp_path):
        dem = _flat_dem()

        ds = dem_to_radar(
            dem=dem,
            orbits=_orbits(),
            working_dir=str(tmp_path),
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            skip_read=False,
            enable_dask=False,
        )

        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars) == SIMUL_VARS
        assert dict(ds.sizes) == {"acquisition_id": 1, "y": N, "x": N, "axis": 3}
        assert (tmp_path / "simulation").is_dir()
        # the flat plane is carried through as exact elevation
        assert np.array_equal(ds.h.values, np.full((N, N), ELEV, dtype=np.float32))
        az_ns, slant, dem_distance = _expected(dem)
        # known orbit/height attributes -> exact values within tolerances
        assert np.abs(ds.azimuth_time.values[0].astype(np.int64) - az_ns).max() <= TOL_NS
        assert np.abs(ds.slant_range_time.values[0] - slant).max() <= TOL_S
        assert np.abs(ds.dem_distance.values[0] - dem_distance.transpose(1, 2, 0)).max() <= TOL_M


class TestSingleSimulation:
    """Unit tests for cop_ard_app.simulation.single_simulation: path management
    for multiburst and single-burst modes, the mandatory reference id, and the
    master-image cache marker."""

    @pytest.mark.unit
    def test_single_simulation_multiburst_writes_the_burst_store(self, tmp_path):
        dem = _flat_dem()
        working = tmp_path / "wd"

        burst_id, ds = single_simulation(
            BURSTS[0],
            dem,
            True,
            _orbits(),
            str(working),
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            enable_dask=False,
        )

        assert burst_id == BURSTS[0]
        assert (working / "bursts" / BURSTS[0] / "simulation").is_dir()
        assert set(ds.data_vars) == SIMUL_VARS
        assert np.array_equal(ds.h.values, np.full((N, N), ELEV, dtype=np.float32))
        az_ns, slant, _ = _expected(dem)
        assert np.abs(ds.azimuth_time.values[0].astype(np.int64) - az_ns).max() <= TOL_NS
        assert np.abs(ds.slant_range_time.values[0] - slant).max() <= TOL_S

    @pytest.mark.unit
    def test_single_simulation_single_burst_uses_the_reference_id(self, tmp_path):
        dem = _flat_dem()
        working = tmp_path / "wd"

        burst_id, ds = single_simulation(
            BURSTS[0],
            dem,
            False,
            _orbits(),
            str(working),
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            reference_id=ACQUISITION,
            enable_dask=False,
        )

        assert burst_id == BURSTS[0]
        # single-burst mode writes under data/{reference_id} instead of bursts/...
        assert (working / "data" / ACQUISITION / "simulation").is_dir()
        assert np.array_equal(ds.h.values, np.full((N, N), ELEV, dtype=np.float32))
        az_ns, slant, _ = _expected(dem)
        assert np.abs(ds.azimuth_time.values[0].astype(np.int64) - az_ns).max() <= TOL_NS
        assert np.abs(ds.slant_range_time.values[0] - slant).max() <= TOL_S

    @pytest.mark.unit
    def test_single_simulation_requires_reference_id_for_single_burst(self, tmp_path):
        with pytest.raises(ValueError, match="reference_id is mandatory"):
            single_simulation(
                BURSTS[0],
                _flat_dem(),
                False,
                _orbits(),
                str(tmp_path / "wd"),
            )

    @pytest.mark.unit
    def test_single_simulation_skips_a_finished_master_image(self, tmp_path):
        dem = _flat_dem()
        working = tmp_path / "wd"

        _, ds = single_simulation(
            BURSTS[1],
            dem,
            True,
            _orbits(),
            str(working),
            master_image=True,
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            enable_dask=False,
        )
        marker = working / "processed_data" / "bursts" / BURSTS[1] / "simulation.ok"
        assert marker.is_file()
        assert np.array_equal(ds.h.values, np.full((N, N), ELEV, dtype=np.float32))

        # seed the store with a foreign elevation...
        seeded = xr.Dataset({"h": xr.DataArray(np.full((N, N), CACHED), dims=("y", "x"))})
        write(seeded, str(working / "bursts" / BURSTS[1] / "simulation"), region={"y": slice(None), "x": slice(None)})
        # ...so keeping it proves the second call skipped the computation
        _, ds = single_simulation(
            BURSTS[1],
            dem,
            True,
            _orbits(),
            str(working),
            master_image=True,
            recompute=False,
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            enable_dask=False,
        )
        assert np.array_equal(ds.h.values, np.full((N, N), CACHED, dtype=np.float32))

        # forcing recompute runs the simulation again and restores ELEV
        _, ds = single_simulation(
            BURSTS[1],
            dem,
            True,
            _orbits(),
            str(working),
            master_image=True,
            recompute=True,
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            enable_dask=False,
        )
        assert marker.is_file()
        assert np.array_equal(ds.h.values, np.full((N, N), ELEV, dtype=np.float32))


class TestStackSimulation:
    """Unit tests for cop_ard_app.simulation.stack_simulation: parallel
    single_simulation over a DEM DataTree, one simulation per burst."""

    def _dem_tree(self, lon_by_burst):
        tree = xr.DataTree()
        for k, lon0 in lon_by_burst.items():
            tree[f"bursts/{k}"] = xr.Dataset({"z": _flat_dem(lon0, Y0 + 0.004)})
        return tree

    @pytest.mark.unit
    def test_stack_simulation_multiburst_returns_one_simulation_per_burst(self, tmp_path):
        lon_by_burst = {k: X0 + 0.004 * i for i, k in enumerate(BURSTS)}

        tree = stack_simulation(
            self._dem_tree(lon_by_burst),
            _orbits(),
            True,
            str(tmp_path),
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            enable_dask=False,
        )

        assert isinstance(tree, xr.DataTree)
        assert set(tree.groups) == {"/", "/bursts", *(f"/bursts/{k}" for k in BURSTS)}
        for k, lon0 in lon_by_burst.items():
            assert (tmp_path / "bursts" / k / "simulation").is_dir()
            ds = tree[f"bursts/{k}"]
            assert set(ds.data_vars) == SIMUL_VARS
            assert np.array_equal(ds.h.values, np.full((N, N), ELEV, dtype=np.float32))
            az_ns, slant, _ = _expected(_flat_dem(lon0, Y0 + 0.004))
            assert np.abs(ds.azimuth_time.values[0].astype(np.int64) - az_ns).max() <= TOL_NS
            assert np.abs(ds.slant_range_time.values[0] - slant).max() <= TOL_S

    @pytest.mark.unit
    def test_stack_simulation_master_image_marks_every_burst_done(self, tmp_path):
        lon_by_burst = {k: X0 + 0.004 * i for i, k in enumerate(BURSTS)}

        tree = stack_simulation(
            self._dem_tree(lon_by_burst),
            _orbits(),
            True,
            str(tmp_path),
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            master_image=True,
            enable_dask=False,
        )

        for k in BURSTS:
            marker = tmp_path / "processed_data" / "bursts" / k / "simulation.ok"
            assert marker.is_file()
            assert (tmp_path / "bursts" / k / "simulation").is_dir()
            assert set(tree[f"bursts/{k}"].data_vars) == SIMUL_VARS


class TestSingleSimulationMl:
    """Unit test for cop_ard_app.simulation.single_simulation_ml: coarsened
    min/max overview of a simulation whose full-resolution values are exact."""

    @pytest.mark.unit
    def test_single_simulation_ml_coarsens_to_exact_block_extremes(self, tmp_path):
        dem = _flat_dem()
        working = tmp_path / "wd"
        _, simulation = single_simulation(
            BURSTS[0],
            dem,
            True,
            _orbits(),
            str(working),
            master_image=True,
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            enable_dask=False,
        )

        burst_id, ml = single_simulation_ml(BURSTS[0], simulation, True, ACQUISITION, str(working))

        assert burst_id == BURSTS[0]
        assert set(ml.groups) == {"/", "/min", "/max"}
        assert (working / "bursts" / BURSTS[0] / "simulation_ml_min").is_dir()
        assert (working / "processed_data" / "bursts" / BURSTS[0] / "simulation_ml.ok").is_file()
        az_ns, slant, _ = _expected(dem)
        cells = (N // CHUNK, N // CHUNK)
        for kind in ("min", "max"):
            values = ml[kind]
            assert dict(values.sizes) == {"y": cells[0], "x": cells[1]}
            # float(ns) rounding of the epoch-sized datetimes bounds the drift
            assert (
                np.abs(values.azimuth_time.values.astype(np.int64) - _block_extrema(az_ns, CHUNK, kind)).max() <= TOL_NS
            )
            assert np.abs(values.slant_range_time.values - _block_extrema(slant, CHUNK, kind)).max() <= TOL_S

    @pytest.mark.unit
    def test_single_simulation_ml_returns_cached_overview(self, tmp_path):
        dem = _flat_dem()
        working = tmp_path / "wd"
        _, simulation = single_simulation(
            BURSTS[0],
            dem,
            True,
            _orbits(),
            str(working),
            master_image=True,
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            enable_dask=False,
        )
        _, ml = single_simulation_ml(BURSTS[0], simulation, True, ACQUISITION, str(working))
        # a second call finds the .ok marker and hands back the stored overview
        _, ml2 = single_simulation_ml(BURSTS[0], simulation, True, ACQUISITION, str(working))
        assert set(ml2.groups) == {"/", "/min", "/max"}
        assert np.array_equal(ml2["min"].slant_range_time.values, ml["min"].slant_range_time.values)
        assert np.array_equal(ml2["max"].slant_range_time.values, ml["max"].slant_range_time.values)


class TestStackSimulationMl:
    """Unit test for cop_ard_app.simulation.stack_simulation_ml: multilooking
    every burst of a simulation DataTree produced by stack_simulation."""

    def _dem_tree(self, lon_by_burst):
        tree = xr.DataTree()
        for k, lon0 in lon_by_burst.items():
            tree[f"bursts/{k}"] = xr.Dataset({"z": _flat_dem(lon0, Y0 + 0.004)})
        return tree

    @pytest.mark.unit
    def test_stack_simulation_ml_coarsens_every_burst(self, tmp_path):
        lon_by_burst = {k: X0 + 0.004 * i for i, k in enumerate(BURSTS)}
        simulation_tree = stack_simulation(
            self._dem_tree(lon_by_burst),
            _orbits(),
            True,
            str(tmp_path),
            chunks={"x": CHUNK, "y": CHUNK, "acquisition_id": 1},
            master_image=True,
            enable_dask=False,
        )

        ml_tree = stack_simulation_ml(simulation_tree, ACQUISITION, True, str(tmp_path))

        assert isinstance(ml_tree, xr.DataTree)
        expected_groups = {"/", "/bursts"}
        for k in BURSTS:
            expected_groups |= {f"/bursts/{k}", f"/bursts/{k}/min", f"/bursts/{k}/max"}
        assert set(ml_tree.groups) == expected_groups
        for k, lon0 in lon_by_burst.items():
            assert set(ml_tree[f"bursts/{k}"].groups) == {f"/bursts/{k}", f"/bursts/{k}/min", f"/bursts/{k}/max"}
            assert (tmp_path / "processed_data" / "bursts" / k / "simulation_ml.ok").is_file()
            az_ns, slant, _ = _expected(_flat_dem(lon0, Y0 + 0.004))
            for kind in ("min", "max"):
                values = ml_tree[f"bursts/{k}"][kind]
                assert dict(values.sizes) == {"y": N // CHUNK, "x": N // CHUNK}
                assert (
                    np.abs(values.azimuth_time.values.astype(np.int64) - _block_extrema(az_ns, CHUNK, kind)).max()
                    <= TOL_NS
                )
                assert np.abs(values.slant_range_time.values - _block_extrema(slant, CHUNK, kind)).max() <= TOL_S


class TestInitializeSimulation:
    """Unit tests for cop_ard_app.simulation.initialize_simulation over a
    synthetic radar grid whose mapping to the geographic simulation grid is the
    exact affine identity in time (one azimuth second per x pixel, one slant
    range second per y pixel), so every pixel coordinate is known a priori."""

    @pytest.mark.unit
    def test_initialize_simulation_selects_the_full_grid(self, tmp_path):
        x, y, sim = _synthetic_simulation()
        slc = _synthetic_slc()

        selection, global_values = initialize_simulation(slc, sim, _synthetic_ml(x, y), CHUNK)

        assert set(selection.data_vars) == {"dem_distance", "h", "az_px", "rg_px"}
        assert dict(selection.sizes) == {"y": N, "x": N, "axis": 3}
        assert np.array_equal(selection.h.values, np.full((N, N), ELEV))
        # radar pixel coordinates are exactly the grid indices
        assert np.array_equal(selection.az_px.values, np.broadcast_to(np.arange(N, dtype=float), (N, N)))
        assert np.array_equal(selection.rg_px.values, np.broadcast_to(np.arange(N, dtype=float)[:, None], (N, N)))
        assert set(global_values.data_vars) == GLOBAL_VARS
        # the footprint spans the whole grid: sizes and origins are exact
        assert int(global_values.geo_xsize.values[0]) == N - 1
        assert int(global_values.geo_ysize.values[0]) == N - 1
        assert int(global_values.radar_xsize.values[0]) == N - 1
        assert int(global_values.radar_ysize.values[0]) == N - 1
        assert int(global_values.radar_xmin.values[0]) == 0
        assert int(global_values.radar_ymin.values[0]) == 0
        assert int(global_values.radar_xmax.values[0]) == N - 1
        assert np.array_equal(global_values.bbox.values, [0, 0, SLC_NPRG, SLC_NAZ])
        assert np.array_equal(global_values.chunks_slc.values, [SLC_CHUNK, SLC_CHUNK])
        assert int(global_values.chunk_length.values[0]) == CHUNK
        assert np.array_equal(global_values.x_coords.values, x)
        assert np.array_equal(global_values.y_coords.values, y)
        assert np.array_equal(global_values.az_coords.values, slc.azimuth_time.values)
        assert np.array_equal(global_values.rg_coords.values, slc.slant_range_time.values)
        # pixel area follows the annotations: Col.SS * Row.SS
        assert abs(global_values.radar_grid_pixel_area.values[0] - speed_of_light / 2) < TOL

    @pytest.mark.unit
    def test_initialize_simulation_rejects_a_footprint_outside_the_simulation(self, tmp_path):
        x, y, sim = _synthetic_simulation()
        # the SLC lies 100 azimuth pixels beyond the covered grid, more than the
        # 10-pixel margin, so the multilooked overview finds no overlap
        slc = _synthetic_slc(az_offset=100)

        with pytest.raises(ValueError, match="Could not determine geographic bounding box"):
            initialize_simulation(slc, sim, _synthetic_ml(x, y), CHUNK)
