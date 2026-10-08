import numpy as np
import pytest
import xarray as xr
from cop_ard_core.warp import sinc

from cop_ard_app.gmlp import blockwise_interp_cmlp, map_geocoding_cmlp, single_gmlp, stack_gmlp, update_conditions_gmlp
from cop_ard_app.io import write

N = 16
POL = "VV"
BURSTS = ["iw1_254", "iw1_255"]
DATE_STR = "2020-01-01"
ACQ_ID = "s1"
REF_ID = "ref-1"
PAST_DATES = ["2019-12-31", "2020-01-01"]
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
RG_DT = 1.0
CRS = "EPSG:4326"
KERNEL_SHAPE = [3, 3]
OV_FACT = 100_000
KAISER_BETA = 6
MARGIN = np.array(KERNEL_SHAPE) // 2 + 1  # sinc kernel margin in pixels (2 for a 3x3 kernel)
CHUNKS = {"acquisition_date": 1, "past_reference_date": 1, "polarization": 1, "y": 2048, "x": 2048}
SPACING = {"x": 1.0, "y": 1.0}
TOL = 1e-6
ETAD_TOL = 1e-5

# The exact expected cstack-like content: an azimuth phase ramp (kept within
# (-pi, pi] so that the phasor round-trip of warp_cmlp is lossless) and a
# constant coherence field. Under the identity geocoding configuration the
# sinc resampling sits exactly on integer source pixels, so the warp output
# must reproduce this content inside the kernel margin and stay zero outside.
PHASE_RAMP = 0.1 * np.arange(N)[:, None] * np.ones((1, N))
COH_FIELD = np.full((N, N), 0.5)
# Full-resolution phase model ramp: affine in the azimuth/range indices, with
# values dyadic so that linear interpolation at exact knots is bitwise exact.
MODEL_RAMP = np.arange(N)[:, None] + 0.25 * np.arange(N)[None, :]


def _annotations():
    return {
        "Grid": {
            "Col": {"SS": 1.0, "DeltaKCOAPoly": {"Coefs": [[0.0]]}},
            "Row": {"SS": 1.0},
        },
        "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0]}}},
    }


def _kernel(shape):
    return {
        "kernel_shape": np.array(shape),
        "ov_fact": OV_FACT,
        "kaiser_beta": KAISER_BETA,
        "sinc_x": sinc(shape[1], OV_FACT, KAISER_BETA),
        "sinc_y": sinc(shape[0], OV_FACT, KAISER_BETA),
    }


def _ground(size):
    return np.arange(size, dtype=np.float64)


def _simul_ds(size, acq=False):
    """Synthetic simulation: ground points map through the identity onto the
    radar grid (azimuth_time ramps along y, slant_range_time along x)."""
    iy = np.arange(size)
    az2 = np.broadcast_to((T0 + iy * AZ_DT)[:, None], (size, size))
    rg2 = np.broadcast_to(iy[None, :].astype(np.float64) * RG_DT, (size, size))
    dem3 = np.broadcast_to(np.array([0.0, 0.0, 100.0]), (size, size, 3))
    dims = ("acquisition_id", "y", "x") if acq else ("y", "x")
    if acq:
        az2 = az2[np.newaxis]
        rg2 = rg2[np.newaxis]
    coords = {
        "y": _ground(size),
        "x": _ground(size),
        "axis": [0, 1, 2],
    }
    if acq:
        coords["acquisition_id"] = [ACQ_ID]
        coords["acquisition_date"] = ("acquisition_id", [DATE_STR])
    ds = xr.Dataset(
        {
            "azimuth_time": (dims, az2),
            "slant_range_time": (dims, rg2),
            "dem_distance": (("y", "x", "axis"), dem3.copy()),
            "h": (("y", "x"), np.zeros((size, size), dtype=np.float32)),
        },
        coords=coords,
    )
    ds.rio.write_crs(CRS, inplace=True)
    chunks = {"y": size, "x": size, "axis": -1} if not acq else {"acquisition_id": 1, "y": size, "x": size, "axis": -1}
    return ds.chunk(chunks)


def _simul_ml(size, block=8, az_shift=0):
    """Coarse min/max multilooked simulation as consumed by get_geo_from_radar_bbox."""
    ys = np.arange(0, size, block, dtype=np.float64)
    ye = ys + block - 1
    az_mn = T0 + az_shift * AZ_DT + ys.astype(int) * AZ_DT
    az_mx = T0 + az_shift * AZ_DT + ye.astype(int) * AZ_DT
    mn = xr.Dataset(
        {
            "azimuth_time": (("y", "x"), np.broadcast_to(az_mn[:, None], (ys.size, ys.size)).copy()),
            "slant_range_time": (("y", "x"), np.broadcast_to(ys * RG_DT, (ys.size, ys.size)).copy()),
        },
        coords={"y": ys, "x": ys},
    )
    mx = xr.Dataset(
        {
            "azimuth_time": (("y", "x"), np.broadcast_to(az_mx[:, None], (ye.size, ye.size)).copy()),
            "slant_range_time": (("y", "x"), np.broadcast_to(ye * RG_DT, (ye.size, ye.size)).copy()),
        },
        coords={"y": ye, "x": ye},
    )
    return {"min": mn, "max": mx}


def _cmlp_ds(size, past=False):
    """CMLP measurements: a phase variable (optionally tagged with past
    reference dates to exercise the long dimensions) and a constant coherence."""
    iy = np.arange(size, dtype=np.float64)
    phase2 = np.broadcast_to(0.1 * iy[:, None], (size, size)).copy()
    coh2 = np.full((size, size), 0.5)
    if past:
        phase = np.stack([phase2, phase2])[np.newaxis]
        dims = ["polarization", "past_reference_date", "azimuth_time", "slant_range_time"]
    else:
        phase = phase2[np.newaxis]
        dims = ["polarization", "azimuth_time", "slant_range_time"]
    coords = {
        "polarization": [POL],
        "azimuth_time": T0 + np.arange(size) * AZ_DT,
        "slant_range_time": np.arange(size, dtype=np.float64) * RG_DT,
        "azimuth_m": ("azimuth_time", np.arange(size, dtype=np.float64)),
        "slant_range_m": ("slant_range_time", np.arange(size, dtype=np.float64)),
    }
    if past:
        coords["past_reference_date"] = PAST_DATES
    ds = xr.Dataset(
        {
            "phase": (dims, phase),
            "short_term_coh": (["polarization", "azimuth_time", "slant_range_time"], coh2[np.newaxis]),
        },
        coords=coords,
        attrs={POL: _annotations()},
    )
    # the geocoding orchestrator reads chunk sizes for the long dimensions
    return ds.chunk({"polarization": 1, **({"past_reference_date": 1} if past else {})})


def _phase_ds(size, offset=0.0):
    """Full-resolution SLC phase model as stored in coreg/measurements: the
    variable references the acquisition_id dimension so the acquisition
    coordinates survive DataTree storage, like real coregistered SLCs."""
    ramp = MODEL_RAMP[:size, :size] + offset
    return xr.Dataset(
        {
            "phase_model": (
                ["acquisition_id", "azimuth_time", "slant_range_time"],
                np.broadcast_to(ramp, (size, size)).astype(np.float32)[np.newaxis],
            )
        },
        coords={
            "acquisition_id": [ACQ_ID],
            "acquisition_date": ("acquisition_id", [DATE_STR]),
            "azimuth_time": T0 + np.arange(size) * AZ_DT,
            "slant_range_time": np.arange(size, dtype=np.float64) * RG_DT,
        },
    )


def _phase_sel(size, offset=0.0):
    """The phase model after the acquisition selection performed by stack_gmlp."""
    return _phase_ds(size, offset=offset).sel(acquisition_id=ACQ_ID)


def _mask_ds(size):
    return xr.Dataset(
        {"mask": (["y", "x"], np.ones((size, size), dtype=np.int8))},
        coords={"y": _ground(size), "x": _ground(size)},
    )


def _write_geo_grid(path, size):
    ds = _mask_ds(size)
    ds.rio.write_crs(CRS, inplace=True)
    write(ds, data_path=str(path), skip_read=True)


def _cmlp_tree(multiburst=False):
    tree = xr.DataTree()
    for k in BURSTS if multiburst else [None]:
        group = f"bursts/{k}/cmlp" if multiburst else "cmlp"
        tree[f"{group}/{DATE_STR}"] = _cmlp_ds(N, past=False)
    return tree


def _slcs_tree(multiburst=False):
    tree = xr.DataTree()
    for i, k in enumerate(BURSTS if multiburst else [None]):
        prefix = f"bursts/{k}" if multiburst else ""
        tree[f"{prefix}/coreg/conditions/orbit"] = xr.DataArray(np.array(0.0))
        tree[f"{prefix}/coreg/measurements"] = _phase_ds(N, offset=100.0 * i)
    return tree


def _simulation_tree(multiburst=False):
    if not multiburst:
        return _simul_ds(N, acq=True)
    tree = xr.DataTree()
    for k in BURSTS:
        tree[f"bursts/{k}"] = _simul_ds(N, acq=True)
    return tree


def _image_da(size):
    image = xr.DataArray(np.zeros((size, size)), dims=("y", "x"), coords={"y": _ground(size), "x": _ground(size)})
    image.rio.write_crs(CRS, inplace=True)
    return image


def _inside(values):
    """Slice of an (y, x) plane inside the sinc-kernel valid region."""
    return values[..., MARGIN[1] : N - MARGIN[0], MARGIN[1] : N - MARGIN[0]]


class TestBlockwiseInterpCmlp:
    """Unit tests for cop_ard_app.gmlp.blockwise_interp_cmlp.

    Uses an identity geocoding configuration: simulation, CMLP and phase model
    all sit exactly on the radar pixels that correspond to each output ground
    pixel, so the resampling must reproduce the inputs bit-for-bit (within
    float tolerances) inside the sinc-kernel margin and leave zeros outside.
    """

    @pytest.mark.unit
    def test_identity_warp_restarts_the_radar_content(self, tmp_path):
        image = _image_da(N)
        simulation = _simul_ds(N)
        cmlp = _cmlp_ds(N)
        phase = _phase_sel(N)
        mask = _mask_ds(N)
        kernel = _kernel(KERNEL_SHAPE)

        out = blockwise_interp_cmlp(
            image,
            simulation,
            cmlp,
            phase,
            mask,
            kernel,
            float(AZ_DT),
            RG_DT,
            float(AZ_DT),
            RG_DT,
        )

        assert set(out.data_vars) == {"phase", "short_term_coh", "phase_model", "mask"}
        assert out["phase"].shape == (1, N, N)
        assert out["phase_model"].shape == (N, N)

        phase_out = out["phase"].values
        expected_phase = np.broadcast_to(0.1 * np.arange(N)[:, None], (N, N))
        assert np.abs(_inside(phase_out[0]) - _inside(expected_phase)).max() < TOL
        outside = phase_out.copy()
        outside[..., MARGIN[1] : N - MARGIN[0], MARGIN[1] : N - MARGIN[0]] = np.nan
        assert np.nansum(np.abs(outside)) == 0.0

        coh_out = out["short_term_coh"].values
        assert np.abs(_inside(coh_out[0]) - COH_FIELD[MARGIN[1] : N - MARGIN[0], MARGIN[1] : N - MARGIN[0]]).max() < TOL
        assert not np.all(coh_out == 0.0)
        outside_coh = coh_out.copy()
        outside_coh[..., MARGIN[1] : N - MARGIN[0], MARGIN[1] : N - MARGIN[0]] = np.nan
        assert np.nansum(np.abs(outside_coh)) == 0.0

        assert np.abs(out["phase_model"].values - MODEL_RAMP).max() < TOL
        assert np.array_equal(out["mask"].values, np.ones((N, N), dtype=np.int8))

    @pytest.mark.unit
    def test_identity_warp_keeps_long_dimensions_of_cmlp(self, tmp_path):
        image = _image_da(N)
        simulation = _simul_ds(N)
        phase = _phase_sel(N)
        kernel = _kernel(KERNEL_SHAPE)
        cmlp = _cmlp_ds(N, past=True)

        out = blockwise_interp_cmlp(
            image, simulation, cmlp, phase, None, kernel, float(AZ_DT), RG_DT, float(AZ_DT), RG_DT
        )

        assert out["phase"].shape == (1, len(PAST_DATES), N, N)
        assert "mask" not in out.data_vars
        plane = out["phase"].values[0]
        expected_phase = np.broadcast_to(0.1 * np.arange(N)[:, None], (N, N))
        assert np.abs(_inside(plane[0]) - _inside(expected_phase)).max() < TOL
        assert np.abs(_inside(plane[1]) - _inside(expected_phase)).max() < TOL

    @pytest.mark.unit
    def test_degenerate_cmlp_grid_leaves_zeros_and_phase_fill(self, tmp_path):
        # a CMLP/phase grid smaller than the kernel footprint has no valid
        # source pixel: the CMLP outputs stay identically zero and the phase
        # interpolation fills the out-of-grid queries with its fill value
        small = 4
        image = _image_da(N)
        simulation = _simul_ds(N)
        cmlp = _cmlp_ds(small)
        phase = _phase_sel(small)
        kernel = _kernel(KERNEL_SHAPE)

        out = blockwise_interp_cmlp(
            image, simulation, cmlp, phase, None, kernel, float(AZ_DT), RG_DT, float(AZ_DT), RG_DT
        )

        assert set(out.data_vars) == {"phase", "short_term_coh", "phase_model"}
        assert np.all(out["phase"].values == 0.0)
        assert np.all(out["short_term_coh"].values == 0.0)
        # the phase model only covers the small grid: the rest is the fill value
        expected_model = np.zeros((N, N))
        expected_model[:small, :small] = MODEL_RAMP[:small, :small]
        assert np.abs(out["phase_model"].values - expected_model).max() < TOL


class TestUpdateConditionsGmlp:
    """Unit tests for cop_ard_app.gmlp.update_conditions_gmlp."""

    @pytest.mark.unit
    def test_look_vector_is_normalized_and_constant(self):
        size = 8
        grid_size = (5, 5)
        simulation = _simul_ds(size)
        gmlp = update_conditions_gmlp(xr.DataTree(), simulation, grid_size=grid_size)

        lv = gmlp["conditions/look_vector"].to_dataset()["look_vector"]
        assert lv.shape == (3, *grid_size)
        assert np.array_equal(lv["axis"].values, [0, 1, 2])
        values = lv.values
        # dem_distance is a constant (0, 0, 100) offset: the unit look vector is (0, 0, 1)
        assert np.abs(values[0]).max() == 0.0
        assert np.abs(values[1]).max() == 0.0
        assert np.abs(values[2] - 1.0).max() < TOL
        norms = np.linalg.norm(values, axis=0)
        assert np.abs(norms - 1.0).max() < TOL

    @pytest.mark.unit
    def test_etad_corrections_land_on_the_coarse_grid(self):
        size = 8
        grid_size = (5, 5)
        simulation = _simul_ds(size)
        az_idx = np.arange(size)
        etad = xr.Dataset(
            {
                "delta_azimuth": (
                    ["acquisition_id", "azimuth_time", "slant_range_time"],
                    np.broadcast_to(2 * az_idx[:, None] + 0.5, (size, size))[np.newaxis],
                ),
                "delta_range": (
                    ["acquisition_id", "azimuth_time", "slant_range_time"],
                    np.broadcast_to(0.25 * az_idx[None, :] + 1.0, (size, size))[np.newaxis],
                ),
            },
            coords={
                "acquisition_id": [ACQ_ID],
                "azimuth_time": T0 + az_idx * AZ_DT,
                "slant_range_time": az_idx.astype(np.float64) * RG_DT,
            },
        )

        gmlp = update_conditions_gmlp(xr.DataTree(), simulation, etad=etad, grid_size=grid_size)

        ds = gmlp["conditions/etad"].to_dataset()
        assert ds["delta_azimuth"].shape == (1, grid_size[0], grid_size[1])
        assert np.array_equal(ds["delta_azimuth"].acquisition_id.values, [ACQ_ID])
        yq = ds["delta_azimuth"].y.values
        xq = ds["delta_azimuth"].x.values
        values = ds["delta_azimuth"].values[0]
        assert np.abs(values - (2 * yq[:, None] + 0.5)).max() < ETAD_TOL
        values = ds["delta_range"].values[0]
        assert np.abs(values - (0.25 * xq[None, :] + 1.0)).max() < ETAD_TOL


class TestMapGeocodingCmlp:
    """Unit tests for cop_ard_app.gmlp.map_geocoding_cmlp."""

    @pytest.mark.unit
    def test_geocoded_zarr_store_contains_every_resampled_variable(self, tmp_path):
        static = tmp_path / "static"
        work = tmp_path / "tmp"
        data_path_static = str(static / "bursts" / BURSTS[0])
        _write_geo_grid(f"{data_path_static}/geo_grid", N)

        cmlp = xr.DataTree()
        cmlp[f"bursts/{BURSTS[0]}/cmlp/{DATE_STR}"] = _cmlp_ds(N, past=True)
        kernel = _kernel(KERNEL_SHAPE)

        burst_id, date, gmlp = map_geocoding_cmlp(
            BURSTS[0],
            DATE_STR,
            cmlp[f"bursts/{BURSTS[0]}/cmlp/{DATE_STR}"],
            _simul_ds(N, acq=True),
            _simul_ml(N),
            None,
            _phase_sel(N),
            data_path_static,
            str(work / "bursts" / BURSTS[0]),
            kernel,
            SPACING,
            CHUNKS,
            enable_dask=False,
        )

        assert (burst_id, date) == (BURSTS[0], DATE_STR)
        assert {"measurements", "conditions/look_vector"} <= {g.lstrip("/") for g in gmlp.groups}

        measurements = gmlp["measurements"].to_dataset()
        assert {"phase", "short_term_coh", "phase_model", "mask"} <= set(measurements.data_vars)
        assert measurements["phase"].shape == (1, len(PAST_DATES), N, N)

        expected_phase = np.broadcast_to(0.1 * np.arange(N)[:, None], (N, N))
        for plane in measurements["phase"].values[0]:
            assert np.abs(_inside(plane) - _inside(expected_phase)).max() < TOL

        assert (
            np.abs(
                _inside(measurements["short_term_coh"].values[0])
                - COH_FIELD[MARGIN[1] : N - MARGIN[0], MARGIN[1] : N - MARGIN[0]]
            ).max()
            < TOL
        )
        assert np.abs(measurements["phase_model"].values - MODEL_RAMP).max() < TOL
        assert np.array_equal(measurements["mask"].values, np.ones((N, N), dtype=np.int8))

        lv = gmlp["conditions/look_vector"].to_dataset()["look_vector"]
        assert lv.shape[0] == 3
        assert np.abs(lv.values[0]).max() < TOL
        assert np.abs(lv.values[2] - 1.0).max() < TOL

    @pytest.mark.unit
    def test_geocoding_rejects_a_radar_chunk_outside_the_simulation(self, tmp_path):
        static = tmp_path / "static"
        data_path_static = str(static / "bursts" / BURSTS[0])
        _write_geo_grid(f"{data_path_static}/geo_grid", N)

        cmlp = xr.DataTree()
        cmlp[f"bursts/{BURSTS[0]}/cmlp/{DATE_STR}"] = _cmlp_ds(N)

        with pytest.raises(ValueError, match="Could not determine geographic bounding box"):
            map_geocoding_cmlp(
                BURSTS[0],
                DATE_STR,
                cmlp[f"bursts/{BURSTS[0]}/cmlp/{DATE_STR}"],
                _simul_ds(N, acq=True),
                # the coarse simulation covers radar times four hours ahead of the chunk
                _simul_ml(N, az_shift=4 * 3600),
                None,
                _phase_sel(N),
                data_path_static,
                str(tmp_path / "tmp" / "bursts" / BURSTS[0]),
                _kernel(KERNEL_SHAPE),
                SPACING,
                CHUNKS,
                enable_dask=False,
            )


class TestSingleGmlp:
    """Unit test for cop_ard_app.gmlp.single_gmlp (single scene, no burst)."""

    @pytest.mark.unit
    def test_single_gmlp_geocodes_the_scene_and_updates_conditions(self, tmp_path):
        static = tmp_path / "static"
        _write_geo_grid(str(static / "data" / REF_ID / "geo_grid"), N)

        cmlp_tree = _cmlp_tree()
        burst_id, date, gmlp = single_gmlp(
            BURSTS[0],
            DATE_STR,
            cmlp_tree[f"cmlp/{DATE_STR}"],
            _simulation_tree(),
            _simul_ml(N),
            None,
            _phase_sel(N),
            False,
            SPACING,
            REF_ID,
            str(static),
            str(tmp_path / "tmp"),
            CHUNKS,
            enable_dask=False,
            warp_kernel_shape=KERNEL_SHAPE,
        )

        assert (burst_id, date) == (BURSTS[0], DATE_STR)
        assert {"measurements", "conditions/look_vector"} <= {g.lstrip("/") for g in gmlp.groups}

        measurements = gmlp["measurements"].to_dataset()
        assert "phase_model" in measurements
        assert np.abs(measurements["phase_model"].values - MODEL_RAMP).max() < TOL
        expected_phase = np.broadcast_to(0.1 * np.arange(N)[:, None], (N, N))
        assert np.abs(_inside(measurements["phase"].values[0]) - _inside(expected_phase)).max() < TOL
        assert np.array_equal(measurements["mask"].values, np.ones((N, N), dtype=np.int8))

        lv = gmlp["conditions/look_vector"].to_dataset()["look_vector"]
        assert np.abs(lv.values[0]).max() < TOL


class TestStackGmlp:
    """Unit test for cop_ard_app.gmlp.stack_gmlp (multiburst mode)."""

    @pytest.mark.unit
    def test_stack_gmlp_reassembles_one_geocoded_node_per_burst_and_date(self, tmp_path):
        static = tmp_path / "static"
        work = tmp_path / "tmp"
        for burst_id in BURSTS:
            _write_geo_grid(str(static / "bursts" / burst_id / "geo_grid"), N)

        gmlp_tree = stack_gmlp(
            _cmlp_tree(multiburst=True),
            _slcs_tree(multiburst=True),
            _simulation_tree(multiburst=True),
            {"bursts": {k: _simul_ml(N) for k in BURSTS}},
            SPACING,
            REF_ID,
            str(static),
            str(work),
            enable_dask=False,
            chunks=CHUNKS,
            warp_kernel_shape=KERNEL_SHAPE,
        )

        expected_groups = {f"bursts/{k}/{DATE_STR}" for k in BURSTS}
        assert expected_groups <= {g.lstrip("/") for g in gmlp_tree.groups}

        expected_phase = np.broadcast_to(0.1 * np.arange(N)[:, None], (N, N))
        for i, burst_id in enumerate(BURSTS):
            node = gmlp_tree[f"bursts/{burst_id}/{DATE_STR}"]
            prefix = f"/bursts/{burst_id}/{DATE_STR}"
            assert {f"{prefix}/measurements", f"{prefix}/conditions/look_vector"} <= set(node.groups)
            measurements = node["measurements"].to_dataset()
            assert np.abs(_inside(measurements["phase"].values[0]) - _inside(expected_phase)).max() < TOL
            assert np.abs(measurements["phase_model"].values - (MODEL_RAMP + 100.0 * i)).max() < TOL
            assert np.array_equal(measurements["mask"].values, np.ones((N, N), dtype=np.int8))
