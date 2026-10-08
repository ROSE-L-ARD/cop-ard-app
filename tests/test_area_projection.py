"""Unit tests for the `cop_ard_app.area_projection` module."""

import numpy as np
import pytest
import xarray as xr
from scipy.constants import speed_of_light
from scipy.sparse import csr_matrix

from cop_ard_app.area_projection import (
    blockwise_rasterization,
    blockwise_rtc_ap,
    map_shadow,
    map_shadow_projection,
    mask_shadow_single_burst,
    single_burst_geometry,
    sparse_to_dense,
    stack_geometry,
    stack_mask,
)
from cop_ard_app.io import exists, read, save_empty_dataset, save_sparse_array, write

# Small exact-dyadic grid geometry.
N_AZ = 5  # Radar grid size along the azimuth time.
N_PRG = 7  # Radar grid size along the slant range time.
GEO_N = 7  # Geogrid size (number of cells, i.e. grid nodes minus one).
DEM_N = 8  # DEM grid size (number of nodes).
GEO_X0 = 10.0  # Geogrid corner in longitude (degrees).
GEO_Y0 = 45.0  # Geogrid corner in latitude (degrees).
DEG_STEP = 2.0**-8  # Spatial step of the geogrid and DEM in degrees (exact dyadic).
ACQUISITION = 0  # Acquisition identifier.
AZIMUTH_ORIGIN = np.datetime64("2020-06-21T00:00:00")  # Origin of the azimuth time.
AZ_DT_S = 1.0  # Azimuth time step in seconds, i.e. geogrid azimuth pixel size.
RG_ORIGIN = 1.0  # Slant range time of the closest radar pixel in seconds.
DRG_S = 1.0  # Slant range time step in seconds, i.e. geogrid range pixel size.
DEM_ELEV = 1000.0  # Constant DEM elevation in meters.
DEM_DISTANCE = 700000.0  # Distance from the satellite to the closest point of the DEM in meters.
ROW_SS = speed_of_light / 2  # Row spacing in time that results in a slant range step of one second.
COL_SS = 1.0  # Column spacing in azimuth time.
TIME_CA_POLY = [0.0, 1.0]  # Time center aperture polynomial coefficients.
CHUNK_RG = 3  # Chunk size along the slant range time.
CHUNK_AZ = 3  # Chunk size along the azimuth time.
CHUNK_XY = 4  # Chunk size along the spatial dimensions of the geogrid and DEM.
LIA_FLAT = 0.0  # Local incidence angle of a plane parallel to the radar beam (radians).
LIA_HORIZONTAL = 1.0  # Local incidence angle of a plane orthogonal to the radar beam (radians).
GRAZING_THRESHOLD = 0.05  # Cosine of the grazing angle under which a pixel is in the shadow.
EPS_NO_OCCLUSION = np.inf  # Epsilon that disables the occlusion (cumulative maximum) mask.
TOLERANCE = 0.0  # Tolerance of the assertions, exact for these analytic fixtures.
REFERENCE_ID = "2020-06-21"  # Reference acquisition identifier of the single scene fixtures.
BURST_A = "iw1_254"  # First burst identifier of the multiburst stack fixtures.
BURST_B = "iw1_255"  # Second burst identifier of the multiburst stack fixtures.
RADAR_BBOX = np.array([0, 0, N_PRG, N_AZ])  # Bounding box of the SLC footprint [min_rg, min_az, max_rg, max_az].
RADAR_PIXEL_AREA = COL_SS * ROW_SS  # Area of a radar pixel, as the pipeline derives it from the annotations.
SINGLE_LOOK = (DEG_STEP, DEG_STEP)  # Output resolution far below a radar pixel, i.e. one expected look.
N_CHUNK_COLUMNS = int(np.ceil(GEO_N / CHUNK_XY))  # Number of geogrid chunk columns of the scene.
EXPECTED_LOOKS = np.where((np.arange(GEO_N) < N_AZ)[:, None], np.ones(GEO_N), np.zeros(GEO_N))  # Geogrid look counts.
# Outputs the radiometric terrain correction kernel would return, which is stubbed out to stay deterministic.
RTC_ANF = np.full((GEO_N, GEO_N), 0.25, dtype="float32")
RTC_LIA = np.full((GEO_N, GEO_N), 0.75, dtype="float32")
RTC_BBOX_RADAR = np.array([3, 0, 7, 0])  # Radar bounding box the correction kernel reports the area block on.
# Weights of a single covered geogrid cell, i.e. the smallest sparse block the correction reads.
WEIGHTS_DATA = np.array([0.25, 0.5, 0.75, 1.0], dtype="float32")
WEIGHTS_INDICES = np.arange(4, dtype="int64")

geo_x = GEO_X0 + (np.arange(GEO_N) + 0.5) * DEG_STEP
geo_y = GEO_Y0 + (np.arange(GEO_N) + 0.5) * DEG_STEP
dem_x = GEO_X0 + np.arange(DEM_N) * DEG_STEP
dem_y = GEO_Y0 + np.arange(DEM_N) * DEG_STEP
azimuth_time = AZIMUTH_ORIGIN + np.arange(N_AZ) * np.timedelta64(int(AZ_DT_S * 1_000_000_000), "ns")
slant_range_time = RG_ORIGIN + np.arange(N_PRG) * DRG_S
cell_x = (dem_x[:-1] + dem_x[1:]) / 2  # Centres of the geogrid cells, i.e. the mid points of the DEM nodes.
cell_y = (dem_y[:-1] + dem_y[1:]) / 2

annotations = {
    "Grid": {"Col": {"SS": COL_SS}, "Row": {"SS": ROW_SS}},
    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": TIME_CA_POLY}}},
}


def _slc() -> xr.DataArray:
    """Build a reference Single Look Complex whose radar grid is flat."""
    slc = xr.DataArray(
        np.ones((1, 1, N_AZ, N_PRG), dtype="float32"),
        dims=["acquisition_id", "polarization", "azimuth_time", "slant_range_time"],
        coords={
            "acquisition_id": [ACQUISITION],
            "polarization": [0],
            "azimuth_time": azimuth_time,
            "slant_range_time": slant_range_time,
        },
    ).chunk({"azimuth_time": CHUNK_AZ, "slant_range_time": CHUNK_RG})
    slc.attrs = {0: annotations}
    return slc


def _simulation() -> xr.Dataset:
    """
    Build a simulation of a flat DEM onto a flat radar grid.

    The DEM is a horizontal plane at constant elevation; the radar is imaged straight
    below the satellite, i.e. the azimuth time increases along the DEM `y` axis and
    the slant range time increases along the DEM `x` axis. The radar distance vectors
    point from the DEM towards the satellite position.
    """
    sim = xr.Dataset(
        coords={
            "acquisition_id": [ACQUISITION],
            "y": dem_y,
            "x": dem_x,
            "axis": [0, 1, 2],
        },
        data_vars={
            "azimuth_time": (
                ["acquisition_id", "y", "x"],
                np.broadcast_to(
                    (AZIMUTH_ORIGIN + np.arange(DEM_N) * np.timedelta64(int(AZ_DT_S * 1_000_000_000), "ns"))[
                        None, :, None
                    ],
                    (1, DEM_N, DEM_N),
                ),
            ),
            "slant_range_time": (
                ["acquisition_id", "y", "x"],
                np.broadcast_to((RG_ORIGIN + np.arange(DEM_N) * DRG_S)[None, None, :], (1, DEM_N, DEM_N)),
            ),
            "dem_distance": (
                ["acquisition_id", "y", "x", "axis"],
                np.broadcast_to([0.0, 0.0, DEM_DISTANCE], (1, DEM_N, DEM_N, 3)),
            ),
            "h": (["y", "x"], np.full((DEM_N, DEM_N), DEM_ELEV, dtype="float32")),
        },
        attrs={"crs": "EPSG:4326"},
    ).chunk({"acquisition_id": 1, "y": CHUNK_XY, "x": CHUNK_XY, "axis": -1})
    sim.rio.write_crs("EPSG:4326", inplace=True)
    return sim


def _simulation_ml() -> dict:
    """Build the multilooked (min/max) simulation bounds covering the whole simulation."""
    bounds = {}
    for name, (y, x, az, rg) in {
        "min": (dem_y[0], dem_x[0], azimuth_time[0], slant_range_time[0]),
        "max": (dem_y[-1], dem_x[-1], azimuth_time[-1], slant_range_time[-1]),
    }.items():
        bounds[name] = xr.Dataset(
            data_vars={
                "azimuth_time": (["y", "x"], np.array([[az]], dtype=np.datetime64)),
                "slant_range_time": (["y", "x"], np.array([[rg]], dtype="float64")),
            },
            coords={"y": [y], "x": [x]},
        )
    return bounds


def _geo_grid(data_path: str, lia: float) -> None:
    """Save a geogrid whose local incidence angle is constant."""
    save_empty_dataset(
        ["lia", "anf", "nlooks", "mask"],
        {"lia": ["y", "x"], "anf": ["y", "x"], "nlooks": ["y", "x"], "mask": ["y", "x"]},
        {"lia": [GEO_N, GEO_N], "anf": [GEO_N, GEO_N], "nlooks": [GEO_N, GEO_N], "mask": [GEO_N, GEO_N]},
        {"y": CHUNK_XY, "x": CHUNK_XY},
        {"lia": np.float32, "anf": np.float32, "nlooks": np.float32, "mask": np.int32},  # type: ignore
        f"{data_path}/geo_grid",
        crs="EPSG:4326",
        coords={"y": geo_y, "x": geo_x},
        attrs={"crs": "EPSG:4326"},
    )
    write(
        xr.Dataset(
            data_vars={"lia": (["y", "x"], np.full((GEO_N, GEO_N), lia, dtype="float32"))},
            coords={"y": geo_y, "x": geo_x},
        ),
        data_path=f"{data_path}/geo_grid",
        region={"y": slice(None), "x": slice(None)},
        consolidated=False,
        mode="r+",
    )


def _radar_grid(data_path: str) -> None:
    """Save an empty radar grid ready to be filled with the masks."""
    save_empty_dataset(
        ["anf", "layover", "shadow"],
        {k: ["azimuth_time", "slant_range_time"] for k in ["anf", "layover", "shadow"]},
        {k: [N_AZ, N_PRG] for k in ["anf", "layover", "shadow"]},
        {"azimuth_time": CHUNK_AZ, "slant_range_time": CHUNK_RG},
        {"anf": np.float32, "layover": bool, "shadow": bool},  # type: ignore
        f"{data_path}/radar_grid",
        coords={"azimuth_time": azimuth_time, "slant_range_time": slant_range_time},
    )


def _global_values(data_path: str, chunk_length: int) -> None:
    """Save the global geogrid values of a scene whose geogrid is `GEO_N` cells wide."""
    write(
        xr.Dataset(
            data_vars={
                "geo_xsize": (["size"], np.array([GEO_N], dtype="int64")),
                "geo_ysize": (["size"], np.array([GEO_N], dtype="int64")),
                "chunk_length": (["size"], np.array([chunk_length], dtype="int64")),
            },
            coords={"size": [0]},
        ),
        data_path=f"{data_path}/global_values",
    )


def _mask(data_path: str) -> xr.DataArray:
    """Read the geocoded mask stored in the geogrid."""
    return read(f"{data_path}/geo_grid", as_type="xr.Dataset", consolidated=False)["mask"]


def _shadow(data_path: str) -> xr.DataArray:
    """Read the shadow mask stored in the radar grid."""
    return read(f"{data_path}/radar_grid", as_type="xr.Dataset", consolidated=False)["shadow"]


def _radar_selection() -> xr.Dataset:
    """
    Build the DEM subset the geometry stage works on, as the simulation maps it onto the radar grid.

    The azimuth time increases along the DEM `y` axis and the slant range time along its `x` axis,
    so the azimuth pixel coordinate of a DEM node is its row index and the range pixel coordinate is
    its column index, which is what the simulation of `_simulation()` yields for the SLC of `_slc()`.
    """
    az_pixel, rg_pixel = np.meshgrid(np.arange(DEM_N), np.arange(DEM_N), indexing="ij")
    return xr.Dataset(
        data_vars={
            "az_px": (["y", "x"], az_pixel.astype("float64")),
            "rg_px": (["y", "x"], rg_pixel.astype("float64")),
            "dem_distance": (
                ["y", "x", "axis"],
                np.broadcast_to([0.0, 0.0, DEM_DISTANCE], (DEM_N, DEM_N, 3)).copy(),
            ),
            "h": (["y", "x"], np.full((DEM_N, DEM_N), DEM_ELEV, dtype="float32")),
        },
        coords={"y": dem_y, "x": dem_x, "axis": [0, 1, 2]},
    ).chunk({"y": CHUNK_XY, "x": CHUNK_XY, "axis": -1})


def _weights_store(data_path: str, block_id: str) -> None:
    """Save the rasterization weights of one geogrid block in the format the correction reads."""
    save_sparse_array(
        WEIGHTS_DATA,
        WEIGHTS_INDICES,
        np.array([0, WEIGHTS_DATA.size], dtype="int64"),
        [2, 2],  # type: ignore
        "CSR",
        WEIGHTS_DATA.size,
        f"{data_path}/w/{block_id}",
    )


def _slc_stack(bursts: tuple[str, ...]) -> xr.DataTree:
    """Build the SLC DataTree of a multiburst, with one reference acquisition per burst."""
    tree = xr.DataTree()
    for burst in bursts:
        tree[f"bursts/{burst}/orig/{REFERENCE_ID}/measurements/slc"] = _slc()
    return tree


def _slc_scene() -> xr.DataTree:
    """Build the SLC DataTree of a single scene, i.e. a stack of one data block."""
    tree = xr.DataTree()
    tree[f"orig/{REFERENCE_ID}/measurements/slc"] = _slc()
    return tree


def _simulation_stack(bursts: tuple[str, ...]) -> xr.DataTree:
    """Build the simulation DataTree of a multiburst, one simulation per burst."""
    tree = xr.DataTree()
    for burst in bursts:
        tree[f"bursts/{burst}"] = _simulation()
    return tree


def _simulation_scene() -> xr.DataTree:
    """Build the simulation DataTree of a single scene, i.e. a simulation at the root."""
    return xr.DataTree(dataset=_simulation())


def _simulation_ml_stack(bursts: tuple[str, ...]) -> dict:
    """Build the multilooked simulation bounds of a multiburst."""
    return {"bursts": {burst: _simulation_ml() for burst in bursts}}


@pytest.mark.unit
class TestMapShadow:
    """Test the mapping of the shadow mask onto the radar grid."""

    @pytest.mark.unit
    def test_plane_parallel_to_beam_is_all_in_shadow(self, tmp_path):
        """A DEM plane parallel to the radar beam has a zero grazing angle, i.e. all its pixels are in the shadow."""
        data_path = str(tmp_path / "bursts/one")
        _geo_grid(data_path, lia=LIA_FLAT)
        _radar_grid(data_path)
        map_shadow(
            _slc(),
            _simulation(),
            _simulation_ml(),
            data_path,
            EPS_NO_OCCLUSION,
            False,
        )
        got = _shadow(data_path).values
        expected = np.full((N_AZ, N_PRG), True)
        np.testing.assert_allclose(got, expected, atol=TOLERANCE)

    @pytest.mark.unit
    def test_plane_orthogonal_to_beam_is_never_in_shadow(self, tmp_path):
        """A DEM plane orthogonal to the radar beam never produces a grazing angle, i.e. no pixel is in the shadow."""
        data_path = str(tmp_path / "bursts/one")
        _geo_grid(data_path, lia=LIA_HORIZONTAL)
        _radar_grid(data_path)
        map_shadow(
            _slc(),
            _simulation(),
            _simulation_ml(),
            data_path,
            EPS_NO_OCCLUSION,
            False,
        )
        got = _shadow(data_path).values
        expected = np.full((N_AZ, N_PRG), False)
        np.testing.assert_allclose(got, expected, atol=TOLERANCE)


@pytest.mark.unit
class TestMapShadowProjection:
    """Test the mapping of the geocoded mask onto the geogrid."""

    @pytest.mark.unit
    def test_absence_of_rasterization_weights_gives_zero_mask(self, tmp_path):
        """Without rasterization weights, the geocoded mask is all zeros, i.e. no point is covered by the radar."""
        data_path = str(tmp_path / "bursts/one")
        _global_values(data_path, CHUNK_XY)
        _geo_grid(data_path, lia=LIA_HORIZONTAL)
        map_shadow_projection(
            _slc(),
            data_path,
            threshold=0.3,
            enable_dask=False,
        )
        got = _mask(data_path).values
        expected = np.zeros((GEO_N, GEO_N), dtype="int32")
        np.testing.assert_allclose(got, expected, atol=TOLERANCE)


@pytest.mark.unit
class TestMaskShadowSingleBurst:
    """Test the computation of the shadow mask of a single burst."""

    @pytest.mark.unit
    def test_reference_id_is_mandatory_for_a_single_scene(self):
        """A single scene (i.e. not a multiburst) cannot be processed without a reference identifier."""
        with pytest.raises(ValueError, match="reference_id is mandatory"):
            mask_shadow_single_burst(
                "one",
                _slc(),
                _simulation(),
                _simulation_ml(),
                False,
                "",
            )

    @pytest.mark.unit
    def test_multiburst_writes_masks_and_finish_marker(self, tmp_path):
        """A multiburst writes the radar grid shadow and the geocoded mask, and finishes with a marker file."""
        working_directory = str(tmp_path)
        data_path = f"{working_directory}/bursts/one"
        _geo_grid(data_path, lia=LIA_FLAT)
        _radar_grid(data_path)
        _global_values(data_path, CHUNK_XY)
        mask_shadow_single_burst(
            "one",
            _slc(),
            _simulation(),
            _simulation_ml(),
            True,
            working_directory,
            eps=EPS_NO_OCCLUSION,
            threshold=0.3,
            enable_dask=False,
        )
        expected_shadow = np.full((N_AZ, N_PRG), True)
        np.testing.assert_allclose(_shadow(data_path).values, expected_shadow, atol=TOLERANCE)
        expected_mask = np.zeros((GEO_N, GEO_N), dtype="int32")
        np.testing.assert_allclose(_mask(data_path).values, expected_mask, atol=TOLERANCE)
        assert exists(f"{working_directory}/processed_data/bursts/one/mask_radar.ok")


@pytest.mark.unit
class TestSparseToDense:
    """Test the accumulation of sparse blocks into a dense radar grid chunk."""

    @pytest.mark.unit
    def test_sparse_blocks_are_accumulated_into_the_dense_chunk(self, tmp_path):
        """The area and layover sparse blocks reproduce the dense arrays they were generated from."""
        az_size, rg_size = 8, 6
        rng = np.random.default_rng(42)
        dense = rng.random((az_size, rg_size)).astype(np.float32)
        lay = np.zeros((az_size, rg_size), dtype="float32")
        lay[1, 2] = 2.0
        lay[5, 3] = 1.0
        expected_layover = lay > 1 + 1e-7
        files_A, files_l1 = [], []
        for az_min, az_max in ((0, 3), (3, az_size)):
            for dense_array, files, prefix in ((dense, files_A, "A"), (lay, files_l1, "L1")):
                band = dense_array[az_min:az_max]
                csr = csr_matrix(band)
                # The kernel indexes the rows of the sparse blocks with their global index,
                # so the row pointers are extended to the whole radar grid.
                indptr = np.concatenate(
                    [
                        np.zeros(az_min, dtype=csr.indptr.dtype),
                        csr.indptr,
                        np.full(az_size - az_max, csr.indptr[-1], dtype=csr.indptr.dtype),
                    ]
                )
                name = f"{tmp_path}/{prefix}/{az_min}.{az_max}.0.{rg_size}"
                save_sparse_array(
                    csr.data,
                    csr.indices,
                    indptr,
                    [az_size, rg_size],
                    "CSR",
                    az_size,
                    name,
                    attrs={"radar_bbox": [0, rg_size - 1, az_min, az_max]},
                )
                files.append(name)
        out = sparse_to_dense(
            xr.Dataset(),
            files_A,
            files_l1,
            [],
            str(tmp_path),
            chunk={"azimuth_time": slice(0, az_size), "slant_range_time": slice(0, rg_size)},
        )
        np.testing.assert_allclose(out.anf.values, dense, atol=TOLERANCE)
        np.testing.assert_array_equal(out.layover.values > 0, expected_layover)


@pytest.mark.unit
class TestBlockwiseRasterization:
    """Test the triangulation weights and look counts of a geogrid block."""

    @pytest.mark.unit
    def test_cells_under_the_radar_footprint_are_looked_once(self):
        """The geogrid cells whose radar coordinates fall in the SLC footprint are looked at exactly once."""
        geo_data, weights, layover = blockwise_rasterization(
            _radar_selection(),
            0,
            0,
            DEM_N - 1,
            GEO_N,
            RADAR_PIXEL_AREA,
            RADAR_BBOX,
            CHUNK_XY,
            SINGLE_LOOK,
            job_id=0,
        )
        np.testing.assert_allclose(geo_data.nlooks.values, EXPECTED_LOOKS, atol=TOLERANCE)
        assert list(weights.children) == ["0.0"]
        # Every covered geogrid cell takes its weight from the four triangles that enclose it.
        assert int(weights.children["0.0"].to_dataset().nnz.values) == 4 * N_AZ * GEO_N
        assert set(layover.children) == {"layover_block_0", "layover_block_1"}
        for name in ("layover_block_0", "layover_block_1"):
            (layover_block,) = [node.to_dataset() for node in layover.children[name].children.values()]
            np.testing.assert_allclose(layover_block.data.values, 1.0, atol=TOLERANCE)

    @pytest.mark.unit
    def test_block_out_of_the_radar_footprint_has_no_weights(self):
        """A DEM whose radar coordinates are all out of the SLC footprint yields no weights nor layover."""
        selection = _radar_selection()
        geo_data, weights, layover = blockwise_rasterization(
            selection.assign(az_px=selection.az_px + N_AZ),
            0,
            0,
            DEM_N - 1,
            GEO_N,
            RADAR_PIXEL_AREA,
            RADAR_BBOX,
            CHUNK_XY,
            SINGLE_LOOK,
            job_id=0,
        )
        np.testing.assert_allclose(geo_data.nlooks.values, 0.0, atol=TOLERANCE)
        assert weights is None
        assert layover is None

    @pytest.mark.unit
    def test_job_identifier_positions_the_block_in_the_geogrid(self):
        """The job identifier is decomposed into the row and the column of the geogrid chunk."""
        selection = _radar_selection()
        for job_id in (1, 2, 3):
            _, weights, _ = blockwise_rasterization(
                selection,
                0,
                0,
                DEM_N - 1,
                GEO_N,
                RADAR_PIXEL_AREA,
                RADAR_BBOX,
                CHUNK_XY,
                SINGLE_LOOK,
                job_id=job_id,
            )
            assert list(weights.children) == [f"{job_id // N_CHUNK_COLUMNS}.{job_id % N_CHUNK_COLUMNS}"]


@pytest.mark.unit
class TestBlockwiseRtcAp:
    """Test the radiometric terrain correction of a geogrid block."""

    @pytest.mark.unit
    def test_block_without_its_weights_is_not_corrected(self, tmp_path):
        """A geogrid block without rasterization weights of its own is not corrected, i.e. stays zero."""
        data_path = str(tmp_path)
        _weights_store(data_path, f"{N_CHUNK_COLUMNS}.0")
        geo_data, area_blocks = blockwise_rtc_ap(
            _radar_selection(),
            GEO_N,
            RADAR_PIXEL_AREA,
            False,
            RADAR_BBOX,
            CHUNK_XY,
            data_path,
            job_id=0,
        )
        expected = np.zeros((GEO_N, GEO_N), dtype="float32")
        np.testing.assert_allclose(geo_data["anf"].values, expected, atol=TOLERANCE)
        np.testing.assert_allclose(geo_data["lia"].values, expected, atol=TOLERANCE)
        assert area_blocks is None

    @pytest.mark.unit
    def test_block_with_its_weights_is_read_and_corrected(self, tmp_path, monkeypatch):
        """The weights stored under the block position are read back and drive the correction."""
        data_path = str(tmp_path)
        block_id = f"1.{N_CHUNK_COLUMNS - 1}"
        _weights_store(data_path, block_id)
        received = []

        def correction(**_unused):
            received.append(_unused["csr_matrix"])
            return (
                xr.DataArray(RTC_ANF, dims=["y", "x"]),
                xr.DataArray(RTC_LIA, dims=["y", "x"]),
                xr.Dataset(data_vars={"a_block": ("point", WEIGHTS_DATA)}),
                RTC_BBOX_RADAR,
            )

        monkeypatch.setattr("cop_ard_app.area_projection.compute_rtc_ap", correction)
        geo_data, area_blocks = blockwise_rtc_ap(
            _radar_selection(),
            GEO_N,
            RADAR_PIXEL_AREA,
            True,
            RADAR_BBOX,
            CHUNK_XY,
            data_path,
            job_id=N_CHUNK_COLUMNS + 1,
        )
        np.testing.assert_allclose(geo_data["anf"].values, RTC_ANF, atol=TOLERANCE)
        np.testing.assert_allclose(geo_data["lia"].values, RTC_LIA, atol=TOLERANCE)
        assert list(area_blocks.children) == [".".join([str(i) for i in RTC_BBOX_RADAR])]
        np.testing.assert_allclose(received[0]["data"].values, WEIGHTS_DATA, atol=TOLERANCE)


@pytest.mark.unit
class TestSingleBurstGeometry:
    """Test the orchestration of the geometry computation of a single burst."""

    @pytest.mark.unit
    def test_slc_must_be_a_data_array_and_a_reference_id_is_mandatory(self):
        """The SLC of a burst must be a data array, and a single scene cannot be processed without a reference."""
        with pytest.raises(Exception, match=r"Expected xr\.DataArray"):
            single_burst_geometry("one", _simulation(), _simulation(), _simulation_ml(), True, CHUNK_XY, "", "")
        with pytest.raises(ValueError, match="reference_id is mandatory"):
            single_burst_geometry("one", _slc(), _simulation(), _simulation_ml(), False, CHUNK_XY, "", "")

    @pytest.mark.unit
    def test_multiburst_writes_the_scene_geometry_and_a_marker(self, tmp_path):
        """A burst writes the global values and the geogrid look counts, and finishes with a completion marker."""
        working_directory = str(tmp_path)
        data_path = f"{working_directory}/bursts/one"
        single_burst_geometry(
            "one",
            _slc(),
            _simulation(),
            _simulation_ml(),
            True,
            CHUNK_XY,
            working_directory,
            working_directory,
            enable_dask=False,
        )
        assert exists(f"{working_directory}/processed_data/bursts/one/area_projection.ok")
        global_values = read(f"{data_path}/global_values", as_type="xr.Dataset", consolidated=False)
        np.testing.assert_array_equal(global_values.bbox.values, RADAR_BBOX)
        np.testing.assert_array_equal(global_values.geo_xsize.values, GEO_N)
        np.testing.assert_array_equal(global_values.geo_ysize.values, GEO_N)
        np.testing.assert_array_equal(global_values.chunks_slc.values, [CHUNK_AZ, CHUNK_RG])
        np.testing.assert_array_equal(global_values.chunk_length.values, CHUNK_XY)
        geo_grid = read(f"{data_path}/geo_grid", as_type="xr.Dataset", consolidated=False)
        np.testing.assert_array_equal(geo_grid.x.values, cell_x)
        np.testing.assert_array_equal(geo_grid.y.values, cell_y)
        np.testing.assert_allclose(geo_grid.nlooks.values, EXPECTED_LOOKS, atol=TOLERANCE)

    @pytest.mark.unit
    def test_single_scene_is_processed_under_its_reference(self, tmp_path, monkeypatch):
        """A single scene is processed under the reference identifier instead of under a burst identifier."""
        working_directory = str(tmp_path)
        processed = []

        def geometry(*positional, **_unused):
            processed.append(positional[2:4])

        monkeypatch.setattr("cop_ard_app.area_projection.map_rasterization_and_rtc_ap", geometry)
        single_burst_geometry(
            "one",
            _slc(),
            _simulation(),
            _simulation_ml(),
            False,
            CHUNK_XY,
            working_directory,
            working_directory,
            reference_id=REFERENCE_ID,
        )
        data_path = f"{working_directory}/data/{REFERENCE_ID}"
        assert processed == [(data_path, data_path)]
        assert exists(f"{working_directory}/processed_data/data/{REFERENCE_ID}/area_projection.ok")

    @pytest.mark.unit
    def test_completion_marker_skips_the_recomputation(self, tmp_path, monkeypatch):
        """A burst whose completion marker exists is only recomputed when explicitly asked for."""
        working_directory = str(tmp_path)
        processed = []

        def geometry(*positional, **_unused):
            processed.append(positional[2])

        def scene(**_unused):
            return xr.Dataset(), xr.Dataset(data_vars={"bbox": (["vertex"], RADAR_BBOX)})

        monkeypatch.setattr("cop_ard_app.area_projection.map_rasterization_and_rtc_ap", geometry)
        monkeypatch.setattr("cop_ard_app.area_projection.initialize_simulation", scene)
        for _ in range(2):
            single_burst_geometry(
                "one", _slc(), _simulation(), _simulation_ml(), True, CHUNK_XY, working_directory, working_directory
            )
        assert processed == [f"{working_directory}/bursts/one"]
        single_burst_geometry(
            "one",
            _slc(),
            _simulation(),
            _simulation_ml(),
            True,
            CHUNK_XY,
            working_directory,
            working_directory,
            recompute=True,
        )
        assert processed == [f"{working_directory}/bursts/one"] * 2


@pytest.mark.unit
class TestStackGeometry:
    """Test the orchestration of the geometry computation of a stack of SLCs."""

    @pytest.mark.unit
    def test_multiburst_scene_of_two_bursts(self, tmp_path):
        """Every burst of the stack gets its own scene geometry and completion marker."""
        working_directory = str(tmp_path)
        stack_geometry(
            _slc_stack((BURST_A, BURST_B)),
            _simulation_stack((BURST_A, BURST_B)),
            _simulation_ml_stack((BURST_A, BURST_B)),
            working_directory,
            working_directory,
            REFERENCE_ID,
            CHUNK_XY,
            True,
            False,
            enable_dask=False,
        )
        for burst in (BURST_A, BURST_B):
            data_path = f"{working_directory}/bursts/{burst}"
            assert exists(f"{working_directory}/processed_data/bursts/{burst}/area_projection.ok")
            global_values = read(f"{data_path}/global_values", as_type="xr.Dataset", consolidated=False)
            np.testing.assert_array_equal(global_values.bbox.values, RADAR_BBOX)
            np.testing.assert_array_equal(global_values.geo_xsize.values, GEO_N)

    @pytest.mark.unit
    def test_single_scene_of_the_stack(self, tmp_path):
        """A stack of a single data block is processed once, under the reference identifier."""
        working_directory = str(tmp_path)
        stack_geometry(
            _slc_scene(),
            _simulation_scene(),
            _simulation_ml(),
            working_directory,
            working_directory,
            REFERENCE_ID,
            CHUNK_XY,
            False,
            False,
            enable_dask=False,
        )
        data_path = f"{working_directory}/data/{REFERENCE_ID}"
        assert exists(f"{working_directory}/processed_data/data/{REFERENCE_ID}/area_projection.ok")
        global_values = read(f"{data_path}/global_values", as_type="xr.Dataset", consolidated=False)
        np.testing.assert_array_equal(global_values.bbox.values, RADAR_BBOX)


@pytest.mark.unit
class TestStackMask:
    """Test the orchestration of the shadow mask computation of a stack of SLCs."""

    @pytest.mark.unit
    def test_multiburst_scene_of_two_bursts(self, tmp_path):
        """Every burst of the stack gets its own shadow, the plane parallel to the beam being all in shadow."""
        working_directory = str(tmp_path)
        for burst in (BURST_A, BURST_B):
            data_path = f"{working_directory}/bursts/{burst}"
            _geo_grid(data_path, lia=LIA_FLAT)
            _radar_grid(data_path)
            _global_values(data_path, CHUNK_XY)
        stack_mask(
            _slc_stack((BURST_A, BURST_B)),
            _simulation_stack((BURST_A, BURST_B)),
            _simulation_ml_stack((BURST_A, BURST_B)),
            working_directory,
            REFERENCE_ID,
            True,
            False,
            enable_dask=False,
        )
        expected_shadow = np.full((N_AZ, N_PRG), True)
        for burst in (BURST_A, BURST_B):
            data_path = f"{working_directory}/bursts/{burst}"
            np.testing.assert_allclose(_shadow(data_path).values, expected_shadow, atol=TOLERANCE)
            assert exists(f"{working_directory}/processed_data/bursts/{burst}/mask_radar.ok")

    @pytest.mark.unit
    def test_single_scene_of_the_stack(self, tmp_path):
        """A stack of a single data block masks the whole scene under the reference identifier."""
        working_directory = str(tmp_path)
        data_path = f"{working_directory}/data/{REFERENCE_ID}"
        _geo_grid(data_path, lia=LIA_HORIZONTAL)
        _radar_grid(data_path)
        _global_values(data_path, CHUNK_XY)
        stack_mask(
            _slc_scene(),
            _simulation_scene(),
            _simulation_ml(),
            working_directory,
            REFERENCE_ID,
            False,
            False,
            enable_dask=False,
        )
        np.testing.assert_allclose(_shadow(data_path).values, np.full((N_AZ, N_PRG), False), atol=TOLERANCE)
        assert exists(f"{working_directory}/processed_data/data/{REFERENCE_ID}/mask_radar.ok")
