import numpy as np
import pytest
import xarray as xr

from cop_ard_app.interf import (
    extract_coreg_aoi,
    extract_interf_multi,
    extract_interf_single,
    get_interf_single_burst,
    get_mosaic_interf,
    mosaic_interf,
    update_interf_coordinates,
)
from cop_ard_app.utils import get_best_pol

N = 16
BURSTS = ["iw1_254", "iw1_255"]
MASTER = "2020-01-01"
MASTER2 = "2020-01-02"
SECOND = "2020-01-12"
SECOND2 = "2020-01-13"
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
POL = "VV"
TOL = 1e-9
ACQ_REF = {"iw1_254": MASTER, "iw1_255": MASTER2}
ACQ_SEC = {"iw1_254": SECOND, "iw1_255": SECOND2}
REF_OFFS = {"iw1_254": 0, "iw1_255": 1}
SEC_OFFS = {"iw1_254": 3, "iw1_255": 11}


def _pattern(n_az: int, n_rg: int, off: int) -> np.ndarray:
    """Analytic phase ramp bounded in [-0.4, 0.4] over a block grid."""
    ii, jj = np.indices((n_az, n_rg))
    return ((ii * 7 + jj * 3 + off) % 17 - 8) * 0.05


def _raw_phase(n_az: int, n_rg: int, ml: int, off: int) -> np.ndarray:
    """Pattern expanded to raw size, constant within each multilook window."""
    pat = _pattern((n_az + ml - 1) // ml, (n_rg + ml - 1) // ml, off)
    raw = np.repeat(np.repeat(pat, ml, axis=0), ml, axis=1)
    return raw[:n_az, :n_rg]


def _magnitude(n_az: int, n_rg: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1.0 + 0.2 * rng.standard_normal((n_az, n_rg))


def _annotations() -> dict:
    return {
        "Grid": {
            "TimeCOAPoly": {"Coefs": [[0.0]]},
            "Col": {"SS": 1.0},
        },
        "Timeline": {"CollectStart": "2020-01-01T00:00:00.000000Z"},
    }


def _offmap(off, pols: tuple[str, ...]) -> dict:
    if isinstance(off, dict):
        return {p: off.get(p, 0) for p in pols}
    return {p: off for p in pols}


def _coreg(n_az: int, n_rg: int, ml: int, seed, acq: str, offs: dict, pols: tuple[str, ...] = (POL,)) -> xr.Dataset:
    """coreg/measurements Dataset with a single cstack variable.

    ``offs`` maps each polarization to the offset of its phase pattern; ``seed=None`` gives unit magnitude.
    """
    data = []
    for k, pol in enumerate(pols):
        z = np.exp(1j * _raw_phase(n_az, n_rg, ml, offs.get(pol, 0)))
        if seed is not None:
            z = z * _magnitude(n_az, n_rg, seed + k)
        data.append(z)
    return xr.Dataset(
        {
            "cstack": (
                ("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
                np.stack(data)[:, None],
                {pol: _annotations() for pol in pols},
            )
        },
        coords={
            "polarization": list(pols),
            "acquisition_id": [acq],
            "azimuth_time": T0 + np.arange(n_az) * AZ_DT,
            "slant_range_time": np.arange(n_rg, dtype=float),
            "azimuth_m": ("azimuth_time", np.arange(n_az, dtype=float)),
            "slant_range_m": ("slant_range_time", np.arange(n_rg, dtype=float)),
        },
    )


def _orig(pols: tuple[str, ...] = (POL,)) -> xr.Dataset:
    """Original SLC node dataset; only its annotations are consumed downstream."""
    return xr.Dataset(
        {
            "slc": (
                ("azimuth_time", "slant_range_time"),
                np.ones((4, 4), dtype=complex),
                {pol: _annotations() for pol in pols},
            )
        }
    )


def _trees(
    n_az: int,
    n_rg: int,
    ml: int,
    off_ref: dict,
    off_sec: dict,
    acq_ref: dict | None = None,
    acq_sec: dict | None = None,
    pols_ref: tuple[str, ...] = (POL,),
    pols_sec: tuple[str, ...] = (POL,),
) -> tuple[xr.DataTree, xr.DataTree]:
    """Reference and secondary trees with one coreg per burst."""
    acq_ref = ACQ_REF if acq_ref is None else acq_ref
    acq_sec = ACQ_SEC if acq_sec is None else acq_sec
    ref, sec = xr.DataTree(), xr.DataTree()
    for bid in BURSTS:
        ref[f"bursts/{bid}/coreg/measurements"] = _coreg(
            n_az, n_rg, ml, None, acq_ref[bid], _offmap(off_ref[bid], pols_ref), pols_ref
        )
        sec[f"bursts/{bid}/coreg/measurements"] = _coreg(
            n_az, n_rg, ml, None, acq_sec[bid], _offmap(off_sec[bid], pols_sec), pols_sec
        )
        ref[f"bursts/{bid}/orig/{acq_ref[bid]}/measurements"] = _orig(pols_ref)
    return ref, sec


def _expected_phase(n_az: int, n_rg: int, ml: int, off_ref: int, off_sec: int, nb_az: int, nb_rg: int) -> np.ndarray:
    ref = _raw_phase(n_az, n_rg, ml, off_ref)
    sec = _raw_phase(n_az, n_rg, ml, off_sec)
    return (sec - ref)[::ml, ::ml][:nb_az, :nb_rg]


def _expected_coh(m: np.ndarray, ml: int, nb_az: int, nb_rg: int) -> np.ndarray:
    """Coherence of a unit-magnitude reference against magnitudes ``m`` per ml x ml window."""
    blk = m[: nb_az * ml, : nb_rg * ml].reshape(nb_az, ml, nb_rg, ml).swapaxes(1, 2).reshape(nb_az, nb_rg, ml * ml)
    return blk.mean(-1) / np.sqrt((blk**2).mean(-1))


def _kept(n_full: int, ml: int) -> int:
    """Rows kept by extract_coreg_aoi for a full image of size ``n_full``.

    The label-based inclusive slice ``sel(y=slice(y_min, y_max + 1))`` keeps
    ``y_max + 2`` rows when the image is 0-indexed.
    """
    return min(n_full, (n_full - 1) // ml * ml + 2)


def _chunklens(n: int, c: int) -> tuple:
    return (c,) * (n // c) + ((n % c,) if n % c else ())


# -----------------------------------------------------------------------------


class TestUpdateInterfCoordinates:
    @pytest.mark.unit
    def test_update_interf_coordinates_multilook(self):
        """Coarsened pixel-centre y/x coordinates are replaced by block indices."""
        for ml, nb in ((2, 7), (3, 5), (1, 4)):
            centres = np.arange(nb) * ml + (ml - 1) / 2
            interf = xr.DataArray(
                np.zeros((nb, nb)),
                dims=("azimuth_time", "slant_range_time"),
                coords={"y": ("azimuth_time", centres), "x": ("slant_range_time", centres)},
            ).to_dataset(name="phase")
            out = update_interf_coordinates(interf, {"azimuth_time": ml, "slant_range_time": ml})
            np.testing.assert_array_equal(out.y.values, np.arange(nb))
            np.testing.assert_array_equal(out.x.values, np.arange(nb))


class TestGetInterfSingleBurst:
    @pytest.mark.unit
    def test_get_interf_single_burst_multilook(self):
        """Constant-window phases and varied secondary magnitude give analytic phase and coherence."""
        ml = 2
        n = 8
        ref = xr.DataArray(
            np.exp(1j * _raw_phase(n, n, ml, 0)),
            dims=("azimuth_time", "slant_range_time"),
        )
        m_s = _magnitude(n, n, 7)
        sec = xr.DataArray(
            m_s * np.exp(1j * _raw_phase(n, n, ml, 3)),
            dims=("azimuth_time", "slant_range_time"),
        )
        res = get_interf_single_burst(ref, sec, {"azimuth_time": ml, "slant_range_time": ml})
        nb = n // ml
        assert res.sizes == {"azimuth_time": nb, "slant_range_time": nb}
        np.testing.assert_allclose(res.phase.data, _expected_phase(n, n, ml, 0, 3, nb, nb), atol=TOL)
        np.testing.assert_allclose(res.coherence.data, _expected_coh(m_s, ml, nb, nb), atol=TOL)

    @pytest.mark.unit
    def test_get_interf_single_burst_no_secondary(self):
        """All-zero secondary yields zero phase and zero coherence (nan filled)."""
        n = 6
        ref = xr.DataArray(np.ones((n, n), dtype=complex), dims=("azimuth_time", "slant_range_time"))
        sec = xr.DataArray(np.zeros((n, n), dtype=complex), dims=("azimuth_time", "slant_range_time"))
        res = get_interf_single_burst(ref, sec, {"azimuth_time": 2, "slant_range_time": 2})
        np.testing.assert_array_equal(res.phase.data, np.zeros((3, 3)))
        np.testing.assert_array_equal(res.coherence.data, np.zeros((3, 3)))


class TestExtractCoregAoi:
    @pytest.mark.unit
    def test_extract_coreg_aoi_multilook(self):
        """Trims rows/columns outside multilook windows and restores the input chunking."""
        coreg = _coreg(N, N, 1, 1, MASTER, {POL: 0}).chunk({"azimuth_time": 4, "slant_range_time": 8})
        for ml in (1, 2, 3, 4):
            kept = _kept(N, ml)
            out = extract_coreg_aoi(coreg, T0, 0.0, {"azimuth_time": ml, "slant_range_time": ml})
            assert out.cstack.sizes == {
                "polarization": 1,
                "acquisition_id": 1,
                "azimuth_time": kept,
                "slant_range_time": kept,
            }
            assert out.cstack.chunks == ((1,), (1,), _chunklens(kept, 4), _chunklens(kept, 8))
            np.testing.assert_array_equal(out.azimuth_time.values, T0 + np.arange(kept) * AZ_DT)
            np.testing.assert_array_equal(out.slant_range_time.values, np.arange(kept, dtype=float))


class TestExtractInterfSingle:
    @pytest.mark.unit
    def test_extract_interf_single_multilook(self):
        """Known phase ramp; each multilook block carries its exact expected phase."""
        for ml in (2, 1):
            ref = _coreg(N, N, ml, None, MASTER, {POL: 0})
            sec = _coreg(N, N, ml, 2, SECOND, {POL: 3})
            burst_id, interf = extract_interf_single(
                "iw1_254",
                "{}",
                ref.sel(polarization=POL),
                sec.sel(polarization=POL),
                POL,
                T0,
                0.0,
                {"azimuth_time": ml, "slant_range_time": ml},
            )
            assert burst_id == "iw1_254"
            nb = _kept(N, ml) // ml
            assert interf.sizes == {"acquisition_id": 1, "azimuth_time": nb, "slant_range_time": nb}
            np.testing.assert_allclose(
                interf.phase.isel(acquisition_id=0).data,
                _expected_phase(N, N, ml, 0, 3, nb, nb),
                atol=TOL,
            )
            np.testing.assert_allclose(
                interf.coherence.isel(acquisition_id=0).data,
                _expected_coh(_magnitude(N, N, 2), ml, nb, nb),
                atol=TOL,
            )
            centres = np.arange(nb) * ml + (ml - 1) / 2
            np.testing.assert_allclose((interf.azimuth_time.values - T0) / AZ_DT, centres)
            np.testing.assert_allclose(interf.slant_range_time.values, centres)
            np.testing.assert_array_equal(
                interf.azimuth_time_raw.values,
                np.full((nb, nb), T0, dtype="datetime64[ns]"),
            )


class TestExtractInterfMulti:
    @pytest.mark.unit
    def test_extract_interf_multi_multiburst(self):
        """Two bursts combined; each node holds its own coarsened phase stack."""
        ml = 2
        ref, sec = _trees(N, N, ml, REF_OFFS, SEC_OFFS)
        res = extract_interf_multi(
            ref,
            sec,
            multiburst=True,
            enable_dask=False,
            a_min=T0,
            r_min=0.0,
            multilook={"azimuth_time": ml, "slant_range_time": ml},
        )
        assert set(res.bursts) == set(BURSTS)
        nb = _kept(N, ml) // ml
        for k, bid in enumerate(BURSTS):
            node = res.bursts[bid].ds
            assert node.sizes == {"acquisition_id": 1, "azimuth_time": nb, "slant_range_time": nb}
            np.testing.assert_allclose(
                node.phase.isel(acquisition_id=0).data,
                _expected_phase(N, N, ml, REF_OFFS[bid], SEC_OFFS[bid], nb, nb),
                atol=TOL,
            )
            np.testing.assert_array_equal(node.id.data, np.full((nb, nb), k, dtype="int8"))

    @pytest.mark.unit
    def test_extract_interf_multi_singleburst(self):
        """Stripmap mode produces a single 'data' node at raw size (no coarsening)."""
        ref = xr.DataTree()
        ref["coreg/measurements"] = _coreg(N, N, 1, None, MASTER, {POL: 0})
        ref["orig/2020-01-01/measurements"] = _orig()
        sec = xr.DataTree()
        sec["coreg/measurements"] = _coreg(N, N, 1, 2, SECOND, {POL: 3})
        res = extract_interf_multi(
            ref,
            sec,
            multiburst=False,
            enable_dask=False,
            a_min=T0,
            r_min=0.0,
            multilook={"azimuth_time": 1, "slant_range_time": 1},
        )
        assert set(res.bursts) == {"data"}
        node = res.bursts["data"].ds
        assert node.sizes == {"acquisition_id": 1, "azimuth_time": N, "slant_range_time": N}
        np.testing.assert_allclose(
            node.phase.isel(acquisition_id=0).data,
            _expected_phase(N, N, 1, 0, 3, N, N),
            atol=TOL,
        )
        np.testing.assert_allclose(
            node.coherence.isel(acquisition_id=0).data,
            _expected_coh(_magnitude(N, N, 2), 1, N, N),
            atol=TOL,
        )
        np.testing.assert_array_equal(node.id.data, np.zeros((N, N), dtype="int8"))

    @pytest.mark.unit
    def test_extract_interf_multi_get_best_pol(self):
        """The best common polarization is selected and used for the phase extraction."""
        assert get_best_pol(["VV", "HH", "VH"]) == "VV"
        assert get_best_pol(["HH", "VH"]) == "HH"
        ml = 2
        pols_ref = ("VV", "HH")
        pols_sec = ("HH", "VH")
        off_ref = {"iw1_254": {"VV": 0, "HH": 1}, "iw1_255": {"VV": 2, "HH": 5}}
        off_sec = {"iw1_254": {"HH": 3, "VH": 9}, "iw1_255": {"HH": 11, "VH": 2}}
        ref, sec = _trees(N, N, ml, off_ref, off_sec, pols_ref=pols_ref, pols_sec=pols_sec)
        res = extract_interf_multi(
            ref,
            sec,
            multiburst=True,
            enable_dask=False,
            a_min=T0,
            r_min=0.0,
            multilook={"azimuth_time": ml, "slant_range_time": ml},
        )
        nb = _kept(N, ml) // ml
        for k, bid in enumerate(BURSTS):
            node = res.bursts[bid].ds
            np.testing.assert_allclose(
                node.phase.isel(acquisition_id=0).data,
                _expected_phase(N, N, ml, off_ref[bid]["HH"], off_sec[bid]["HH"], nb, nb),
                atol=TOL,
            )


class TestMosaicInterf:
    @pytest.mark.unit
    def test_mosaic_interf_margins(self):
        """Per-burst margins are cropped and different acquisitions are stacked."""
        interf = xr.DataTree()
        dims = ("acquisition_id", "azimuth_time", "slant_range_time")
        for k, bid in enumerate(BURSTS):
            phase = -0.4 if k == 0 else 0.65
            node = xr.Dataset(
                {
                    "phase": xr.DataArray(np.full((1, 13, 16), phase), dims=dims),
                    "coherence": xr.DataArray(np.ones((1, 13, 16)), dims=dims),
                    "id": xr.DataArray(np.full((13, 16), k, dtype=np.int8), dims=("azimuth_time", "slant_range_time")),
                    "azimuth_time_raw": xr.DataArray(np.full((13, 16), T0, dtype="datetime64[ns]"), dims=dims[1:]),
                },
                coords={
                    "acquisition_id": [SECOND if k == 0 else SECOND2],
                    "azimuth_time": np.arange(13),
                    "slant_range_time": np.arange(16, dtype=float),
                },
            )
            interf[f"bursts/{bid}"] = node
        res = mosaic_interf(interf, {"acquisition_id": 1, "azimuth_time": 4, "slant_range_time": 3})
        assert res.sizes == {"acquisition_id": 2, "azimuth_time": 7, "slant_range_time": 5}
        assert res.phase.chunks == ((1, 1), (4, 3), (3, 2))
        for k, exp_phase in enumerate((-0.4, 0.65)):
            np.testing.assert_allclose(res.phase.isel(acquisition_id=k).data, np.full((7, 5), exp_phase), atol=TOL)
            np.testing.assert_allclose(res.coherence.isel(acquisition_id=k).data, np.ones((7, 5)), atol=TOL)
        np.testing.assert_array_equal(res.id.data, np.zeros((7, 5), dtype="int8"))
        np.testing.assert_array_equal(np.asarray(res.coords["azimuth_time"].values), np.arange(3, 10))
        np.testing.assert_array_equal(np.asarray(res.coords["slant_range_time"].values), np.arange(3, 8, dtype=float))


class TestGetMosaicInterf:
    @pytest.mark.unit
    def test_get_mosaic_interf_multiburst(self):
        """End-to-end mosaic: cropped margins of the coarsened stacks, per-burst phases."""
        ml = 2
        n_az, n_rg = 32, 33
        ref, sec = _trees(n_az, n_rg, ml, REF_OFFS, SEC_OFFS)
        res = get_mosaic_interf(
            ref,
            sec,
            multiburst=True,
            chunks={"acquisition_id": 1, "azimuth_time": 4, "slant_range_time": 3},
            enable_dask=False,
            a_min=T0,
            r_min=0.0,
            multilook={"azimuth_time": ml, "slant_range_time": ml},
        )
        nb_az = _kept(n_az, ml) // ml
        nb_rg = _kept(n_rg, ml) // ml
        assert res.sizes == {"acquisition_id": 2, "azimuth_time": 10, "slant_range_time": 5}
        for k, bid in enumerate(BURSTS):
            exp = _expected_phase(n_az, n_rg, ml, REF_OFFS[bid], SEC_OFFS[bid], nb_az, nb_rg)[3:13, 3:8]
            np.testing.assert_allclose(res.phase.isel(acquisition_id=k).data, exp, atol=TOL)
        np.testing.assert_array_equal(res.id.data, np.zeros((10, 5), dtype="int8"))

    @pytest.mark.unit
    def test_get_mosaic_interf_singleburst(self):
        """Stripmap mosaic: one 'data' node at raw size, cropped margins."""
        ref = xr.DataTree()
        ref["coreg/measurements"] = _coreg(N, N, 1, None, MASTER, {POL: 0})
        ref["orig/2020-01-01/measurements"] = _orig()
        sec = xr.DataTree()
        sec["coreg/measurements"] = _coreg(N, N, 1, 2, SECOND, {POL: 3})
        res = get_mosaic_interf(
            ref,
            sec,
            multiburst=False,
            chunks={},
            enable_dask=False,
            a_min=T0,
            r_min=0.0,
            multilook={"azimuth_time": 1, "slant_range_time": 1},
        )
        assert res.sizes == {"acquisition_id": 1, "azimuth_time": 10, "slant_range_time": 5}
        np.testing.assert_allclose(
            res.phase.isel(acquisition_id=0).data,
            _expected_phase(N, N, 1, 0, 3, N, N)[3:13, 3:8],
            atol=TOL,
        )
        np.testing.assert_array_equal(res.id.data, np.zeros((10, 5), dtype="int8"))
