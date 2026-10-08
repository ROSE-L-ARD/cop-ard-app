import numpy as np
import pytest
import xarray as xr

from cop_ard_app.cmlp import _correct_drift, _prepare_cstack, _update_stable_reference, single_cmlp, stack_cmlp
from cop_ard_app.io import Path, save_empty_dataset

N = 16
POL = "VV"
BURSTS = ["iw1_254", "iw1_255"]
REFERENCE_ID = "ref-1"
DAYS = ["2020-01-01", "2020-01-02", "2020-01-03"]
DATE_NS = np.array([np.datetime64(d, "ns") for d in DAYS])
DATE_STR = [str(d) for d in DATE_NS]
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
POSTING = {"azimuth_time": 1, "slant_range_time": 1}
CHUNKS = {"azimuth_time": 4, "slant_range_time": 4}
SHORT_CHUNKS = {"polarization": 1, "azimuth_time": 4, "slant_range_time": 4}
SHORT_DIMS = ["polarization", "azimuth_time", "slant_range_time"]
STABLE_DIMS = ["acquisition_date", *SHORT_DIMS]
MEMORY = 0.5
COH_MIN = 0.5
PHASE_TOL = 1e-6
TOL = 1e-5
PHASE_DRIFT = 0.7

# With posting 1 (multilook 2), the normalized interferogram keeps N - 1
# pixels per direction and the block-based corrections leave a one-pixel
# margin at the end of the reference grid untouched.
COARSE = N - 1


def _annotations():
    return {
        "Grid": {
            "Col": {"SS": 1.0, "DeltaKCOAPoly": {"Coefs": [[0.0]]}},
            "Row": {"SS": 1.0},
        },
        "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0]}}},
    }


def _make_cstack(dates, values):
    """Full multi-date cstack as found in coreg/measurements, dask-chunked."""
    data = np.stack([np.full((N, N), v, dtype=np.complex64) for v in values])[np.newaxis]
    return xr.DataArray(
        data,
        dims=["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
        coords={
            "polarization": [POL],
            "acquisition_id": list(dates),
            "acquisition_date": ("acquisition_id", [np.datetime64(d, "ns") for d in dates]),
            "azimuth_time": T0 + np.arange(N) * AZ_DT,
            "slant_range_time": np.arange(N, dtype=np.float64),
            "azimuth_m": ("azimuth_time", np.arange(N, dtype=np.float64)),
            "slant_range_m": ("slant_range_time", np.arange(N, dtype=np.float64)),
        },
        name="cstack",
        attrs=_annotations(),
    ).chunk(CHUNKS)


def _make_tree(**burst_values):
    slcs = xr.DataTree()
    for burst_id, value in burst_values.items():
        slcs[f"bursts/{burst_id}/coreg/measurements"] = xr.Dataset({"cstack": _make_cstack(DAYS, [value] * len(DAYS))})
    return slcs


def _short(data):
    """Single-plane reference DataArray over the short dimensions."""
    return xr.DataArray(
        np.asarray(data, dtype=np.complex64),
        dims=SHORT_DIMS,
        coords={
            "polarization": [POL],
            "azimuth_time": T0 + np.arange(N) * AZ_DT,
            "slant_range_time": np.arange(N, dtype=np.float64),
        },
        name="cstack",
    )


def _reference(value):
    return _short(np.full((1, N, N), value, dtype=np.complex64))


def _stable_stack(per_date, dates):
    """Stack of references along the stable-reference (past) acquisition dates."""
    data = np.stack([np.broadcast_to(np.asarray(v, dtype=np.complex64), (1, N, N)).copy() for v in per_date])
    return xr.DataArray(
        data,
        dims=STABLE_DIMS,
        coords={"acquisition_date": np.asarray(dates, dtype="datetime64[ns]")},
        name="cstack",
    )


def _phase_ramp():
    """Analytic unit-amplitude image with a spatial phase ramp."""
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    return (np.exp(1j * (0.1 * i + 0.05 * j))).astype(np.complex64)[np.newaxis]


def _reference_state(value, steps):
    """Expected reference content after folding `steps` identical SLCs into it.

    The recursive update multiplies the whole grid by memory_param at each step
    and adds the (phase-aligned) SLC back into the valid block only, so the
    valid pixels converge towards value / (1 - MEMORY) while the one-pixel
    margin decays as value * MEMORY ** steps.
    """
    valid = value * (1 - MEMORY ** (steps + 1)) / (1 - MEMORY)
    margin = value * MEMORY**steps
    return valid, margin


class TestPrepareCstack:
    """Unit test for cop_ard_app.cmlp._prepare_cstack."""

    @pytest.mark.unit
    def test_prepare_cstack_sorts_dates_and_keeps_chunks(self):
        # dates given out of order, each plane tagged with its index in DAYS
        dates = [DAYS[1], DAYS[0], DAYS[2]]
        markers = [1.0, 0.0, 2.0]
        sel, chunks = _prepare_cstack(_make_cstack(dates, markers))

        assert chunks == SHORT_CHUNKS
        assert np.array_equal(sel.acquisition_date.values, DATE_NS)
        planes = sel.values[:, :, 0, 0]
        assert planes.shape == (1, len(DAYS))
        assert np.all(np.abs(planes - np.arange(len(DAYS))) < TOL)
        # auxiliary coordinates are not dimension coordinates and get dropped,
        # while the dimensions and the SICD annotations are preserved
        assert "azimuth_m" not in sel.coords
        assert "slant_range_m" not in sel.coords
        assert "acquisition_date" in sel.coords
        assert sel.dims == ("polarization", "acquisition_id", "azimuth_time", "slant_range_time")
        assert dict(sel.attrs) == _annotations()


class TestCorrectDrift:
    """Unit test for cop_ard_app.cmlp._correct_drift."""

    @pytest.mark.unit
    def test_correct_drift_recovers_a_known_phase_drift(self):
        base = _phase_ramp()
        drifted = (np.exp(1j * PHASE_DRIFT) * base).astype(np.complex64)
        current = _short(drifted)

        stable = _stable_stack([base, base], DATE_NS[1:])

        corrected = _correct_drift(
            stable_reference=stable,
            current_reference=current,
            posting=POSTING,
            coherence_threshold=0.1,
            shape=(1, COARSE, COARSE),
            output_path="/unused",
            save_metrics=False,
        )

        assert corrected.shape == current.shape
        assert corrected.dims == current.dims
        # the drift phasor is recovered as exp(1j * PHASE_DRIFT) and removed
        assert np.abs(corrected.values[:, :COARSE, :COARSE] - base[:, :COARSE, :COARSE]).max() < TOL
        # the untouched margin still holds the drifted values
        assert np.abs(corrected.values[:, COARSE:, :] - drifted[:, COARSE:, :]).max() < TOL
        assert np.abs(corrected.values[:, :, COARSE:] - drifted[:, :, COARSE:]).max() < TOL

    @pytest.mark.unit
    def test_correct_drift_leaves_the_reference_untouched_for_incoherent_history(self):
        base = _phase_ramp()
        drifted = (np.exp(1j * PHASE_DRIFT) * base).astype(np.complex64)
        current = _short(drifted)

        # a ±1 pattern alternating along azimuth sums to zero in every 2x2
        # multilook window, so every stable reference is fully incoherent
        checker = (1 - 2 * (np.arange(N) % 2)).astype(np.complex64)
        stable = _stable_stack([(checker * base)[0]] * 2, DATE_NS[1:])

        corrected = _correct_drift(
            stable_reference=stable,
            current_reference=current,
            posting=POSTING,
            coherence_threshold=0.9,
            shape=(1, COARSE, COARSE),
            output_path="/unused",
            save_metrics=False,
        )

        assert np.abs(corrected.values - current.values).max() == 0.0


class TestUpdateStableReference:
    """Unit test for cop_ard_app.cmlp._update_stable_reference."""

    @pytest.mark.unit
    def test_update_stable_reference_writes_and_returns_the_current_state(self, tmp_path):
        value = 0.6 + 0.8j
        date = DATE_NS[1]
        data_path = str(tmp_path / "data")
        save_empty_dataset(
            data_vars=["cstack"],
            dims={"cstack": SHORT_DIMS},
            coords={
                "polarization": [POL],
                "azimuth_time": T0 + np.arange(N) * AZ_DT,
                "slant_range_time": np.arange(N, dtype=np.float64),
                "acquisition_date": date,
            },
            shape={"cstack": [1, N, N]},
            chunks=SHORT_CHUNKS,
            dtype={"cstack": "complex64"},
            data_path=f"{data_path}/stable_reference/{date}",
            skip_read=True,
            storage_options=None,
        )

        updated = _update_stable_reference(
            current_reference=_reference(value),
            date=date,
            data_path=data_path,
            chunks=SHORT_CHUNKS,
            initial_stable_reference=None,
        )

        assert updated.dims == ("acquisition_date", *SHORT_DIMS)
        assert updated.shape == (1, 1, N, N)
        assert np.array_equal(updated.acquisition_date.values, [date])
        assert np.abs(updated.values[0] - value).max() < TOL

    @pytest.mark.unit
    def test_update_stable_reference_concatenates_initial_history(self, tmp_path):
        data_path = str(tmp_path / "data")
        for date in DATE_NS[1:]:
            save_empty_dataset(
                data_vars=["cstack"],
                dims={"cstack": SHORT_DIMS},
                coords={
                    "polarization": [POL],
                    "azimuth_time": T0 + np.arange(N) * AZ_DT,
                    "slant_range_time": np.arange(N, dtype=np.float64),
                    "acquisition_date": date,
                },
                shape={"cstack": [1, N, N]},
                chunks=SHORT_CHUNKS,
                dtype={"cstack": "complex64"},
                data_path=f"{data_path}/stable_reference/{date}",
                skip_read=True,
                storage_options=None,
            )

        updated = _update_stable_reference(
            current_reference=_reference(0.6 + 0.8j),
            date=DATE_NS[2],
            data_path=data_path,
            chunks=SHORT_CHUNKS,
            initial_stable_reference=_stable_stack([1.0 + 0.0j], [DATE_NS[1]]),
        )

        assert updated.shape == (2, 1, N, N)
        assert np.array_equal(updated.acquisition_date.values, DATE_NS[1:])
        assert np.abs(updated.values[0] - 1.0).max() < TOL
        assert np.abs(updated.values[1] - (0.6 + 0.8j)).max() < TOL


class TestSingleCmlp:
    """Unit test for cop_ard_app.cmlp.single_cmlp.

    Feeds a stack of identical constant-phase SLCs: the normalized interferogram
    is then exactly 1, so the metrics are exact and the recursive reference
    accumulates the geometric series value * (1 + MEMORY + MEMORY**2 + ...).
    """

    @pytest.mark.unit
    def test_single_cmlp_constant_stack_recovers_reference_state(self, tmp_path):
        value = 0.6 + 0.8j
        burst_id, cmlp, current_reference, stable_reference = single_cmlp(
            BURSTS[0],
            _make_cstack(DAYS, [value] * len(DAYS)),
            reference_database_path=Path(str(tmp_path / "references")),
            posting=POSTING,
            multiburst=True,
            working_directory=str(tmp_path / "work"),
            memory_param=MEMORY,
            coherence_threshold=0.1,
            stable_interval_days=1,
            save_metrics=True,
        )

        assert burst_id == BURSTS[0]
        assert [str(d) for d, _ in cmlp] == DATE_STR
        for date, ds in cmlp:
            coh = ds["short_term_coh"].values
            phase = ds["phase"].values
            assert coh.shape == (1, COARSE, COARSE)
            assert coh.min() > COH_MIN
            assert np.abs(phase).max() < PHASE_TOL

        expected, margin = _reference_state(value, len(DAYS))
        assert current_reference.shape == (1, N, N)
        assert np.abs(current_reference.values[:, :COARSE, :COARSE] - expected).max() < TOL
        assert np.abs(current_reference.values[:, COARSE:, :] - margin).max() < TOL
        assert np.abs(current_reference.values[:, :, COARSE:] - margin).max() < TOL
        assert current_reference.attrs["starting_date"] == DATE_STR[0]

        # with one-day spacing and a one-day interval every date but the first
        # becomes a stable reference milestone
        assert stable_reference is not None
        assert np.array_equal(stable_reference.acquisition_date.values, DATE_NS[1:])
        expected_second, _ = _reference_state(value, 2)
        assert np.abs(stable_reference.values[0, :, :COARSE, :COARSE] - expected_second).max() < TOL

    @pytest.mark.unit
    def test_single_cmlp_rejects_wrong_posting_keys(self, tmp_path):
        with pytest.raises(ValueError, match="Wrong keys"):
            single_cmlp(
                BURSTS[0],
                _make_cstack(DAYS, [1.0] * len(DAYS)),
                reference_database_path=Path(str(tmp_path / "references")),
                posting={"azimuth": 1, "slant_range": 1},
                multiburst=True,
                working_directory=str(tmp_path / "work"),
            )

    @pytest.mark.unit
    def test_single_cmlp_requires_reference_id_for_single_burst(self, tmp_path):
        with pytest.raises(ValueError, match="reference_id is mandatory"):
            single_cmlp(
                BURSTS[0],
                _make_cstack(DAYS, [1.0] * len(DAYS)),
                reference_database_path=Path(str(tmp_path / "references")),
                posting=POSTING,
                multiburst=False,
                working_directory=str(tmp_path / "work"),
            )


class TestStackCmlp:
    """Unit test for cop_ard_app.cmlp.stack_cmlp."""

    @pytest.mark.unit
    def test_stack_cmlp_multiburst_builds_one_group_per_burst_and_date(self, tmp_path):
        burst_values = {BURSTS[0]: 1.0 + 0.0j, BURSTS[1]: 0.6 + 0.8j}
        cmlp_tree, current_reference_tree, stable_references_tree = stack_cmlp(
            _make_tree(**burst_values),
            POSTING,
            str(tmp_path / "work"),
            reference_database_path=Path(str(tmp_path / "references")),
            memory_param=MEMORY,
            coherence_threshold=0.1,
            stable_interval_days=1,
            enable_dask=False,
            save_metrics=True,
        )

        for burst_id, value in burst_values.items():
            expected, margin = _reference_state(value, len(DAYS))
            current_reference = current_reference_tree[f"/bursts/{burst_id}/measurements/cstack"].to_dataset()["cstack"]
            assert current_reference.shape == (1, N, N)
            assert np.abs(current_reference.values[:, :COARSE, :COARSE] - expected).max() < TOL
            assert np.abs(current_reference.values[:, COARSE:, :] - margin).max() < TOL
            assert f"/bursts/{burst_id}/measurements" in stable_references_tree.groups

            expected_groups = {f"/bursts/{burst_id}/cmlp/{date}" for date in DATE_STR}
            assert expected_groups <= set(cmlp_tree.groups)
            for group in sorted(expected_groups):
                ds = cmlp_tree[group].to_dataset()
                coh = ds["short_term_coh"].values
                assert coh.shape == (1, COARSE, COARSE)
                assert coh.min() > COH_MIN
                assert np.abs(ds["phase"].values).max() < PHASE_TOL

    @pytest.mark.unit
    def test_stack_cmlp_marks_empty_stable_references_with_placeholder(self, tmp_path):
        cmlp_tree, _current_reference_tree, stable_references_tree = stack_cmlp(
            _make_tree(**{BURSTS[0]: 1.0 + 0.0j}),
            POSTING,
            str(tmp_path / "work"),
            reference_database_path=Path(str(tmp_path / "references")),
            memory_param=MEMORY,
            # a 365-day interval is never reached by one-day-spaced dates
            stable_interval_days=365,
            enable_dask=False,
            save_metrics=True,
        )

        assert set(stable_references_tree.groups) == {"/", "/status"}
        assert stable_references_tree["status"].attrs["is_empty"] is True
        assert "long_term_coh" not in cmlp_tree[f"/bursts/{BURSTS[0]}/cmlp/{DATE_STR[0]}"].to_dataset()

    @pytest.mark.unit
    def test_stack_cmlp_single_burst_uses_reference_id_for_paths(self, tmp_path):
        slcs = xr.DataTree()
        slcs["coreg/measurements"] = xr.Dataset({"cstack": _make_cstack(DAYS, [0.6 + 0.8j] * len(DAYS))})
        value = 0.6 + 0.8j

        cmlp_tree, current_reference_tree, _ = stack_cmlp(
            slcs,
            POSTING,
            str(tmp_path / "work"),
            reference_database_path=Path(str(tmp_path / "references")),
            memory_param=MEMORY,
            stable_interval_days=365,
            enable_dask=False,
            reference_id=REFERENCE_ID,
        )

        assert {f"/cmlp/{date}" for date in DATE_STR} <= set(cmlp_tree.groups)
        assert "/measurements" in current_reference_tree.groups

        expected, margin = _reference_state(value, len(DAYS))
        current_reference = current_reference_tree["/measurements/cstack"].to_dataset()["cstack"]
        assert np.abs(current_reference.values[:, :COARSE, :COARSE] - expected).max() < TOL
        assert np.abs(current_reference.values[:, COARSE:, :] - margin).max() < TOL
