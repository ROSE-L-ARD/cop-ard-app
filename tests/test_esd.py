import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter

from cop_ard_app.esd import estimate

N = 64
OVERLAP = 16  # realistic partial overlap (16 out of 64 azimuth lines = 25%)
BURSTS = ["iw1_254", "iw1_255"]
DOPPLER = {"iw1_254": 10.0, "iw1_255": 12.0}
MASTER = "2020-01-01"
SECONDARIES = ["2020-01-12"]
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
POL = "VV"
TOL = 1e-6


def _annotations(f_k, ss_az=1.0, ss_rg=1.0):
    return {
        "Grid": {
            "Col": {"SS": ss_rg, "DeltaKCOAPoly": {"Coefs": [[f_k]]}},
            "Row": {"SS": ss_az},
        },
        "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0]}}},
    }


def _exact_bandlimited_shift(sig: np.ndarray, delta_az: float) -> np.ndarray:
    """Shift a 2D complex signal along azimuth (axis 0) by `delta_az` pixels via FFT."""
    freqs = np.fft.fftfreq(sig.shape[0])[:, None]
    shifted = np.fft.ifft(np.fft.fft(sig, axis=0) * np.exp(-2j * np.pi * freqs * delta_az), axis=0)
    return shifted.astype(np.complex64)


def _build_overlapping_trees(
    deltas_by_date,
    burst_ids=("iw1_254", "iw1_255", "iw1_256"),
    doppler_map=None,
    realistic_shift=False,
    corrupt_half_overlap=False,
):
    """Build DataTrees with partial burst overlap and shared physical ground scatterers."""
    if doppler_map is None:
        doppler_map = {"iw1_254": 10.0, "iw1_255": 12.0, "iw1_256": 14.0}

    rng = np.random.default_rng(42)
    step = N - OVERLAP
    total_az = step * (len(burst_ids) - 1) + N

    if realistic_shift:
        # Smooth bandlimited scene so FFT sub-pixel shifting has negligible wrap-around at burst edges
        scene_re = gaussian_filter(rng.standard_normal((total_az, N)), sigma=1.5)
        scene_im = gaussian_filter(rng.standard_normal((total_az, N)), sigma=1.5)
        global_scene = (scene_re + 1j * scene_im).astype(np.complex64)
    else:
        # Unit-modulus complex speckle
        raw = rng.standard_normal((total_az, N)) + 1j * rng.standard_normal((total_az, N))
        global_scene = (raw / np.abs(raw)).astype(np.complex64)

    sec_dates = list(deltas_by_date.keys())
    reference_slcs = xr.DataTree()
    secondary_slcs = xr.DataTree()

    for b_idx, burst_id in enumerate(burst_ids):
        start_idx = b_idx * step
        f_k = doppler_map[burst_id]
        t = np.arange(start_idx, start_idx + N, dtype=np.float64)
        burst_scene = global_scene[start_idx : start_idx + N]

        # Reference (master) burst at zero shift
        ref_signal = (burst_scene * np.exp(2j * np.pi * f_k * t[:, None])).astype(np.complex64)
        ref_data = ref_signal[None, None, :, :]

        # Secondary acquisitions shifted in azimuth by each date's delta
        sec_data = np.empty((1, len(sec_dates), N, N), dtype=np.complex64)
        for a_idx, date in enumerate(sec_dates):
            delta = deltas_by_date[date]
            if realistic_shift:
                sec_img = _exact_bandlimited_shift(ref_signal, delta)
            else:
                sec_img = (burst_scene * np.exp(2j * np.pi * f_k * (t - delta)[:, None])).astype(np.complex64)

            if corrupt_half_overlap:
                # Replace left half of range with uncorrelated noise (simulates water / decorrelation)
                noise = rng.standard_normal((N, N // 2)) + 1j * rng.standard_normal((N, N // 2))
                sec_img[:, : N // 2] = (noise / np.abs(noise)).astype(np.complex64)

            sec_data[0, a_idx] = sec_img

        coords = {
            "polarization": [POL],
            "azimuth_time": T0 + np.arange(start_idx, start_idx + N) * AZ_DT,
            "slant_range_time": np.arange(N, dtype=np.float64),
            "azimuth_m": ("azimuth_time", np.arange(start_idx, start_idx + N, dtype=np.float64)),
            "slant_range_m": ("slant_range_time", np.arange(N, dtype=np.float64)),
        }

        reference_slcs[f"bursts/{burst_id}/coreg/measurements"] = xr.Dataset(
            {
                "cstack": (
                    ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
                    ref_data,
                    {POL: _annotations(f_k)},
                )
            },
            coords={**coords, "acquisition_id": [MASTER], "acquisition_date": ("acquisition_id", [MASTER])},
        )
        secondary_slcs[f"bursts/{burst_id}/coreg/measurements"] = xr.Dataset(
            {
                "cstack": (
                    ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
                    sec_data,
                    {POL: _annotations(f_k)},
                )
            },
            coords={**coords, "acquisition_id": sec_dates, "acquisition_date": ("acquisition_id", sec_dates)},
        )

    return reference_slcs, secondary_slcs


class TestEstimate:
    """Unit tests for cop_ard_app.esd.estimate."""

    @pytest.mark.unit
    @pytest.mark.parametrize("delta", [-0.015, 0.0, 0.005, 0.02])
    def test_partial_overlap_positive_negative_and_zero_shifts(self, delta):
        """ESD must extract the partial overlap window between bursts and recover signed shifts."""
        ref_tree, sec_tree = _build_overlapping_trees(
            deltas_by_date={"2020-01-12": delta},
            burst_ids=("iw1_254", "iw1_255"),
        )

        esd = estimate(
            ref_tree,
            sec_tree,
            poly_degree=0,
            enable_dask=False,
            client_options={"type": "DUMMY"},
        )

        shifts = esd["mean_shift"].values
        coherence = esd["coherence"].values
        mean_phase = esd["mean_phase"].values

        assert shifts.shape == (1, 1)
        np.testing.assert_allclose(np.abs(shifts), abs(delta), atol=TOL)
        # Effective coherence weight scales as sqrt(OVERLAP) = sqrt(16) = 4.0 for noiseless data
        np.testing.assert_allclose(coherence, np.sqrt(OVERLAP), atol=1e-4)
        # Double-difference phase magnitude must equal 2 * pi * |f1 - f0| * |delta|
        expected_phase_mag = 2.0 * np.pi * (12.0 - 10.0) * abs(delta)
        np.testing.assert_allclose(np.abs(mean_phase), expected_phase_mag, atol=1e-4)

    @pytest.mark.unit
    def test_multi_burst_and_multi_acquisition_stack(self):
        """ESD across 3 consecutive bursts (2 overlap zones) and 3 dates recovers per-date shifts."""
        deltas = {
            "2020-01-12": 0.008,
            "2020-01-24": -0.012,
            "2020-02-05": 0.018,
        }
        ref_tree, sec_tree = _build_overlapping_trees(
            deltas_by_date=deltas,
            burst_ids=("iw1_254", "iw1_255", "iw1_256"),
        )

        esd = estimate(
            ref_tree,
            sec_tree,
            poly_degree=0,
            enable_dask=False,
            client_options={"type": "DUMMY"},
        )

        # Shape is (n_overlap_pairs=2, n_acquisitions=3)
        shifts = esd["mean_shift"].values
        expected = np.broadcast_to(np.array(list(deltas.values())), shifts.shape)
        np.testing.assert_allclose(np.abs(shifts), np.abs(expected), atol=TOL)
        # Verify sign consistency across opposite-sign acquisitions
        assert np.all(np.sign(shifts[:, 0]) == -np.sign(shifts[:, 1]))
        assert np.all(np.sign(shifts[:, 0]) == np.sign(shifts[:, 2]))

    @pytest.mark.unit
    def test_realistic_subpixel_complex_signal_shift(self):
        """Spatially shifting a baseband complex SAR signal in azimuth recovers the sub-pixel offset."""
        delta = 0.015
        # Use baseband Doppler centroids within Nyquist [-0.5, 0.5] cycles/pixel to avoid aliasing
        baseband_doppler = {"iw1_254": -0.15, "iw1_255": 0.20}
        ref_tree, sec_tree = _build_overlapping_trees(
            deltas_by_date={"2020-01-12": delta},
            burst_ids=("iw1_254", "iw1_255"),
            doppler_map=baseband_doppler,
            realistic_shift=True,
        )

        esd = estimate(
            ref_tree,
            sec_tree,
            poly_degree=0,
            enable_dask=False,
            client_options={"type": "DUMMY"},
        )

        shifts = esd["mean_shift"].values
        np.testing.assert_allclose(np.abs(shifts), abs(delta), atol=5e-4)
        assert np.all(esd["coherence"].values > 3.5)

    @pytest.mark.unit
    def test_low_coherence_masking_in_overlap(self):
        """Decorrelated noise in half of the overlap must not corrupt the estimated shift."""
        delta = 0.012
        ref_tree, sec_tree = _build_overlapping_trees(
            deltas_by_date={"2020-01-12": delta},
            burst_ids=("iw1_254", "iw1_255"),
            corrupt_half_overlap=True,
        )

        esd = estimate(
            ref_tree,
            sec_tree,
            poly_degree=0,
            enable_dask=False,
            client_options={"type": "DUMMY"},
        )

        shifts = esd["mean_shift"].values
        np.testing.assert_allclose(np.abs(shifts), abs(delta), atol=2e-3)
        # With 50% of the overlap decorrelated, effective coherence weight drops by sqrt(0.5)
        np.testing.assert_allclose(esd["coherence"].values, np.sqrt(OVERLAP * 0.5), atol=0.1)
