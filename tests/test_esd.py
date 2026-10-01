import numpy as np
import pytest
import xarray as xr

from cop_ard_app.esd import estimate

N = 64
BURSTS = ["iw1_254", "iw1_255"]
DOPPLER = {"iw1_254": 10.0, "iw1_255": 12.0}
MASTER = "2020-01-01"
SECONDARIES = ["2020-01-12"]
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
POL = "VV"
TOL = 1e-6


def _annotations(f_k):
    return {
        "Grid": {
            "Col": {"SS": 1.0, "DeltaKCOAPoly": {"Coefs": [[f_k]]}},
            "Row": {"SS": 1.0},
        },
        "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0]}}},
    }


def _make_dataset(burst_id, dates, speckle, delta):
    """Build a coreg/measurements Dataset for one burst.

    The secondary image at azimuth pixel i is c * exp(2j*pi*f_k*i); when delta is
    not None the reference (master) image is the same speckle shifted in azimuth
    by delta pixels, i.e. c * exp(2j*pi*f_k*(i-delta)). The normalized
    interferogram between the two is then exactly exp(2j*pi*f_k*delta), so the
    ESD between two bursts with Doppler centroids f0 and f1 yields
    mean_phase = 2*pi*(f1-f0)*delta and mean_shift = delta / SS.
    """
    f_k = DOPPLER[burst_id]
    n = N
    i = np.arange(n, dtype=np.float64)
    d = delta if delta is not None else 0.0
    phase = np.exp(2j * np.pi * f_k * (i - d)[:, None])
    data = np.empty((1, len(dates), n, n), dtype=np.complex64)
    for a in range(len(dates)):
        data[0, a] = (speckle * phase).astype(np.complex64)

    return xr.Dataset(
        {
            "cstack": (
                ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
                data,
                {POL: _annotations(f_k)},
            )
        },
        coords={
            "polarization": [POL],
            "acquisition_id": list(dates),
            "acquisition_date": ("acquisition_id", list(dates)),
            "azimuth_time": T0 + np.arange(n) * AZ_DT,
            "slant_range_time": np.arange(n, dtype=np.float64),
            "azimuth_m": ("azimuth_time", np.arange(n, dtype=np.float64)),
            "slant_range_m": ("slant_range_time", np.arange(n, dtype=np.float64)),
        },
    )


def _build_trees(delta):
    rng = np.random.default_rng(42)
    speckle = {
        b: (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex64)
        for b in BURSTS
    }

    reference_slcs = xr.DataTree()
    secondary_slcs = xr.DataTree()
    for burst_id in BURSTS:
        reference_slcs[f"bursts/{burst_id}/coreg/measurements"] = _make_dataset(
            burst_id, [MASTER], speckle[burst_id], delta
        )
        secondary_slcs[f"bursts/{burst_id}/coreg/measurements"] = _make_dataset(
            burst_id, SECONDARIES, speckle[burst_id], None
        )

    return reference_slcs, secondary_slcs



class TestEstimate():
    """Unit test for cop_ard_app.esd.estimate.

    Simulates a pair of consecutive bursts (iw1_254, iw1_255) whose master (reference)
    images are shifted in azimuth by a sub-pixel amount (0.005 and 0.01 pixel) with
    respect to the secondary images, and verifies that estimate() recovers the shift
    within an interval around the true value.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("delta", [0.005, 0.01])
    def test_estimate_recovers_master_azimuth_shift(self, delta):
        reference_slcs, secondary_slcs = _build_trees(delta)

        esd = estimate(
            reference_slcs,
            secondary_slcs,
            poly_degree=0,
            enable_dask=False,
            client_options={"type": "DUMMY"},
        )

        shifts = esd["mean_shift"].values
        assert shifts.ndim == 2
        assert shifts.size == len(SECONDARIES)
        assert np.all(np.isfinite(shifts))
        # the estimated residual shift must fall within an interval around the
        # shift imposed on the master burst
        assert np.all(np.abs(shifts - delta) < TOL)
        # sanity checks on the companion outputs
        assert np.all(esd["coherence"].values > 0)
        assert np.all(np.isfinite(esd["mean_phase"].values))
