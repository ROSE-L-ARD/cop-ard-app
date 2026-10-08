import numpy as np
import pytest
import xarray as xr

from cop_ard_app.io import Path
from cop_ard_app.split_spectrum import (
    get_phase_corr_from_x_poly,
    remove_iono_surface,
    remove_iono_surface_chunk,
    remove_iono_surface_single_burst,
    separate_dispersive_spectrum_components,
    separate_dispersive_spectrum_components_delta_mean,
    split_spectrum,
)

N = 32
AZ = 32
# sub-band centres in range-FFT frequency units; bin q maps to coordinate
# q/31 (q < 16) or (q - 32)/31 (q >= 16), so the bands at +/-F_OFF pick the
# spectrum peaks at bins Q_LO and Q_HI respectively
F_OFF = 2.0 / 31.0
BW = 1.0 / 31.0
Q_LO, Q_HI = 30, 2
F0 = 16.0 / 31.0
F_L = F0 - F_OFF
F_H = F0 + F_OFF
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
POL = "VV"
MASTER = "2020-01-01"
SECOND = "2020-01-12"
DELTA_LO = 0.005
DELTA_HI = (9.0 / 7.0) * 15.0 * DELTA_LO
C_INJ = 0.05
C_SURF = 0.3
COHE_TH = 0.5
ML = {"azimuth_time": 1, "slant_range_time": 1}
TOL = 1e-9
# analytic phases of the shifted dual-ramp interferogram in each sub-band
PH_L = 2.0 * np.pi * Q_LO * DELTA_LO / N
PH_H = 2.0 * np.pi * Q_HI * DELTA_HI / N
# analytic split-spectrum outputs for those phases (split_spectrum_core)
C_IO_RAMP = (PH_L * (F_H / F0) - PH_H * (F_L / F0)) * F_L * F_H / (F_H**2 - F_L**2)
C_ND_RAMP = (PH_H * (F_H / F0) - PH_L * (F_L / F0)) * F0 / (2.0 * (F_H - F_L))
# analytic delta/mean phases for those phases (joint_unwrap_delta_mean)
DPHI_RAMP = PH_H - PH_L
PHBAR_RAMP = (PH_H + PH_L) / 2.0
# analytic split-spectrum outputs for the injected constant x_poly correction
K_INJ = 2.0 * np.pi * F_OFF * C_INJ
C_IO_INJ = (K_INJ * (F_H / F0) - (-K_INJ) * (F_L / F0)) * F_L * F_H / (F_H**2 - F_L**2)
C_ND_INJ = ((-K_INJ) * (F_H / F0) - K_INJ * (F_L / F0)) * F0 / (2.0 * (F_H - F_L))
# analytic delta/mean outputs for the injected correction
DPHI_INJ = -2.0 * K_INJ
PHBAR_INJ = 0.0


def _annotations() -> dict:
    return {
        "Grid": {
            "TimeCOAPoly": {"Coefs": [[0.0]]},
            "Col": {"SS": 1.0},
            "Row": {"SS": 1.0},
        },
        "Timeline": {"CollectStart": "2020-01-01T00:00:00.000000Z"},
        "RadarCollection": {"TxFrequency": {"Min": F0, "Max": F0}},
        "SCPCOA": {"ARPPos": {"X": 6524834.0, "Y": 0.0, "Z": 0.0}},
    }


def _dual_ramp(delta_lo: float, delta_hi: float) -> np.ndarray:
    """Image whose spectrum is concentrated at the exact FFT bins Q_LO, Q_HI.

    The range FFT (N samples, fftshifted frequency coordinates k/31 for k < 16
    and (k - 32)/31 for k >= 16) of the returned image has its only two
    non-zero bins at k=Q_LO and k=Q_HI, i.e. exactly at the sub-band centres
    -F_OFF and +F_OFF. Shifting each ramp in range by its ``delta`` only
    rotates the corresponding spectral peak, so the normalized interferogram
    between an unshifted image and this one has the constant phase
    +2*pi*Q*delta/N inside each sub-band (exactly, modulo float rounding).
    """
    r = np.arange(N, dtype=np.float64)
    return np.exp(2j * np.pi * Q_LO * (r - delta_lo) / N) + np.exp(2j * np.pi * Q_HI * (r - delta_hi) / N)


def _coreg(acq: str, delta_lo: float, delta_hi: float) -> xr.Dataset:
    """coreg/measurements Dataset carrying the dual-ramp image."""
    img = np.broadcast_to(_dual_ramp(delta_lo, delta_hi), (AZ, N))
    data = np.empty((1, 1, AZ, N), dtype=complex)
    data[0, 0] = img
    coreg = xr.Dataset(
        {
            "cstack": (
                ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
                data,
                {POL: _annotations()},
            )
        },
        coords={
            "polarization": [POL],
            "acquisition_id": [acq],
            "acquisition_date": ("acquisition_id", [acq]),
            "azimuth_time": T0 + np.arange(AZ) * AZ_DT,
            "slant_range_time": np.arange(N, dtype=np.float64),
            "azimuth_m": ("azimuth_time", np.arange(AZ, dtype=np.float64)),
            "slant_range_m": ("slant_range_time", np.arange(N, dtype=np.float64)),
        },
    )
    return coreg.chunk({"azimuth_time": 8, "slant_range_time": 8})


def _split_trees(delta_lo: float, delta_hi: float) -> tuple[xr.DataTree, xr.DataTree]:
    """Reference/secondary trees: master shifted, secondary unshifted."""
    reference_slcs = xr.DataTree()
    secondary_slcs = xr.DataTree()
    reference_slcs["coreg/measurements"] = _coreg(MASTER, delta_lo, delta_hi)
    reference_slcs[f"orig/{MASTER}/measurements"] = xr.Dataset(
        {
            "slc": (
                ["azimuth_time", "slant_range_time"],
                np.ones((4, 4), dtype=complex),
                {POL: _annotations()},
            )
        }
    )
    secondary_slcs["coreg/measurements"] = _coreg(SECOND, 0.0, 0.0)
    return reference_slcs, secondary_slcs


def _full_tree() -> xr.DataTree:
    """Stripmap tree holding both the reference and the secondary acquisition."""
    slcs = xr.DataTree()
    ref = _coreg(MASTER, DELTA_LO, DELTA_HI)
    sec = _coreg(SECOND, 0.0, 0.0)
    slcs["coreg/measurements"] = xr.concat([ref, sec], dim="acquisition_id")
    slcs[f"orig/{MASTER}/measurements"] = xr.Dataset(
        {
            "slc": (
                ["azimuth_time", "slant_range_time"],
                np.ones((4, 4), dtype=complex),
                {POL: _annotations()},
            )
        }
    )
    return slcs


def _x_poly(value: float) -> xr.Dataset:
    """Constant range-polynomial phase-correction model."""
    return xr.Dataset(
        {
            SECOND: xr.DataArray(
                np.array([[value]]),
                dims=("i", "j"),
                attrs={
                    "off_y": "2020-01-01T00:00:00",
                    "offset": np.array([0.0, 0.0]),
                    "scale_fact": np.array([1.0, 1.0]),
                },
            )
        }
    )


def _const_poly(value: float) -> xr.DataArray:
    """Constant 2-D correction polynomial with the expected attributes."""
    return xr.DataArray(
        np.array([[value]]),
        dims=("i", "j"),
        attrs={
            "off_y": "2020-01-01T00:00:00",
            "offset": np.array([0.0, 0.0]),
            "scale_fact": np.array([1.0, 1.0]),
        },
    )


class TestGetPhaseCorrFromXPoly:
    @pytest.mark.unit
    def test_get_phase_corr_from_x_poly_linear(self):
        """A known 2-D polynomial is evaluated exactly on the given grid."""
        az = xr.DataArray(T0 + np.arange(8) * AZ_DT, dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(8, dtype=np.float64), dims=["slant_range_time"])
        off_y = "2020-01-01T00:00:00"
        offset = np.array([1.0e9, 3.0])
        scale = np.array([1.0e9, 4.0])
        coeffs = np.array([[0.0, 2.0], [1.0, 0.0]])
        x_poly = xr.Dataset(
            {
                SECOND: xr.DataArray(
                    coeffs,
                    dims=("i", "j"),
                    attrs={"off_y": off_y, "offset": offset, "scale_fact": scale},
                )
            }
        )
        freq = 1000.0
        out = get_phase_corr_from_x_poly(az, rg, x_poly[SECOND], freq)

        az_sc = ((az - np.datetime64(off_y, "ns")).astype(float) - offset[0]) / scale[0]
        rg_sc = (rg - offset[1]) / scale[1]
        expected = 2.0 * np.pi * freq * (az_sc + 2.0 * rg_sc)

        assert out.dims == ("azimuth_time", "slant_range_time")
        assert out.sizes == {"azimuth_time": 8, "slant_range_time": 8}
        np.testing.assert_allclose(out.data, expected.data, atol=TOL)

    @pytest.mark.unit
    def test_get_phase_corr_from_x_poly_constant(self):
        """A constant polynomial yields the same correction on every pixel."""
        az = xr.DataArray(T0 + np.arange(4) * AZ_DT, dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(5, dtype=np.float64), dims=["slant_range_time"])
        freq = 12.5
        out = get_phase_corr_from_x_poly(az, rg, _x_poly(C_INJ)[SECOND], freq)
        assert np.all(np.isfinite(out.data))
        np.testing.assert_allclose(out.data, np.full((4, 5), 2.0 * np.pi * freq * C_INJ), atol=TOL)


class TestSeparateDispersiveSpectrumComponents:
    """Split spectrum on an SLC whose interferogram is a pure spectrum ramp.

    The master image is an exact shift of the secondary image, so the
    interferogram phase is frequency-independent within each spectral
    component: the non-dispersive component must be recovered exactly and the
    ionospheric one must vanish. An x_poly-injected constant shift is the
    mirror case: it is recovered exactly as ionospheric phase.
    """

    @pytest.mark.unit
    def test_separate_pure_spectrum_ramp(self):
        """A pure (frequency-proportional) ramp gives a zero ionospheric phase."""
        reference_slcs, secondary_slcs = _split_trees(DELTA_LO, DELTA_HI)
        phase_iono, phase_nd = separate_dispersive_spectrum_components(
            reference_slcs,
            secondary_slcs,
            frequency_offset=F_OFF,
            bandwidth=BW,
            cohe_th=COHE_TH,
            a_min=T0,
            r_min=0.0,
            x_poly=None,
            y_poly=None,
            wdir_path=None,
            multilook=ML,
            multiburst=False,
            enable_dask=False,
        )

        assert phase_iono.sizes["azimuth_time"] == AZ - 6
        assert phase_iono.sizes["slant_range_time"] == N - 11
        np.testing.assert_allclose(phase_iono.phase.data, np.full_like(phase_iono.phase, C_IO_RAMP), atol=1e-9)
        assert np.max(np.abs(phase_iono.phase.data)) < 1e-9
        np.testing.assert_allclose(phase_nd.data, np.full_like(phase_nd.data, C_ND_RAMP), atol=1e-9)
        assert np.all(phase_iono.coherence.data > COHE_TH)
        assert np.all(np.isfinite(phase_iono["azimuth_time_raw"].values.astype(np.int64)))

    @pytest.mark.unit
    def test_separate_recovers_injected_shift(self):
        """A constant x_poly shift is recovered exactly as ionospheric phase."""
        reference_slcs, secondary_slcs = _split_trees(0.0, 0.0)
        phase_iono, phase_nd = separate_dispersive_spectrum_components(
            reference_slcs,
            secondary_slcs,
            frequency_offset=F_OFF,
            bandwidth=BW,
            cohe_th=COHE_TH,
            a_min=T0,
            r_min=0.0,
            x_poly=_x_poly(C_INJ),
            y_poly=None,
            wdir_path=None,
            multilook=ML,
            multiburst=False,
            enable_dask=False,
        )

        np.testing.assert_allclose(phase_iono.phase.data, np.full_like(phase_iono.phase, C_IO_INJ), atol=1e-9)
        np.testing.assert_allclose(phase_nd.data, np.full_like(phase_nd.data, C_ND_INJ), atol=1e-9)
        assert np.all(phase_iono.coherence.data > COHE_TH)


class TestSeparateDispersiveSpectrumComponentsDeltaMean:
    """Delta/mean formulation on the same analytic fixtures."""

    @pytest.mark.unit
    def test_separate_delta_mean_pure_spectrum_ramp(self):
        """A pure spectrum ramp leaves a zero ionospheric phase (delta/mean)."""
        reference_slcs, secondary_slcs = _split_trees(DELTA_LO, DELTA_HI)
        phase_iono, phase_nd = separate_dispersive_spectrum_components_delta_mean(
            reference_slcs,
            secondary_slcs,
            frequency_offset=F_OFF,
            bandwidth=BW,
            cohe_th=COHE_TH,
            a_min=T0,
            r_min=0.0,
            x_poly=None,
            y_poly=None,
            wdir_path=None,
            multilook=ML,
            multiburst=False,
            enable_dask=False,
        )

        np.testing.assert_allclose(phase_iono.phase.data, np.full_like(phase_iono.phase, 0.0), atol=1e-9)
        expected_nd = 0.25 * (F_L + F_H) / (F_H - F_L) * DPHI_RAMP + 0.5 * PHBAR_RAMP
        np.testing.assert_allclose(phase_nd.data, np.full_like(phase_nd.data, expected_nd), atol=1e-9)
        assert np.all(phase_iono.coherence.data > COHE_TH)

    @pytest.mark.unit
    def test_separate_delta_mean_recovers_injected_shift(self):
        """The x_poly shift is recovered identically by the delta/mean core."""
        reference_slcs, secondary_slcs = _split_trees(0.0, 0.0)
        phase_iono, phase_nd = separate_dispersive_spectrum_components_delta_mean(
            reference_slcs,
            secondary_slcs,
            frequency_offset=F_OFF,
            bandwidth=BW,
            cohe_th=COHE_TH,
            a_min=T0,
            r_min=0.0,
            x_poly=_x_poly(C_INJ),
            y_poly=None,
            wdir_path=None,
            multilook=ML,
            multiburst=False,
            enable_dask=False,
        )

        expected_io = -0.5 * (F_L * F_H) / (F0 * (F_H - F_L)) * DPHI_INJ
        expected_nd = 0.25 * (F_H + F_L) / (F_H - F_L) * DPHI_INJ
        np.testing.assert_allclose(phase_iono.phase.data, np.full_like(phase_iono.phase, expected_io), atol=1e-9)
        np.testing.assert_allclose(phase_nd.data, np.full_like(phase_nd.data, expected_nd), atol=1e-9)
        assert np.allclose(expected_io, C_IO_INJ, atol=1e-12)
        assert np.allclose(expected_nd, C_ND_INJ, atol=1e-12)


class TestRemoveIonosurfaceChunk:
    @pytest.mark.unit
    def test_remove_iono_surface_chunk_constant_surface(self):
        """A constant correction polynomial rotates cstack exactly by -C."""
        values = np.arange(AZ * N, dtype=np.float64).reshape(AZ, N)
        cstack = xr.DataArray(
            np.exp(1j * values)[None],
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={
                "acquisition_id": [SECOND],
                "azimuth_time": T0 + np.arange(AZ) * AZ_DT,
                "slant_range_time": np.arange(N, dtype=np.float64),
                "azimuth_m": ("azimuth_time", np.arange(AZ, dtype=np.float64)),
                "slant_range_m": ("slant_range_time", np.arange(N, dtype=np.float64)),
            },
        )
        original = cstack.values.copy()
        ds = remove_iono_surface_chunk(
            cstack,
            _const_poly(C_SURF),
            np.array([[0.0]]),
            T0,
            250000.0,
            3900000.0,
        )

        assert ds.sizes == {"acquisition_id": 1, "azimuth_time": AZ, "slant_range_time": N}
        np.testing.assert_allclose(ds.cstack.data, original * np.exp(-1j * C_SURF), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ds.phase_iono.data, np.full(ds.phase_iono.shape, C_SURF), atol=1e-9)


class TestRemoveIonosurfaceSingleBurst:
    @pytest.mark.unit
    def test_remove_iono_surface_single_burst_constant_surface(self, tmp_path):
        """Disk-based removal of a constant ionospheric surface from one burst."""
        coreg = _coreg(SECOND, 0.0, 0.0)
        original = coreg.cstack.values.copy()
        burst_id, coreg_ss = remove_iono_surface_single_burst(
            "data",
            coreg,
            _const_poly(C_SURF),
            250000.0,
            3900000.0,
            Path(str(tmp_path)),
            False,
            enable_dask=False,
            reference_id=SECOND,
        )

        assert burst_id == "data"
        assert "cstack" in coreg_ss.data_vars
        assert "phase_iono" in coreg_ss.data_vars
        np.testing.assert_allclose(
            coreg_ss.cstack.isel(polarization=0, acquisition_id=0).data,
            original[0, 0] * np.exp(-1j * C_SURF),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(coreg_ss.phase_iono.data, np.full(coreg_ss.phase_iono.shape, C_SURF), atol=1e-5)


class TestRemoveIonosurface:
    @pytest.mark.unit
    def test_remove_iono_surface_stripmap_constant_surface(self, tmp_path):
        """The ionospheric surface is removed and recorded in the SLC tree."""
        slcs = xr.DataTree()
        slcs["coreg/measurements"] = _coreg(SECOND, 0.0, 0.0)
        original = slcs.coreg.measurements.cstack.values.copy()
        out = remove_iono_surface(
            slcs,
            _const_poly(C_SURF),
            250000.0,
            3900000.0,
            wdir_path=Path(str(tmp_path)),
            multiburst=False,
            enable_dask=False,
            reference_id=SECOND,
        )

        assert "phase_iono" in out.coreg.measurements.data_vars
        np.testing.assert_allclose(
            out.coreg.measurements.cstack.isel(polarization=0, acquisition_id=0).data,
            original[0, 0] * np.exp(-1j * C_SURF),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out.coreg.measurements.phase_iono.data,
            np.full((1, AZ, N), C_SURF),
            atol=1e-5,
        )


class TestSplitSpectrum:
    @pytest.mark.unit
    def test_split_spectrum_stripmap_injected_shift(self, tmp_path):
        """Full stripmap workflow: the injected ionospheric shift is removed."""
        slcs = _full_tree()
        original_second = (
            xr.concat([_coreg(MASTER, DELTA_LO, DELTA_HI), _coreg(SECOND, 0.0, 0.0)], dim="acquisition_id")
            .cstack.isel(polarization=0)
            .sel(acquisition_id=SECOND)
            .data.copy()
        )
        out = split_spectrum(
            slcs,
            ML,
            F_OFF,
            BW,
            COHE_TH,
            str(tmp_path),
            False,
            MASTER,
            esd_poly_degree=2,
            split_spectrum_poly_degree=1,
            h_iono=250000.0,
            enable_dask=False,
            x_poly=_x_poly(C_INJ),
            dbfile=None,
        )

        meas = out["coreg/measurements"].ds
        assert "phase_iono" in meas.data_vars
        np.testing.assert_allclose(meas.phase_iono.data, np.full(meas.phase_iono.shape, C_IO_INJ), atol=1e-5)
        np.testing.assert_allclose(
            meas.cstack.isel(polarization=0).sel(acquisition_id=SECOND).data,
            original_second * np.exp(-1j * C_IO_INJ),
            rtol=1e-5,
            atol=1e-5,
        )
        poly = out.attrs["coregistration_polynomials"]["split_spectrum"]
        assert poly["description"] == "Split Spectrum"
        assert SECOND in poly["acquisitions"]
