import numpy as np
import pytest
import xarray as xr
from cop_ard_core.warp import GRID, SIGMA_ORB_2
from scipy.constants import speed_of_light

from cop_ard_app.coreg import (
    blockwise_warp_cslc,
    finalize_coreg,
    initialize_LUT_coords,
    update_conditions,
    update_LUT,
    update_quality,
    updateLut,
)

N_AZ = 8
N_RG = 8
COARSE = (4, 4)
AZ_STEP = N_AZ // COARSE[0]
RG_STEP = N_RG // COARSE[1]
POL = "VV"
ACQ = "2020-01-01"
SECONDARY = "2020-01-12"
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
AZ_DT = np.timedelta64(1_000_000_000, "ns")
AZ = T0 + np.arange(N_AZ) * AZ_DT
RG = np.arange(N_RG, dtype=np.float64)
AZ_COARSE = AZ[::AZ_STEP]
RG_COARSE = RG[::RG_STEP]
SCP_ROW = 2
SCP_COL = 3
RG0 = RG[SCP_ROW]
AZ0 = AZ[SCP_COL]
REF_TCA = 3.0
ACQS = [ACQ, SECONDARY]
DISPLACEMENT_NS = [3, 7]
DISPLACEMENT_RG = [0.5, -1.5]
KERNEL = {"kernel_shape": 7, "ov_fact": 2}
CHUNK = {"azimuth_time": slice(2, 5), "slant_range_time": slice(0, 4)}


def _annotations(dkcoa, tca_c, *, col_ss=1.0, row_ss=1.0, min_tx=0.0, max_tx=0.0, scp_row=SCP_ROW, scp_col=SCP_COL):
    """Full SAR metadata annotation of the shape coreg's LUT and warp helpers expect."""
    return {
        "Grid": {
            "Col": {"SS": col_ss, "DeltaKCOAPoly": {"Coefs": dkcoa}},
            "Row": {"SS": row_ss},
        },
        "RadarCollection": {"TxFrequency": {"Min": min_tx, "Max": max_tx}},
        "ImageData": {"SCPPixel": {"Row": scp_row, "Col": scp_col}},
        "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, tca_c]}}},
    }


def _measurements(cstack, acqs=(ACQ,), az=AZ, rg=RG, attrs=None):
    """A `coreg/measurements`-shaped Dataset: the coregistered grid."""
    return xr.Dataset(
        {"cstack": cstack},
        coords={
            "polarization": [POL],
            "acquisition_id": list(acqs),
            "acquisition_date": ("acquisition_id", list(acqs)),
            "azimuth_time": az,
            "slant_range_time": rg,
        },
        attrs=attrs,
    )


def _orig(acq, slc, az=AZ, rg=RG):
    """An `orig/<acq>/measurements`-shaped Dataset: one original SLC image."""
    return xr.Dataset(
        {"slc": slc},
        coords={
            "acquisition_id": [acq],
            "azimuth_time": az,
            "slant_range_time": rg,
        },
    )


def _build_tree(coreg_ds, orig_ds):
    """The SLC tree `updateLut` walks: a coregistered stack plus its acquisitions."""
    return xr.DataTree.from_dict(
        {
            "/coreg/measurements": coreg_ds,
            f"/orig/{ACQ}/measurements": orig_ds,
        }
    )


def _coreg_stack(acqs):
    """An empty `coreg_cstack` of the given acquisitions, as consumed by the warper."""
    return xr.DataArray(
        np.zeros((1, len(acqs), N_AZ, N_RG), dtype=np.complex64),
        dims=("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
        coords={"polarization": [POL], "acquisition_id": list(acqs)},
    )


def _phase_model(n_acq):
    """The companion phase model, addressed positionally by the warper."""
    return xr.DataArray(
        np.zeros((n_acq, N_AZ, N_RG)),
        dims=("acquisition_date_index", "azimuth_time", "slant_range_time"),
    )


def _daz_drg(az_values, rg_values, daz_ns, drg):
    """Displacement fields (`daz`, `drg`) on a coarse grid, as produced upstream."""
    return xr.Dataset(
        {
            "daz": (
                ["acquisition_id", "azimuth_time", "slant_range_time"],
                np.asarray(daz_ns, dtype=np.int64)[np.newaxis, :, :].astype("timedelta64[ns]"),
            ),
            "drg": (
                ["acquisition_id", "azimuth_time", "slant_range_time"],
                np.asarray(drg, dtype=np.float64)[np.newaxis, :, :],
            ),
        },
        coords={
            "acquisition_id": [ACQ],
            "azimuth_time": np.asarray(az_values, dtype="datetime64[ns]"),
            "slant_range_time": np.asarray(rg_values, dtype=np.float64),
        },
    )


def _displacements(acqs, az_ns, az_rg):
    """Per-acquisition `daz`/`drg` fields on the full-resolution grid."""
    coords = {"acquisition_id": np.array(list(acqs)), "azimuth_time": AZ, "slant_range_time": RG}
    dims = ("acquisition_id", "azimuth_time", "slant_range_time")
    daz = np.stack([np.full((N_AZ, N_RG), v, dtype=np.int64) for v in az_ns])
    drg = np.stack([np.full((N_AZ, N_RG), v, dtype=np.float64) for v in az_rg])
    return (
        xr.DataArray(daz.astype("timedelta64[ns]"), dims=dims, coords=coords),
        xr.DataArray(drg, dims=dims, coords=coords),
    )


def _zero_lut_shifts():
    """Zero `daz`/`drg` on the coarse grid: the secondary grid *is* the reference grid."""
    return _daz_drg(AZ_COARSE, RG_COARSE, np.zeros((COARSE[0], COARSE[1])), np.zeros((COARSE[0], COARSE[1])))


class TestInitializeLutCoords:
    """Unit test for cop_ard_app.coreg.initialize_LUT_coords."""

    @pytest.mark.unit
    @pytest.mark.parametrize("grid", [(3, 5), (2 * N_AZ, 2 * N_RG)])
    def test_decimates_measurement_grid(self, grid):
        acq_ids = np.array([ACQ, SECONDARY])
        out = initialize_LUT_coords(
            xr.DataArray(acq_ids, dims="acquisition_id"),
            xr.DataArray(acq_ids, dims="acquisition_id"),
            xr.DataArray(AZ, dims="azimuth_time"),
            xr.DataArray(RG, dims="slant_range_time"),
            grid=grid,
        )

        # the decimation step sizes follow ceil(size / grid), azimuth and range apart
        az_step = -(-N_AZ // grid[0])
        rg_step = -(-N_RG // grid[1])
        assert np.array_equal(out.azimuth_time.values, AZ[::az_step])
        assert np.array_equal(out.slant_range_time.values, RG[::rg_step])

        # the acquisition axis is never decimated, only the two spatial axes
        shape = (acq_ids.size, AZ[::az_step].size, RG[::rg_step].size)
        assert out.daz.dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert out.daz.shape == shape
        assert out.drg.dims == out.daz.dims
        assert out.drg.shape == shape

        # both placeholders are zero filled, and typed as they are consumed downstream
        assert out.drg.dtype == np.dtype(np.float64)
        assert np.all(out.drg.values == 0.0)
        assert out.daz.dtype == np.dtype("timedelta64[ns]")
        assert np.all(out.daz.values.astype(np.int64) == 0)

        # the companion acquisition_date coordinate is carried along
        assert np.array_equal(out.acquisition_date.values, acq_ids)


class TestUpdateLUT:
    """Unit test for cop_ard_app.coreg.update_LUT.

    An update writes the coarse displacement fields into the LUT grid, but only on the
    box delimited by the displacements' own coordinates: every LUT node inside that
    bounding box gets the label-matched value, every node outside stays zero.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sub_az,sub_rg",
        [
            (AZ_COARSE[1:3], RG_COARSE[1:]),
            (AZ_COARSE + 20 * AZ_DT, RG_COARSE + 20.0),
        ],
        ids=["partial box", "disjoint box"],
    )
    def test_copies_displacements_inside_bounding_box(self, sub_az, sub_rg):
        lut_coords = initialize_LUT_coords(
            xr.DataArray([ACQ], dims="acquisition_id"),
            xr.DataArray([ACQ], dims="acquisition_id"),
            xr.DataArray(AZ, dims="azimuth_time"),
            xr.DataArray(RG, dims="slant_range_time"),
            grid=COARSE,
        )

        az_sec = np.array([(a - T0) / AZ_DT for a in sub_az], dtype=np.float64)
        rg_sec = np.array(sub_rg, dtype=np.float64)
        daz_ns = 1000.0 * az_sec[:, None] + rg_sec[None, :]
        drg = 100.0 * az_sec[:, None] + rg_sec[None, :]
        daz_drg = _daz_drg(sub_az, sub_rg, daz_ns, drg)

        out = update_LUT(daz_drg, lut_coords)

        # every coarse node inside the displacements' bounding box takes the matching value
        az_grid = np.array([(a - T0) / AZ_DT for a in AZ_COARSE], dtype=np.float64)[:, None]
        rg_grid = RG_COARSE[None, :]
        inside = ((AZ_COARSE >= sub_az.min()) & (AZ_COARSE <= sub_az.max()))[:, None] & (
            (RG_COARSE >= sub_rg.min()) & (RG_COARSE <= sub_rg.max())
        )[None, :]
        expected_drg = np.where(inside, 100.0 * az_grid + rg_grid, 0.0)
        expected_daz = np.where(inside, 1000.0 * az_grid + rg_grid, 0.0)

        assert out.drg.dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert np.array_equal(out.drg.isel(acquisition_id=0).values, expected_drg)
        assert np.array_equal(out.daz.isel(acquisition_id=0).values.astype(np.int64), expected_daz.astype(np.int64))
        # the LUT extent is never resized by an update
        assert out.daz.shape == lut_coords.daz.shape


class TestUpdateLut:
    """Unit test for cop_ard_app.coreg.updateLut.

    A LUT holds a metadata polynomial evaluated on the coarse grid carried by
    ``daz_drg``. With zero displacements that grid is exactly the decimated coreg
    grid, so the values are the polynomial evaluated at the offsets (seconds in
    azimuth, metres in range) of those nodes from the scene centre pixel. ``getLUT``
    builds its surface on an (azimuth, range) grid while it is stored as
    (slant_range_time, azimuth_time), hence an azimuth-only polynomial is constant
    along the last axis of the stored array.
    """

    def _tree(self, coefs):
        coreg = _measurements(
            xr.DataArray(
                np.zeros((1, 1, N_AZ, N_RG), dtype=np.complex64),
                dims=("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
                attrs={POL: _annotations([[1.0]], REF_TCA)},
            )
        )
        slc = xr.DataArray(
            np.zeros((N_AZ, N_RG), dtype=np.complex64),
            dims=("azimuth_time", "slant_range_time"),
            coords={"azimuth_time": AZ, "slant_range_time": RG},
            attrs={POL: _annotations(coefs, 1.0)},
        )
        return _build_tree(coreg, _orig(ACQ, slc))

    def _lut(self, coefs, out_name):
        return updateLut(
            self._tree(coefs),
            _zero_lut_shifts(),
            "Grid.Col.DeltaKCOAPoly.Coefs",
            out_name,
            grid=COARSE,
        )

    @pytest.mark.unit
    def test_constant_coefficient_gives_constant_lut(self):
        out = self._lut([[7.5]], "doppler")

        assert "doppler" in out.data_vars
        assert out.doppler.dims == ("acquisition_id", "slant_range_time", "azimuth_time")
        assert out.doppler.shape == (1, COARSE[0], COARSE[1])
        assert np.array_equal(out.doppler.values, np.full((1, COARSE[0], COARSE[1]), 7.5))

        # the coarse grid of daz_drg becomes the LUT coordinate system
        assert np.array_equal(out.slant_range_time.values, RG_COARSE)
        assert np.array_equal(out.azimuth_time.values, AZ_COARSE)

        # the bookkeeping attributes describe the reference point and the unit scales
        assert out.doppler.attrs["reference"] == {
            "azimuth_time": str(AZ0),
            "slant_range_time": float(RG0),
        }
        assert out.doppler.attrs["scale"] == {
            "azimuth_time": 1.0 / (1e9 * REF_TCA),
            "slant_range_time": speed_of_light / 2.0,
        }
        assert out.doppler.attrs["deg"] == {"azimuth_time": 1, "slant_range_time": 1}

    @pytest.mark.unit
    def test_azimuth_polynomial_evaluated_on_coarse_grid(self):
        out = self._lut([[0.0, 1.0]], "doppler")

        offsets = np.array([(a - AZ0) / AZ_DT for a in AZ_COARSE], dtype=np.float64)
        expected = np.broadcast_to(offsets[:, None], (1, COARSE[0], COARSE[1]))
        assert np.array_equal(out.doppler.values, expected)
        assert out.doppler.attrs["deg"] == {"azimuth_time": 2, "slant_range_time": 1}

    @pytest.mark.unit
    def test_radiometric_names_gain_a_polarization_axis(self):
        out = self._lut([[7.5]], "sigma0")

        assert "sigma0" in out.data_vars
        assert out.sigma0.dims == ("polarization", "acquisition_id", "slant_range_time", "azimuth_time")
        assert out.sigma0.shape == (1, 1, COARSE[0], COARSE[1])
        assert np.array_equal(out.sigma0.values, np.full((1, 1, COARSE[0], COARSE[1]), 7.5))
        assert np.array_equal(out.polarization.values, [POL])

    @pytest.mark.unit
    def test_missing_coefficients_are_rejected(self):
        annot = _annotations([[1.0]], 1.0)
        del annot["Grid"]["Col"]["DeltaKCOAPoly"]
        coreg = _measurements(
            xr.DataArray(
                np.zeros((1, 1, N_AZ, N_RG), dtype=np.complex64),
                dims=("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
                attrs={POL: annot},
            )
        )
        slc = xr.DataArray(
            np.zeros((N_AZ, N_RG), dtype=np.complex64),
            dims=("azimuth_time", "slant_range_time"),
            coords={"azimuth_time": AZ, "slant_range_time": RG},
            attrs={POL: annot},
        )
        with pytest.raises(ValueError, match="Could not find coefficients"):
            updateLut(
                _build_tree(coreg, _orig(ACQ, slc)),
                _zero_lut_shifts(),
                "Grid.Col.DeltaKCOAPoly.Coefs",
                "doppler",
                grid=COARSE,
            )


class TestBlockwiseWarpCslc:
    """Unit test for cop_ard_app.coreg.blockwise_warp_cslc.

    The sinc warping itself belongs to cop_ard_core, so ``warp_cslc`` is replaced by
    a recorder: this test checks what the coregistration layer derives from the SLC
    metadata and feeds to it, once per acquisition and in stack order, and that the
    warped stack is returned as a ``cstack``/``phase_model`` Dataset.
    """

    @pytest.mark.unit
    def test_warps_every_acquisition_in_order(self, monkeypatch):
        calls = []
        out = self._run(monkeypatch, calls, ACQS)

        assert [c["acquisition_id"] for c in calls] == ACQS
        assert [c["acquisition_date_index"] for c in calls] == list(range(len(ACQS)))

        assert sorted(out.data_vars) == ["cstack", "phase_model"]
        assert out.cstack.shape == (1, len(ACQS), N_AZ, N_RG)
        assert out.phase_model.shape == (len(ACQS), N_AZ, N_RG)
        # the warper works in place and the warped stack is handed back
        for i, _ in enumerate(ACQS):
            assert np.array_equal(out.cstack.isel(acquisition_id=i).values, np.full((1, N_AZ, N_RG), complex(i + 1, 0)))

    @pytest.mark.unit
    def test_derives_geometry_from_slc_metadata(self, monkeypatch):
        calls = []
        self._run(monkeypatch, calls, ACQS)

        for acq, call in zip(ACQS, calls, strict=True):
            annot = self._annotations_for(acq)
            assert (
                call["azimuth_grid_spacing"]
                == annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
            )
            assert call["range_grid_spacing"] == 2 * annot["Grid"]["Row"]["SS"] / speed_of_light
            assert call["margin"] == KERNEL["kernel_shape"] // 2 + 1
            assert call["min_tx_frequency"] == annot["RadarCollection"]["TxFrequency"]["Min"]
            assert call["max_tx_frequency"] == annot["RadarCollection"]["TxFrequency"]["Max"]
            assert call["scp_pixel_row"] == annot["ImageData"]["SCPPixel"]["Row"]
            assert call["scp_pixel_col"] == annot["ImageData"]["SCPPixel"]["Col"]
            assert np.array_equal(
                call["timeca_polynomial_coeffs"], np.array(annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"])
            )
            assert np.array_equal(
                call["dkcoa_polynomial_coeffs"],
                np.array(annot["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"]),
            )
            # only the slice of the displacements belonging to this acquisition is passed on
            assert np.array_equal(
                call["daz"].values.astype(np.int64), np.full((N_AZ, N_RG), DISPLACEMENT_NS[ACQS.index(acq)])
            )
            assert np.array_equal(call["drg"].values, np.full((N_AZ, N_RG), DISPLACEMENT_RG[ACQS.index(acq)]))
            # dask-chunk offsets are where polynomials must be evaluated
            assert call["evaluation_offset"] == {k: v.start for k, v in CHUNK.items()}

    @pytest.mark.unit
    def test_forwards_per_acquisition_polynomials(self, monkeypatch):
        calls = []
        x_poly = {acq: xr.DataArray(np.zeros((2, 3)), dims=("slant_range_time", "azimuth_time")) for acq in ACQS}
        y_poly = {acq: xr.DataArray(np.ones((2, 3)), dims=("slant_range_time", "azimuth_time")) for acq in ACQS}
        self._run(monkeypatch, calls, ACQS, x_poly=x_poly, y_poly=y_poly)

        for acq, call in zip(ACQS, calls, strict=True):
            assert call["x_poly"] is x_poly[acq]
            assert call["y_poly"] is y_poly[acq]

    @pytest.mark.unit
    def test_missing_coefficients_are_rejected(self, monkeypatch):
        annotations = {acq: self._annotations_for(acq) for acq in ACQS}
        del annotations[SECONDARY]["Grid"]["Col"]["DeltaKCOAPoly"]
        with pytest.raises(ValueError, match="Failed to extract polynomial coefficients"):
            self._run(monkeypatch, [], ACQS, annotations=annotations)

    def _annotations_for(self, acq):
        """The SLC metadata each acquisition is annotated with, also used as expectation."""
        return {
            ACQ: _annotations([[1.0]], 2.0, col_ss=0.5, row_ss=1.5, min_tx=5.0e9, max_tx=6.0e9, scp_row=1, scp_col=5),
            SECONDARY: _annotations([[3.0]], 4.0, col_ss=1.5, row_ss=2.5, min_tx=4.0e9, max_tx=7.0e9),
        }[acq]

    def _run(self, monkeypatch, calls, acqs, x_poly=None, y_poly=None, annotations=None):
        """Invoke blockwise_warp_cslc with a recorded stand-in for cop_ard_core.warp_cslc."""
        if annotations is None:
            annotations = {acq: self._annotations_for(acq) for acq in acqs}
        orig_slcs = xr.DataTree.from_dict(
            {
                f"orig/{acq}": xr.Dataset(
                    {
                        "slc": xr.DataArray(
                            np.zeros((N_AZ, N_RG), dtype=np.complex64),
                            dims=("azimuth_time", "slant_range_time"),
                            coords={"azimuth_time": AZ, "slant_range_time": RG},
                            attrs={POL: annotations[acq]},
                        )
                    }
                )
                for acq in acqs
            }
        )
        etads = xr.DataTree.from_dict(
            {acq: xr.Dataset({"dlat": (["azimuth_time"], np.zeros(N_AZ))}, coords={"azimuth_time": AZ}) for acq in acqs}
        )
        daz, drg = _displacements(acqs, DISPLACEMENT_NS, DISPLACEMENT_RG)

        def fake_warp(**kwargs):
            calls.append(kwargs)
            index = kwargs["acquisition_date_index"]
            kwargs["coreg_cstack"].values[:, index, :, :] = complex(index + 1, 0.0)

        monkeypatch.setattr("cop_ard_app.coreg.warp_cslc", fake_warp)
        return blockwise_warp_cslc(
            coreg_cstack=_coreg_stack(acqs),
            coreg_phase_model=_phase_model(len(acqs)),
            daz=daz,
            drg=drg,
            coreg_acquisition_id=xr.DataArray(np.array(acqs), dims="acquisition_id"),
            coregistration_reference_id=acqs[0],
            orig_slcs=orig_slcs,
            orig_etad=etads,
            kernel=KERNEL,
            x_poly=x_poly,
            y_poly=y_poly,
            chunk=CHUNK,
        )


# `update_conditions` and `update_quality` ask `updateLut` for a LUT without naming a
# grid, so the fixtures below span the default `cop_ard_core.warp.GRID` of decimation:
# one LUT node per axis label. The range axis starts at 1 s, the accuracy estimator
# divides by the slant range of every node.
LUT_AZ = T0 + np.arange(GRID[0]) * AZ_DT
LUT_RG = np.arange(1.0, 1.0 + GRID[1], dtype=np.float64)
LUT_ACQS = [ACQ, SECONDARY]
INCIDENCE_ANGLE = 39.0
COLLECT_START = "2020-01-01T00:00:00.000000Z"
SECONDARY_START = "2020-01-12T00:00:00.000000Z"
ANNOTATED_ORBIT = "AUX_ANNOTATED"
PRECISE_ORBIT = "AUX_PREORB"
ARP_COEFS = {"X": [1.0, 2.0], "Y": [-3.0, 4.0], "Z": [5.0, -6.0]}
CALIB_VALUE = {"SigmaZero": 2.0, "GammaZero": 3.0, "BetaZero": 4.0, "RCS": 11.0}
NOISE_VALUE = 5.0
PHASE_DEMOD = 7.5


def _scene_annot(calib=(), noise=True, *, dkcoa=(1.0,), tca_c=REF_TCA, start=COLLECT_START, arp=ARP_COEFS):
    """`_annotations` plus the scene, orbit and radiometric keys the quality path reads."""
    annot = _annotations([list(dkcoa)], tca_c)
    annot["SCPCOA"] = {"IncidenceAng": INCIDENCE_ANGLE}
    annot["Position"] = {"ARPPoly": {e: {"Coefs": list(c)} for e, c in arp.items()}}
    annot["Timeline"] = {"CollectStart": start}
    radiometric = {}
    if noise:
        radiometric["NoiseLevel"] = {"NoisePoly": {"Coefs": [[NOISE_VALUE]]}}
    for calib_name in calib:
        radiometric[f"{calib_name}SFPoly"] = {"Coefs": [[CALIB_VALUE[calib_name]]]}
    if radiometric:
        annot["Radiometric"] = radiometric
    return annot


def _annotated_stack(acqs, annot, az=LUT_AZ, rg=LUT_RG):
    """A `cstack` of the given grid annotated with the coregistration's own metadata."""
    return xr.DataArray(
        np.zeros((1, len(acqs), az.size, rg.size), dtype=np.complex64),
        dims=("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
        coords={
            "polarization": [POL],
            "acquisition_id": list(acqs),
            "azimuth_time": az,
            "slant_range_time": rg,
        },
        attrs={POL: annot},
    )


def _annotated_slc(annot, az=LUT_AZ, rg=LUT_RG):
    """An `slc` DataArray annotated as the original product it stands for."""
    return xr.DataArray(
        np.zeros((az.size, rg.size), dtype=np.complex64),
        dims=("azimuth_time", "slant_range_time"),
        coords={"azimuth_time": az, "slant_range_time": rg},
        attrs={POL: annot},
    )


def _annotated_tree(coreg_ds, slc_annots, acqs, *, etad_acqs=(), orbit_type=None):
    """A single-burst tree with annotated originals, with ETAD and orbit companions on demand."""
    groups = {"/coreg/measurements": coreg_ds}
    for acq in acqs:
        groups[f"/orig/{acq}/measurements"] = _orig(acq, _annotated_slc(slc_annots[acq], LUT_AZ, LUT_RG))
        if acq in etad_acqs:
            groups[f"/orig/{acq}/conditions/ETAD"] = xr.Dataset(
                {"dlat": (["azimuth_time"], np.zeros(LUT_AZ.size))}, coords={"azimuth_time": LUT_AZ}
            )
        if orbit_type:
            groups[f"/orig/{acq}/conditions/orbit"] = xr.Dataset(attrs={"orbit_type": orbit_type})
    return xr.DataTree.from_dict(groups)


def _lut_displacements(acqs=(ACQ,), daz_ns=0, drg=0.0):
    """Uniform LUT displacements covering the whole default `GRID` reference grid."""
    shape = (len(acqs), LUT_AZ.size, LUT_RG.size)
    return xr.Dataset(
        {
            "daz": (
                ("acquisition_id", "azimuth_time", "slant_range_time"),
                np.full(shape, daz_ns, dtype=np.int64).astype("timedelta64[ns]"),
            ),
            "drg": (("acquisition_id", "azimuth_time", "slant_range_time"), np.full(shape, drg)),
        },
        coords={
            "acquisition_id": np.array(list(acqs)),
            "azimuth_time": LUT_AZ,
            "slant_range_time": LUT_RG,
        },
    )


class TestUpdateConditions:
    """Unit test for cop_ard_app.coreg.update_conditions.

    The orbit, ETAD and look-vector producers belong to `cop_ard_app.slc` and have their
    own test module, so they are replaced by recorders here: the test pins what the
    coregistration layer hands them, which groups end up in the `conditions` subtree
    (the ETAD one only when every original carries ETAD metadata) and leaves untouched,
    and the exact `phase_demod` LUT `updateLut` derives for the coregistered grid.
    """

    def _tree(self, acqs=(ACQ,), etad_acqs=()):
        coreg = _measurements(_annotated_stack(acqs, _scene_annot(dkcoa=(PHASE_DEMOD,))), acqs, LUT_AZ, LUT_RG)
        originals = {acq: _scene_annot(dkcoa=(PHASE_DEMOD,), tca_c=1.0) for acq in acqs}
        return _annotated_tree(coreg, originals, acqs, etad_acqs=etad_acqs)

    def _stubbed(self, monkeypatch, calls):
        """Record the arguments of the three `cop_ard_app.slc` collaborators."""

        def orbit(slcs):
            calls["orbit"] = slcs
            return xr.Dataset({"coeffs": (["n"], np.arange(3.0))})

        def etad(slcs, measurements):
            calls["etad"] = (slcs, measurements)
            return xr.Dataset({"dlat": (["azimuth_time"], np.zeros(LUT_AZ.size))}, coords={"azimuth_time": LUT_AZ})

        def look_vector(measurements, simulation):
            calls["look_vector"] = (measurements, simulation)
            return xr.Dataset({"look_vector": (["axis"], np.zeros(3))})

        for name, stub in (("updateOrbits", orbit), ("updateETAD", etad), ("updateLookVector", look_vector)):
            monkeypatch.setattr(f"cop_ard_app.coreg.{name}", stub)

    @pytest.mark.unit
    def test_populates_orbit_look_vector_and_phase_demod(self, monkeypatch):
        calls = {}
        self._stubbed(monkeypatch, calls)
        slcs = self._tree()
        measurements = xr.Dataset(
            {
                "cstack": (
                    ["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
                    np.zeros((1, 1, LUT_AZ.size, LUT_RG.size), dtype=np.complex64),
                )
            }
        )
        simulation = xr.Dataset({"dem_distance": (["axis_a", "axis_b", "axis_c"], np.zeros((3, 2, 2)))})
        conditions = xr.DataTree.from_dict({"instrument": xr.Dataset({"processor": "cop-ard"})})

        out = update_conditions(slcs, measurements, simulation, _lut_displacements(), conditions)

        # an update merges into the target subtree, it never drops what is already there
        assert sorted(out.children) == ["instrument", "look_vector", "orbit", "phase_demod"]

        # the ancillary producers get the stack and the measurements they are documented for
        assert calls["orbit"] is slcs
        assert calls["look_vector"] == (measurements, simulation)
        assert "etad" not in calls

        # the Doppler LUT is the one `updateLut` yields for the default grid
        phase_demod = out.phase_demod.to_dataset()
        assert list(phase_demod.data_vars) == ["phase_demod"]
        assert phase_demod.phase_demod.dims == ("acquisition_id", "slant_range_time", "azimuth_time")
        assert phase_demod.phase_demod.shape == (1, *GRID)
        assert np.array_equal(phase_demod.phase_demod.values, np.full((1, *GRID), PHASE_DEMOD))
        assert np.array_equal(phase_demod.azimuth_time.values, LUT_AZ)
        assert np.array_equal(phase_demod.slant_range_time.values, LUT_RG)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "etad_acqs",
        [(ACQ, SECONDARY), (ACQ,)],
        ids=["every original has ETAD", "one original lacks ETAD"],
    )
    def test_etad_group_requires_every_original_to_have_one(self, monkeypatch, etad_acqs):
        calls = {}
        self._stubbed(monkeypatch, calls)
        acqs = (ACQ, SECONDARY)
        slcs = self._tree(acqs, etad_acqs)

        out = update_conditions(slcs, xr.Dataset(), xr.Dataset(), _lut_displacements(acqs), xr.DataTree())

        # the whole `conditions` subtree is only refreshed as a set, never per acquisition
        expected = sorted(["look_vector", "orbit", "phase_demod"] + (["ETAD"] if etad_acqs == acqs else []))
        assert sorted(out.children) == expected
        assert ("etad" in calls) is (etad_acqs == acqs)

    @pytest.mark.unit
    def test_phase_demod_polynomial_is_taken_per_acquisition(self, monkeypatch):
        calls = {}
        self._stubbed(monkeypatch, calls)
        acqs = (ACQ, SECONDARY)
        coreg = _measurements(_annotated_stack(acqs, _scene_annot()), acqs, LUT_AZ, LUT_RG)
        originals = {
            ACQ: _scene_annot(dkcoa=(PHASE_DEMOD,), tca_c=1.0),
            SECONDARY: _scene_annot(dkcoa=(0.0, 1.0), tca_c=1.0),
        }
        slcs = _annotated_tree(coreg, originals, acqs)

        out = update_conditions(slcs, xr.Dataset(), xr.Dataset(), _lut_displacements(acqs), xr.DataTree())

        phase_demod = out.phase_demod.to_dataset()
        assert list(phase_demod.data_vars) == ["phase_demod"]
        assert phase_demod.phase_demod.shape == (len(acqs), *GRID)
        # an azimuth-only polynomial is constant along the stored range axis
        offsets = np.array([(a - LUT_AZ[SCP_COL]) / AZ_DT for a in LUT_AZ], dtype=np.float64)
        expected = np.stack([np.full(GRID, PHASE_DEMOD), np.broadcast_to(offsets[:, None], GRID)])
        assert np.array_equal(phase_demod.phase_demod.values, expected)


class TestUpdateQuality:
    """Unit test for cop_ard_app.coreg.update_quality.

    Every radiometric LUT is the one `updateLut` yields for the default grid, stored under
    the name its calibration is mapped to and only when the coregistration metadata declares
    it; of those, only `sigma0`, `gamma0` and `noise` keep a polarization axis. `crg_accuracy`
    comes from `cop_ard_core.coreg_qlt`: both products carry the same orbit polynomials and no
    displacement is applied, so the geometric term vanishes and what remains per acquisition is
    exactly the tabulated error of the orbit data used for it, `AUX_ANNOTATED` unless the
    acquisition ships a `conditions/orbit` node stating otherwise. The reference acquisition
    contributes no error at all, and the reference clock is the one used for every acquisition,
    whatever the acquisition's own `Timeline.CollectStart` says. The orbit polynomials of the
    coregistration and of every original are required for that metric to be computable at all,
    and it is labelled on the axes of the LUTs it is stored beside transposed.
    """

    def _tree(self, calib=(), noise=True, orbit_type=None, coreg_arp=ARP_COEFS, original_arp=ARP_COEFS):
        coreg = _measurements(
            _annotated_stack(LUT_ACQS, _scene_annot(calib=calib, noise=noise, arp=coreg_arp)),
            LUT_ACQS,
            LUT_AZ,
            LUT_RG,
            attrs={"coregistration_reference_id": ACQ},
        )
        originals = {
            ACQ: _scene_annot(calib=calib, noise=noise, tca_c=1.0, arp=original_arp),
            SECONDARY: _scene_annot(calib=calib, noise=noise, tca_c=1.0, start=SECONDARY_START, arp=original_arp),
        }
        return _annotated_tree(coreg, originals, LUT_ACQS, orbit_type=orbit_type)

    @pytest.mark.unit
    def test_luts_are_named_after_their_calibration(self):
        quality = update_quality(self._tree(calib=tuple(CALIB_VALUE)), _lut_displacements(LUT_ACQS), xr.Dataset())

        assert sorted(quality.data_vars) == sorted(["crg_accuracy", "beta0", "gamma0", "noise", "RCS", "sigma0"])
        for name, value in (
            ("sigma0", CALIB_VALUE["SigmaZero"]),
            ("gamma0", CALIB_VALUE["GammaZero"]),
            ("noise", NOISE_VALUE),
        ):
            assert quality[name].dims == ("polarization", "acquisition_id", "slant_range_time", "azimuth_time")
            assert np.array_equal(quality[name].values, np.full((1, len(LUT_ACQS), *GRID), value))
        for name, value in (("RCS", CALIB_VALUE["RCS"]), ("beta0", CALIB_VALUE["BetaZero"])):
            assert quality[name].dims == ("acquisition_id", "slant_range_time", "azimuth_time")
            assert np.array_equal(quality[name].values, np.full((len(LUT_ACQS), *GRID), value))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "calib,noise,expected",
        [
            ((), True, ["crg_accuracy", "noise"]),
            (("SigmaZero", "RCS"), False, ["crg_accuracy", "RCS", "sigma0"]),
            (("GammaZero",), False, ["crg_accuracy", "gamma0"]),
        ],
        ids=["noise only", "two calibrations", "one calibration"],
    )
    def test_luts_are_only_made_for_declared_calibrations(self, calib, noise, expected):
        quality = update_quality(
            self._tree(calib=calib, noise=noise),
            _lut_displacements(LUT_ACQS),
            xr.Dataset({"pre_existing": (["sample"], np.arange(3.0))}),
        )

        # only the calibrations the coregistration declares are added, and what the caller
        # already had is kept
        assert sorted(quality.data_vars) == sorted([*expected, "pre_existing"])

    @pytest.mark.unit
    @pytest.mark.parametrize("orbit_type", [None, PRECISE_ORBIT], ids=["annotated orbits", "precise orbits"])
    def test_accuracy_is_the_error_of_each_acquisitions_orbits(self, orbit_type):
        quality = update_quality(self._tree(orbit_type=orbit_type), _lut_displacements(LUT_ACQS), xr.Dataset())

        # unlike the LUTs it is assigned next to, the metric is labelled the other way round
        assert quality.crg_accuracy.dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert quality.crg_accuracy.shape == (len(LUT_ACQS), *GRID)
        # nothing is displaced and both products have the same orbit polynomials, so the
        # reference is exactly where it is and the rest pays only its orbit error
        assert np.array_equal(quality.crg_accuracy.isel(acquisition_id=0).values, np.zeros(GRID))
        resolved = orbit_type or ANNOTATED_ORBIT
        expected = np.full(GRID, np.sqrt(SIGMA_ORB_2[resolved]))
        assert np.array_equal(quality.crg_accuracy.isel(acquisition_id=1).values, expected)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwargs", [{"coreg_arp": {}}, {"original_arp": {}}], ids=["coregistration metadata", "one original"]
    )
    def test_both_products_must_state_their_orbit_polynomials(self, kwargs):
        tree = self._tree(**kwargs)

        with pytest.raises(ValueError, match="Missing ARP polynomial coefficients in metadata."):
            update_quality(tree, _lut_displacements(LUT_ACQS), xr.Dataset())


WORKING_DIRECTORY = "/wd/"
REFERENCE_ID = "2019-06-01"
REFERENCE_DATE = "2019-06-01"
COLLIDING_DATE = "2018-01-01"
REFERENCE_VALUE = complex(7.0, 0.0)
STORAGE_OPTIONS = {"anon": True}
AZIMUTH_M = np.arange(N_AZ, dtype=np.float64)
SLANT_RANGE_M = np.arange(N_RG, dtype=np.float64)
BURST_ONE = "bursts/b1"
BURST_TWO = "bursts/b2"
BURST_REFERENCE = "/wd/bursts/b1/current_reference"
SECOND_BURST_REFERENCE = "/wd/bursts/b2/current_reference"
DATAKE_REFERENCE = "/wd/data/2019-06-01/current_reference"


def _coreg_product(acqs, dates=None, value=0.0j):
    """A `coreg/measurements` node with the metric coordinates `finalize_coreg` copies."""
    dates = list(dates) if dates else list(acqs)
    cstack = xr.DataArray(
        np.full((1, len(acqs), N_AZ, N_RG), value, dtype=np.complex64),
        dims=("polarization", "acquisition_id", "azimuth_time", "slant_range_time"),
        coords={
            "polarization": [POL],
            "acquisition_id": list(acqs),
            "azimuth_time": AZ,
            "slant_range_time": RG,
        },
    )
    return xr.Dataset(
        {"cstack": cstack},
        coords={
            "polarization": [POL],
            "acquisition_id": list(acqs),
            "acquisition_date": ("acquisition_id", dates),
            "azimuth_time": AZ,
            "slant_range_time": RG,
            "azimuth_m": ("azimuth_time", AZIMUTH_M),
            "slant_range_m": ("slant_range_time", SLANT_RANGE_M),
        },
    )


def _product_tree(nodes, acqs, dates=None, value=0.0j):
    """A stack per node path: "bursts/<name>" is a multiburst branch, "" a single datatake."""
    groups = {}
    for node in nodes:
        prefix = f"/{node}" if node else ""
        groups[f"{prefix}/coreg/measurements"] = _coreg_product(acqs, dates, value)
        groups.update({f"{prefix}/orig/{a}/measurements": xr.Dataset() for a in acqs})
    return xr.DataTree.from_dict(groups)


def _incoming_reference(date, value=0.0j):
    """What reading a `current_reference` yields: one standalone acquisition image."""
    cstack = xr.DataArray(
        np.full((1, N_AZ, N_RG), value, dtype=np.complex64),
        dims=("acquisition_id", "azimuth_time", "slant_range_time"),
        coords={"acquisition_id": ["incoming"], "azimuth_time": AZ, "slant_range_time": RG},
    )
    return xr.Dataset({"cstack": cstack}, coords={"acquisition_date": date})


class TestFinalizeCoreg:
    """Unit test for cop_ard_app.coreg.finalize_coreg.

    A `current_reference` product is normally read back from the product tree, so the `cop_ard_app.io`
    lookup and read are replaced by recorders here: the test pins the exact path every node is looked
    up under and that its storage options reach both of them, which acquisitions gain an `orig` node
    holding the incoming image labelled with the coregistration's metric coordinates, how an
    acquisition that is already part of the stack is renamed to a `_sec` entry instead, and how the
    nodes whose reference is missing are dropped or refused. The concatenated stack itself is never
    reached through the node view the assignment goes through, so the coregistered grid keeps the
    acquisitions it had.
    """

    def _io(self, monkeypatch, references, looked_up):
        """Replace the product-tree lookup with a recorded in-memory reference."""

        def fake_exists(path, storage_options=None):
            looked_up.append((path, storage_options))
            return path in references

        def fake_read(path, **kwargs):
            return references[path]

        monkeypatch.setattr("cop_ard_app.coreg.exists", fake_exists)
        monkeypatch.setattr("cop_ard_app.coreg.read", fake_read)

    @pytest.mark.unit
    def test_disabled_options_leave_the_stack_untouched(self):
        slcs = _product_tree([BURST_ONE, BURST_TWO], [SECONDARY])

        out = finalize_coreg(slcs, WORKING_DIRECTORY, REFERENCE_ID)

        assert out is slcs
        assert sorted(out.bursts.children) == ["b1", "b2"]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "nodes,path",
        [([BURST_ONE], BURST_REFERENCE), ([""], DATAKE_REFERENCE)],
        ids=["multiburst", "single datatake"],
    )
    def test_appends_the_current_reference_of_every_node(self, monkeypatch, nodes, path):
        looked_up = []
        self._io(monkeypatch, {path: _incoming_reference(REFERENCE_DATE, REFERENCE_VALUE)}, looked_up)
        slcs = _product_tree(nodes, [SECONDARY])

        out = finalize_coreg(slcs, WORKING_DIRECTORY, REFERENCE_ID, do_esd=True, storage_options=STORAGE_OPTIONS)

        # a single datatake is looked up under `data/`, and the path is normalized
        assert looked_up == [(path, STORAGE_OPTIONS)]
        node = out[nodes[0]] if nodes[0] else out
        assert sorted(node.children) == ["coreg", "orig"]
        assert sorted(node.orig.children) == sorted([SECONDARY, REFERENCE_ID])

        incoming = node.orig[REFERENCE_ID].measurements
        assert list(incoming.data_vars) == ["slc"]
        assert list(incoming.acquisition_id.values) == [REFERENCE_ID]
        assert np.array_equal(incoming.acquisition_date.values, np.array([REFERENCE_DATE]))
        assert incoming.slc.dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert np.array_equal(incoming.slc.values, np.full((1, N_AZ, N_RG), REFERENCE_VALUE, dtype=np.complex64))
        assert np.array_equal(incoming.azimuth_m.values, AZIMUTH_M)
        assert np.array_equal(incoming.slant_range_m.values, SLANT_RANGE_M)

        assert [str(x) for x in node["coreg/measurements"].acquisition_id.values] == [SECONDARY]

    @pytest.mark.unit
    def test_a_reference_the_stack_already_has_is_kept_as_a_secondary(self, monkeypatch):
        looked_up = []
        self._io(monkeypatch, {BURST_REFERENCE: _incoming_reference(REFERENCE_DATE, REFERENCE_VALUE)}, looked_up)
        slcs = _product_tree([BURST_ONE], [ACQ, REFERENCE_ID], [ACQ, COLLIDING_DATE])

        out = finalize_coreg(slcs, WORKING_DIRECTORY, REFERENCE_ID, do_image_matching=True)

        assert looked_up == [(BURST_REFERENCE, None)]
        node = out[BURST_ONE]
        assert sorted(node.orig.children) == sorted([ACQ, f"{REFERENCE_ID}_sec", REFERENCE_ID])
        assert sorted(node.orig[f"{REFERENCE_ID}_sec"].children) == ["measurements"]
        incoming = node.orig[REFERENCE_ID].measurements
        assert list(incoming.data_vars) == ["slc"]
        assert incoming.slc.dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert np.array_equal(incoming.slc.values, np.full((1, N_AZ, N_RG), REFERENCE_VALUE, dtype=np.complex64))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "nodes,references,error,looked_up",
        [
            ([BURST_ONE, BURST_TWO], [BURST_REFERENCE], None, [BURST_REFERENCE, SECOND_BURST_REFERENCE]),
            (
                [BURST_ONE, BURST_TWO],
                [],
                "Current reference missing for all bursts. DataTree is empty.",
                [BURST_REFERENCE, SECOND_BURST_REFERENCE],
            ),
            (
                [""],
                [],
                "Current reference missing for the datatake. DataTree might be empty.",
                [DATAKE_REFERENCE],
            ),
        ],
        ids=["one burst without one", "every burst without one", "datatake without one"],
    )
    def test_a_missing_reference_drops_or_refuses_the_node(self, monkeypatch, nodes, references, error, looked_up):
        calls = []
        self._io(monkeypatch, {path: _incoming_reference(REFERENCE_DATE) for path in references}, calls)
        slcs = _product_tree(nodes, [SECONDARY])

        if error is None:
            out = finalize_coreg(slcs, WORKING_DIRECTORY, REFERENCE_ID, do_split_spectrum=True)
            assert sorted(out.bursts.children) == ["b1"]
        else:
            with pytest.raises(ValueError, match=error):
                finalize_coreg(slcs, WORKING_DIRECTORY, REFERENCE_ID, do_split_spectrum=True)
        assert [path for path, _ in calls] == looked_up
