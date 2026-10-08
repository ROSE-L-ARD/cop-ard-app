"""SarPy reader for simulated ROSE-L GPP Aresys XML + raw complex rasters.

The GPP package is expected to contain one metadata XML and one raw binary
raster for every swath/polarization pair.  Each XML may describe multiple
ScanSAR bursts.  The reader returns one SICD and one data segment per burst.

This implementation intentionally targets the supplied AresysXmlDoc 2.1
layout rather than attempting to support arbitrary ROSE-L products.
"""

from typing import Any

__classification__ = "UNCLASSIFIED"
__author__ = "ROSE-L GPP adaptation"

import copy
import glob
import logging
import os
import re
from collections.abc import Sequence
from datetime import datetime
from xml.etree import ElementTree

import numpy
import scipy
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Grid import DirParamType, GridType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.RadarCollection import (
    ChanParametersType,
    RadarCollectionType,
    WaveformParametersType,
)
from sarpy.io.complex.sicd_elements.RMA import INCAType, RMAType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.Timeline import IPPSetType, TimelineType
from sarpy.io.complex.utils import fit_position_xvalidation, fit_time_coa_polynomial
from sarpy.io.general.base import SarpyIOError
from sarpy.io.general.data_segment import NumpyMemmapSegment, SubsetSegment
from sarpy.io.general.utils import get_seconds, is_file_like
from scipy.constants import speed_of_light
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

_WGS84_A = 6378137.0
_WGS84_B = 6356752.314245179


def _clean_text(value: str | None, keep_not_available: bool = True) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if not keep_not_available and value.upper() == "NOT_AVAILABLE":
        return None
    return value


def poly_fit_2d(
    x: numpy.ndarray,
    y: numpy.ndarray,
    z: numpy.ndarray,
    x_order: int = 2,
    y_order: int = 2,
    rcond: float | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, int, numpy.ndarray]:
    # --- Input Validation ---
    if (isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray) and isinstance(z, numpy.ndarray)) is False:
        raise TypeError("x, y, z must be numpy arrays")
    if ((x.size == z.size) and (y.size == z.size)) is False:
        raise ValueError("x, y, z must have the same cardinality size.")
    # --- Data Normalization ---
    # Scale coordinates to the range [0, 1] to improve numerical stability of the fit.
    x_scale = 1.0 / x.max()
    y_scale = 1.0 / y.max()
    x = x.flatten() * x_scale
    y = y.flatten() * y_scale
    z = z.flatten()
    # --- Construct the Design Matrix (A) ---
    # This matrix relates the polynomial coefficients to the z-values.
    # Each column corresponds to a term in the polynomial (e.g., 1, x, y, xy, x^2, ...).
    A = numpy.empty((x.size, (x_order + 1) * (y_order + 1)), dtype=numpy.float64)
    for i, index in enumerate(numpy.ndindex((x_order + 1, y_order + 1))):
        A[:, i] = numpy.power(x, index[0]) * numpy.power(y, index[1])
    # --- Perform Least Squares Fit ---
    # Solve the linear system A*t = z for the coefficient vector t.
    sol, residuals, rank, sing_values = scipy.linalg.lstsq(A, z, cond=rcond)
    # --- Post -processing ---
    # Normalize the residuals by the number of data points.
    if isinstance(residuals, (numpy.ndarray, numpy.number)):
        residuals /= float(x.size)
    # Rescale the solution coefficients to match the original , unnormalized coordinates.
    sol = (
        numpy.power(x_scale, numpy.arange(x_order + 1))[:, numpy.newaxis]
        * numpy.reshape(sol, (x_order + 1, y_order + 1))
        * numpy.power(y_scale, numpy.arange(y_order + 1))
    )
    return sol, residuals, rank, sing_values


def parse_timestring(time_string: str, precision: Any = "us") -> numpy.datetime64 | None:
    """Parse timestamps such as ``31-DEC-1999 23:59:58.077657806819``."""
    value = _clean_text(time_string, keep_not_available=False)
    if value is None:
        return None
    match = re.fullmatch(r"(\d{2})-([A-Za-z]{3})-(\d{4})\s+(\d{2}:\d{2}:\d{2})(?:\.(\d+))?", value)
    if match is None:
        try:
            return numpy.datetime64(value.replace("Z", ""), precision)
        except (TypeError, ValueError):
            return None
    day, mon, year, hms, frac = match.groups()
    base = datetime.strptime(f"{day}-{mon.upper()}-{year} {hms}", "%d-%b-%Y %H:%M:%S")  # noqa: DTZ007
    # numpy can preserve more precision than datetime. Build an ISO string directly.
    iso = base.strftime("%Y-%m-%dT%H:%M:%S")
    if frac:
        iso += "." + frac
    return numpy.datetime64(iso, precision)


def _parse_xml(file_name: str) -> ElementTree.Element:
    with open(file_name, "rb") as fi:
        xml_string = fi.read()
    # The supplied file uses a prefixed namespace declaration but unqualified
    # element names.  Removing a default namespace also supports minor variants.
    xml_string = re.sub(rb'\sxmlns="[^"]+"', b"", xml_string, count=1)
    return ElementTree.fromstring(xml_string)


def _required(node: ElementTree.Element, path: str) -> ElementTree.Element:
    out = node.find(path)
    if out is None or out.text is None:
        raise SarpyIOError(f"Required ROSE-L metadata element is missing: {path}")
    return out


def _float(node: ElementTree.Element, path: str) -> float:
    node_text = _required(node, path).text
    if node_text is None:
        raise TypeError("Expected text, got None")
    return float(node_text)


def _int(node: ElementTree.Element, path: str) -> int:
    node_text = _required(node, path).text
    if node_text is None:
        raise TypeError("Expected text, got None")
    return int(node_text)


def _vals(node: ElementTree.Element, path: str) -> numpy.ndarray:
    return numpy.asarray([float(entry.text) for entry in node.findall(path)], dtype="float64")  # type: ignore


def _polarization(value: str) -> tuple[str, str, str]:
    parts = [part.strip().upper() for part in value.replace(":", "/").split("/")]
    if len(parts) != 2:
        raise SarpyIOError(f"Unexpected ROSE-L polarization {value!r}")
    tx, rcv = parts
    return tx, rcv, f"{tx}:{rcv}"


def _ecf_scp_from_zero_doppler(
    arp: numpy.ndarray, velocity: numpy.ndarray, slant_range: float, side: str
) -> numpy.ndarray:
    """Derive the ellipsoid/zero-Doppler/range-sphere intersection.

    This is not an invented geolocation value: it is the geometric solution
    implied by the XML state vector, slant range, side-looking direction, and
    WGS-84 Earth model.
    """
    arp = numpy.asarray(arp, dtype="float64")
    velocity = numpy.asarray(velocity, dtype="float64")
    rhat = arp / numpy.linalg.norm(arp)
    vhat = velocity / numpy.linalg.norm(velocity)
    cross = numpy.cross(vhat, rhat)
    cross /= numpy.linalg.norm(cross)
    # cross(v, radial) points approximately to one side of track. Try both and
    # select by the requested side after solving.
    sign = 1.0 if side.upper().startswith("R") else -1.0
    look = sign * cross
    earth_radius_guess = 0.5 * (_WGS84_A + _WGS84_B)
    nadir_distance = max(numpy.linalg.norm(arp) - earth_radius_guess, 0.0)
    along = numpy.sqrt(max(slant_range * slant_range - nadir_distance * nadir_distance, 0.0))
    x0 = arp - rhat * nadir_distance + look * along

    def residual(x):
        los = x - arp
        return numpy.asarray(
            [
                numpy.dot(los, los) - slant_range * slant_range,
                numpy.dot(velocity, los),
                (x[0] * x[0] + x[1] * x[1]) / (_WGS84_A * _WGS84_A) + x[2] * x[2] / (_WGS84_B * _WGS84_B) - 1.0,
            ],
            dtype="float64",
        )

    scales = numpy.asarray([slant_range * slant_range, numpy.linalg.norm(velocity) * slant_range, 1.0])
    result = least_squares(lambda x: residual(x) / scales, x0, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    if not result.success:
        raise SarpyIOError(f"Could not derive SCP from ROSE-L orbit/range geometry: {result.message}")
    return result.x


class ROSELSwathDetails:
    """Interpret one GPP XML/binary pair (one swath and polarization)."""

    def __init__(self, file_name: str):
        if not os.path.isfile(file_name):
            raise SarpyIOError(f"Path {file_name} does not exist or is not a file")
        self._file_name = os.path.abspath(file_name)
        self._directory_name = os.path.dirname(self._file_name)
        self._root_node = _parse_xml(self._file_name)
        if self._root_node.tag != "AresysXmlDoc":
            raise SarpyIOError(f"File {file_name} is not an AresysXmlDoc")
        self._state_time = None
        self._state_position = None
        self._state_velocity = None
        self._parse_state_vectors()
        self._bursts = self._parse_bursts()

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def directory_name(self) -> str:
        return self._directory_name

    @property
    def swath(self) -> str:
        swath_node = _required(self._root_node, "./Channel/SwathInfo/Swath").text
        if swath_node is None:
            raise TypeError()
        return swath_node.strip()

    @property
    def polarization(self) -> str:
        pol_node = _required(self._root_node, "./Channel/SwathInfo/Polarization").text
        if pol_node is None:
            raise TypeError()
        return pol_node.strip()

    @property
    def bursts(self) -> list[dict]:
        return copy.deepcopy(self._bursts)

    @property
    def binary_file(self) -> str:
        name = _required(self._root_node, "./Channel/RasterInfo/FileName").text
        if name is None:
            raise TypeError()
        name = name.strip()
        candidates = [os.path.join(self.directory_name, name)]
        _, ext = os.path.splitext(name)
        if not ext:
            candidates.extend(os.path.join(self.directory_name, name + suffix) for suffix in (".bin", ".raw", ".dat"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        # Return the metadata-declared path so the error is informative at reader construction.
        return candidates[0]

    def _parse_state_vectors(self) -> None:
        pos = _vals(self._root_node, "./Channel/StateVectorData/pSV_m/val")
        vel = _vals(self._root_node, "./Channel/StateVectorData/vSV_mOs/val")
        if pos.size != vel.size or pos.size % 3 != 0:
            raise SarpyIOError("Position and velocity state-vector arrays are incompatible")
        n = pos.size // 3
        declared = _int(self._root_node, "./Channel/StateVectorData/nSV_n")
        if n != declared:
            raise SarpyIOError(f"StateVectorData declares {declared} vectors but contains {n}")
        ref_time_node = _required(self._root_node, "./Channel/StateVectorData/t_ref_Utc").text
        if ref_time_node is None:
            raise TypeError()
        start = parse_timestring(ref_time_node, "us")
        step = _float(self._root_node, "./Channel/StateVectorData/dtSV_s")
        self._state_time = start + numpy.rint(numpy.arange(n) * step * 1e6).astype("timedelta64[us]")
        self._state_position = pos.reshape((-1, 3))  # type: ignore
        self._state_velocity = vel.reshape((-1, 3))  # type: ignore

    def _lines_per_burst(self, burst_number: int) -> int:
        changes = []
        for entry in self._root_node.findall("./Channel/BurstInfo/LinesPerBurstChangeList/Lines"):
            if entry.text is None:
                raise TypeError()
            changes.append((int(entry.attrib.get("FromBurst", "1")), int(entry.text)))
        if not changes:
            total_lines = _int(self._root_node, "./Channel/RasterInfo/Lines")
            count = _int(self._root_node, "./Channel/BurstInfo/NumberOfBursts")
            return total_lines // count
        changes.sort()
        value = changes[0][1]
        for first_burst, lines in changes:
            if burst_number >= first_burst:
                value = lines
        return value

    def _parse_bursts(self) -> list[dict]:
        bursts = []
        line_offset = 0
        samples = _int(self._root_node, "./Channel/RasterInfo/Samples")
        for entry in self._root_node.findall("./Channel/BurstInfo/Burst"):
            number = int(entry.attrib["N"])
            lines = self._lines_per_burst(number)
            range_start_time = _required(entry, "./RangeStartTime").text
            azimuth_start_time = _required(entry, "./AzimuthStartTime").text
            center_shift = _required(entry, "./BurstCenterAzimuthShift").text
            if range_start_time is None or azimuth_start_time is None or center_shift is None:
                raise TypeError()
            bursts.append(
                {
                    "number": number,
                    "line_offset": line_offset,
                    "num_lines": lines,
                    "num_samples": samples,
                    "range_start_time": float(range_start_time),
                    "azimuth_start_time": parse_timestring(azimuth_start_time, "us"),
                    "center_shift": float(center_shift),
                }
            )
            line_offset += lines
        if line_offset != _int(self._root_node, "./Channel/RasterInfo/Lines"):
            raise SarpyIOError(
                f"Burst lines sum to {line_offset}, but RasterInfo/Lines is "
                f"{_int(self._root_node, './Channel/RasterInfo/Lines')}"
            )
        return bursts

    def _position(self, collect_start: numpy.datetime64) -> PositionType:
        if self._state_time is None:
            raise TypeError()
        times = numpy.asarray(
            [get_seconds(entry, collect_start, precision="us") for entry in self._state_time], dtype="float64"
        )
        px, py, pz = fit_position_xvalidation(
            times, self._state_position, self._state_velocity, max_degree=min(8, len(times) - 1)
        )
        return PositionType(ARPPoly=XYZPolyType(X=px, Y=py, Z=pz))

    def _doppler_entries(self, tag: str) -> list[dict]:
        output: list[dict] = []
        for node in self._root_node.findall(f"./Channel/{tag}"):
            time_node = _required(node, "./taz0_Utc").text
            if time_node is None:
                raise TypeError("Expected text, got None")
            output.append(
                {
                    "number": int(node.attrib.get("Number", len(output) + 1)),
                    "coeffs": _vals(node, "./pol/val"),
                    "range_ref": _float(node, "./trg0_s"),
                    "az_time": parse_timestring(time_node, "us"),
                }
            )
        if not output:
            raise SarpyIOError(f"No {tag} entries found")
        return output

    @staticmethod
    def _nearest_entry(entries: Sequence[dict], time: numpy.datetime64) -> dict:
        return min(entries, key=lambda item: abs(get_seconds(item["az_time"], time, precision="us")))

    def _collection_info(self, burst: dict) -> CollectionInfoType:
        mode_id_node = _required(self._root_node, "./Channel/DataSetInfo/AcquisitionMode")
        if mode_id_node.text is None:
            raise TypeError("Expected string, got None")
        mode_id = mode_id_node.text.strip()
        # A single ScanSAR burst is spotlight-like in SICD's formation geometry.
        mode_type = "SPOTLIGHT" if mode_id.upper() == "SCANSAR" else "STRIPMAP"
        date = burst["azimuth_start_time"].astype(datetime).strftime("%Y%m%dT%H%M%S")
        core = f"ROSEL_{self.swath}_{self.polarization.replace('/', '')}_B{burst['number']:03d}_{date}"
        return CollectionInfoType(
            CollectorName="ROSE-L-GPP",
            CoreName=core,
            CollectType="MONOSTATIC",
            Classification="UNCLASSIFIED",
            RadarMode=RadarModeType(ModeID=mode_id, ModeType=mode_type),
            Parameters={
                "GPP_XML": os.path.basename(self.file_name),
                "SWATH": self.swath,
                "BURST": str(burst["number"]),
                "BURST_CENTER_AZIMUTH_SHIFT_S": repr(burst["center_shift"]),
            },
        )

    def _image_creation(self) -> ImageCreationType:
        processing_date = _required(self._root_node, "./Channel/DataSetInfo/ProcessingDate").text
        if processing_date is None:
            raise TypeError("Expected string, got None")
        processing = parse_timestring(processing_date, "ms")
        return ImageCreationType(Application="ROSE-L GPP", DateTime=processing, Site="N/A", Profile="sarpy")

    def _image_data(self, burst: dict) -> ImageDataType:
        samples = _int(self._root_node, "./Channel/RasterInfo/Samples")
        cell_type_node = _required(self._root_node, "./Channel/RasterInfo/CellType")
        if cell_type_node.text is None:
            raise TypeError("Expected text, got None")
        cell_type = cell_type_node.text.strip().upper()
        if cell_type != "FLOAT_COMPLEX":
            raise SarpyIOError(f"Only FLOAT_COMPLEX is supported, got {cell_type}")
        return ImageDataType(
            PixelType="RE32F_IM32F",
            NumRows=samples,
            NumCols=burst["num_lines"],
            FirstRow=0,
            FirstCol=0,  # burst["line_offset"],
            FullImage=(samples, burst["num_lines"]),
            SCPPixel=(samples // 2, burst["num_lines"] // 2),
        )

    def _radar_collection(self) -> RadarCollectionType:
        center = _float(self._root_node, "./Channel/DataSetInfo/fc_hz")
        pulse_length = _float(self._root_node, "./Channel/Pulse/PulseLength")
        bandwidth = _float(self._root_node, "./Channel/Pulse/Bandwidth")
        adc = _float(self._root_node, "./Channel/Pulse/PulseSamplingRate")
        start_freq_offset = _float(self._root_node, "./Channel/Pulse/PulseStartFrequency")
        tx, _, tx_rcv = _polarization(self.polarization)
        samples = _int(self._root_node, "./Channel/RasterInfo/Samples")
        waveform = WaveformParametersType(
            index=1,
            TxPulseLength=pulse_length,
            TxRFBandwidth=bandwidth,
            TxFreqStart=center + start_freq_offset,
            TxFMRate=bandwidth / pulse_length,
            RcvWindowLength=samples / adc,
            ADCSampleRate=adc,
            RcvIFBandwidth=bandwidth,
            RcvDemodType="CHIRP",
            RcvFMRate=0.0,
        )
        return RadarCollectionType(
            TxFrequency=(center - 0.5 * bandwidth, center + 0.5 * bandwidth),
            TxPolarization=tx,
            Waveform=[waveform],
            RcvChannels=[ChanParametersType(index=1, TxRcvPolarization=tx_rcv)],
        )

    def _timeline(self, burst: dict) -> TimelineType:
        line_step = _float(self._root_node, "./Channel/RasterInfo/LinesStep")
        duration = burst["num_lines"] * line_step
        prf_node = self._root_node.find("./Channel/AcquisitionTimeLine/PRF_changes_values/val")
        prf = float(prf_node.text) if prf_node is not None and prf_node.text is not None else 1.0 / line_step
        ipp_end = max(round(duration * prf) - 1, 0)
        ipp = IPPSetType(
            index=1,
            TStart=0.0,
            TEnd=duration,
            IPPStart=0,
            IPPEnd=ipp_end,
            IPPPoly=Poly1DType(Coefs=(0.0, prf)),
        )
        return TimelineType(CollectStart=burst["azimuth_start_time"], CollectDuration=duration, IPP=[ipp])

    def _get_demodulation_phase(
        self,
        burst: dict,
    ):
        def _evaluate_gpp_poly(poly, az_axis, rg_axis):
            coeffs = numpy.zeros((2, poly["coeffs"].size - 2), dtype="float64")
            coeffs[0, :2] = poly["coeffs"][:2]
            coeffs[1, :2] = poly["coeffs"][2:4]
            coeffs[0, 2:] = poly["coeffs"][4:]
            coeffs = coeffs.T

            x = rg_axis - poly["range_ref"]
            y = (az_axis - poly["az_time"]).astype("float64") / 1e9

            return numpy.polynomial.polynomial.polygrid2d(x, y, coeffs)

        # some usefull parameters for the burst (TODO Remove all not necessary)
        sample_time = _float(self._root_node, "./Channel/RasterInfo/SamplesStep")
        line_step = _float(self._root_node, "./Channel/RasterInfo/LinesStep")

        dop_rate = self._nearest_entry(self._doppler_entries("DopplerRate"), burst["azimuth_start_time"])
        dop_cent = self._nearest_entry(self._doppler_entries("DopplerCentroid"), burst["azimuth_start_time"])

        # Implementation of the formula on the ARBD of GPP
        # t_a is the abosolute azimuth time
        # k_a is the doppler rate (dop_rate) evaluated on the grid I use to make the fit
        # f_dc is the doppler centroid (dop_cent) evaluated on the grid I use to make the fit
        az_axis = (
            numpy.round(numpy.linspace(0, burst["num_lines"], 50) * line_step * 1e9).astype("timedelta64[ns]")
            + burst["azimuth_start_time"]
        )
        rg_axis = numpy.linspace(0, burst["num_samples"], 50) * sample_time + burst["range_start_time"]

        t_r = numpy.tile(rg_axis, az_axis.size).reshape((az_axis.size, rg_axis.size))
        k_a = _evaluate_gpp_poly(dop_rate, az_axis, rg_axis)
        f_dc = _evaluate_gpp_poly(dop_cent, az_axis, rg_axis)

        k_t = -k_a
        t_c = -f_dc / k_a
        t_ref = burst["num_lines"] // 2 * line_step + t_c - t_c[:, 0]
        t_a = numpy.repeat(numpy.linspace(0, burst["num_lines"], 50) * line_step, rg_axis.size).reshape(
            (az_axis.size, rg_axis.size)
        )

        # This is the SCANSAR component of the demodulation phase
        phi_demod = numpy.pi * k_t * (t_a - t_ref) ** 2

        phi_demod += 2 * numpy.pi * f_dc * t_a

        return phi_demod, t_a, t_r - t_r[:, 0]

    def _grid_rma_geo(
        self,
        burst: dict,
        image_data: ImageDataType,
        position: PositionType,
        collection_info: CollectionInfoType,
    ) -> tuple[GridType, RMAType, GeoDataType, SCPCOAType]:
        center_frequency = _float(self._root_node, "./Channel/DataSetInfo/fc_hz")
        sample_time = _float(self._root_node, "./Channel/RasterInfo/SamplesStep")
        row_ss = 0.5 * speed_of_light * sample_time
        row_bw = 2.0 * _float(self._root_node, "./Channel/Pulse/Bandwidth") / speed_of_light
        line_step = _float(self._root_node, "./Channel/RasterInfo/LinesStep")
        burst_duration = burst["num_lines"] * line_step
        time_ca = 0.5 * burst_duration + burst["center_shift"]
        scp_range_time = burst["range_start_time"] + image_data.SCPPixel.Row * sample_time
        r_ca_scp = 0.5 * speed_of_light * scp_range_time  # Range in meters corresponding to the SCP
        arp = position.ARPPoly(time_ca)
        vel = position.ARPPoly.derivative_eval(time_ca, der_order=1)
        side_node = _required(self._root_node, "./Channel/DataSetInfo/SideLooking")
        if side_node.text is None:
            raise TypeError("Expected text, got None")
        side = side_node.text.strip().upper()
        scp_ecf = _ecf_scp_from_zero_doppler(arp, vel, r_ca_scp, side)
        geo = GeoDataType(SCP=SCPType(ECF=scp_ecf))
        scpcoa = SCPCOAType(SideOfTrack=side[0])
        row = DirParamType(
            SS=row_ss,
            ImpRespBW=row_bw,
            ImpRespWid=1.0 / row_bw,
            Sgn=-1,
            KCtr=2.0 * center_frequency / speed_of_light,
            DeltaKCOAPoly=Poly2DType(Coefs=((0.0,),)),
        )
        col = DirParamType(Sgn=-1, KCtr=0.0)
        grid = GridType(ImagePlane="SLANT", Type="RGZERO", Row=row, Col=col)

        dop_rate = self._nearest_entry(self._doppler_entries("DopplerRate"), burst["azimuth_start_time"])
        dop_cent = self._nearest_entry(self._doppler_entries("DopplerCentroid"), burst["azimuth_start_time"])
        alpha = 2.0 / speed_of_light
        velocity_sq = float(numpy.dot(vel, vel))
        r_ca = numpy.asarray([r_ca_scp, 1.0], dtype="float64")

        rate_poly = Poly1DType(Coefs=dop_rate["coeffs"])
        rate_t0 = dop_rate["range_ref"] - alpha * r_ca_scp
        rate_scaled = rate_poly.shift(rate_t0, alpha, return_poly=False)
        drate_coeffs = -numpy.convolve(rate_scaled, r_ca) / (alpha * center_frequency * velocity_sq)
        inca = INCAType(
            R_CA_SCP=r_ca_scp,
            FreqZero=center_frequency,
            DRateSFPoly=Poly2DType(Coefs=drate_coeffs.reshape((-1, 1))),
            TimeCAPoly=Poly1DType(Coefs=(time_ca,)),
        )
        ss_scale = abs(numpy.sqrt(velocity_sq) * inca.DRateSFPoly[0, 0])
        if not numpy.isfinite(ss_scale) or ss_scale <= 0:
            raise SarpyIOError("Invalid azimuth sample-spacing scale derived from DopplerRate")
        look = -1.0 if side.startswith("R") else 1.0
        inca.TimeCAPoly = Poly1DType(Coefs=(time_ca, -look / ss_scale))
        grid.Col.SS = line_step * ss_scale

        # Baz_hz is the processed azimuth bandwidth in this GPP metadata.
        az_bw = _float(self._root_node, "./Channel/SamplingConstants/Baz_hz")
        grid.Col.ImpRespBW = az_bw / ss_scale
        grid.Col.ImpRespWid = 1.0 / grid.Col.ImpRespBW

        cent_poly = Poly1DType(Coefs=dop_cent["coeffs"])
        cent_t0 = dop_cent["range_ref"] - alpha * r_ca_scp
        cent_scaled = cent_poly.shift(cent_t0, alpha, return_poly=False)

        # Construct the burst Doppler centroid as a 2-D SICD polynomial
        # f_DC(x_rg, y_az) = f_annotated(x_rg) + y_az * f_geometry(x_rg).
        #
        # This mirrors the branch in SarPy's Radarsat reader, with
        # one deliberate correction: the Radarsat excerpt often circulated
        # externally allocates `dop_poly` from cent_scaled.shape but does not
        # visibly copy cent_scaled into column zero.  For ROSE-L we retain the
        # annotated range Doppler and add the acquisition-geometry term.
        if collection_info.RadarMode.ModeType != "SPOTLIGHT":
            raise SarpyIOError(
                "ROSE-L ScanSAR bursts are expected to use SICD SPOTLIGHT "
                "mode for the burst-local Doppler geometry model"
            )

        dop_poly = numpy.zeros((cent_scaled.shape[0], 2), dtype="float64")
        dop_poly[:, 0] = cent_scaled

        speed_ca = numpy.sqrt(velocity_sq)
        # Linear azimuth-frequency slope at R_CA_SCP and its first-order
        # range variation.  alpha*f0 = 2/lambda.
        dop_poly[0, 1] = -look * center_frequency * alpha * speed_ca / r_ca_scp
        if dop_poly.shape[0] > 1:
            dop_poly[1, 1] = -center_frequency * alpha * speed_ca / (r_ca_scp**2)

        # The annotation time is absolute.  Position.ARPPoly is referenced to
        # this burst's CollectStart, so convert to burst-relative seconds.
        doppler_cent_est = get_seconds(dop_cent["az_time"], burst["azimuth_start_time"], precision="us")
        one_way_time = r_ca_scp / speed_of_light
        use_time = doppler_cent_est + one_way_time
        pos_est = position.ARPPoly(use_time)
        vel_est = position.ARPPoly.derivative_eval(use_time, der_order=1)
        vel_norm = numpy.linalg.norm(vel_est)
        if not numpy.isfinite(vel_norm) or vel_norm <= 0:
            raise SarpyIOError("Invalid ARP velocity at Doppler centroid estimate time")
        los = geo.SCP.ECF.get_array() - pos_est
        vel_hat = vel_est / vel_norm
        along_track_offset = float(numpy.dot(los, vel_hat))

        # Translate the geometry polynomial from the estimate-time along-track
        # origin to the SICD azimuth-coordinate origin.
        dop_poly[:, 0] += -look * dop_poly[:, 1] * along_track_offset

        inca.DopCentroidPoly = Poly2DType(Coefs=dop_poly)
        grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=inca.DopCentroidPoly.get_array() * line_step / grid.Col.SS)

        phi_demod, t_a, t_r = self._get_demodulation_phase(burst)
        azimuth_m = t_a / line_step * grid.Col.SS - (image_data.SCPPixel.Col * grid.Col.SS)
        slant_range_m = t_r / sample_time * grid.Row.SS - (image_data.SCPPixel.Row * grid.Row.SS)

        coeffs, _, _, _ = poly_fit_2d(slant_range_m, azimuth_m, phi_demod, 2, 3)

        grid.Col.DeltaKCOAPoly = Poly2DType(
            Coefs=numpy.polynomial.polynomial.polyder(coeffs / (2 * numpy.pi), 1, axis=1)
        )

        coeffs = grid.Col.DeltaKCOAPoly.get_array()[:, 0]

        # SarPy's helper supplies the SICD Grid.TimeCOAPoly consistent with INCA.
        try:
            time_coa = fit_time_coa_polynomial(inca, image_data, grid, rate_scaled, poly_order=2)
            if collection_info.RadarMode.ModeType == "SPOTLIGHT":
                grid.TimeCOAPoly = Poly2DType(Coefs=((time_coa.Coefs[0, 0],),))
                inca.DopCentroidPoly = None
            else:
                grid.TimeCOAPoly = time_coa
                inca.DopCentroidCOA = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fit Grid.TimeCOAPoly: %s", exc)
            grid.TimeCOAPoly = Poly2DType(Coefs=((time_ca,),))
            if collection_info.RadarMode.ModeType == "SPOTLIGHT":
                inca.DopCentroidPoly = None

        return grid, RMAType(RMAlgoType="OMEGA_K", INCA=inca), geo, scpcoa

    def get_sicds(self) -> list[SICDType]:
        output = []
        _, _, tx_rcv = _polarization(self.polarization)
        radar_collection = self._radar_collection()
        image_creation = self._image_creation()
        for burst in self._bursts:
            collection = self._collection_info(burst)
            image_data = self._image_data(burst)
            timeline = self._timeline(burst)
            position = self._position(timeline.CollectStart)
            grid, rma, geo, scpcoa = self._grid_rma_geo(burst, image_data, position, collection)
            image_formation = ImageFormationType(
                RcvChanProc=RcvChanProcType(NumChanProc=1, PRFScaleFactor=1.0, ChanIndices=[1]),
                TxRcvPolarizationProc=tx_rcv,
                ImageFormAlgo="RMA",
                TStartProc=0.0,
                TEndProc=timeline.CollectDuration,
                TxFrequencyProc=(radar_collection.TxFrequency.Min, radar_collection.TxFrequency.Max),
                STBeamComp="GLOBAL",
                ImageBeamComp="SV",
                AzAutofocus="NO",
                RgAutofocus="NO",
            )
            sicd = SICDType(
                CollectionInfo=collection,
                ImageCreation=image_creation,
                ImageData=image_data,
                GeoData=geo,
                Grid=grid,
                Timeline=timeline,
                Position=position,
                RadarCollection=copy.deepcopy(radar_collection),
                ImageFormation=image_formation,
                SCPCOA=scpcoa,
                RMA=rma,
            )
            sicd.derive()
            output.append(sicd)
        return output

    def get_data_segments(self) -> list[SubsetSegment]:
        path = self.binary_file
        if not os.path.isfile(path):
            raise SarpyIOError(f"Binary raster declared by {self.file_name} was not found: {path}")
        lines = _int(self._root_node, "./Channel/RasterInfo/Lines")
        samples = _int(self._root_node, "./Channel/RasterInfo/Samples")
        offset = _int(self._root_node, "./Channel/RasterInfo/HeaderOffsetBytes")
        row_prefix = _int(self._root_node, "./Channel/RasterInfo/RowPrefixBytes")
        if row_prefix != 0:
            raise SarpyIOError("RowPrefixBytes other than zero is not currently supported")
        byte_order_node = _required(self._root_node, "./Channel/RasterInfo/ByteOrder")
        if byte_order_node.text is None:
            raise TypeError("Expected text, got None")
        byte_order = byte_order_node.text.strip().upper()
        dtype = numpy.dtype("<c8" if byte_order == "LITTLEENDIAN" else ">c8")
        parent = NumpyMemmapSegment(
            path,
            data_offset=offset,
            raw_dtype=dtype,
            raw_shape=(lines, samples),
            formatted_dtype=numpy.dtype("complex64"),
            formatted_shape=(samples, lines),
            transpose_axes=(1, 0),
            close_file=True,
        )
        segments = []
        for index, burst in enumerate(self._bursts):
            start = burst["line_offset"]
            stop = start + burst["num_lines"]
            segments.append(
                SubsetSegment(
                    parent,
                    (slice(0, samples), slice(start, stop)),
                    coordinate_basis="formatted",
                    squeeze=False,
                    close_parent=(index == len(self._bursts) - 1),
                )
            )
        return segments


class ROSELDetails:
    """Discover every swath/polarization XML in a GPP product folder."""

    def __init__(self, path: str):
        if os.path.isdir(path):
            xml_files = sorted(glob.glob(os.path.join(os.path.abspath(path), "*.xml")))
        elif os.path.isfile(path):
            xml_files = [os.path.abspath(path)]
        else:
            raise SarpyIOError(f"Path {path} does not exist")
        details = []
        for xml_file in xml_files:
            try:
                details.append(ROSELSwathDetails(xml_file))
            except (SarpyIOError, ElementTree.ParseError, ValueError):
                continue
        if not details:
            raise SarpyIOError(f"No ROSE-L GPP Aresys XML files found under {path}")
        self._directory_name = os.path.dirname(details[0].file_name) if os.path.isfile(path) else os.path.abspath(path)
        self._swaths = details

    @property
    def directory_name(self) -> str:
        return self._directory_name

    @property
    def swaths(self) -> tuple[ROSELSwathDetails, ...]:
        return tuple(self._swaths)

    def get_sicds(self) -> list[SICDType]:
        return [sicd for detail in self._swaths for sicd in detail.get_sicds()]


class ROSELReader(SICDTypeReader):
    """SarPy complex reader for a ROSE-L GPP folder or one metadata XML."""

    __slots__ = ("_rosel_details",)

    def __init__(self, rosel_details: str | ROSELDetails):
        if isinstance(rosel_details, str):
            rosel_details = ROSELDetails(rosel_details)
        if not isinstance(rosel_details, ROSELDetails):
            raise TypeError("rosel_details must be a path or ROSELDetails")
        self._rosel_details = rosel_details
        sicds = []
        segments = []
        for detail in rosel_details.swaths:
            detail_sicds = detail.get_sicds()
            detail_segments = detail.get_data_segments()
            if len(detail_sicds) != len(detail_segments):
                raise SarpyIOError("ROSE-L SICD/data segment count mismatch")
            sicds.extend(detail_sicds)
            segments.extend(detail_segments)
        super().__init__(segments, sicds, close_segments=True)
        self._check_sizes()

    @property
    def rosel_details(self) -> ROSELDetails:
        return self._rosel_details

    @property
    def file_name(self) -> str:
        return self._rosel_details.directory_name


def is_a(file_name: str) -> ROSELReader | None:
    """Return a reader when *file_name* is a ROSE-L GPP XML/folder, else None."""
    if is_file_like(file_name):
        return None
    try:
        details = ROSELDetails(file_name)
        return ROSELReader(details)
    except (SarpyIOError, ElementTree.ParseError, ValueError, OSError, OverflowError):
        return None


__all__ = [
    "ROSELDetails",
    "ROSELReader",
    "ROSELSwathDetails",
    "is_a",
    "parse_timestring",
]
