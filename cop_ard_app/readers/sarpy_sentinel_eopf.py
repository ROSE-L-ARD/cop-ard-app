# MIT License

# Copyright (c) 2018-2021 National Geospatial-Intelligence Agency

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Functionality for reading Sentinel-1 data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Daniel Haverporth")


import logging
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy
import numpy.typing
import pandas as pd  # FIXME find an alternative method for datetime64 -> datetime.datetime conversion
import xarray as xr
from numpy.polynomial import polynomial
from sarpy.geometry.geocoords import geodetic_to_ecf
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Grid import DirParamType, GridType, WgtTypeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.RadarCollection import (
    ChanParametersType,
    RadarCollectionType,
    WaveformParametersType,
)
from sarpy.io.complex.sicd_elements.Radiometric import NoiseLevelType_, RadiometricType
from sarpy.io.complex.sicd_elements.RMA import INCAType, RMAType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.Timeline import IPPSetType, TimelineType
from sarpy.io.complex.utils import get_im_physical_coords, two_dim_poly_fit
from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.data_segment import DataSegment
from sarpy.io.general.format_function import FormatFunction
from sarpy.io.general.slice_parsing import get_subscript_result_size
from sarpy.io.general.utils import get_seconds, parse_timestring
from scipy.constants import speed_of_light
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


###########
# parser and interpreter for sentinel-1 manifest.safe file


class SentinelDetails(object):
    __slots__ = ("_data", "_satellite", "_product_type", "_base_sicd")

    def __init__(self, data: xr.DataTree):
        """
        Parameters
        ----------
        data : xr.DataTree
            Input data to initialize SentinelDetails.
        """
        self._data = data

        self._satellite = self.get_platform_family_name().upper()
        if self._satellite != "SENTINEL-1":
            raise ValueError(
                "The platform in the manifest.safe file is required to be SENTINEL-1, got {}".format(self._satellite)
            )
        self._product_type = self.get_product_type()
        if self._product_type != "SLC":
            raise ValueError(
                'The product type in the manifest.safe file is required to be "SLC", got {}'.format(self._product_type)
            )
        self._base_sicd = self._get_base_sicd()

    @property
    def satellite(self) -> str:
        """
        str: the satellite
        """

        return self._satellite

    @property
    def product_type(self) -> str:
        """
        str: the product type
        """

        return self._product_type

    @staticmethod
    def _parse_pol(str_in: str) -> str:
        return "{}:{}".format(str_in[0], str_in[1])

    def get_all_swaths(self) -> List[dict]:
        """
        Extracts paths for measurement and metadata files from a Sentinel manifest.safe file.
        These files will be grouped according to "measurement data unit" implicit in the
        Sentinel structure.

        Returns
        -------
        List[dict]
        """
        mappings = []
        for k, child in self._data.children.items():
            swath = child.attrs["stac_discovery"]["properties"].get("sar:swath")
            name = child.name
            if not isinstance(name, str):
                raise ValueError(f"Expected swath name, got {name}")
            if swath is None:  # fallback to group name
                if child.attrs["stac_discovery"]["properties"]["sar:instrument_mode"] in ("IW", "EW", "WV"):
                    swath = name.split("_")[-2].upper()  # e.g., IW1, EW1, WV1
                else:
                    swath = name[3:6]
            pol = child.attrs["stac_discovery"]["properties"].get("sar:polarization")
            if pol is None:  # fallback to group name
                fields = [field for field in name.split("_") if len(field) == 2]  # only pol
                pol_candidates = [pol for pol in ("VV", "VH", "HH", "HV") if pol in fields]
                if len(pol_candidates) == 0:
                    raise ValueError("Unable to determine polarization")
                elif len(pol_candidates) > 1:
                    raise ValueError("Multiple polarizations found: {}".format(pol_candidates))
                pol = pol_candidates[0]
            mappings.append(((swath, pol), (child, child.measurements.ds["slc"])))
        groups: dict[Any, Any] = {}
        for pair, child in mappings:
            if pair not in groups:
                groups[pair] = []
            groups[pair].append(child)
        groups_values = cast(list[dict], groups.values())
        return groups_values

    def get_platform_family_name(self) -> str:
        """
        Retrieves the platform family name.

        Returns
        -------
        str
        """
        return self._data.attrs["stac_discovery"]["properties"]["constellation"]

    def get_product_type(self) -> str:
        """
        Retrieves the product type.

        Returns
        -------
        str
        """
        return self._data.attrs["stac_discovery"]["properties"]["product:type"][6:]

    def get_transmitter_receiver_polarizations(self) -> List[str]:
        """
        Retrieves the transmitter-receiver polarizations.

        Returns
        -------
        List[ElementTree.Element]
        """
        return self._data.attrs["stac_discovery"]["properties"]["sar:polarizations"]

    def get_processing_facility(self) -> str:
        """
        Retrieves the processing facility.

        Returns
        -------
        ElementTree.Element
        """
        for burst in self._data.values():
            facility = burst.attrs["processing_history"]["Level-1 Intermediate SLC Product"][0]["facility"]
            break
        return facility

    def get_processing_software(self) -> str:
        """
        Retrieves the processing software.

        Returns
        -------
        ElementTree.Element
        """
        for burst in self._data.values():
            processor = burst.attrs["processing_history"]["Level-1 Intermediate SLC Product"][0]["processor"]
            break
        return processor

    def get_processing_software_version(self) -> str:
        """
        Retrieves the processing software.

        Returns
        -------
        ElementTree.Element
        """
        for burst in self._data.values():
            version = burst.attrs["processing_history"]["Level-1 Intermediate SLC Product"][0]["version"]
            break
        return version

    def get_processing_stop_time(self) -> str:
        """
        Retrieves the processing stop time.

        Returns
        -------
        str
        """
        for burst in self._data.values():
            time = burst.attrs["processing_history"]["Level-1 Intermediate SLC Product"][0]["time"]
            break
        return time

    def get_platform_instrument_mode(self) -> str:
        """
        Retrieves the platform instrument mode.

        Returns
        -------
        str
        """
        return self._data.attrs["stac_discovery"]["properties"]["sar:instrument_mode"]

    def get_platform_number(self) -> str:
        family_name = self.get_platform_family_name()
        return str(self._data.attrs["stac_discovery"]["properties"]["platform"]).removeprefix(family_name)

    def _get_base_sicd(self) -> SICDType:
        """
        Gets the base SICD element.

        Returns
        -------
        SICDType
        """
        from sarpy.__about__ import __version__

        # CollectionInfo
        collector_name = (self.get_platform_family_name() + self.get_platform_number()).upper()
        mode_id = self.get_platform_instrument_mode()
        mode_type = "STRIPMAP" if mode_id == "SM" else "DYNAMIC STRIPMAP"
        collection_info = CollectionInfoType(
            Classification="UNCLASSIFIED",
            CollectorName=collector_name,
            CollectType="MONOSTATIC",
            RadarMode=RadarModeType(ModeID=mode_id, ModeType=mode_type),
        )
        # ImageCreation
        facility = self.get_processing_facility()
        software = self.get_processing_software()
        software_version = self.get_processing_software_version()
        image_creation = ImageCreationType(
            Application="{name} {version}".format(name=software, version=software_version),
            DateTime=self.get_processing_stop_time(),
            Site=facility,
            Profile="sarpy {}".format(__version__),
        )
        # RadarCollection
        polarizations = self.get_transmitter_receiver_polarizations()
        radar_collection = RadarCollectionType(
            RcvChannels=[
                ChanParametersType(TxRcvPolarization=self._parse_pol(pol), index=i)
                for i, pol in enumerate(polarizations)
            ]
        )
        return SICDType(CollectionInfo=collection_info, ImageCreation=image_creation, RadarCollection=radar_collection)

    def _parse_product_sicd(self, burst_list) -> Union[SICDType, List[SICDType]]:
        """

        Parameters
        ----------
        product_file_name : str

        Returns
        -------
        SICDType|List[SICDType]
        """

        DT_FMT = "%Y-%m-%dT%H:%M:%S.%f"
        # parse the geolocation information - for SCP calculation
        geo_pixels, geo_coords = self.get_geo_coords(burst_list[0])

        def get_center_frequency():
            return burst_list[0].attrs["stac_discovery"]["properties"]["sar:center_frequency"]

        def get_image_col_spacing_zdt():
            # Image column spacing in zero doppler time (seconds)
            # Sentinel-1 is always right-looking, so this should always be positive
            delta = burst_list[0].attrs["other_metadata"]["image_annotation"]["image_information"][
                "azimuth_time_interval"
            ]
            return float(delta)

        def get_image_data():  # type: () -> ImageDataType
            pixel_type = get_image_data_type()
            num_rows, num_cols = get_total_image_size()
            # SCP pixel within single burst image is the same for all burst
            return ImageDataType(
                PixelType=pixel_type,
                NumRows=num_rows,
                NumCols=num_cols,
                FirstRow=0,
                FirstCol=0,
                FullImage=(num_rows, num_cols),
                SCPPixel=(int((num_rows - 1) / 2), int((num_cols - 1) / 2)),
            )

        def get_total_image_size():
            num_cols = burst_list[0].measurements.slc.azimuth_time.size
            num_rows = burst_list[0].measurements.slc.slant_range_time.size
            return num_rows, num_cols

        def get_image_data_type():
            if numpy.iscomplexobj(burst_list[0].measurements.slc.data):
                pixel_type = "RE16I_IM16I"
            else:
                # NB: we only handle SLC
                raise ValueError("SLC data should be 16-bit complex, got pixelValue = .")
            return pixel_type

        def get_common_grid():  # type: () -> GridType
            center_frequency = get_center_frequency()
            image_plane = get_image_projection()
            # get range processing node
            range_proc = None
            delta_tau_s = 1.0 / get_range_sampling_rate()
            row_window_name, row_params = get_range_window_params(range_proc)
            row = DirParamType(
                SS=(speed_of_light / 2) * delta_tau_s,
                Sgn=-1,
                KCtr=2 * center_frequency / speed_of_light,
                ImpRespBW=2.0 * get_range_processing_bandwidth(range_proc) / speed_of_light,
                DeltaKCOAPoly=Poly2DType(
                    Coefs=[
                        [
                            0,
                        ]
                    ]
                ),
                WgtType=WgtTypeType(WindowName=row_window_name, Parameters=row_params),
            )
            # get azimuth processing node
            az_proc = None
            col_ss = get_azimuth_pixel_spacing()
            dop_bw = get_azimuth_processing_bandwidth(az_proc)
            ss_zd_s = get_image_col_spacing_zdt()
            col_window_name, col_params = get_azimuth_window_params(az_proc)
            col = DirParamType(
                SS=col_ss,
                Sgn=-1,
                KCtr=0,
                ImpRespBW=dop_bw * ss_zd_s / col_ss,
                WgtType=WgtTypeType(WindowName=col_window_name, Parameters=col_params),
            )
            return GridType(ImagePlane=image_plane, Type="RGZERO", Row=row, Col=col)

        def get_azimuth_window_params(az_proc):
            col_window_name = (
                burst_list[0]
                .attrs["other_metadata"]["image_annotation"]["processing_information"]["swath_proc_params_list"][0][
                    "azimuth_processing"
                ]["window_type"]
                .upper()
            )
            col_params = None
            if col_window_name == "NONE":
                col_window_name = "UNIFORM"
            elif col_window_name == "HAMMING":
                col_window_param = burst_list[0].attrs["other_metadata"]["image_annotation"]["processing_information"][
                    "swath_proc_params_list"
                ][0]["azimuth_processing"]["window_coefficient"]
                col_params = {"COEFFICIENT": str(col_window_param)}
            return col_window_name, col_params

        def get_azimuth_processing_bandwidth(az_proc):
            return burst_list[0].attrs["other_metadata"]["image_annotation"]["processing_information"][
                "swath_proc_params_list"
            ][0]["azimuth_processing"]["processing_bandwidth"]

        def get_azimuth_pixel_spacing():
            col_ss = burst_list[0].attrs["stac_discovery"]["properties"]["sar:pixel_spacing_azimuth"]  # [m]
            return col_ss

        def get_range_processing_bandwidth(range_proc):
            return burst_list[0].attrs["other_metadata"]["image_annotation"]["processing_information"][
                "swath_proc_params_list"
            ][0]["range_processing"]["processing_bandwidth"]

        def get_range_window_params(range_proc):
            row_window_name = (
                burst_list[0]
                .attrs["other_metadata"]["image_annotation"]["processing_information"]["swath_proc_params_list"][0][
                    "range_processing"
                ]["window_type"]
                .upper()
            )
            row_params = None
            if row_window_name == "NONE":
                row_window_name = "UNIFORM"
            elif row_window_name == "HAMMING":
                row_window_param = burst_list[0].attrs["other_metadata"]["image_annotation"]["processing_information"][
                    "swath_proc_params_list"
                ][0]["range_processing"]["window_coefficient"]
                row_params = {"COEFFICIENT": str(row_window_param)}
            return row_window_name, row_params

        def get_range_sampling_rate():
            return burst_list[0].attrs["other_metadata"]["general_annotation"]["product_information"][
                "range_sampling_rate"
            ]

        def get_image_projection():
            image_plane = "SLANT" if "slc" in burst_list[0].measurements else None

            return image_plane

        def get_common_timeline():  # type: () -> TimelineType
            prf = get_prf()
            # NB: TEnd and IPPEnd are nonsense values which will be corrected
            return TimelineType(
                IPP=[
                    IPPSetType(TStart=0, TEnd=0, IPPStart=0, IPPEnd=0, IPPPoly=(0, prf), index=0),
                ]
            )

        def get_common_radar_collection():  # type: () -> RadarCollectionType
            radar_collection = out_sicd.RadarCollection.copy()
            center_frequency = get_center_frequency()
            min_frequency = center_frequency + get_tx_pulse_start_frequency()
            tx_pulse_length = get_tx_pulse_length()
            tx_fm_rate = get_tx_pulse_fm_rate()
            band_width = tx_pulse_length * tx_fm_rate
            pol = get_polarization()
            radar_collection.TxPolarization = pol[0]
            radar_collection.TxFrequency = (min_frequency, min_frequency + band_width)
            adc_sample_rate = get_range_sampling_rate()  # Raw not decimated
            swl_list = get_all_swl()
            radar_collection.Waveform = [
                WaveformParametersType(
                    index=j,
                    TxFreqStart=min_frequency,
                    TxPulseLength=tx_pulse_length,
                    TxFMRate=tx_fm_rate,
                    TxRFBandwidth=band_width,
                    RcvFMRate=0,
                    ADCSampleRate=adc_sample_rate,
                    RcvWindowLength=float(swl["value"]),
                )
                for j, swl in enumerate(swl_list)
            ]
            return radar_collection

        def get_all_swl():
            swls = burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                "downlink_values"
            ]["swl_list"]
            return [{"azimuth_time": swl["azimuth_time"], "value": swl["value"]} for swl in swls]

        def get_polarization():
            pol = burst_list[0].attrs["stac_discovery"]["properties"].get("sar:polarization")
            if pol is None:  # fallback to group name
                fields = [field for field in burst_list[0].name.split("_") if len(field) == 2]  # only pol
                pol_candidates = [pol for pol in ("VV", "VH", "HH", "HV") if pol in fields]
                if len(pol_candidates) == 0:
                    raise ValueError("Unable to determine polarization")
                elif len(pol_candidates) > 1:
                    raise ValueError("Multiple polarizations found: {}".format(pol_candidates))
                pol = pol_candidates[0]
            return pol

        def get_tx_pulse_fm_rate():
            return float(
                burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                    "downlink_values"
                ]["tx_pulse_ramp_rate"]
            )

        def get_tx_pulse_length():
            return float(
                burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                    "downlink_values"
                ]["tx_pulse_length"]
            )

        def get_tx_pulse_start_frequency():
            return float(
                burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                    "downlink_values"
                ]["tx_pulse_start_frequency"]
            )

        def get_image_formation():  # type: () -> ImageFormationType
            st_beam_comp = "GLOBAL" if out_sicd.CollectionInfo.RadarMode.ModeID[0] == "S" else "SV"
            pol = self._parse_pol(get_polarization())
            # which channel does this pol correspond to?
            chan_indices = None
            for element in out_sicd.RadarCollection.RcvChannels:
                if element.TxRcvPolarization == pol:
                    chan_indices = [
                        element.index,
                    ]

            return ImageFormationType(
                RcvChanProc=RcvChanProcType(NumChanProc=1, PRFScaleFactor=1, ChanIndices=chan_indices),
                TxRcvPolarizationProc=pol,
                TStartProc=0,
                TxFrequencyProc=(out_sicd.RadarCollection.TxFrequency.Min, out_sicd.RadarCollection.TxFrequency.Max),
                ImageFormAlgo="RMA",
                ImageBeamComp="SV",
                AzAutofocus="NO",
                RgAutofocus="NO",
                STBeamComp=st_beam_comp,
            )

        def get_rma():  # type: () -> RMAType
            center_frequency = get_center_frequency()
            tau_0 = get_first_slant_range_time()
            delta_tau_s = 1.0 / get_range_sampling_rate()
            return RMAType(
                RMAlgoType="RG_DOP",
                INCA=INCAType(
                    FreqZero=center_frequency,
                    DopCentroidCOA=True,
                    R_CA_SCP=(0.5 * speed_of_light) * (tau_0 + out_sicd.ImageData.SCPPixel.Row * delta_tau_s),
                ),
            )

        def get_first_slant_range_time():
            return float(burst_list[0].measurements.slc.slant_range_time[0])

        def get_slice() -> int:
            return burst_list[0].attrs["other_metadata"]["image_annotation"]["image_information"]["slice_number"]

        def get_swath() -> str:
            instrument_mode = self.get_platform_instrument_mode()
            swath = burst_list[0].attrs["stac_discovery"]["properties"].get("sar:swath")
            if swath is None:  # fallback to group name
                if instrument_mode in ("IW", "EW", "WV"):
                    swath = burst_list[0].name.split("_")[-2].upper()  # e.g., IW1, EW1, WV1
                else:
                    swath = burst_list[0].name[3:6]
            return swath

        def get_collection_info():  # type: () -> CollectionInfoType
            collection_info = out_sicd.CollectionInfo.copy()
            collection_info.CollectorName = get_mission_id()
            collection_info.RadarMode.ModeID = get_radar_mode()
            t_slice = get_slice()
            swath = get_swath()
            collection_info.Parameters = {
                "SLICE": t_slice,
                "BURST": "1",
                "SWATH": swath,
                "ORBIT_SOURCE": "SLC_INTERNAL",
            }
            return collection_info

        def get_radar_mode():
            return self.get_platform_instrument_mode()

        def get_mission_id():
            mission_id = f"{self.get_platform_family_name().lower().replace(r'sentinel-1', r'S1')}"
            platform_number = f"{self.get_platform_number()}"
            return f"{mission_id}{platform_number}".upper()

        def get_state_vectors(start: numpy.datetime64):
            orbit_dataset = burst_list[0].conditions.orbit.ds
            Ts: numpy.typing.NDArray[numpy.datetime64] = orbit_dataset["azimuth_time"].values
            d_Ts = Ts - start
            d_Ts = d_Ts / numpy.timedelta64(1, "s")
            positions = orbit_dataset["position"]
            Xs = positions.isel(axis=0).values.astype(numpy.float64)
            Ys = positions.isel(axis=1).values.astype(numpy.float64)
            Zs = positions.isel(axis=2).values.astype(numpy.float64)
            return d_Ts, Xs, Ys, Zs

        def get_doppler_estimates(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]
            doppler_centroid_dataset = burst_list[0].conditions.doppler_centroid.ds
            dc_az_time = doppler_centroid_dataset["azimuth_time"].values
            d_dc_az_time = dc_az_time - start
            dc_az_time = d_dc_az_time / numpy.timedelta64(1, "s")
            dc_t0 = doppler_centroid_dataset["t0"].values
            data_dc_poly = [numpy.array(val) for val in doppler_centroid_dataset["data_dc_polynomial"].values.tolist()]
            return dc_az_time, dc_t0, data_dc_poly

        def get_azimuth_fm_estimates(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]
            azimuth_fm_estimates = burst_list[0].attrs["other_metadata"]["general_annotation"]["azimuth_fm_rate_list"]
            az_t = numpy.array([fm["azimuth_time"] for fm in azimuth_fm_estimates], dtype=numpy.datetime64)
            # azimuth_fm_rate_dataset = burst_list[0].conditions.azimuth_fm_rate.ds
            # az_t = azimuth_fm_rate_dataset['azimuth_time'].values
            d_az_t = az_t - start
            az_t = d_az_t / numpy.timedelta64(1, "s")
            # az_t0 = azimuth_fm_rate_dataset['t0'].values
            az_t0 = numpy.array([fm["t_0"] for fm in azimuth_fm_estimates])
            k_a_poly = [numpy.array(fm["azimuth_fm_rate_polynomial"]) for fm in azimuth_fm_estimates]
            return az_t, az_t0, k_a_poly

        def set_core_name(sicd, start_dt, burst_id):
            # type: (SICDType, datetime, int) -> None
            t_slice = int(get_slice())
            swath = get_swath()
            sicd.CollectionInfo.CoreName = "{0:s}{1:s}{2:s}_{3:02d}_{4:s}_{5}".format(
                start_dt.strftime("%d%b%y"),
                get_mission_id(),
                get_mission_datatake_id(sicd),
                t_slice,
                swath,
                burst_id,
            )
            sicd.CollectionInfo.Parameters["BURST"] = str(burst_id)

        def get_mission_datatake_id(sicd):
            return self._data.attrs["stac_discovery"]["properties"]["eopf:datatake_id"]

        def set_timeline(sicd, start, duration):
            # type: (SICDType, numpy.datetime64, float) -> None
            prf = get_prf()
            timeline = sicd.Timeline
            timeline.CollectStart = start
            timeline.CollectDuration = duration
            timeline.IPP[0].TEnd = duration
            timeline.IPP[0].IPPEnd = round(timeline.CollectDuration * prf) - 1
            sicd.ImageFormation.TEndProc = duration

        def get_prf():
            return (
                1
                / burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                    "downlink_values"
                ]["pri"]
            )
            # return burst_list[0].attrs['other_metadata']['image_annotation']['image_information']['azimuth_frequency']

        def set_position(sicd, start):
            # type: (SICDType, numpy.datetime64) -> None
            Ts, Xs, Ys, Zs = get_state_vectors(start)
            poly_order = min(5, Ts.size - 1)
            P_X = polynomial.polyfit(Ts.astype(numpy.float64), Xs, poly_order)
            P_Y = polynomial.polyfit(Ts.astype(numpy.float64), Ys, poly_order)
            P_Z = polynomial.polyfit(Ts.astype(numpy.float64), Zs, poly_order)
            sicd.Position = PositionType(ARPPoly=XYZPolyType(X=P_X, Y=P_Y, Z=P_Z))

        def update_rma_and_grid(sicd, first_line_relative_start, start, return_time_dets=False):

            center_frequency = get_center_frequency()
            # set TimeCAPoly
            ss_zd_s = get_image_col_spacing_zdt()
            eta_mid = ss_zd_s * float(out_sicd.ImageData.SCPPixel.Col)
            sicd.RMA.INCA.TimeCAPoly = Poly1DType(
                Coefs=[first_line_relative_start + eta_mid, ss_zd_s / out_sicd.Grid.Col.SS]
            )
            range_time_scp = sicd.RMA.INCA.R_CA_SCP * 2 / speed_of_light
            # get velocity polynomial
            vel_poly = sicd.Position.ARPPoly.derivative(1, return_poly=True)
            # We pick a single velocity magnitude at closest approach to represent
            # the entire burst.  This is valid, since the magnitude of the velocity
            # changes very little.
            vm_ca = numpy.linalg.norm(vel_poly(sicd.RMA.INCA.TimeCAPoly[0]))
            az_rate_times, az_rate_t0, k_a_poly = get_azimuth_fm_estimates(start)
            # find the closest fm rate polynomial
            az_rate_poly_ind = int(numpy.argmin(numpy.abs(az_rate_times - sicd.RMA.INCA.TimeCAPoly[0])))
            az_rate_poly = Poly1DType(Coefs=k_a_poly[az_rate_poly_ind])
            dr_ca_poly = az_rate_poly.shift(
                t_0=az_rate_t0[az_rate_poly_ind] - range_time_scp, alpha=2 / speed_of_light, return_poly=False
            )
            r_ca = numpy.array([sicd.RMA.INCA.R_CA_SCP, 1], dtype=numpy.float64)
            sicd.RMA.INCA.DRateSFPoly = numpy.reshape(
                -numpy.convolve(dr_ca_poly, r_ca) * (speed_of_light / (2 * center_frequency * vm_ca * vm_ca)), (-1, 1)
            )
            # Doppler Centroid
            dc_est_times, dc_t0, data_dc_poly = get_doppler_estimates(start)
            # find the closest doppler centroid polynomial
            dc_poly_ind = int(numpy.argmin(numpy.abs(dc_est_times - sicd.RMA.INCA.TimeCAPoly[0])))
            # we are going to move the respective polynomial from reference point as dc_t0 to
            # reference point at SCP time.
            dc_poly = Poly1DType(Coefs=data_dc_poly[dc_poly_ind])

            # Fit DeltaKCOAPoly, DopCentroidPoly, and TimeCOAPoly from data
            tau_0 = get_first_slant_range_time()
            delta_tau_s = 1.0 / get_range_sampling_rate()
            image_data = sicd.ImageData
            grid = sicd.Grid
            inca = sicd.RMA.INCA
            # common use for the fitting efforts
            poly_order = 2
            grid_samples = poly_order + 4
            cols = numpy.linspace(0, image_data.NumCols - 1, grid_samples, dtype=numpy.int64)
            rows = numpy.linspace(0, image_data.NumRows - 1, grid_samples, dtype=numpy.int64)
            coords_az = get_im_physical_coords(cols, grid, image_data, "col")
            coords_rg = get_im_physical_coords(rows, grid, image_data, "row")
            coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)

            # fit DeltaKCOAPoly
            tau = tau_0 + delta_tau_s * rows
            # Azimuth steering rate (constant, not dependent on burst or range)
            k_psi = numpy.deg2rad(get_azimuth_steering_rate(sicd))
            k_s = vm_ca * center_frequency * k_psi * 2 / speed_of_light
            k_a = az_rate_poly(tau - az_rate_t0[az_rate_poly_ind])
            k_t = (k_a * k_s) / (k_a - k_s)
            f_eta_c = dc_poly(tau - dc_t0[dc_poly_ind])
            eta = (cols - image_data.SCPPixel.Col) * ss_zd_s
            eta_c = -f_eta_c / k_a  # Beam center crossing time (TimeCOA)
            eta_ref = eta_c - eta_c[0]
            eta_2d, eta_ref_2d = numpy.meshgrid(eta, eta_ref)
            eta_arg = eta_2d - eta_ref_2d
            deramp_phase = 0.5 * k_t[:, numpy.newaxis] * eta_arg * eta_arg
            demod_phase = eta_arg * f_eta_c[:, numpy.newaxis]
            total_phase = deramp_phase + demod_phase
            phase, residuals, rank, sing_values = two_dim_poly_fit(
                coords_rg_2d,
                coords_az_2d,
                total_phase,
                x_order=poly_order,
                y_order=poly_order,
                x_scale=1e-3,
                y_scale=1e-3,
                rcond=1e-35,
            )
            logger.info(
                "The phase polynomial fit details:\n\t"
                "root mean square residuals = {}\n\t"
                "rank = {}\n\t"
                "singular values = {}".format(residuals, rank, sing_values)
            )

            # DeltaKCOAPoly is derivative of phase in azimuth/Col direction
            delta_kcoa_poly = polynomial.polyder(phase, axis=1)
            grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=delta_kcoa_poly)

            # derive the DopCentroidPoly directly
            dop_centroid_poly = delta_kcoa_poly * grid.Col.SS / ss_zd_s
            inca.DopCentroidPoly = Poly2DType(Coefs=dop_centroid_poly)

            # complete deriving the TimeCOAPoly, which depends on the DOPCentroidPoly
            time_ca_sampled = inca.TimeCAPoly(coords_az_2d)
            doppler_rate_sampled = polynomial.polyval(coords_rg_2d, dr_ca_poly)
            dop_centroid_sampled = inca.DopCentroidPoly(coords_rg_2d, coords_az_2d)
            time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled
            time_coa_poly, residuals, rank, sing_values = two_dim_poly_fit(
                coords_rg_2d,
                coords_az_2d,
                time_coa_sampled,
                x_order=poly_order,
                y_order=poly_order,
                x_scale=1e-3,
                y_scale=1e-3,
                rcond=1e-40,
            )
            logger.info(
                "The TimeCOAPoly fit details:\n\t"
                "root mean square residuals = {}\n\t"
                "rank = {}\n\t"
                "singular values = {}".format(residuals, rank, sing_values)
            )
            grid.TimeCOAPoly = Poly2DType(Coefs=time_coa_poly)
            return time_coa_sampled.min(), time_coa_sampled.max()

        def get_azimuth_steering_rate(sicd):
            return burst_list[0].attrs["other_metadata"]["general_annotation"]["product_information"][
                "azimuth_steering_rate"
            ]

        def adjust_time(sicd, time_offset):
            # type: (SICDType, float) -> None
            # adjust TimeCOAPoly
            sicd.Grid.TimeCOAPoly.Coefs[0, 0] -= time_offset
            # adjust TimeCAPoly
            sicd.RMA.INCA.TimeCAPoly.Coefs[0] -= time_offset
            # shift ARPPoly
            sicd.Position.ARPPoly = sicd.Position.ARPPoly.shift(-time_offset, return_poly=True)

        def update_geodata(sicd):  # type: (SICDType) -> None
            scp_pixel = [sicd.ImageData.SCPPixel.Row, sicd.ImageData.SCPPixel.Col]
            ecf = sicd.project_image_to_ground(scp_pixel, projection_type="HAE")
            sicd.update_scp(ecf, coord_system="ECF")

        def get_scps(count):
            # SCPPixel - points at which to interpolate geo_pixels & geo_coords data
            scp_pixels = numpy.zeros((count, 2), dtype=numpy.float64)
            scp_pixels[:, 0] = int((out_sicd.ImageData.NumRows - 1) / 2.0)
            scp_pixels[:, 1] = int((out_sicd.ImageData.NumCols - 1) / 2.0) + out_sicd.ImageData.NumCols * (
                numpy.arange(count, dtype=numpy.float64)
            )
            scps = numpy.zeros((count, 3), dtype=numpy.float64)
            for j in range(3):
                scps[:, j] = griddata(geo_pixels, geo_coords[:, j], scp_pixels)
            return scps

        def finalize_stripmap():  # type: () -> SICDType
            # out_sicd is the one that we return, just complete it
            # set preliminary geodata (required for projection)
            scp = get_scps(1)
            out_sicd.GeoData = GeoDataType(SCP=SCPType(ECF=scp[0, :]))  # EarthModel & LLH are implicitly set
            # NB: SCPPixel is already set to the correct thing
            im_dat = out_sicd.ImageData
            im_dat.ValidData = (
                (0, 0),
                (0, im_dat.NumCols - 1),
                (im_dat.NumRows - 1, im_dat.NumCols - 1),
                (im_dat.NumRows - 1, 0),
            )
            start_dt = datetime.strptime(
                get_first_line_sensing_time(),
                DT_FMT,
            )
            start = numpy.datetime64(start_dt, "us")
            stop = parse_timestring(get_last_line_sensing_time())
            set_core_name(out_sicd, start_dt, 0)
            set_timeline(out_sicd, start, get_seconds(stop, start, precision="us"))
            set_position(out_sicd, start)

            azimuth_time_first_line = parse_timestring(get_first_line_utc_time())
            first_line_relative_start = get_seconds(azimuth_time_first_line, start, precision="us")
            update_rma_and_grid(out_sicd, first_line_relative_start, start)
            update_geodata(out_sicd)
            return [out_sicd]

        def get_first_line_utc_time():
            return burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                "first_line_sensing_time"
            ]

        def get_last_line_sensing_time():
            return burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                "last_line_sensing_time"
            ]

        def get_first_line_sensing_time():
            return burst_list[0].attrs["other_metadata"]["general_annotation"]["downlink_information_list"][0][
                "first_line_sensing_time"
            ]

        def finalize_bursts():  # type: () -> List[SICDType]
            # we will have one sicd per burst.
            sicds = []
            scps = get_scps(len(burst_list))

            for j, burst in enumerate(burst_list):
                t_sicd = out_sicd.copy()
                # set preliminary geodata (required for projection)
                t_sicd.GeoData = GeoDataType(SCP=SCPType(ECF=scps[j, :]))  # EarthModel & LLH are implicitly set

                xml_first_cols = get_first_burst_sample(burst)
                xml_last_cols = get_last_burst_sample(burst)
                valid = (xml_first_cols >= 0) & (xml_last_cols >= 0)
                valid_cols = numpy.arange(xml_first_cols.size, dtype=numpy.int64)[valid]
                first_row = int(numpy.min(xml_first_cols[valid]))
                last_row = int(numpy.max(xml_last_cols[valid]))
                first_col = valid_cols[0]
                last_col = valid_cols[-1]
                t_sicd.ImageData.ValidData = (
                    (first_row, first_col),
                    (first_row, last_col),
                    (last_row, last_col),
                    (last_row, first_col),
                )
                burst_id = burst.attrs["other_metadata"]["burst_id"]

                # This is the first and last zero doppler times of the columns in the burst.
                # Not really CollectStart and CollectDuration in SICD (first last pulse time)
                start_dt = get_burst_first_azimuth_time(burst)
                start = numpy.datetime64(start_dt, "us")
                set_core_name(t_sicd, start_dt, burst_id)
                set_position(t_sicd, start)
                early, late = update_rma_and_grid(t_sicd, 0, start, return_time_dets=True)
                new_start = start + numpy.timedelta64(numpy.int64(early * 1e6), "us")
                duration = late - early
                set_timeline(t_sicd, new_start, duration)
                # adjust my time offset
                adjust_time(t_sicd, early)
                update_geodata(t_sicd)
                sicds.append(t_sicd)
            return sicds

        def get_burst_first_azimuth_time(burst):  # FIXME  NO INFO AVAILABLE IN MAPPED DATA
            return pd.to_datetime(burst.measurements.azimuth_time.values[0])

        def get_last_burst_sample(burst):  # FIXME  NO INFO AVAILABLE IN MAPPED DATA
            try:
                val = burst.conditions.burst_info.last_valid_sample.values
            except AttributeError:
                val = numpy.ones(burst.measurement.slc.azimtuh_time.size)
            return val

        def get_first_burst_sample(burst):  # FIXME  NO INFO AVAILABLE IN MAPPED DATA
            try:
                val = burst.conditions.burst_info.first_valid_sample.values
            except AttributeError:
                val = numpy.ones(burst.measurement.slc.azimtuh_time.size)
            return val

        ######
        # create a common sicd with shared basic information here
        out_sicd = self._base_sicd.copy()
        out_sicd.ImageData = get_image_data()
        out_sicd.Grid = get_common_grid()
        out_sicd.Timeline = get_common_timeline()
        out_sicd.RadarCollection = get_common_radar_collection()
        out_sicd.ImageFormation = get_image_formation()
        out_sicd.RMA = get_rma()
        out_sicd.CollectionInfo = get_collection_info()
        ######
        # consider the burst situation, and split into the appropriate sicd collection
        if ("IW" in burst_list[0].name) or ("EW" in burst_list[0].name):
            return finalize_bursts()
        else:
            return finalize_stripmap()

    def get_geo_coords(self, burst):
        gcp_dataset = burst.conditions.gcp
        pixels = gcp_dataset.pixel.values
        lines = gcp_dataset.line.values
        pixels, lines = numpy.meshgrid(pixels, lines)
        pixels = pixels.ravel()
        lines = lines.ravel()
        geo_pixels = numpy.vstack((pixels, lines)).T
        lats = gcp_dataset.latitude.values.ravel()
        lons = gcp_dataset.longitude.values.ravel()
        heights = gcp_dataset.height.values.ravel()
        geo_coords = numpy.vstack((lats, lons, heights)).T
        geo_coords = geodetic_to_ecf(geo_coords)
        return geo_pixels, geo_coords

    def _refine_using_calibration(self, burst_list, sicds: Union[SICDType, List[SICDType]]) -> None:
        """

        Parameters
        ----------
        cal_file_name : str
        sicds : SICDType|List[SICDType]

        Returns
        -------
        None
        """

        # do not use before Sentinel baseline processing calibration update on 25 Nov 2015.
        # if self._base_sicd.ImageCreation.DateTime < numpy.datetime64("2015-11-25"):
        #     return
        root_node = None
        if isinstance(sicds, SICDType):
            sicds = [
                sicds,
            ]

        def update_sicd(sicd, index):  # type: (SICDType, int) -> None
            # NB: in the deprecated version, there is a check if beta is constant,
            #   in which case constant values are used for beta/sigma/gamma.
            #   This has been removed.

            valid_lines = (line >= (index - 1) * lines_per_burst) & (line < (index + 2) * lines_per_burst)
            valid_count = numpy.sum(valid_lines)
            if valid_count == 0:
                # this burst contained no useful calibration data
                return

            coords_rg = (pixel[valid_lines] + sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row) * sicd.Grid.Row.SS
            coords_az = (
                line[valid_lines] + sicd.ImageData.FirstCol - sicd.ImageData.SCPPixel.Col - index * lines_per_burst
            ) * sicd.Grid.Col.SS
            # NB: coords_rg = (valid_count, M) and coords_az = (valid_count, )
            coords_az = numpy.repeat(coords_az, pixel.shape[1])
            if valid_count > 1:
                coords_az = coords_az.reshape((valid_count, -1))

            def create_poly(arr, poly_order=2, poly_name=""):
                poly, residuals, rank, sing_values = two_dim_poly_fit(
                    coords_rg,
                    coords_az,
                    arr,
                    x_order=poly_order,
                    y_order=poly_order,
                    x_scale=1e-3,
                    y_scale=1e-3,
                    rcond=1e-35,
                )
                logger.info(
                    "The {} polynomial fit details:\n\t"
                    "root mean square residuals = {}\n\t"
                    "rank = {}\n\t"
                    "singular values = {}".format(poly_name, residuals, rank, sing_values)
                )
                return Poly2DType(Coefs=poly)

            if sicd.Radiometric is None:
                sicd.Radiometric = RadiometricType()
            sicd.Radiometric.SigmaZeroSFPoly = create_poly(
                sigma[valid_lines, :], poly_order=2, poly_name="SigmaZeroSFPoly"
            )
            sicd.Radiometric.BetaZeroSFPoly = create_poly(
                beta[valid_lines, :], poly_order=2, poly_name="BetaZeroSFPoly"
            )
            sicd.Radiometric.GammaZeroSFPoly = create_poly(
                gamma[valid_lines, :], poly_order=2, poly_name="GammaZeroSFPoly"
            )
            return

        lines_per_burst = sicds[0].ImageData.NumCols
        line, pixel, sigma, beta, gamma = self.get_calibration_matrices(burst_list, root_node)
        # adjust sentinel values for sicd convention (square and invert)
        sigma = 1.0 / (sigma * sigma)
        beta = 1.0 / (beta * beta)
        gamma = 1.0 / (gamma * gamma)

        for ind, sic in enumerate(sicds):
            update_sicd(sic, ind)

    def get_calibration_matrices(self, burst_list, root_node):
        cal_vector_dataset = burst_list[0].quality.calibration.ds
        line = cal_vector_dataset["line"].values.astype(int)
        pixel = cal_vector_dataset["pixel"].values
        pixel, _ = numpy.meshgrid(pixel, line)
        sigma = cal_vector_dataset["sigma_nought"].values
        beta = cal_vector_dataset["beta_nought"].values
        gamma = cal_vector_dataset["gamma"].values
        return line, pixel, sigma, beta, gamma

    def _refine_using_noise(self, burst_list, sicds: Union[SICDType, List[SICDType]]) -> None:
        """

        Parameters
        ----------
        noise_file_name : str
        sicds : SICDType|List[SICDType]

        Returns
        -------
        None
        """

        # do not use before Sentinel baseline processing calibration update on 25 Nov 2015.
        # if self._base_sicd.ImageCreation.DateTime < numpy.datetime64("2015-11-25"):
        #     return
        root_node = None
        if isinstance(sicds, SICDType):
            sicds = [
                sicds,
            ]

        lines_per_burst = sicds[0].ImageData.NumCols

        def get_azimuth_noise(_):
            try:
                azimuth_noise_ds = burst_list[0].quality.noise_azimuth.ds.squeeze()
                line = azimuth_noise_ds.line.values.tolist()
                noise = [azimuth_noise_ds.noise_azimuth_lut.values]
            except AttributeError:
                line = [None]
                noise = [None]
            return line, [None], noise

        def get_range_noise(_):
            range_noise_ds = burst_list[0].quality.noise_range.ds.squeeze()
            line = range_noise_ds.line.values
            pixel = range_noise_ds.pixel.values
            noise = [numpy.array(val) for val in range_noise_ds.noise_range_lut.values.tolist()]
            pixel = [pixel for i in noise]
            return line.tolist(), pixel, noise

        def populate_noise(sicd, index):  # type: (SICDType, int) -> None
            # NB: the default order was 7 before refactoring...that seems excessive.
            rg_poly_order = min(5, range_pixel[0].size - 1)
            if sicd.CollectionInfo.RadarMode.ModeID[0] == "S":
                # STRIPMAP - all LUTs apply
                # Treat range and azimuth polynomial components as fully independent
                az_poly_order = min(4, len(range_line) - 1)

                # NB: the previous rammed together two one-dimensional polys, but
                # we should do a 2-d fit.
                coords_rg = (range_pixel[0] + sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row) * sicd.Grid.Row.SS
                coords_az = (
                    numpy.array(range_line) + sicd.ImageData.FirstCol - sicd.ImageData.SCPPixel.Col
                ) * sicd.Grid.Col.SS

                coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)
                noise_poly, res, rank, sing_vals = two_dim_poly_fit(
                    coords_rg_2d,
                    coords_az_2d,
                    numpy.array(range_noise),
                    x_order=rg_poly_order,
                    y_order=az_poly_order,
                    x_scale=1e-3,
                    y_scale=1e-3,
                    rcond=1e-40,
                )
                logger.info(
                    "NoisePoly fit details:\n\t"
                    "root mean square residuals = {}\n\t"
                    "rank = {}\n\t"
                    "singular values = {}".format(res, rank, sing_vals)
                )
            else:
                # TOPSAR has single LUT per burst
                # Treat range and azimuth polynomial components as weakly independent
                if index >= len(range_pixel):
                    logger.error(
                        "We have run out of noise information.\n\t"
                        "Current index = {}, length of noise array = {}.\n\t"
                        "The previous noise information will be used to populate the NoisePoly.".format(
                            index, len(range_pixel)
                        )
                    )
                rp_array = range_pixel[min(index, len(range_pixel) - 1)]
                rn_array = range_noise[min(index, len(range_pixel) - 1)]
                coords_rg = (rp_array + sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row) * sicd.Grid.Row.SS
                rg_poly = numpy.array(polynomial.polyfit(coords_rg, rn_array, rg_poly_order))
                az_poly = None
                if azimuth_noise is not None:
                    line0 = lines_per_burst * index
                    coords_az = (azimuth_line[0] - line0 - sicd.ImageData.SCPPixel.Col) * sicd.Grid.Col.SS
                    valid_lines = (azimuth_line[0] >= line0) & (azimuth_line[0] < line0 + lines_per_burst)
                    valid_count = numpy.sum(valid_lines)
                    if valid_count > 1:
                        az_poly_order = min(2, valid_lines.size - 1)
                        az_poly = numpy.array(
                            polynomial.polyfit(coords_az[valid_lines], azimuth_noise[valid_lines], az_poly_order)
                        )
                if az_poly is not None:
                    noise_poly = numpy.zeros((rg_poly.size, az_poly.size), dtype=numpy.float64)
                    noise_poly[:, 0] += rg_poly
                    noise_poly[0, :] += az_poly
                else:
                    noise_poly = numpy.reshape(rg_poly, (-1, 1))
            if sicd.Radiometric is None:
                sicd.Radiometric = RadiometricType()
            sicd.Radiometric.NoiseLevel = NoiseLevelType_(
                NoiseLevelType="ABSOLUTE", NoisePoly=Poly2DType(Coefs=noise_poly)
            )

        def extract_noise(root_node):
            range_line, range_pixel, range_noise = get_range_noise("noiseRange")
            azimuth_line, azimuth_pixel, azimuth_noise = get_azimuth_noise("noiseAzimuth")
            range_noise = 10 * numpy.log10(range_noise) if range_noise[0] is not None else None
            azimuth_noise = 10 * numpy.log10(azimuth_noise) if azimuth_noise[0] is not None else None
            return (range_line, range_pixel, range_noise), (azimuth_line, azimuth_pixel, azimuth_noise)

        # extract our noise vectors (used in populate_noise through implicit reference)
        (range_line, range_pixel, range_noise), (azimuth_line, azimuth_pixel, azimuth_noise) = extract_noise(root_node)
        # NB: range_line is actually a list of 1 element arrays - probably should have been parsed better
        # range_line = numpy.concatenate(range_line, axis=0)
        for ind, sic in enumerate(sicds):
            populate_noise(sic, ind)

    @staticmethod
    def _derive(sicds: Union[SICDType, List[SICDType]]) -> None:
        if isinstance(sicds, SICDType):
            sicds.derive()
            sicds.populate_rniirs(override=False)
        else:
            for sicd in sicds:
                sicd.derive()
                sicd.populate_rniirs(override=False)

    def get_sicd_collection(self) -> List[Tuple[str, Union[SICDType, List[SICDType]]]]:
        """
        Get the data file location(s) and corresponding sicd collection for each file.

        Returns
        -------
        List[Tuple[str, SICDType|List[SICDType]]]
            list of the form `(file, sicds)`. Here `file` is the data filename (tiff).
            `sicds` is either a single `SICDType` (STRIPMAP collect),
            or a list of `SICDType` (TOPSAR with multiple bursts).
        """

        out = []
        for pair in self.get_all_swaths():
            entry, datafiles = zip(*pair)
            # get the sicd collection for each product
            sicds = self._parse_product_sicd(entry)
            # refine our sicds(s) using the calibration data (if sensible)
            self._refine_using_calibration(entry, sicds)
            # refine our sicd(s) using the noise data (if sensible)
            self._refine_using_noise(entry, sicds)
            # populate our derived fields for the sicds
            self._derive(sicds)
            out.append((datafiles, sicds))
        return cast(list, out)


class SentinelReader(SICDTypeReader):
    """
    A Sentinel-1 SLC file package reader implementation.

    **Changed in version 1.3.0** for reading changes.
    """

    __slots__ = ("_sentinel_details", "_parent_segments")

    def __init__(self, sentinel_details: Union[str, SentinelDetails]):
        """

        Parameters
        ----------
        sentinel_details : str|SentinelDetails
        """

        if isinstance(sentinel_details, str):
            sentinel_details = SentinelDetails(cast(xr.DataTree, sentinel_details))
        if not isinstance(sentinel_details, SentinelDetails):
            raise TypeError("Input argument for SentinelReader must be a file name or SentinelReader object.")

        self._sentinel_details = sentinel_details  # type: SentinelDetails

        reverse_axes = None
        transpose_axes = (1, 0)  # True for all Sentinel-1 data

        parent_segments: list[Any] = []
        segments = []
        sicd_collection = self._sentinel_details.get_sicd_collection()
        sicd_collection_out = []
        for data_files, sicds in sicd_collection:
            for da, sicd in zip(data_files, sicds):
                segments.append(
                    XarraySegment(
                        cast(xr.DataArray, da), reverse_axes=reverse_axes, transpose_axes=transpose_axes, mode="r"
                    )
                )
                sicd_collection_out.append(sicd)

        self._parent_segments = parent_segments
        SICDTypeReader.__init__(self, segments, sicd_collection_out, close_segments=True)
        self._check_sizes()

    @property
    def sentinel_details(self) -> SentinelDetails:
        """
        SentinelDetails: The sentinel details object.
        """

        return self._sentinel_details

    def close(self):
        BaseReader.close(self)
        if hasattr(self, "_parent_segments") and self._parent_segments is not None:
            # noinspection PyBroadException
            try:
                while len(self._parent_segments) > 0:
                    segment = self._parent_segments.pop()
                    segment.close()
            except Exception as e:
                logger.warning(f"Exception is: {e.__class__.__name__}")
                pass
            self._parent_segments = None


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name: str) -> Optional[SentinelReader]:
    """
    Tests whether a given file_name corresponds to a Sentinel file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SentinelReader|None
        `SentinelReader` instance if Sentinel-1 file, `None` otherwise
    """

    if not isinstance(file_name, xr.DataTree):
        return None

    try:
        if file_name.attrs["other_metadata"]["eopf_category"] != "eocontainer":
            return None
        if file_name.attrs["stac_discovery"]["properties"]["constellation"] != "sentinel-1":
            return None
        sentinel_details = SentinelDetails(file_name)
        return SentinelReader(sentinel_details)
    except (SarpyIOError, AttributeError, KeyError):
        return None


class XarraySegment(DataSegment):
    _allowed_modes = ("r",)

    __slots__ = (
        "_data",
        "_closed",
        "_mode",
        "_raw_dtype",
        "_raw_shape",
        "_formatted_dtype",
        "_formatted_shape",
        "_reverse_axes",
        "_transpose_axes",
        "_reverse_transpose_axes",
        "_format_function",
    )

    def __init__(
        self,
        data: xr.DataArray,
        reverse_axes: Union[None, int, Sequence[int]] = None,
        transpose_axes: Optional[Tuple[int, ...]] = None,
        format_function: Optional[FormatFunction] = None,
        mode: str = "r",
    ):
        """

        Parameters
        ----------
        raw_dtype : str|numpy.dtype
        raw_shape : Tuple[int, ...]
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
            Note that the format function instance
        mode : str
        """
        self._data = data
        raw_dtype = data.dtype
        raw_shape = data.shape
        formatted_dtype = raw_dtype
        formatted_shape = data.data.transpose(transpose_axes).shape
        DataSegment.__init__(
            self,
            raw_dtype=raw_dtype,
            raw_shape=raw_shape,
            formatted_dtype=formatted_dtype,
            formatted_shape=formatted_shape,
            reverse_axes=reverse_axes,
            transpose_axes=transpose_axes,
            format_function=format_function,
            mode=mode,
        )

    def read_raw(
        self, subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]], squeeze=True
    ) -> numpy.ndarray:
        self._validate_closed()
        if self.mode != "r":
            raise ValueError('Requires mode == "r"')

        subscript, out_shape = get_subscript_result_size(subscript, self.raw_shape)
        out = self._data[subscript].data  # squeezed by default

        return out.squeeze() if squeeze else out

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        raise Exception(f"{self._data.nbytes}")

    def flush(self) -> None:
        """
        Should perform, if possible, any necessary steps to flush any unwritten
        data to the file.

        Returns
        -------
        None
        """

        return

    def close(self):
        """
        This should perform any necessary clean-up operations, like closing
        open file handles, deleting any temp files, etc
        """

        if not hasattr(self, "_closed") or self._closed:
            return
        self._data.close()
        self._closed = True

    def __del__(self):
        # NB: this is called when the object is marked for garbage collection
        # (i.e. reference count goes to 0)
        # This order in which this happens may be unreliable
        self.close()
