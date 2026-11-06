# type: ignore
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
import os
from datetime import datetime
from typing import Any, cast
from xml.etree import ElementTree

import numpy
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
from sarpy.io.general.data_segment import SubsetSegment
from sarpy.io.general.tiff import NativeTiffDataSegment, TiffDetails
from sarpy.io.general.utils import get_seconds, is_file_like, parse_timestring
from scipy.constants import speed_of_light
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


##########
# helper functions


def _parse_xml(file_name: str, without_ns: bool = False) -> ElementTree.Element | tuple[dict, ElementTree.Element]:
    """
    Parses an XML file and returns its root node, and (unless suppressed) its namespace map.

    Parameters
    ----------
    file_name : str
        The path to the XML file to parse.
    without_ns : bool, optional
        Whether to return only the root node, skipping the namespace
        prefix parsing. Defaults to False.

    Returns
    -------
    tuple[dict, ElementTree.Element] | ElementTree.Element
        The `(namespaces, root node)` tuple, where `namespaces` maps
        namespace prefixes to URIs; or just the root node if `without_ns` is True.
    """

    root_node = ElementTree.parse(file_name).getroot()
    if without_ns:
        return root_node
    else:
        nodes = cast(
            list[tuple[Any, Any]], [node for _, node in ElementTree.iterparse(file_name, events=("start-ns",))]
        )
        ns: dict[Any, Any] = dict(nodes)
        return ns, root_node


def find_string(root_node, path) -> str:
    """
    Finds the XML node at the given path, and returns its text content.

    Parameters
    ----------
    root_node : ElementTree.Element
        The root XML node to search within.
    path : str
        The element path (XPath-style) of the node whose text is sought.

    Returns
    -------
    str
        The text content of the found node.

    Raises
    ------
    TypeError
        If the node found at `path` is not an `ElementTree.Element`,
        or if its text content is None.
    """

    node = root_node.find(path)
    if not isinstance(node, ElementTree.Element):
        raise TypeError(f"Expected Element for {path}")
    text = node.text
    if text is None:
        raise TypeError(f"Expected str for {path} got None")
    return text


###########
# parser and interpreter for sentinel-1 manifest.safe file


class SentinelDetails:
    """
    Parser and interpreter for a Sentinel-1 SAFE package.

    This class locates and validates the `manifest.safe` file within a Sentinel-1
    SLC package, extracts the platform and product information from it, and builds
    the SICD metadata collections for each measurement data unit described by the
    manifest.

    Attributes
    ----------
    file_name : str
        The path to the `manifest.safe` file.
    directory_name : str
        The Sentinel-1 package directory containing the `manifest.safe` file.
    satellite : str
        The platform family name, required to be `SENTINEL-1`.
    product_type : str
        The product type, required to be `SLC`.

    Methods
    -------
    _get_file_sets :
        Groups the measurement, product, noise, and calibration file paths
        by measurement data unit.
    _parse_product_sicd :
        Builds the SICD collection(s) from a product (SAFE) annotation file.
    _refine_using_calibration :
        Adds radiometric scale-factor polynomials from a calibration file.
    _refine_using_noise :
        Adds radiometric noise polynomials from a noise file.
    get_sicd_collection :
        Returns the `(data file, sicds)` pairs for each measurement data unit.
    """

    __slots__ = ("_base_sicd", "_directory_name", "_file_name", "_ns", "_product_type", "_root_node", "_satellite")

    def __init__(self, file_name: str):
        """

        Parameters
        ----------
        file_name : str
        """

        if os.path.isdir(file_name):  # its directory - point it at the manifest.safe file
            t_file_name = os.path.join(file_name, "manifest.safe")
            if os.path.exists(t_file_name):
                file_name = t_file_name
        if not os.path.exists(file_name) or not os.path.isfile(file_name):
            raise SarpyIOError(f"path {file_name} does not exist or is not a file")
        if os.path.split(file_name)[1] != "manifest.safe":
            raise SarpyIOError(f"The sentinel file is expected to be named manifest.safe, got path {file_name}")
        self._file_name = file_name
        absolute_path = os.path.abspath(file_name)
        self._directory_name, _ = os.path.split(absolute_path)

        parsed_xml = _parse_xml(file_name)
        if not isinstance(parsed_xml, tuple):
            raise TypeError(f"Expeted tuple, got {type(parsed_xml)}")
        self._ns: dict[Any, Any] = parsed_xml[0]
        self._root_node = parsed_xml[1]
        # note that the manifest.safe apparently does not have a default namespace,
        # so we have to explicitly enter no prefix in the namespace dictionary

        if not isinstance(self._ns, dict):
            raise TypeError(f"Expected dict, got {type(self._ns)}")
        self._ns[""] = ""
        _satellite = self._find(
            './metadataSection/metadataObject[@ID="platform"]/metadataWrap/xmlData/safe:platform/safe:familyName'
        )

        self._satellite = _satellite.text if _satellite is not None else None
        if self._satellite != "SENTINEL-1":
            raise ValueError(
                f"The platform in the manifest.safe file is required to be SENTINEL-1, got {self._satellite}"
            )

        _product_type = self._find(
            "./metadataSection"
            '/metadataObject[@ID="generalProductInformation"]'
            "/metadataWrap"
            "/xmlData"
            "/s1sarl1:standAloneProductInformation"
            "/s1sarl1:productType"
        )

        self._product_type = _product_type.text if _product_type is not None else None

        if self._product_type != "SLC":
            raise ValueError(
                f'The product type in the manifest.safe file is required to be "SLC", got {self._product_type}'
            )
        self._base_sicd = self._get_base_sicd()

    @property
    def file_name(self) -> str:
        """
        str: the file name
        """

        return self._file_name

    @property
    def directory_name(self) -> str:
        """
        str: the package directory name
        """

        return self._directory_name

    @property
    def satellite(self) -> str | None:
        """
        str: the satellite
        """

        return self._satellite

    @property
    def product_type(self) -> str | None:
        """
        str: the product type
        """

        return self._product_type

    def _find(self, tag: str) -> ElementTree.Element | None:
        """
        Pass through to ElementTree.Element.find(tag, ns).

        Parameters
        ----------
        tag : str

        Returns
        -------
        ElementTree.Element
        """

        res = self._root_node.find(tag, self._ns)
        return res

    def _findall(self, tag: str) -> list[ElementTree.Element]:
        """
        Pass through to ElementTree.Element.findall(tag, ns).

        Parameters
        ----------
        tag : str

        Returns
        -------
        List[ElementTree.Element
        """

        return self._root_node.findall(tag, self._ns)

    @staticmethod
    def _parse_pol(str_in: str) -> str:
        """
        Reformats a Sentinel polarization code (e.g. `RR`) into the SICD
        colon-separated form (e.g. `R:R`).

        Parameters
        ----------
        str_in : str
            The polarization code string.

        Returns
        -------
        str
            The first two characters of `str_in`, separated by a colon.
        """

        return f"{str_in[0]}:{str_in[1]}"

    def _get_file_sets(self) -> list[dict]:
        """
        Extracts paths for measurement and metadata files from a Sentinel manifest.safe file.
        These files will be grouped according to "measurement data unit" implicit in the
        Sentinel structure.

        Returns
        -------
        List[dict]
        """

        def get_file_location(schema_type, tids):
            """
            Resolves a data-object ID (or list of IDs) of the given schema type to the
            corresponding file path within the package directory, or None if not found.
            """

            if isinstance(tids, str):
                tids = [
                    tids,
                ]
            for tid in tids:
                do = self._find(f'dataObjectSection/dataObject[@repID="{schema_type}"]/[@ID="{tid}"]')
                if do is None:
                    continue
                return os.path.join(base_path, do.find("./byteStream/fileLocation").attrib["href"])
            return None

        base_path = os.path.dirname(self._file_name)

        files = []
        for mdu in self._findall(
            './informationPackageMap/xfdu:contentUnit/xfdu:contentUnit/[@repID="s1Level1MeasurementSchema"]'
        ):
            # get the data file for this measurement
            fnames = {
                "data": get_file_location(
                    "s1Level1MeasurementSchema",
                    cast(ElementTree.Element, mdu.find("dataObjectPointer")).attrib["dataObjectID"],
                )
            }
            # get the ids for product, noise, and calibration associated with this measurement data unit
            ids = mdu.attrib["dmdID"].split()
            # translate these ids to data object ids=file ids for the data files
            _fids = [self._find(f'./metadataSection/metadataObject[@ID="{did}"]/dataObjectPointer') for did in ids]
            fids = [fid.attrib["dataObjectID"] if fid is not None else None for fid in _fids]
            # NB: there is (at most) one of these per measurement data unit
            fnames["product"] = get_file_location("s1Level1ProductSchema", fids)
            fnames["noise"] = get_file_location("s1Level1NoiseSchema", fids)
            fnames["calibration"] = get_file_location("s1Level1CalibrationSchema", fids)
            files.append(fnames)
        return files

    def _get_base_sicd(self) -> SICDType:
        """
        Gets the base SICD element.

        Returns
        -------
        SICDType
        """

        from sarpy.__about__ import __version__

        # CollectionInfo
        platform = self._find('./metadataSection/metadataObject[@ID="platform"]/metadataWrap/xmlData/safe:platform')
        if platform is None:
            raise TypeError("Platform is None")
        family_name = cast(ElementTree.Element, platform.find("safe:familyName", self._ns)).text
        safe_number = cast(ElementTree.Element, platform.find("safe:number", self._ns)).text
        collector_name = f"{family_name}{safe_number}"
        mode_id = cast(
            ElementTree.Element,
            platform.find("./safe:instrument/safe:extension/s1sarl1:instrumentMode/s1sarl1:mode", self._ns),
        ).text
        if mode_id == "SM":
            mode_type = "STRIPMAP"
        else:
            # TOPSAR - closest SICD analog is Dynamic Stripmap
            mode_type = "DYNAMIC STRIPMAP"
        collection_info = CollectionInfoType(
            Classification="UNCLASSIFIED",
            CollectorName=collector_name,
            CollectType="MONOSTATIC",
            RadarMode=RadarModeType(ModeID=mode_id, ModeType=mode_type),
        )
        # ImageCreation
        processing = self._find(
            './metadataSection/metadataObject[@ID="processing"]/metadataWrap/xmlData/safe:processing'
        )
        if processing is None:
            raise TypeError("Processing is None")
        facility = cast(ElementTree.Element, processing.find("safe:facility", self._ns))
        software = cast(ElementTree.Element, facility.find("safe:software", self._ns))
        image_creation = ImageCreationType(
            Application="{name} {version}".format(**software.attrib),
            DateTime=processing.attrib["stop"],
            Site="{name}, {site}, {country}".format(**facility.attrib),
            Profile=f"sarpy {__version__}",
        )
        # RadarCollection
        polarizations = self._findall(
            "./metadataSection"
            '/metadataObject[@ID="generalProductInformation"]'
            "/metadataWrap"
            "/xmlData"
            "/s1sarl1:standAloneProductInformation"
            "/s1sarl1:transmitterReceiverPolarisation"
        )

        radar_collection = RadarCollectionType(
            RcvChannels=[
                ChanParametersType(TxRcvPolarization=self._parse_pol(str(pol.text)), index=i)
                for i, pol in enumerate(polarizations)
            ]
        )
        return SICDType(CollectionInfo=collection_info, ImageCreation=image_creation, RadarCollection=radar_collection)

    def _parse_product_sicd(self, product_file_name: str) -> SICDType | list[SICDType]:
        """

        Parameters
        ----------
        product_file_name : str

        Returns
        -------
        SICDType|List[SICDType]
        """

        DT_FMT = "%Y-%m-%dT%H:%M:%S.%f"
        root_node = _parse_xml(product_file_name, without_ns=True)
        if not isinstance(root_node, ElementTree.Element):
            raise TypeError(f"Expected root_node type: ElementTree.Element. Got {type(root_node)}")
        burst_list = root_node.findall("./swathTiming/burstList/burst")
        # parse the geolocation information - for SCP calculation
        geo_grid_point_list = root_node.findall("./geolocationGrid/geolocationGridPointList/geolocationGridPoint")
        geo_pixels = numpy.zeros((len(geo_grid_point_list), 2), dtype=numpy.float64)
        geo_coords = numpy.zeros((len(geo_grid_point_list), 3), dtype=numpy.float64)
        for i, grid_point in enumerate(geo_grid_point_list):
            geo_pixels[i, :] = (
                float(find_string(grid_point, "./pixel")),
                float(find_string(grid_point, "./line")),
            )  # (row, col) order
            geo_coords[i, :] = (
                float(find_string(grid_point, "./latitude")),
                float(find_string(grid_point, "./longitude")),
                float(find_string(grid_point, "./height")),
            )
        geo_coords = geodetic_to_ecf(geo_coords)

        def get_center_frequency():  # type: () -> float
            """
            Reads the radar center frequency in Hz from the
            `generalAnnotation/productInformation/radarFrequency` field.

            Returns
            -------
            float
                The center frequency in Hz.
            """

            return float(find_string(root_node, "./generalAnnotation/productInformation/radarFrequency"))

        def get_image_col_spacing_zdt():  # type: () -> float
            """
            Returns the image column spacing in zero-doppler time (seconds),
            read from `imageAnnotation/imageInformation/azimuthTimeInterval`.

            Returns
            -------
            float
                The per-column zero-doppler time spacing in seconds.
            """

            # Image column spacing in zero doppler time (seconds)
            # Sentinel-1 is always right-looking, so this should always be positive
            return float(find_string(root_node, "./imageAnnotation/imageInformation/azimuthTimeInterval"))

        def get_image_data():  # type: () -> ImageDataType
            """
            Builds the SICD `ImageData` element from the product annotation file.

            Reads `imageAnnotation/imageInformation/pixelValue` (only 16-bit complex
            SLC data is supported); the row and column counts come from
            `swathTiming/samplesPerBurst` and `swathTiming/linesPerBurst` when the
            collect contains bursts (TOPSAR), and otherwise from `numberOfSamples`
            and `numberOfLines` (STRIPMAP).

            Returns
            -------
            ImageDataType
                The image data element with `NumRows`, `NumCols`, full-image extent,
                and an `SCPPixel` at the center of a single-burst image.
            """

            _pv = find_string(root_node, "./imageAnnotation/imageInformation/pixelValue")
            if _pv == "Complex":
                pixel_type = "RE16I_IM16I"
            else:
                # NB: we only handle SLC
                raise ValueError(f"SLC data should be 16-bit complex, got pixelValue = {_pv}.")
            if len(burst_list) > 0:
                # should be TOPSAR
                num_rows = int(find_string(root_node, "./swathTiming/samplesPerBurst"))
                num_cols = int(find_string(root_node, "./swathTiming/linesPerBurst"))
            else:
                # STRIPMAP
                # NB - these fields always contain the number of rows/cols in the entire tiff,
                # even if there are multiple bursts
                num_rows = int(find_string(root_node, "./imageAnnotation/imageInformation/numberOfSamples"))
                num_cols = int(find_string(root_node, "./imageAnnotation/imageInformation/numberOfLines"))
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

        def get_common_grid():  # type: () -> GridType
            """
            Builds the SICD `Grid` element common to all bursts of the product.

            The row spacing comes from the inverse of
            `generalAnnotation/productInformation/rangeSamplingRate`, the column
            spacing from `imageAnnotation/imageInformation/azimuthPixelSpacing`, and
            the windowing parameters from the
            `imageAnnotation/processingInformation/swathProcParamsList` entries.

            Returns
            -------
            GridType
                The grid element (RG-ZERO type, with slant image plane when the
                projection is `Slant Range`, and row/column `DirParamType`s).
            """

            center_frequency = get_center_frequency()
            image_plane = (
                "SLANT"
                if cast(ElementTree.Element, root_node.find("./generalAnnotation/productInformation/projection")).text
                == "Slant Range"
                else None
            )
            # get range processing node
            range_proc = root_node.find(
                "./imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing"
            )
            if not isinstance(range_proc, ElementTree.Element):
                raise TypeError(f"Expected range_proc Element, got {type(range_proc)}")
            delta_tau_s = 1.0 / float(
                find_string(root_node, "./generalAnnotation/productInformation/rangeSamplingRate")
            )
            row_window_name = find_string(range_proc, "./windowType").upper()
            row_params = None
            if row_window_name == "NONE":
                row_window_name = "UNIFORM"
            elif row_window_name == "HAMMING":
                row_params = {"COEFFICIENT": cast(ElementTree.Element, range_proc.find("./windowCoefficient")).text}
            row = DirParamType(
                SS=(speed_of_light / 2) * delta_tau_s,
                Sgn=-1,
                KCtr=2 * center_frequency / speed_of_light,
                ImpRespBW=2.0 * float(find_string(range_proc, "./processingBandwidth")) / speed_of_light,
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
            az_proc = root_node.find(
                "./imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/azimuthProcessing"
            )
            if not isinstance(az_proc, ElementTree.Element):
                raise TypeError(f"Expected az_proc Element, got {type(az_proc)}")
            col_ss = float(find_string(root_node, "./imageAnnotation/imageInformation/azimuthPixelSpacing"))
            dop_bw = float(find_string(az_proc, "./processingBandwidth"))  # Doppler bandwidth
            ss_zd_s = get_image_col_spacing_zdt()
            col_window_name = find_string(az_proc, "./windowType").upper()
            col_params = None
            if col_window_name == "NONE":
                col_window_name = "UNIFORM"
            elif col_window_name == "HAMMING":
                col_params = {"COEFFICIENT": cast(ElementTree.Element, az_proc.find("./windowCoefficient")).text}
            col = DirParamType(
                SS=col_ss,
                Sgn=-1,
                KCtr=0,
                ImpRespBW=dop_bw * ss_zd_s / col_ss,
                WgtType=WgtTypeType(WindowName=col_window_name, Parameters=col_params),
            )
            return GridType(ImagePlane=image_plane, Type="RGZERO", Row=row, Col=col)

        def get_common_timeline():  # type: () -> TimelineType
            """
            Builds the preliminary SICD `Timeline` element common to all bursts.

            The pulse read frequency (PRF) is read from
            `generalAnnotation/downlinkInformationList/downlinkInformation/prf`; the
            `TEnd` and `IPPEnd` are placeholder values, to be corrected per collect
            later.

            Returns
            -------
            TimelineType
                The timeline element with a single IPPSet carrying the
                `(0, prf)` IPP polynomial.
            """

            prf = float(find_string(root_node, "./generalAnnotation/downlinkInformationList/downlinkInformation/prf"))
            # NB: TEnd and IPPEnd are nonsense values which will be corrected
            return TimelineType(
                IPP=[
                    IPPSetType(TStart=0, TEnd=0, IPPStart=0, IPPEnd=0, IPPPoly=(0, prf), index=0),
                ]
            )

        def get_common_radar_collection():  # type: () -> RadarCollectionType
            """
            Builds the SICD `RadarCollection` element for the product.

            Copies the collection-wide radar collection and sets the TX
            polarization from `adsHeader/polarisation`, the TX frequency range
            from the center frequency plus
            `downlinkValues/txPulseStartFrequency` and the pulse length times
            the ramp rate, and the waveform parameters from the downlink values
            and the `swlList` receive window lengths.

            Returns
            -------
            RadarCollectionType
                The completed radar collection element.
            """

            radar_collection = out_sicd.RadarCollection.copy()
            center_frequency = get_center_frequency()
            downlink_values = "./generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues"
            min_frequency = center_frequency + float(
                find_string(
                    root_node,
                    f"{downlink_values}/txPulseStartFrequency",
                )
            )
            tx_pulse_length = float(find_string(root_node, f"{downlink_values}/txPulseLength"))
            tx_fm_rate = float(find_string(root_node, f"{downlink_values}/txPulseRampRate"))
            band_width = tx_pulse_length * tx_fm_rate
            pol = find_string(root_node, "./adsHeader/polarisation")
            radar_collection.TxPolarization = pol[0]
            radar_collection.TxFrequency = (min_frequency, min_frequency + band_width)
            adc_sample_rate = float(
                find_string(root_node, "./generalAnnotation/productInformation/rangeSamplingRate")
            )  # Raw not decimated
            swl_list = root_node.findall(
                "./generalAnnotation/downlinkInformationList/" + "downlinkInformation/downlinkValues/swlList/swl"
            )
            radar_collection.Waveform = [
                WaveformParametersType(
                    index=j,
                    TxFreqStart=min_frequency,
                    TxPulseLength=tx_pulse_length,
                    TxFMRate=tx_fm_rate,
                    TxRFBandwidth=band_width,
                    RcvFMRate=0,
                    ADCSampleRate=adc_sample_rate,
                    RcvWindowLength=float(find_string(swl, "./value")),
                )
                for j, swl in enumerate(swl_list)
            ]
            return radar_collection

        def get_image_formation():  # type: () -> ImageFormationType
            """
            Builds the SICD `ImageFormation` element for the product.

            Determines the channel index matching the `adsHeader/polarisation`
            of this data file within the collection-wide receive channels, and
            records the processing polarization, TX frequency range, and RMA
            image formation algorithm.

            Returns
            -------
            ImageFormationType
                The image formation element (with `RcvChanProc`, polarization,
                and TX frequency processing information).
            """

            st_beam_comp = "GLOBAL" if out_sicd.CollectionInfo.RadarMode.ModeID[0] == "S" else "SV"
            pol = self._parse_pol(find_string(root_node, "./adsHeader/polarisation"))
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
            """
            Builds the preliminary SICD `RMA` element common to all bursts.

            Uses `imageAnnotation/imageInformation/slantRangeTime` together with the
            range sampling rate to compute the range to closest approach (RCA) at the
            SCP, and sets the frequency zero to the center frequency.

            Returns
            -------
            RMAType
                The RMA element, with an `INCA` sub-element.
            """

            center_frequency = get_center_frequency()
            tau_0 = float(find_string(root_node, "./imageAnnotation/imageInformation/slantRangeTime"))
            delta_tau_s = 1.0 / float(
                find_string(root_node, "./generalAnnotation/productInformation/rangeSamplingRate")
            )
            return RMAType(
                RMAlgoType="RG_DOP",
                INCA=INCAType(
                    FreqZero=center_frequency,
                    DopCentroidCOA=True,
                    R_CA_SCP=(0.5 * speed_of_light) * (tau_0 + out_sicd.ImageData.SCPPixel.Row * delta_tau_s),
                ),
            )

        def get_slice():  # type: () -> str
            """
            Returns the slice number of the product.

            Returns
            -------
            str
                The slice number from `imageAnnotation/imageInformation/sliceNumber`,
                or `"0"` if that field is absent.
            """

            slice_number = root_node.find("./imageAnnotation/imageInformation/sliceNumber")
            if slice_number is None:
                return "0"
            else:
                return str(slice_number.text)

        def get_swath():  # type: () -> str
            """
            Returns the swath indicator (e.g. `iw` or `ew`) of the product.

            Returns
            -------
            str
                The swath identifier from `adsHeader/swath`.
            """

            return find_string(root_node, "./adsHeader/swath")

        def get_collection_info():  # type: () -> CollectionInfoType
            """
            Builds the SICD `CollectionInfo` element for the product.

            Sets the collector name from `adsHeader/missionId` and the radar mode
            from `adsHeader/mode`, and computes the relative orbit number from the
            manifest `safe:relativeOrbitNumber` field if present, otherwise falling
            back to a formula on `adsHeader/absoluteOrbitNumber`. The slice, swath,
            and orbit values are stored in the `Parameters` dictionary.

            Returns
            -------
            CollectionInfoType
                The collection information element.
            """

            collection_info = out_sicd.CollectionInfo.copy()
            mission_id = cast(ElementTree.Element, root_node.find("./adsHeader/missionId")).text
            collection_info.CollectorName = mission_id
            collection_info.RadarMode.ModeID = cast(ElementTree.Element, root_node.find("./adsHeader/mode")).text

            t_slice = get_slice()
            swath = get_swath()
            abs_orbit_str = find_string(root_node, "./adsHeader/absoluteOrbitNumber")
            rel_orbit_node = self._find(".//safe:relativeOrbitNumber")
            if rel_orbit_node is not None:
                rel_orbit_str = rel_orbit_node.text
            else:
                # Fallback
                try:
                    abs_orbit_int = int(abs_orbit_str)
                    if mission_id == "S1A":
                        rel_orbit_str = str(((abs_orbit_int - 73) % 175) + 1)
                    elif mission_id == "S1B":
                        rel_orbit_str = str(((abs_orbit_int - 27) % 175) + 1)
                    else:
                        rel_orbit_str = "UNKNOWN"
                except ValueError:
                    rel_orbit_str = "UNKNOWN"

            collection_info.Parameters = {
                "SLICE": t_slice,
                "BURST": "1",
                "SWATH": swath,
                "ORBIT_SOURCE": "SLC_INTERNAL",
                "ABSOLUTE_ORBIT": abs_orbit_str,
                "RELATIVE_ORBIT": rel_orbit_str,
            }
            return collection_info

        def get_state_vectors(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            """
            Extracts the orbit state vectors from `generalAnnotation/orbitList`.

            Parameters
            ----------
            start : numpy.datetime64
                The reference time relative to which the state-vector times
                (parsed with microsecond precision) are computed.

            Returns
            -------
            tuple
                A tuple `(Ts, Xs, Ys, Zs)` where `Ts` are time offsets in seconds
                from `start`, and `Xs`, `Ys`, `Zs` are the ECF position components
                (in meters) of each state vector.
            """

            orbit_list = root_node.findall("./generalAnnotation/orbitList/orbit")
            shp = (len(orbit_list),)
            Ts = numpy.empty(shp, dtype=numpy.float64)
            Xs = numpy.empty(shp, dtype=numpy.float64)
            Ys = numpy.empty(shp, dtype=numpy.float64)
            Zs = numpy.empty(shp, dtype=numpy.float64)
            for j, orbit in enumerate(orbit_list):
                Ts[j] = get_seconds(parse_timestring(find_string(orbit, "./time")), start, precision="us")
                Xs[j] = float(find_string(orbit, "./position/x"))
                Ys[j] = float(find_string(orbit, "./position/y"))
                Zs[j] = float(find_string(orbit, "./position/z"))
            return Ts, Xs, Ys, Zs

        def get_doppler_estimates(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]
            """
            Extracts the Doppler centroid estimates from the
            `dopplerCentroid/dcEstimateList` annotation.

            Parameters
            ----------
            start : numpy.datetime64
                The reference time relative to which the azimuth times
                (parsed with microsecond precision) are computed.

            Returns
            -------
            tuple
                A tuple `(dc_az_time, dc_t0, data_dc_poly)` with the azimuth
                times in seconds, the slant-range time references `t0` (seconds),
                and the coefficient arrays of each `dataDcPolynomial`.
            """

            dc_estimate_list = root_node.findall("./dopplerCentroid/dcEstimateList/dcEstimate")
            shp = (len(dc_estimate_list),)
            dc_az_time = numpy.empty(shp, dtype=numpy.float64)
            dc_t0 = numpy.empty(shp, dtype=numpy.float64)
            data_dc_poly = []
            for j, dc_estimate in enumerate(dc_estimate_list):
                dc_az_time[j] = get_seconds(
                    parse_timestring(find_string(dc_estimate, "./azimuthTime")), start, precision="us"
                )
                dc_t0[j] = float(find_string(dc_estimate, "./t0"))
                data_dc_poly.append(numpy.fromstring(find_string(dc_estimate, "./dataDcPolynomial"), sep=" "))
            return dc_az_time, dc_t0, data_dc_poly

        def get_azimuth_fm_estimates(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]
            """
            Extracts the azimuth fill-mode rate estimates from the
            `generalAnnotation/azimuthFmRateList` annotation.

            Handles both the old `c0/c1/c2` coefficient style and the newer
            `azimuthFmRatePolynomial` style.

            Parameters
            ----------
            start : numpy.datetime64
                The reference time relative to which the azimuth times
                (parsed with microsecond precision) are computed.

            Returns
            -------
            tuple
                A tuple `(az_t, az_t0, k_a_poly)` with the azimuth times in
                seconds, the slant-range time references `t0` (seconds), and the
                fill-mode rate polynomial coefficient arrays.
            """

            azimuth_fm_rate_list = root_node.findall("./generalAnnotation/azimuthFmRateList/azimuthFmRate")
            shp = (len(azimuth_fm_rate_list),)
            az_t = numpy.empty(shp, dtype=numpy.float64)
            az_t0 = numpy.empty(shp, dtype=numpy.float64)
            k_a_poly = []
            for j, az_fm_rate in enumerate(azimuth_fm_rate_list):
                az_t[j] = get_seconds(parse_timestring(find_string(az_fm_rate, "./azimuthTime")), start, precision="us")
                az_t0[j] = float(find_string(az_fm_rate, "./t0"))
                if az_fm_rate.find("c0") is not None:
                    # old style annotation xml file
                    k_a_poly.append(
                        numpy.array(
                            [
                                float(find_string(az_fm_rate, "./c0")),
                                float(find_string(az_fm_rate, "./c1")),
                                float(find_string(az_fm_rate, "./c2")),
                            ],
                            dtype=numpy.float64,
                        )
                    )
                else:
                    k_a_poly.append(numpy.fromstring(find_string(az_fm_rate, "./azimuthFmRatePolynomial"), sep=" "))
            return az_t, az_t0, k_a_poly

        def set_core_name(sicd, start_dt, burst_num):
            # type: (SICDType, datetime, int) -> None
            """
            Sets the SICD `CollectionInfo.CoreName` and updates the `BURST`
            parameter of the given metadata object in place.

            The name is built from the start date, mission ID, mission data-take
            ID, slice number, swath, and burst number.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            start_dt : datetime
                The collection start date/time.
            burst_num : int
                The burst number to record.
            """

            t_slice = int(get_slice())
            swath = get_swath()
            sicd.CollectionInfo.CoreName = "{:s}{:s}{:s}_{:02d}_{:s}_{}".format(
                start_dt.strftime("%d%b%y"),
                find_string(root_node, "./adsHeader/missionId"),
                find_string(root_node, "./adsHeader/missionDataTakeId"),
                t_slice,
                swath,
                burst_num,
            )
            sicd.CollectionInfo.Parameters["BURST"] = str(burst_num)

        def set_timeline(sicd, start, duration):
            # type: (SICDType, numpy.datetime64, float) -> None
            """
            Sets the collection timeline fields on the given metadata object in place.

            Sets `Timeline.CollectStart` and `CollectDuration`, completes the IPP
            set `TEnd` and `IPPEnd` using the PRF from the downlink annotation,
            and sets `ImageFormation.TEndProc`.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            start : numpy.datetime64
                The collection start time.
            duration : float
                The collection duration in seconds.
            """

            prf = float(find_string(root_node, "./generalAnnotation/downlinkInformationList/downlinkInformation/prf"))
            timeline = sicd.Timeline
            timeline.CollectStart = start
            timeline.CollectDuration = duration
            timeline.IPP[0].TEnd = duration
            timeline.IPP[0].IPPEnd = round(timeline.CollectDuration * prf) - 1
            sicd.ImageFormation.TEndProc = duration

        def set_position(sicd, start):
            # type: (SICDType, numpy.datetime64) -> None
            """
            Fits polynomial position to the orbit state vectors and sets it
            as the `Position.ARPPoly` of the given metadata object in place.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            start : numpy.datetime64
                The reference time for the state vectors.
            """

            Ts, Xs, Ys, Zs = get_state_vectors(start)
            poly_order = min(5, Ts.size - 1)
            P_X = polynomial.polyfit(Ts, Xs, poly_order)
            P_Y = polynomial.polyfit(Ts, Ys, poly_order)
            P_Z = polynomial.polyfit(Ts, Zs, poly_order)
            sicd.Position = PositionType(ARPPoly=XYZPolyType(X=P_X, Y=P_Y, Z=P_Z))

        def update_rma_and_grid(sicd, first_line_relative_start, start, return_time_dets=False):
            # type: (SICDType, Union[float, int], numpy.datetime64, bool) -> Union[None, Tuple[float, float]]
            """
            Completes the RMA INCA and Grid time/rate polynomials for one burst,
            modifying the given metadata object in place.

            Sets `RMA.INCA.TimeCAPoly` from the first-line zero-doppler time and
            the column spacing, derives `DRateSFPoly` from the closest azimuth
            fill-mode rate polynomial, and fits `DeltaKCOAPoly`, `DopCentroidPoly`,
            and `TimeCOAPoly` over sampled image coordinates with a degree-2
            two-dimensional fit. The fit residuals, rank, and singular values are
            logged at debug level.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            first_line_relative_start : float
                The zero-doppler time of the first image line, in seconds relative
                to `start`.
            start : numpy.datetime64
                The collection start time, i.e. the reference time of the
                state vectors and rate estimates.
            return_time_dets : bool, optional
                Whether to return the earliest and latest sampled TimeCOA times.
                Defaults to False.

            Returns
            -------
            tuple or None
                The `(minimum, maximum)` sampled TimeCOA times in seconds if
                `return_time_dets` is True, otherwise None.
            """

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
            tau_0 = float(find_string(root_node, "./imageAnnotation/imageInformation/slantRangeTime"))
            delta_tau_s = 1.0 / float(
                find_string(root_node, "./generalAnnotation/productInformation/rangeSamplingRate")
            )
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
            k_psi = numpy.deg2rad(
                float(find_string(root_node, "./generalAnnotation/productInformation/azimuthSteeringRate"))
            )
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
            logger.debug(
                "The phase polynomial fit details:\n\t"
                f"root mean square residuals = {residuals}\n\t"
                f"rank = {rank}\n\t"
                f"singular values = {sing_values}"
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
            logger.debug(
                "The TimeCOAPoly fit details:\n\t"
                f"root mean square residuals = {residuals}\n\t"
                f"rank = {rank}\n\t"
                f"singular values = {sing_values}"
            )
            grid.TimeCOAPoly = Poly2DType(Coefs=time_coa_poly)
            if return_time_dets:
                return time_coa_sampled.min(), time_coa_sampled.max()
            else:
                return None

        def adjust_time(sicd, time_offset):
            # type: (SICDType, float) -> None
            """
            Shifts the time-referenced polynomials of a burst by a time offset,
            in place.

            Subtracts the offset from the constant terms of `Grid.TimeCOAPoly`
            and `RMA.INCA.TimeCAPoly`, and shifts `Position.ARPPoly` by the
            negative offset.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            time_offset : float
                The time offset (seconds) by which to shift the polynomials.
            """

            # adjust TimeCOAPoly
            sicd.Grid.TimeCOAPoly.Coefs[0, 0] -= time_offset
            # adjust TimeCAPoly
            sicd.RMA.INCA.TimeCAPoly.Coefs[0] -= time_offset
            # shift ARPPoly
            sicd.Position.ARPPoly = sicd.Position.ARPPoly.shift(-time_offset, return_poly=True)

        def update_geodata(sicd):  # type: (SICDType) -> None
            """
            Projects the SCP pixel to the ground and updates the SCP of the
            SICD `GeoData` element in place.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            """

            scp_pixel = [sicd.ImageData.SCPPixel.Row, sicd.ImageData.SCPPixel.Col]
            ecf = sicd.project_image_to_ground(scp_pixel, projection_type="HAE")
            sicd.update_scp(ecf, coord_system="ECF")

        def get_scps(count):
            """
            Interpolates the SCP ground positions for the images of each burst.

            Constructs SCPPixel locations at the center of each burst image and
            interpolates them through the geolocation grid points of the product
            annotation using `scipy.interpolate.griddata`.

            Parameters
            ----------
            count : int
                The number of SCPs, i.e. the number of burst images, to compute.

            Returns
            -------
            numpy.ndarray
                Array of shape `(count, 3)` with the ECF (meters) SCP coordinates.
            """

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
            """
            Completes the SICD metadata for a STRIPMAP (no bursts) collect.

            Sets the preliminary geodata and valid-data extent, then the core name,
            timeline, position, RMA/grid elements, and updated geodata using the
            first/last line sensing times from the downlink annotation.

            Returns
            -------
            SICDType
                The completed SICD metadata object for the stripmap collect.
            """

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
            downlink_info = "./generalAnnotation/downlinkInformationList/downlinkInformation"
            start_dt = datetime.strptime(  # noqa: DTZ007
                find_string(root_node, f"{downlink_info}/firstLineSensingTime"),
                DT_FMT,
            )
            start = numpy.datetime64(start_dt, "us")
            stop = parse_timestring(find_string(root_node, f"{downlink_info}/lastLineSensingTime"))
            set_core_name(out_sicd, start_dt, 0)
            set_timeline(out_sicd, start, get_seconds(stop, start, precision="us"))
            set_position(out_sicd, start)

            azimuth_time_first_line = parse_timestring(
                find_string(root_node, "./imageAnnotation/imageInformation/productFirstLineUtcTime")
            )
            first_line_relative_start = get_seconds(azimuth_time_first_line, start, precision="us")
            update_rma_and_grid(out_sicd, first_line_relative_start, start)
            update_geodata(out_sicd)
            return out_sicd

        def compute_burst_id(relative_orbit_number, anx_time, swath):
            """
            Computes the Sentinel-1 ANX-relative burst ID string.

            Estimates the burst index from the relative orbit number and the burst
            ANX time using time-per-beam and orbital-period constants along with
            swath-specific time offsets; east-look IDs are offset by 11.

            Parameters
            ----------
            relative_orbit_number : int
                The relative orbit number of the collection.
            anx_time : float
                The ANX time of the burst in seconds, from `azimuthAnxTime`.
            swath : str
                The swath indicator (lowercase, e.g. `iw` or `ew`).

            Returns
            -------
            str
                The zero-padded 6-digit burst ID string.
            """

            mode = swath[:-1]
            TBEAM = {"iw": 2.758273, "ew": 3.0388819516}  # Time per beam in seconds
            TORB = 12 * 86400 / 175  # Orbital period in seconds
            DELTA = {
                "iw1": -0.5469335,
                "iw2": 0.2852695,
                "iw3": 1.3633065,
                "ew1": -0.20158940051821037,
                "ew2": 0.4289565994970726,
                "ew3": 0.9952805994812225,
                "ew4": 1.5820385994932618,
                "ew5": 2.192150599480655,
            }  # Swath-specific time offsets
            t = (relative_orbit_number - 1) * TORB + anx_time - DELTA[swath]
            burst_id = int(numpy.round(t / TBEAM[mode]))
            if mode == "ew":
                burst_id += 11

            return f"{burst_id:06}"

        def get_relative_orbit_number():
            """
            Returns the relative orbit number from the manifest
            `safe:relativeOrbitNumber` field.

            Returns
            -------
            int
                The relative orbit number.
            """

            return int(self._find(".//safe:relativeOrbitNumber").text)

        def finalize_bursts():  # type: () -> List[SICDType]
            """
            Splits the collection into one completed SICD per burst.

            For each burst, sets the preliminary geodata and the valid-data extent
            from the `firstValidSample`/`lastValidSample` fields of the burst list,
            computes the burst ID and core name, fits the position and the RMA/grid
            polynomials, adjusts the times to the burst's zero-doppler extent, and
            updates the geodata.

            Returns
            -------
            list[SICDType]
                One completed SICD metadata object per burst.
            """

            # we will have one sicd per burst.
            sicds = []
            scps = get_scps(len(burst_list))

            for j, burst in enumerate(burst_list):
                t_sicd = out_sicd.copy()
                # set preliminary geodata (required for projection)
                t_sicd.GeoData = GeoDataType(SCP=SCPType(ECF=scps[j, :]))  # EarthModel & LLH are implicitly set

                xml_first_cols = numpy.fromstring(find_string(burst, "./firstValidSample"), sep=" ", dtype=numpy.int64)
                xml_last_cols = numpy.fromstring(find_string(burst, "./lastValidSample"), sep=" ", dtype=numpy.int64)
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
                relative_orbit_number = get_relative_orbit_number()
                burst_anx_time = float(find_string(burst, "./azimuthAnxTime"))
                swath = t_sicd.CollectionInfo.Parameters.get("SWATH").lower()
                burst_id = compute_burst_id(relative_orbit_number, burst_anx_time, swath)

                # This is the first and last zero doppler times of the columns in the burst.
                # Not really CollectStart and CollectDuration in SICD (first last pulse time)
                start_dt = datetime.strptime(find_string(burst, "./azimuthTime"), DT_FMT)  # noqa: DTZ007
                start = numpy.datetime64(start_dt, "us")
                set_core_name(t_sicd, start_dt, burst_id)
                set_position(t_sicd, start)
                res = update_rma_and_grid(t_sicd, 0, start, return_time_dets=True)
                if res is None:
                    raise TypeError(f"Expected res tuple, got {type(res)}")
                early, late = res
                new_start = start + numpy.timedelta64(numpy.int64(early * 1e6), "us")
                duration = late - early
                set_timeline(t_sicd, new_start, duration)
                # adjust my time offset
                adjust_time(t_sicd, early)
                update_geodata(t_sicd)
                sicds.append(t_sicd)
            return sicds

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
        if len(burst_list) > 0:
            return finalize_bursts()
        else:
            return finalize_stripmap()

    def _refine_using_calibration(self, cal_file_name: str, sicds: SICDType | list[SICDType]) -> None:
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
        if self._base_sicd.ImageCreation.DateTime < numpy.datetime64("2015-11-25"):
            return
        root_node = _parse_xml(cal_file_name, without_ns=True)
        if not isinstance(root_node, ElementTree.Element):
            raise TypeError(f"Expected Element, got {type(root_node)}")

        if isinstance(sicds, SICDType):
            sicds = [
                sicds,
            ]

        def update_sicd(sicd, index):  # type: (SICDType, int) -> None
            """
            Fits the radiometric scale-factor polynomials for one SICD of the
            collection and sets `SigmaZeroSFPoly`, `BetaZeroSFPoly`, and
            `GammaZeroSFPoly` on its `Radiometric` element in place.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            index : int
                The index of the SICD within the collection, used to select the
                calibration lines belonging to this burst.
            """

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
            if not isinstance(pixel, numpy.ndarray):
                raise TypeError(f"Expected type(pixel)=numpy.ndarray, got {type(pixel)}")
            coords_az = numpy.repeat(coords_az, pixel.shape[1])
            if valid_count > 1:
                coords_az = coords_az.reshape((valid_count, -1))

            def create_poly(arr, poly_order=2, poly_name=""):
                """
                Fits a two-dimensional polynomial to `arr` over the range/azimuth
                coordinates captured in the enclosing scope, logs the fit details,
                and returns the result as a `Poly2DType`.

                Parameters
                ----------
                arr : numpy.ndarray
                    The gridded values to fit.
                poly_order : int, optional
                    The polynomial order in each direction. Defaults to 2.
                poly_name : str, optional
                    A name used only to identify the fit in the debug log.
                    Defaults to ''.

                Returns
                -------
                Poly2DType
                    The fitted 2D polynomial.
                """

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
                logger.debug(
                    f"The {poly_name} polynomial fit details:\n\t"
                    f"root mean square residuals = {residuals}\n\t"
                    f"rank = {rank}\n\t"
                    f"singular values = {sing_values}"
                )
                return Poly2DType(Coefs=poly)

            if sicd.Radiometric is None:
                sicd.Radiometric = RadiometricType()
            if not isinstance(sigma, numpy.ndarray):
                raise TypeError(f"Expected sigma np.ndarray, got {type(sigma)}")
            if not isinstance(beta, numpy.ndarray):
                raise TypeError(f"Expected beta np.ndarray, got {type(beta)}")
            if not isinstance(gamma, numpy.ndarray):
                raise TypeError(f"Expected sigma np.ndarray, got {type(sigma)}")

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

        cal_vector_list = root_node.findall("./calibrationVectorList/calibrationVector")
        line = numpy.empty((len(cal_vector_list),), dtype=numpy.float64)
        pixel, sigma, beta, gamma = [], [], [], []
        for i, cal_vector in enumerate(cal_vector_list):
            line[i] = float(find_string(cal_vector, "./line"))
            pixel.append(numpy.fromstring(find_string(cal_vector, "./pixel"), sep=" ", dtype=numpy.float64))
            sigma.append(numpy.fromstring(find_string(cal_vector, "./sigmaNought"), sep=" ", dtype=numpy.float64))
            beta.append(numpy.fromstring(find_string(cal_vector, "./betaNought"), sep=" ", dtype=numpy.float64))
            gamma.append(numpy.fromstring(find_string(cal_vector, "./gamma"), sep=" ", dtype=numpy.float64))
        lines_per_burst = sicds[0].ImageData.NumCols
        pixel = numpy.array(pixel)
        sigma = numpy.array(sigma)
        beta = numpy.array(beta)
        gamma = numpy.array(gamma)
        # adjust sentinel values for sicd convention (square and invert)
        sigma = 1.0 / (sigma * sigma)
        beta = 1.0 / (beta * beta)
        gamma = 1.0 / (gamma * gamma)

        for ind, sic in enumerate(sicds):
            update_sicd(sic, ind)

    def _refine_using_noise(self, noise_file_name: str, sicds: SICDType | list[SICDType]) -> None:
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
        if self._base_sicd.ImageCreation.DateTime < numpy.datetime64("2015-11-25"):
            return
        root_node = _parse_xml(noise_file_name, without_ns=True)
        if not isinstance(root_node, ElementTree.Element):
            raise TypeError(f"Expected Element, got {type(root_node)}")
        if isinstance(sicds, SICDType):
            sicds = [
                sicds,
            ]

        mode_id = sicds[0].CollectionInfo.RadarMode.ModeID
        lines_per_burst = sicds[0].ImageData.NumCols
        range_size_pixels = sicds[0].ImageData.NumRows

        def extract_vector(stem):
            # type: (str) -> Tuple[List[numpy.ndarray], List[Union[None, numpy.ndarray]], List[numpy.ndarray]]
            """
            Extracts the noise LUT vectors of `stemVectorList` (e.g. `noise`,
            `noiseRange`, or `noiseAzimuth`) from the noise annotation file.

            Ignores vectors whose lines are all negative and vectors whose LUT
            values are all zero, and converts the noise values to dB.

            Parameters
            ----------
            stem : str
                The XML element stem prefix of the vector list to extract.

            Returns
            -------
            tuple
                A tuple `(lines, pixels, noises)` with, per vector, the line
                numbers, the pixel indices (None for azimuth noise), and the
                noise LUT values in dB.
            """

            lines = []
            pixels = []
            noises = []
            noise_vector_list = root_node.findall(f"./{stem:s}VectorList/{stem:s}Vector")
            for i, noise_vector in enumerate(noise_vector_list):
                line = numpy.fromstring(find_string(noise_vector, "./line"), dtype=numpy.int64, sep=" ")
                # some datasets have noise vectors for negative lines - ignore these
                if numpy.all(line < 0):
                    continue
                pixel_node = noise_vector.find("./pixel")  # does not exist for azimuth noise
                if pixel_node is not None:
                    pixel = numpy.fromstring(str(pixel_node.text), dtype=numpy.int64, sep=" ")
                else:
                    pixel = None
                noise = numpy.fromstring(find_string(noise_vector, f"./{stem}Lut"), dtype=numpy.float64, sep=" ")
                # some datasets do not have any noise data (all 0's) - skipping these will throw things into disarray
                if not numpy.all(noise == 0):
                    # convert noise to dB - what about -inf values?
                    noise = 10 * numpy.log10(noise)
                if not isinstance(noise, numpy.ndarray):
                    raise TypeError(f"Expected noise numpy.ndarray, got {type(noise)}")

                # do some validity checks
                if (mode_id == "IW") and numpy.any((line % lines_per_burst) != 0) and (i != len(noise_vector_list) - 1):
                    # NB: the final burst has different timing
                    logger.error(
                        "Noise file should have one lut per burst, but more are present.\n\tThis may lead to confusion."
                    )
                if (pixel is not None) and (pixel[-1] > range_size_pixels):
                    logger.error("Noise file has more pixels in LUT than range size.\n\tThis may lead to confusion.")

                lines.append(line)
                pixels.append(pixel)
                noises.append(noise)
            return lines, pixels, noises

        def populate_noise(sicd, index):  # type: (SICDType, int) -> None
            """
            Fits the noise polynomial from the noise LUT data captured in the
            enclosing scope and sets it as an absolute `NoiseLevel` on the
            SICD `Radiometric` element in place.

            For STRIPMAP mode, all LUTs apply and a full 2D fit is made; for
            TOPSAR, a per-burst range polynomial is combined weakly with an
            azimuth polynomial from the (may be absent) azimuth noise data.

            Parameters
            ----------
            sicd : SICDType
                The SICD metadata object to modify in place.
            index : int
                The index of the SICD within the collection, used to select the
                relevant burst noise data.
            """

            # NB: the default order was 7 before refactoring...that seems excessive.

            if not isinstance(range_pixel[0], numpy.ndarray):
                raise TypeError(f"Expected numpy.ndarray, got {type(range_pixel)}")
            rg_poly_order = min(5, range_pixel[0].size - 1)
            if sicd.CollectionInfo.RadarMode.ModeID[0] == "S":
                # STRIPMAP - all LUTs apply
                # Treat range and azimuth polynomial components as fully independent
                az_poly_order = min(4, len(range_line) - 1)

                # NB: the previous rammed together two one-dimensional polys, but
                # we should do a 2-d fit.
                coords_rg = (range_pixel[0] + sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row) * sicd.Grid.Row.SS
                coords_az = (range_line + sicd.ImageData.FirstCol - sicd.ImageData.SCPPixel.Col) * sicd.Grid.Col.SS

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
                logger.debug(
                    "NoisePoly fit details:\n\t"
                    f"root mean square residuals = {res}\n\t"
                    f"rank = {rank}\n\t"
                    f"singular values = {sing_vals}"
                )
            else:
                # TOPSAR has single LUT per burst
                # Treat range and azimuth polynomial components as weakly independent
                if index >= len(range_pixel):
                    logger.error(
                        "We have run out of noise information.\n\t"
                        f"Current index = {index}, length of noise array = {len(range_pixel)}.\n\t"
                        "The previous noise information will be used to populate the NoisePoly."
                    )
                rp_array = range_pixel[min(index, len(range_pixel) - 1)]
                rn_array = range_noise[min(index, len(range_pixel) - 1)]
                coords_rg = (rp_array + sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row) * sicd.Grid.Row.SS
                rg_poly = numpy.array(polynomial.polyfit(coords_rg, rn_array, rg_poly_order))
                az_poly = None
                if azimuth_noise is not None:
                    if not isinstance(azimuth_line, numpy.ndarray):
                        raise TypeError(f"Expected numpy.ndarray, got {type(azimuth_line)}")
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

        # extract our noise vectors (used in populate_noise through implicit reference)
        if root_node.find("./noiseVectorList") is not None:
            # probably prior to March 2018
            range_line, range_pixel, range_noise = extract_vector("noise")
            azimuth_line, _, azimuth_noise = None, None, None
        else:
            # noiseRange and noiseAzimuth fields began in March 2018
            range_line, range_pixel, range_noise = extract_vector("noiseRange")
            azimuth_line, _, azimuth_noise = extract_vector("noiseAzimuth")
            try:
                azimuth_line = numpy.concatenate(azimuth_line, axis=0)
            except Exception as e:  # noqa: BLE001
                # Fallback for test cases or missing metadata.
                logger.warning(f"Error: {e.__class__.__name__}")
                azimuth_line = None
        # NB: range_line is actually a list of 1 element arrays - probably should have been parsed better
        range_line = numpy.concatenate(range_line, axis=0)
        for ind, sic in enumerate(sicds):
            populate_noise(sic, ind)

    @staticmethod
    def _derive(sicds: SICDType | list[SICDType]) -> None:
        """
        Populates the derived SICD fields (via `SICDType.derive()` and
        `populate_rniirs`) for a single SICD or a list of them, in place.

        Parameters
        ----------
        sicds : SICDType | list[SICDType]
            The SICD metadata object(s) whose derived fields are populated.
        """

        if isinstance(sicds, SICDType):
            sicds.derive()
            sicds.populate_rniirs(override=False)
        else:
            for sicd in sicds:
                sicd.derive()
                sicd.populate_rniirs(override=False)

    def get_sicd_collection(self) -> list[tuple[str, SICDType | list[SICDType]]]:
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
        for entry in self._get_file_sets():
            # get the sicd collection for each product
            sicds = self._parse_product_sicd(entry["product"])
            # refine our sicds(s) using the calibration data (if sensible)
            self._refine_using_calibration(entry["calibration"], sicds)
            # refine our sicd(s) using the noise data (if sensible)
            self._refine_using_noise(entry["noise"], sicds)
            # populate our derived fields for the sicds
            self._derive(sicds)
            out.append((entry["data"], sicds))
        return out


class SentinelReader(SICDTypeReader):
    """
    A Sentinel-1 SLC file package reader implementation.

    **Changed in version 1.3.0** for reading changes.
    """

    __slots__ = ("_parent_segments", "_sentinel_details")

    def __init__(self, sentinel_details: str | SentinelDetails):
        """

        Parameters
        ----------
        sentinel_details : str|SentinelDetails
        """

        if isinstance(sentinel_details, str):
            sentinel_details = SentinelDetails(sentinel_details)
        if not isinstance(sentinel_details, SentinelDetails):
            raise TypeError("Input argument for SentinelReader must be a file name or SentinelReader object.")

        self._sentinel_details = sentinel_details  # type: SentinelDetails

        reverse_axes = None
        transpose_axes = (1, 0, 2)  # True for all Sentinel-1 data

        parent_segments = []
        segments = []
        sicd_collection = self._sentinel_details.get_sicd_collection()
        sicd_collection_out = []
        for data_file, sicds in sicd_collection:
            tiff_details = TiffDetails(data_file)
            if isinstance(sicds, SICDType):
                segments.append(
                    NativeTiffDataSegment(tiff_details, reverse_axes=reverse_axes, transpose_axes=transpose_axes)
                )
                sicd_collection_out.append(sicds)
            elif len(sicds) == 1:
                segments.append(
                    NativeTiffDataSegment(tiff_details, reverse_axes=reverse_axes, transpose_axes=transpose_axes)
                )
                sicd_collection_out.append(sicds[0])
            else:
                p_segment = NativeTiffDataSegment(
                    tiff_details, reverse_axes=reverse_axes, transpose_axes=transpose_axes
                )
                parent_segments.append(p_segment)
                begin_col = 0
                for sicd in sicds:
                    end_col = begin_col + sicd.ImageData.NumCols
                    subet_def = (slice(0, tiff_details.tags["ImageWidth"], 1), slice(begin_col, end_col, 1))
                    segments.append(SubsetSegment(p_segment, subet_def, "formatted", close_parent=False))
                    begin_col = end_col
                    sicd_collection_out.append(sicd)

        self._parent_segments = parent_segments  # type: List[NativeTiffDataSegment]
        SICDTypeReader.__init__(self, segments, sicd_collection_out, close_segments=True)
        self._check_sizes()

    @property
    def sentinel_details(self) -> SentinelDetails:
        """
        SentinelDetails: The sentinel details object.
        """

        return self._sentinel_details

    @property
    def file_name(self) -> str:
        """
        str: the Sentinel-1 package directory name
        """

        return self.sentinel_details.directory_name

    def close(self):
        """
        Closes the reader and any parent TIFF data segments.

        Calls `BaseReader.close` on the reader, and then pops and closes each
        parent segment held for multi-burst products, logging (rather than
        raising) any exception encountered.
        """

        BaseReader.close(self)
        if hasattr(self, "_parent_segments") and self._parent_segments is not None:
            # noinspection PyBroadException
            try:
                while len(self._parent_segments) > 0:
                    segment = self._parent_segments.pop()
                    segment.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Exception is: {e.__class__.__name__}")
            self._parent_segments = None


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name: str) -> SentinelReader | None:
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

    if is_file_like(file_name):
        return None

    try:
        sentinel_details = SentinelDetails(file_name)
        logger.debug(f"Path {file_name} is determined to be or contain a Sentinel-1 manifest.safe file.")
        return SentinelReader(sentinel_details)
    except (SarpyIOError, AttributeError, SyntaxError, ElementTree.ParseError):
        return None


__all__ = [
    "SentinelDetails",
    "SentinelReader",
    "is_a",
]
