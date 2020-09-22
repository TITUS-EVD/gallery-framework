
import ROOT
#from ROOT import larutil, galleryfmwk
from ROOT import galleryfmwk
import numpy as np

import os

class geoBase(object):

    """docstring for geometry"""

    def __init__(self):
        super(geoBase, self).__init__()
        self._nViews = 2
        self._nTPCs = 1
        self._nCryos = 1
        self._nPlanes = 2
        self._split_wire = False
        self._view_names = ['U', 'V', 'Y']
        self._tRange = 1600
        self._wRange = [240, 240]
        self._aspectRatio = 4
        self._time2Cm = 0.1
        self._wire2Cm = 0.4
        self._samplingRate = 0.5
        self._levels = [(-15, 15), (-10, 30)]
        self._pedestals = [0, 0]
        self._name = "null"
        self._offset = [0, 0]
        self._halfwidth = 1.0
        self._halfheight = 1.0
        self._length = 1.0
        self._haslogo = False
        self._logo = None
        self._path = os.path.dirname(os.path.realpath(__file__))
        self._logopos = [0,0]
        self._logoscale = 1.0
        self._triggerOffset = 60
        self._readoutWindowSize = 2408
        self._planeOriginX = [-0.2, -0.6]
        self._planeOriginXTicks = [-0.2/0.4, -0.6/0.4]
        self._readoutPadding = 0
        self._timeOffsetTicks = 0
        self._timeOffsetCm = 0
        self._cathodeGap = 0
        self._opdet_radius = 10.16 #cm
        self._opdet_x = [0]
        self._opdet_y = [0]
        self._opdet_name = ['pmt']
        self._opdet_default = -9999
        self._n_optical_frames = 1
        self._n_optical_offset = 0
        self._plane_mix = {}
        self._plane_flip = []

        self._geometryCore = None
        self._detectorProperties = None
        self._clockProperties = None
        self._lar_properties = None

    def name(self):
        return self._name

    def halfwidth(self):
       return self._halfwidth

    def halfheight(self):
       return self._halfheight

    def length(self):
       return self._length

    def nViews(self):
        return self._nViews

    def nTPCs(self):
        return self._nTPCs

    def nCryos(self):
        return self._nCryos

    def nPlanes(self):
        return self._nPlanes

    def splitWire(self):
        return self._split_wire

    def viewNames(self):
        return self._view_names

    def planeFlip(self):
        return self._plane_flip

    def planeMix(self):
        return self._plane_mix

    def tRange(self):
        return self._tRange

    def wRange(self, plane):
        return self._wRange[plane]

    def getLevels(self, plane):
        return self._levels[plane]

    def getPedestal(self, plane):
        return self._pedestals[plane]

    def aspectRatio(self):
        return self._aspectRatio

    def getBlankData(self, plane):
        return np.ones((self._wRange[plane], self._tRange))

    def wire2cm(self):
        return self._wire2Cm

    def time2cm(self):
        return self._time2Cm

    def samplingRate(self):
        return self._samplingRate

    def name(self):
        return self._name

    def offset(self, plane):
        return self._offset[plane]

    def hasLogo(self):
        return self._haslogo

    def logo(self):
        return self._logo

    def logoScale(self):
        return self._logoscale

    def logoPos(self):
        return self._logopos

    def readoutWindowSize(self):
        return self._readoutWindowSize

    def readoutPadding(self):
        return self._readoutPadding

    def triggerOffset(self):
        return self._triggerOffset

    def planeOriginX(self, plane):
        return self._planeOriginX[plane]

    def timeOffsetTicks(self, plane):
        return self._timeOffsetTicks
        # return self._timeOffsetTicks + self._planeOriginXTicks[plane]

    def timeOffsetCm(self, plane):
        return self._timeOffsetCm

    def cathodeGap(self):
        return self._cathodeGap

    def opdetLoc(self):
        return self._opdet_x, self._opdet_y, self._opdet_z

    def opdetName(self):
        return self._opdet_name

    def opdetRadius(self):
        return self._opdet_radius

    def opdetToTPC(self):
        return

    def opdetDefaultValue(self):
        return self._opdet_default

    def nOpticalFrames(self):
        return self._n_optical_frames

    def opticalOffset(self):
        return self._n_optical_offset

    def getGeometryCore(self):
        return self._geometryCore

    def getDetectorProperties(self):
        return self._detectorProperties

    def getDetectorClocks(self):
        return self._detectorClocks

    def getLArProperties(self):
        return self._lar_properties

    def getPlaneID(self, plane, tpc, cryo):
        if (plane, tpc, cryo) not in self._planeid_map:
            raise Exception (f'{plane}, {tpc}, {cryo} not available.')
        return self._planeid_map[(plane, tpc, cryo)]

    def getOtherPlanes(self, plane_id):
        if plane_id not in self._planeid_to_other_planes:
            raise Exception (f'{plane_id} not available.')
        return self._planeid_to_other_planes[plane_id]

    def flipPlane(self, plane_id):
        if plane_id > len(self._plane_flip):
            raise Exception (f'{plane_id} not available.')
        return self._plane_flip[plane_id]

    def shiftPlane(self, plane_id):
        if plane_id > len(self._plane_shift):
            raise Exception (f'{plane_id} not available.')
        return self._plane_shift[plane_id]


class geometry(geoBase):

    def __init__(self):
        super(geometry, self).__init__()
        self._defaultColorScheme = []
        self._colorScheme = {}

        self._defaultColorScheme = [(
            {'ticks': [(1, (22, 30, 151, 255)),
                       (0.791, (0, 181, 226, 255)),
                       (0.645, (76, 140, 43, 255)),
                       (0.47, (0, 206, 24, 255)),
                       (0.33333, (254, 209, 65, 255)),
                       (0, (255, 0, 0, 255))],
             'mode': 'rgb'})]
        self._defaultColorScheme.append(
            {'ticks': [(0, (22, 30, 151, 255)),
                       (0.33333, (0, 181, 226, 255)),
                       (0.47, (76, 140, 43, 255)),
                       (0.645, (0, 206, 24, 255)),
                       (0.791, (254, 209, 65, 255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'})
        self._defaultColorScheme.append(
            {'ticks': [(0, (22, 30, 151, 255)),
                       (0.33333, (0, 181, 226, 255)),
                       (0.47, (76, 140, 43, 255)),
                       (0.645, (0, 206, 24, 255)),
                       (0.791, (254, 209, 65, 255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'})
        self._defaultColorScheme.append(
            {'ticks': [(1, (22, 30, 151, 255)),
                       (0.791, (0, 181, 226, 255)),
                       (0.645, (76, 140, 43, 255)),
                       (0.47, (0, 206, 24, 255)),
                       (0.33333, (254, 209, 65, 255)),
                       (0, (255, 0, 0, 255))],
             'mode': 'rgb'})
        self._defaultColorScheme.append(
            {'ticks': [(0, (22, 30, 151, 255)),
                       (0.33333, (0, 181, 226, 255)),
                       (0.47, (76, 140, 43, 255)),
                       (0.645, (0, 206, 24, 255)),
                       (0.791, (254, 209, 65, 255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'})
        self._defaultColorScheme.append(
            {'ticks': [(0, (22, 30, 151, 255)),
                       (0.33333, (0, 181, 226, 255)),
                       (0.47, (76, 140, 43, 255)),
                       (0.645, (0, 206, 24, 255)),
                       (0.791, (254, 209, 65, 255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'})

    def configure(self):
        '''
        This is the default configuration
        that uses the singleton implemetation of
        the Geometry and GeometryHelper
        '''
        self._halfwidth = larutil.Geometry.GetME().DetHalfWidth()
        self._halfheight = larutil.Geometry.GetME().DetHalfHeight()
        self._length = larutil.Geometry.GetME().DetLength()
        self._time2Cm = larutil.GeometryHelper.GetME().TimeToCm()
        self._wire2Cm = larutil.GeometryHelper.GetME().WireToCm()
        self._samplingRate = larutil.DetectorProperties.GetME().SamplingRate()
        self._aspectRatio = self._wire2Cm / self._time2Cm
        self._nViews = larutil.Geometry.GetME().Nviews()
        self._nTPCs = int(larutil.Geometry.GetME().NTPC())
        self._nPlanes = int(larutil.Geometry.GetME().Nplanes())
        # self._tRange = larutil.DetectorProperties.GetME().ReadOutWindowSize()
        self._wRange = []
        self._offset = []
        for v in range(0, self._nViews):
            self._wRange.append(larutil.Geometry.GetME().Nwires(v))

        self._opdet_x = []
        self._opdet_y = []
        self._opdet_z = []
        self._opdet_name = []
        for d in range (0, larutil.Geometry.GetME().NOpDets()):
            self._opdet_x.append(larutil.Geometry.GetME().OpDetX(d))
            self._opdet_y.append(larutil.Geometry.GetME().OpDetY(d))
            self._opdet_z.append(larutil.Geometry.GetME().OpDetZ(d))
            self._opdet_name.append(larutil.Geometry.GetME().OpDetNameFromOpChannel(d))


    def configure(self, geometryCore, detProperties, detClocks, lar_properties):
        '''
        This is a new implementation that
        uses LArSoft services to get the
        GeometryCore and DetectorProperties
        '''
        if geometryCore is None or detProperties is None or detClocks is None:
            self.configure()
            return

        print ('Configuring geometry from services.')

        self._geometryCore = geometryCore
        self._detectorClocks = detClocks.DataForJob()
        self._detectorProperties = detProperties.DataFor(self._detectorClocks)
        self._lar_properties = lar_properties

        self._halfwidth = geometryCore.DetHalfWidth()
        self._halfheight = geometryCore.DetHalfHeight()
        self._length = geometryCore.DetLength()
        #self._time2Cm = detProperties.SamplingRate() / 1000.0 * detProperties.DriftVelocity(detProperties.Efield(), detProperties.Temperature())
        self._time2Cm = self._detectorClocks.TPCClock().TickPeriod() * self._detectorProperties.DriftVelocity(self._detectorProperties.Efield(), self._detectorProperties.Temperature())
        self._wire2Cm = geometryCore.WirePitch()
        self._samplingRate = self._detectorClocks.TPCClock().TickPeriod() * 1000. #detProperties.SamplingRate()
        self._aspectRatio = self._wire2Cm / self._time2Cm
        self._nViews = geometryCore.Nviews() * geometryCore.NTPC() * geometryCore.Ncryostats()
        self._nPlanes = geometryCore.Nplanes()
        self._nTPCs = int(geometryCore.NTPC())
        self._nCryos = int(geometryCore.Ncryostats())
        self._tRange = self._detectorProperties.NumberTimeSamples()
        self._readoutWindowSize = self._detectorProperties.NumberTimeSamples()
        #self._triggerOffset = detProperties.TriggerOffset()
        self._triggerOffset = self._detectorClocks.TPCClock().Ticks(self._detectorClocks.TriggerOffsetTPC() * -1.)

        self._wRange = []
        self._offset = []
        for v in range(0, self._nViews):
            self._wRange.append(geometryCore.Nwires(v))

        self._opdet_x = []
        self._opdet_y = []
        self._opdet_z = []
        self._opdet_name = []
        for opch in range(0, geometryCore.NOpDets()):
            xyz = geometryCore.OpDetGeoFromOpChannel(opch).GetCenter();
            self._opdet_x.append(xyz.X())
            self._opdet_y.append(xyz.Y())
            self._opdet_z.append(xyz.Z())
            shape_name = geometryCore.OpDetGeoFromOpChannel(opch).Shape().IsA().GetName()
            if shape_name == 'TGeoSphere':
                self._opdet_name.append('pmt_coated')
            elif shape_name == 'TGeoBBox':
                self._opdet_name.append('xarapuca_vis')
            else:
                self._opdet_name.append('unknown')

            # print ('opch', opch, 'shape', geometryCore.OpDetGeoFromOpChannel(opch).Shape().IsA().GetName())
            # self._opdet_radius = geometryCore.OpDetGeoFromOpChannel(opch).RMax()

    def recalculateOffsets(self):
        self._offset = []
        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )


    def colorMap(self, plane, colormaptype='default'):
        if colormaptype == 'default':
            return self._defaultColorScheme[plane]
        else:
            return self._colorScheme[colormaptype][plane]

class sbnd(geometry):


    def __init__(self, geometryCore=None, detProperties=None, detClocks=None, lar_properties=None):
        # Try to get the values from the geometry file.  Configure for sbnd
        # and then call the base class __init__
        super(sbnd, self).__init__()
        # larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kSBND)
        self.configure(geometryCore, detProperties, detClocks, lar_properties)

        # self._pedestals = [2048, 2048, 400, 2048, 2048, 400]
        # self._levels = [[-100, 10], [-10, 100], [-10, 200], [-100, 10], [-10, 100], [-10, 200]]
        self._pedestals = [0, 0, 0]
        self._levels = [(-80, 0), (-10, 100), (-10, 200)]

        self._view_names = ['U', 'V', 'Y']
        self._plane_mix = {0: [4], 1: [3], 2: [5]}
        self._plane_flip = [False, False, False, True, True, True]
        self._plane_shift = [False, False, False, False, False, False]

        self._name = "sbnd"
        self._logo = self._path + "/logos/SBND-color.png"
        self._logoRatio = 1.0
        self._haslogo = False
        self._logopos = [30, 30]
        self._logoscale = 0.13
        from .mapping import sbnd_opdet_map
        self._opdet_radius = 6
        self._opdet_name = sbnd_opdet_map
        self._tRange = 3000 #7500
        self._triggerOffset = 0 #2500
        self._readoutWindowSize = 3000 #7500
        self._planeOriginX = [0.0, -0.3, -0.6, 0.0, -0.3, -0.6]
        self._planeOriginXTicks = [0.0, -0.3/self._time2Cm, -0.6/self._time2Cm, 0.0, -0.3/self._time2Cm, -0.6/self._time2Cm]
        self._cathodeGap = 8.5 / self._time2Cm # 5.3 cm   # 100

        color_scheme = [(
            {'ticks': [(1, (255, 255, 255, 255)),
                       (0, (0, 0, 0, 255))],
             'mode': 'rgb'})]
        color_scheme.append(
            {'ticks': [(0, (255, 255, 255, 255)),
                       (1, (0, 0, 0, 255))],
             'mode': 'rgb'})
        color_scheme.append(
            {'ticks': [(0, (255, 255, 255, 255)),
                       (1, (0, 0, 0, 255))],
             'mode': 'rgb'})
        color_scheme.append(
            {'ticks': [(1, (255, 255, 255, 255)),
                       (0, (0, 0, 0, 255))],
             'mode': 'rgb'})
        color_scheme.append(
            {'ticks': [(0, (255, 255, 255, 255)),
                       (1, (0, 0, 0, 255))],
             'mode': 'rgb'})
        color_scheme.append(
            {'ticks': [(0, (255, 255, 255, 255)),
                       (1, (0, 0, 0, 255))],
             'mode': 'rgb'})

        self._colorScheme['grayscale'] = color_scheme

        self._n_optical_frames = 3
        self._n_optical_offset = 1250

        self._offset = []
        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )

        self._planeid_map = {
            # [plane, tpc, cryo]: titus_plane_id
            (0, 0, 0): 0,
            (1, 0, 0): 1,
            (2, 0, 0): 2,
            (0, 1, 0): 3,
            (1, 1, 0): 4,
            (2, 1, 0): 5,
            (0, 0, 1): 6,
            (1, 0, 1): 7,
            (2, 0, 1): 8,
            (0, 1, 1): 9,
            (1, 1, 1): 10,
            (2, 1, 1): 11
        }

        self._planeid_to_other_planes = {
            0: [4],
            1: [3],
            2: [5]
        }

    def opdetToTPC(self, ch):
        if (self._opdet_x[ch] < 0):
            return 0
        else:
            return 1


class icarus(geometry):


    def __init__(self, geometryCore=None, detProperties=None, detClocks=None, lar_properties=None, no_split_wire=False):
        # Try to get the values from the geometry file.  Configure for sbnd
        # and then call the base class __init__
        super(icarus, self).__init__()
        self.configure(geometryCore, detProperties, detClocks, lar_properties)

        self._pedestals = [0, 0, 0, 0, 0, 0]
        self._levels = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
        self._view_names = ['H', 'U', 'V']
        # self._plane_mix = {0: [3, 6, 9], 1: [5, 8, 11], 2: [4, 7, 10]}
        self._plane_mix = {0: [3], 1: [5], 2: [4], 6: [9], 7: [11], 8: [10]}
        self._plane_flip = [False, False, False, True, True, True, False, False, False, True, True, True]

        self._name = "icarus"
        self._logo = self._path + "/logos/logo_icarus.png"
        self._logoRatio = 1.0
        self._haslogo = False
        self._logopos = [30, 30]
        self._logoscale = 0.35
        self._opdet_radius = 7
        # self._tRange = 7500
        # self._triggerOffset = 2500
        # self._readoutWindowSize = 7500
        self._planeOriginX = [0.0, -0.3, -0.6,
                              0.0, -0.3, -0.6,
                              0.0, -0.3, -0.6,
                              0.0, -0.3, -0.6]
        self._planeOriginXTicks = [0.0, -0.3/self._time2Cm, -0.6/self._time2Cm,
                                   0.0, -0.3/self._time2Cm, -0.6/self._time2Cm,
                                   0.0, -0.3/self._time2Cm, -0.6/self._time2Cm,
                                   0.0, -0.3/self._time2Cm, -0.6/self._time2Cm]
        self._cathodeGap = 6 / self._time2Cm # 5.3 cm   # 100

        color_scheme = [(
            {'ticks': [(0, (255, 255, 255, 255)),
                       (1, (0, 0, 0, 255))],
             'mode': 'rgb'})]
        color_scheme.append(
            {'ticks': [(0, (255, 255, 255, 255)),
                       (1, (0, 0, 0, 255))],
             'mode': 'rgb'})
        color_scheme.append(
            {'ticks': [(0, (255, 255, 255, 255)),
                       (1, (0, 0, 0, 255))],
             'mode': 'rgb'})

        self._colorScheme['grayscale'] = color_scheme

        self._offset = []
        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.
            self._offset.append(0)
            # self._offset.append(
            #     self.triggerOffset()
            #     * self.time2cm()
            #     - self.planeOriginX(v) )

        self._planeid_map = {
            # [plane, tpc, cryo]: titus_plane_id
            (0, 0, 0): 0,
            (1, 0, 0): 1,
            (2, 0, 0): 2,
            (0, 1, 0): 3,
            (1, 1, 0): 4,
            (2, 1, 0): 5,
            (0, 2, 0): 6,
            (1, 2, 0): 7,
            (2, 2, 0): 8,
            (0, 3, 0): 9,
            (1, 3, 0): 10,
            (2, 3, 0): 11,
            (0, 0, 1): 12,
            (1, 0, 1): 13,
            (2, 0, 1): 14,
            (0, 1, 1): 15,
            (1, 1, 1): 16,
            (2, 1, 1): 17,
            (0, 2, 1): 18,
            (1, 2, 1): 19,
            (2, 2, 1): 20,
            (0, 3, 1): 21,
            (1, 3, 1): 22,
            (2, 3, 1): 23
        }

        self._planeid_to_other_planes = {
            0: [3, 6, 9],
            1: [4, 8, 11],
            2: [5, 7, 10],

            12: [15, 18, 21],
            13: [16, 20, 23],
            14: [17, 19, 22],
        }

        if not no_split_wire:
            self._split_wire = True
            self._nTPCs = int(self._nTPCs / 2)
            self._plane_mix = {0: [6], 1: [8], 2: [7], 12: [18], 13: [20], 14: [19]}
            self._plane_flip = [False, False, False, # TPC 0
                                False, False, False, # TPC 1
                                True, True, True,    # TPC 2
                                True, True, True,    # TPC 3
                                False, False, False, # TPC 4
                                False, False, False, # TPC 5
                                True, True, True,    # TPC 6
                                True, True, True]    # TPC 7
            self._plane_shift = [False, False, False, # TPC 0
                                True, True, True,     # TPC 1
                                False, False, False,    # TPC 2
                                True, True, True,    # TPC 3
                                False, False, False, # TPC 4
                                True, True, True, # TPC 5
                                False, False, False,    # TPC 6
                                True, True, True]    # TPC 7


            for v in range(0, len(self._wRange)):
                self._wRange[v] = self._wRange[v] * 2 + self._cathodeGap

    def opdetToTPC(self, ch):
        if (self._opdet_x[ch] < -100):
            return 0
        elif (self._opdet_x[ch] > -100 and self._opdet_x[ch] < 0):
            return 1
        elif (self._opdet_x[ch] > 0 and self._opdet_x[ch] < 100):
            return 2
        else:
            return 3


class microboone(geometry):

    def __init__(self):
        # Try to get the values from the geometry file.  Configure for microboone
        # and then call the base class __init__
        super(microboone, self).__init__()
        larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kMicroBooNE)
        self.configure()
        self._levels = [(-100, 10), (-10, 100), (-10, 200)]
        # self._colorScheme =
        # self._time2Cm = 0.05515
        self._pedestals = [2000, 2000, 440]
        self._name = "uboone"
        self._logo = self._path + "/logos/uboone_logo_bw_transparent.png"
        self._logoRatio = 1.0
        self._haslogo = True
        self._logopos = [1250,10]
        self._logoscale = 0.1
        self._tRange = 9600
        self._triggerOffset = 3200
        self._readoutWindowSize = 9600
        self._planeOriginX = [0.0, -0.3, -0.6]
        self._planeOriginXTicks = [0.0, -0.3/self._time2Cm, -0.6/self._time2Cm]
        # remove = larutil.DetectorProperties.GetME().TriggerOffset() \
        #           * larutil.GeometryHelper.GetME().TimeToCm()
        # self._offset[:] = [x - remove for x in self._offset]

        self._offset = []
        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )


class microboonetruncated(microboone):

    def __init__(self):
        super(microboonetruncated, self).__init__()

        # The truncated readouts change the trigger offset and
        self._tRange = 9600
        self._triggerOffset = 3200
        self._planeOriginX = [0.3, -0.3, -0.6]
        self._planeOriginXTicks = [0.3/self.time2cm(), -0.3/self.time2cm(), -0.6/self.time2cm()]
        self._readoutWindowSize = 9600
        self._readoutPadding = 2400
        self._offset = []
        self._timeOffsetTicks = 2400
        self._timeOffsetCm = 2400 * self._time2Cm
        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )

class argoneut(geometry):

    def __init__(self):
        # Try to get the values from the geometry file.  Configure for microboone
        # and then call the base class __init__
        super(argoneut, self).__init__()
        larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kArgoNeuT)
        self.configure()
        self._levels = [(-15, 60), (-25, 100)]
        self._pedestals = [0, 0]
        self._name = "argoneut"
        self._offset = []

        self._tRange = 1800
        self._triggerOffset = 60
        self._planeOriginX = [-0.2, -0.6]
        self._readoutWindowSize = 2048

        self._offset = []

        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )

        self._defaultColorScheme = [
            {'ticks': [(0.0,  (30,  30, 255, 255)),
                       (0.32,  (0,  255, 255, 255)),
                       (0.8,  (0,  255, 0,   255)),
                       (1,    (255,  0, 0,   255))],
             'mode': 'rgb'}]
        self._defaultColorScheme.append(
            {'ticks': [(0.0,  (30,  30, 255, 255)),
                       (0.4,  (0,  255, 255, 255)),
                       (0.8,  (0,  255, 0,   255)),
                       (1,    (255,  0, 0,   255))],
             'mode': 'rgb'})
        # self._offset = [1.7226813611, 2.4226813611]


class lariat(geometry):

    def __init__(self):
        # Try to get the values from the geometry file.  Configure for microboone
        # and then call the base class __init__
        super(lariat, self).__init__()
        larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kArgoNeuT)
        self.configure()
        # lariat has a different number of time ticks
        # fix it directly:
        self._tRange = 3072
        self._levels = [(-40, 160), (-80, 320)]
        self._pedestals = [0, 0]
        self._name = "lariat"
        # Get the logo too!
        self._logo = self._path + "/logos/LArIAT_simple_outline.png"
        self._haslogo = True
        self._logopos = [1200,10]
        self._logoscale = 0.2
        # Make default color schemes here:
        self._defaultColorScheme = [
            {'ticks': [(0, (30, 30, 255, 255)),
                       (0.33333, (0, 255, 255, 255)),
                       (0.66666, (255,255,100,255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'}]
        self._defaultColorScheme.append(
            {'ticks': [(0, (30, 30, 255, 255)),
                       (0.33333, (0, 255, 255, 255)),
                       (0.66666, (255,255,100,255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'})

