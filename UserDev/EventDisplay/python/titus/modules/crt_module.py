#/usr/bin/env python3

"""
This module adds the CRT view & associated controls
"""
import math

from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg 
import numpy as np

import ROOT
from ROOT import vector, string
from ROOT import set as ROOTset

# set up gallery to access some larcoreobj types
import galleryUtils
sc = galleryUtils.SourceCode
sc.loadHeaderFromUPS('larcorealg/Geometry/AuxDetSensitiveGeo.h')

from titus.modules import Module
import titus.drawables as drawables

_SBND_CRT_FEBDATA = 'sbnd::crt::FEBData'
_CRT_COLORMAP = pg.colormap.get('CET-L17')
# _CRT_COLORMAP.reverse()


class CrtModule(Module):
    def __init__(self, larsoft_module, geom_module):
        super().__init__()
        self._gm = geom_module
        self._lsm = larsoft_module
        self._central_widget = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout()
        self._central_widget.setLayout(self._layout)

        self._draw_crt_strips = True
        self._crt_strip_drawer = None
        self._crt_view = None

        self._dock =  QtWidgets.QDockWidget('CRT Controls', self._gui, objectName='crt_dock')
        self._dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._dock_widgets = set([self._dock])

    def _initialize(self):
        self._gm.geometryChanged.connect(self.init_ui)
        self._gui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock)

    def init_ui(self):
        if self._gm.current_geom is None:
            return
        self._crt_view = CrtView(self._gm.current_geom)
        self._layout.addWidget(self._crt_view)

        self.init_crt_controls()

    def init_crt_controls(self):
        frame = QtWidgets.QWidget(self._dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._dock.setWidget(frame)

        main_layout.addStretch()

    def update(self):
        all_producers = self._gi.get_producers(_SBND_CRT_FEBDATA, self._lsm.current_stage)
        if all_producers is None:
            self._draw_crt_strips = False
            if self._crt_strip_drawer is not None:
                self.remove_drawable(self._crt_strip_drawer)
                self._crt_strip_drawwer = None
            return

        self._draw_crt_strips = True
        if self._crt_strip_drawer is None:
            self._crt_strip_drawer = self.register_drawable(
                drawables.FEBData(self._gi, self._gm.current_geom)
            )
            self._crt_strip_drawer.set_producer(all_producers[0].full_name())
            self._crt_strip_drawer.analyze()

        self._crt_view.drawCrtData(self._crt_strip_drawer.getData())


class CrtView(QtWidgets.QSplitter):
    ''' Widget holding CRT PyQtGraph window and CRT controls '''
    def __init__(self, geometry, plane=-1):
        super().__init__()
        self._geometry = geometry
        self.setOrientation(QtCore.Qt.Vertical)
        self._view_widget = CrtViewWidget(self._geometry)
        self._time_widget = CrtTimeViewWidget()
        self.addWidget(self._view_widget)
        self.addWidget(self._time_widget)
        self._time_widget.timeWindowChanged.connect(self._view_widget.set_time_range)

    def drawCrtData(self, data):
        if data is None:
            return

        if data.size == 0:
            return
        
        self._view_widget.drawCrtData(data)
        self._time_widget.drawCrtHitTimes(data)

    def getWidget(self):
        return self, self._layout


class CrtHitsItem(pg.GraphicsObject):
    '''
    class to draw all CRT hits to QPicture first, so that QPicture can be drawn
    in one go. QPicture can also be cached for faster drawing when re-visiting
    events
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.picture = QtGui.QPicture()

    def add_hits(self, coords):
        painter = QtGui.QPainter(self.picture)
        if not coords:
            painter.end()
            return

        for coord_info in coords:
            coord_min, coord_max, tfrac = coord_info

            x, y = coord_min
            l = coord_max[0] - coord_min[0]
            w = coord_max[1] - coord_min[1]
            color = _CRT_COLORMAP.mapToQColor(tfrac)
            painter.setBrush(color)
            painter.setOpacity(0.8)
            painter.setPen(color)
            painter.drawRect(QtCore.QRectF(x, y, l, w))

        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class CrtViewWidget(pg.GraphicsLayoutWidget):
    ''' PyQtGraph widget to draw CRT data '''
    def __init__(self, geometry, plane=-1):
        super(CrtViewWidget, self).__init__(border=None)

        self._geometry = geometry

        self._data = None

        self._plot = pg.PlotItem(name="CRTPlot")
        self._plot.setAspectLocked(1)
        self._plot.hideAxis('left')
        self._plot.hideAxis('bottom')
        self.addItem(self._plot)

        self._crt_strips = {}
        self._crt_strip_bounds_map = {}
        self._drawn_crt_modules = []
        self._drawn_crt_hits = None

        self._init_crt_strips()
        self.init_geometry()
        self._crt_hit_picture_cache = {}

        self._min_time = -30e6
        self._max_time = 30e6

        # only draw hits within this range, separate from the min and max times of all hits
        self._draw_min_time = -30e6
        self._draw_max_time = 30e6

    def set_time_range(self, min_max_tuple):
        """Update the min and max times."""
        self._clear_cache()

        min_, max_ = min_max_tuple
        if min_ >= max_:
            raise ValueError(f"Attempt to set min/max time on CRT widget with invalid times [{min_}, {max_}]")

        self._draw_min_time = min_
        self._draw_max_time = max_
        self.drawCrtData(self._data)

    def _init_crt_strips(self):
        ''' create initial map of mac and strip ID to GDML objects '''
        geo_core = self._geometry.getGeometryCore()
        for ad_i in range(geo_core.NAuxDets()):
            # Get module from parent of the aux det in the GDML
            ad = geo_core.AuxDet(ad_i)
            ad_name = ad.TotalVolume().GetName()
            # FindAllVolumePaths needs a C++ set argument, can't initialize in one
            # line for some reason
            name_set = ROOTset(string)()
            name_set.insert(ad_name)
            geo_paths = geo_core.FindAllVolumePaths(name_set)

            path = ''
            for p in geo_paths[0]:
                path += p.GetName()
                path += '/'
            path = path[:-1]
                
            manager = geo_core.ROOTGeoManager()
            manager.cd(path)
            module = manager.GetMother(1);
            local_mac = module.GetNumber()

            nstrip = ad.NSensitiveVolume()

            for sv_i in range(nstrip):
                channel0 = 2 * sv_i + 0
                channel1 = 2 * sv_i + 1
                strip = ad.SensitiveVolume(sv_i)
                self._crt_strips[(local_mac, sv_i)] = strip
                self._crt_strip_bounds_map[strip] = self._crt_strip_world_bounds(strip)

    def _crt_strip(self, mac, sipm):
        ''' 
        return GDML volume for CRT strip associated with specified module and
        channel
        '''

        # two sipms per strip. They are sequential --> If odd, the other sipm
        # is the previous index, otherwise it's the next index
        strip_idx = sipm
        if sipm % 2 == 0:
            strip_idx /= 2
        else:
            strip_idx = (sipm - 1) / 2

        try:
            return self._crt_strips[(mac, strip_idx)]
        except KeyError:
            return None

    def _crt_plane_world_bounds(self):
        '''
        return min. and max. corners of CRT module plane defined by strips
        contained within. This is done coarsely, so that adjacent planes
        separated small amounts are grouped together. I'm not sure if there is
        a GDML way to do it, but AuxDet bounding boxes seem to be too large
        '''
        plane_number = 0
        # a dict because named planes may be useful for other geometries
        result = {}

        # AuxDets are the CRT strip arrays. The GDML mother of the AuxDets is the CRT module
        geo_core = self._geometry.getGeometryCore()
        nauxdet = geo_core.NAuxDets()
        for ad_i in range(nauxdet):
            ad = geo_core.AuxDet(ad_i)
            nstrip = ad.NSensitiveVolume()

            ad_name = ad.TotalVolume().GetName()
            name_set = ROOTset(string)()
            name_set.insert(ad_name)
            geo_paths = geo_core.FindAllVolumePaths(name_set)

            path = ''
            for p in geo_paths[0]:
                path += p.GetName()
                path += '/'
            # remove trailing /
            path = path[:-1]
                
            manager = geo_core.ROOTGeoManager()
            manager.cd(path)
            array_node = manager.GetCurrentNode()
            mod_node = manager.GetMother(1)
            tagger_node = manager.GetMother(2)
            det_node = manager.GetMother(3)
            
            hw = array_node.GetVolume().GetShape().GetDX();
            hh = array_node.GetVolume().GetShape().GetDY();
            hl = array_node.GetVolume().GetShape().GetDZ()/2;
            limits_min = np.array([-hw, -hh, -hl])
            limits_max = np.array([hw, hh, hl])

            def to_world_coord(limits):
                for node in [array_node, mod_node, tagger_node, det_node]:
                    new_limits = np.zeros(3)
                    node.LocalToMaster(limits, new_limits)
                    limits = new_limits.copy()
                return limits

            result[plane_number] = [to_world_coord(limits_min), to_world_coord(limits_max)]
            plane_number += 1

        return result

    def _crt_strip_world_bounds(self, strip):
        ''' return min. and max. corners of CRT strip bounding box in world coordinates '''

        try:
            return self._crt_strip_bounds_map[strip]
        except KeyError:
            pass

        hw = strip.HalfWidth1()
        hh = strip.HalfHeight()
        hl = strip.Length() / 2

        local_pt_min = ROOT.geo.AuxDetSensitiveGeo.LocalPoint_t(-hw, -hh, -hl)
        local_pt_max = ROOT.geo.AuxDetSensitiveGeo.LocalPoint_t(hw, hh, hl)
        world_coord_min = strip.toWorldCoords(local_pt_min)
        world_coord_max = strip.toWorldCoords(local_pt_max)

        self._crt_strip_bounds_map[strip] = (np.array([world_coord_min.X(), world_coord_min.Y(), world_coord_min.Z()]),\
            np.array([world_coord_max.X(), world_coord_max.Y(), world_coord_max.Z()]))
        
        return self._crt_strip_bounds_map[strip]

    def _crt_draw_pos(self, coord):
        '''
        convert world coordinates to drawn coordinates, unrolled CRT box view
        note: does not check if the point is actually on the CRT plane surface,
        just draws the point on the side chosen by some loose cuts
        '''

        # TODO replace the remaining magic numbers with detector-specific ones
        top = coord[1] > self._geometry.crt_top_ymin
        bot = coord[1] < self._geometry.crt_bot_ymax
        if top or bot:
            # z along xdir, x along ydir (top down)
            x = coord[2]
            y = coord[0]

            if coord[1] > 680:
                # SBND special 2nd top CRT
                return x + 2500, y + 1000
            if coord[1] > self._geometry.crt_top_ymin:
                return x + 1500, y + 1000
            else:
                return x, y

        # front or back
        back = coord[2] < self._geometry.crt_back_zmax
        front = coord[2] > self._geometry.crt_front_zmin
        if front or back:
            x = coord[1]
            y = coord[0]
            if back:
                return -x -600, y
            return x + 1200, y

        left = coord[0] > -380.0
        right = coord[0] < 381.3
        if left or right:
            # left or right side: z along xdir, y along ydir (side view)
            x = coord[2]
            y = coord[1]
            if coord[0] < 0: 
                return x, y + 800
            return x, -y - 800
        print('Warning: Got invalid point when trying to convert to CRT view ', coord)


    def init_geometry(self):
        ''' draw the permanent CRT module outlines '''
        plane_bounds = self._crt_plane_world_bounds()
        for plane, bounds in plane_bounds.items():
            pt_min, pt_max = bounds

            draw_min = self._crt_draw_pos(pt_min)
            draw_max = self._crt_draw_pos(pt_max)

            # positive widths and heights only!
            draw_min_sort = np.array([min(draw_min[i], draw_max[i]) for i in range(2)])
            draw_max_sort = np.array([max(draw_min[i], draw_max[i]) for i in range(2)])
            w = draw_max_sort[0] - draw_min_sort[0]
            h = draw_max_sort[1] - draw_min_sort[1]

            rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(draw_min_sort[0], draw_min_sort[1], w, h))
            rect.setPen(QtGui.QColor(255, 255, 255))
            self._drawn_crt_modules.append(rect)
            self._plot.addItem(rect)

    def drawCrtData(self, hit_array):
        ''' draw all hit strips '''
        self.clear_crt_hits()
        self._data = hit_array
        self._min_time = np.min(hit_array[:,1])
        self._max_time = np.max(hit_array[:,1])

        key = hit_array.data.tobytes()
        if key not in self._crt_hit_picture_cache:
            picture = CrtHitsItem()

            # draw hit strips
            draw_coords = []
            for hit in hit_array:
                # idx = module * 32 + sipm
                idx = int(hit[0])
                mod = idx // 32
                sipm = idx % 32 
                adc = hit[3]
                time = hit[1]

                if time < self._draw_min_time:
                    continue

                if time > self._draw_max_time:
                    continue

                if adc <= 0:
                    continue

                strip = self._crt_strip(mod, sipm)
                if strip is None:
                    print(f"Warning: Couldn't find strip with module={mod} sipm={sipm}")
                    continue

                pt_min, pt_max = self._crt_strip_world_bounds(strip)
                draw_min = self._crt_draw_pos(pt_min)
                draw_max = self._crt_draw_pos(pt_max)
                draw_min_sort = np.array([min(draw_min[i], draw_max[i]) for i in range(2)])
                draw_max_sort = np.array([max(draw_min[i], draw_max[i]) for i in range(2)])
                tfrac = (time - self._min_time) / (self._max_time - self._min_time)
                draw_coords.append((draw_min_sort, draw_max_sort, tfrac))
            
            picture.add_hits(draw_coords)
            self._crt_hit_picture_cache[key] = picture
                    
        self._plot.addItem(self._crt_hit_picture_cache[key])
        self._drawn_crt_hits = self._crt_hit_picture_cache[key]
        self._plot.update()

    def clear_crt_hits(self):
        ''' removes the currently-drawn CrtHitsItem '''
        if self._drawn_crt_hits is not None:
            self._plot.removeItem(self._drawn_crt_hits)
        self._drawn_crt_hits = None

    def _clear_cache(self):
        ''' removes all cached CrtHitsItems from this object '''
        self.clear_crt_hits()
        self._crt_hit_picture_cache = {}


class CrtTimeViewWidget(pg.GraphicsLayoutWidget):

    timeWindowChanged = QtCore.pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self._time_plot = pg.PlotItem(name="CRT Hit Times")
        self._time_plot.setLabel(axis='left', text='Hits')
        self._time_plot.setLabel(axis='bottom', text='Time (ns)')
        self.addItem(self._time_plot)

        self._time_range = [-30e6, 30e6]
        self._time_window = pg.LinearRegionItem(
            values=self._time_range, orientation=pg.LinearRegionItem.Vertical)
        self._time_window.sigRegionChangeFinished.connect(
            lambda: self.timeWindowChanged.emit(self._time_window.getRegion()))
        
        self._time_plot.addItem(self._time_window)


    def drawCrtHitTimes(self, crt_data):
        if crt_data is None:
            return

        if crt_data.size == 0:
            return
        times = crt_data[crt_data[:,3] > 0,1]

        # times are all zeros?
        if times.size == 0:
            return

        self._time_plot.clear()

        t_min = self._time_range[0]
        t_max = self._time_range[1]
        n_bins = min(200, int(t_max - t_min))

        if len(crt_data) == 1:
            t_min -= 100
            t_max += 100
            n_bins = 200

        data_y, data_x = np.histogram(times, bins=np.linspace(t_min, t_max, n_bins))
        self._time_plot.plot(x=data_x, y=data_y, stepMode=True, fillLevel=0, brush=_CRT_COLORMAP.getBrush(span=(t_min, t_max), orientation='horizontal'))
        self._time_plot.addItem(self._time_window)
        self._time_plot.autoRange()
