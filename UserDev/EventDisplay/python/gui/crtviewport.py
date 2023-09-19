#!/usr/bin/env python
import math

from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

import ROOT
from ROOT import vector, string
from ROOT import set as ROOTset

# set up gallery to access some larcoreobj types
import galleryUtils
sc = galleryUtils.SourceCode
sc.loadHeaderFromUPS('larcorealg/Geometry/AuxDetSensitiveGeo.h')


class CrtViewport(QtWidgets.QWidget):
    ''' Widget holding CRT PyQtGraph window and CRT controls '''
    def __init__(self, geometry, plane=-1):
        super(CrtViewport, self).__init__()

        self._geometry = geometry


        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self._view_widget = CrtViewWidget(self._geometry)
        self._layout.addWidget(self._view_widget)
        self._layout.addStretch(1)

    def drawCrtData(self, data):
        if data is not None:
            self._view_widget.drawCrtData(data)

    def getWidget(self):
        return self, self._layout


class CrtViewWidget(pg.GraphicsLayoutWidget):
    ''' PyQtGraph widget to draw CRT data '''
    def __init__(self, geometry, plane=-1):
        super(CrtViewWidget, self).__init__(border=None)

        self._geometry = geometry

        self._data = None

        self._plot = pg.PlotItem(name="CRTPlot")
        # TODO axis labels
        self._plot.setLabel(axis='left', text='')
        self._plot.setLabel(axis='bottom', text='')
        self.addItem(self._plot)

        self._crt_strips = {}
        self._crt_strip_bounds_map = {}
        self._drawn_crt_modules = []
        self._drawn_crt_hits = set()

        self._init_crt_strips()
        self.init_geometry()

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

        left = coord[0] < -380.0
        right = coord[0] > 381.3
        if left or right:
            # left or right side: z along xdir, y along ydir (side view)
            x = coord[2]
            y = coord[1]
            if coord[0] < 0: 
                return x, -y - 800
            return x, y + 800
        print('invalid point', coord)


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

    def draw_crt_hit(self, module, sipm, adc):
        ''' draw a hit CRT strip '''

        # skip low adc strips
        idx = int(min(255, adc / 10))
        if idx < 50:
            return

        strip = self._crt_strip(module, sipm)
        pt_min, pt_max = self._crt_strip_world_bounds(strip)
        draw_min = self._crt_draw_pos(pt_min)
        draw_max = self._crt_draw_pos(pt_max)
        draw_min_sort = np.array([min(draw_min[i], draw_max[i]) for i in range(2)])
        draw_max_sort = np.array([max(draw_min[i], draw_max[i]) for i in range(2)])
        w = draw_max_sort[0] - draw_min_sort[0]
        h = draw_max_sort[1] - draw_min_sort[1]

        # rect = RectItem(QtCore., fc=color, lc=color)
        rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(draw_min_sort[0], draw_min_sort[1], w, h))
        rect.setBrush(QtGui.QColor(idx, 0, 0))
        self._plot.addItem(rect)
        self._drawn_crt_hits.add(rect)

    def drawCrtData(self, adc_array):
        ''' draw all hit strips '''
        self.clear_crt_hits()

        for mod in range(len(adc_array)):
            for sipm in range(len(adc_array[mod,:])):
                adc = adc_array[mod, sipm]
                if adc <= 0:
                    continue
                
                self.draw_crt_hit(mod, sipm, adc)
        self._plot.update()

    def clear_crt_hits(self):
        for strip in self._drawn_crt_hits:
            self._plot.removeItem(strip)
        self._drawn_crt_hits = set()


if __name__ == '__main__':
    # TODO a basic test to initialize the CRT view
    import SBNDservices as services
    from evdmanager import geometry
    detClocks = services.ServiceManager('DetectorClocks')
    detProperties = services.ServiceManager('DetectorProperties')
    geometryCore = services.ServiceManager('Geometry')
    lar_properties = services.ServiceManager('LArProperties')
    geom = geometry.sbnd(geometryCore, detProperties, detClocks, lar_properties)
    cgv = CrtViewWidget(geom)
    print(cgv._crt_plane_world_bounds())
