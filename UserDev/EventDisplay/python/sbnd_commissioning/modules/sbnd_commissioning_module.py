#/usr/bin/env python3

"""
This module is shows SBND TPC, optical, and CRT information in the same view.
It utilizes components from the base TITUS modules
"""

import numpy as np
import pyqtgraph as pg 
from PyQt5 import QtWidgets, QtGui, QtCore
# these functions are used to make tracks artificially thicker
from scipy.ndimage import grey_dilation, generate_binary_structure

from titus.modules import Module
from titus.modules import TpcModule, WireView, CrtHitsItem
import titus.drawables as drawables
from titus.gui.optical_elements import _bordercol_
from titus.gui.widgets import MultiSelectionBox, recoBox

import ROOT
from ROOT import vector, string
from ROOT import set as ROOTset
# set up gallery to access some larcoreobj types
import galleryUtils
sc = galleryUtils.SourceCode
sc.loadHeaderFromUPS('larcorealg/Geometry/AuxDetSensitiveGeo.h')


# _RAW_RAWDIGIT = 'raw::RawDigit'
_RAW_RAWDIGIT = 'recob::Wire'
_RAW_OPDETWAVEFORM = 'raw::OpDetWaveform'
_SBND_CRT_FEBDATA = 'sbnd::crt::FEBData'

_OPDET_COLORMAP = pg.colormap.get('CET-L4')
# _OPDET_COLORMAP.reverse()

class SBNDCommissioningModule(Module):
    def __init__(self, larsoft_module, geom_module):
        super().__init__()
        self._lsm = larsoft_module
        self._gm = geom_module

        self._central_widget = QtWidgets.QWidget()
        self._layout = QtWidgets.QHBoxLayout()
        self._central_widget.setLayout(self._layout)
        
        # tpc-related
        self._wire_drawer = None
        self._wire_views = {}

        # optical-related
        self._opdet_waveform_drawer = None

        # crt-related
        self._crt_drawer = None

        self._draw_dock =  QtWidgets.QDockWidget('Draw Controls', self._gui, objectName='sbnd_comm_dock_draw')
        self._draw_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._dock_widgets = set([self._draw_dock])

    def _initialize(self):
        # main TPC view widget with multiple WireViews and waveform view
        self._gui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._draw_dock)

        self._gm.geometryChanged.connect(self.init_views)
        self._gm.geometryChanged.connect(self.init_controls)

        # TODO temporary to trigger wire drawing
        self._lsm.stageChanged.connect(self.draw_wires)
        self._lsm.stageChanged.connect(self.draw_opdet_waveforms)
        self._lsm.stageChanged.connect(self.draw_crts)

    def init_views(self):
        """ Set up wire views and wire drawers """
        self._wire_views = {}
        if self._gm.current_geom is None:
            return

        # for p in range(self._gm.current_geom.nPlanes()):
        view = XZDetectorView(self._gm.current_geom, 2, 0)
        view.connectStatusBar(self._gui.statusBar())
        self._wire_views[(2, 0)] = view
        self._layout.addWidget(view.getWidgetAndLayout()[0])
        view.setVisible(True)

    def init_controls(self):
        """ set up controls for enabling/disabling subsystem drawing """
        frame = QtWidgets.QWidget(self._draw_dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._draw_dock.setWidget(frame)

        sub_group_box = QtWidgets.QGroupBox('Subsystems')

        # wires
        self._wire_checkbox = QtWidgets.QCheckBox('Wires')
        self._wire_checkbox.clicked.connect(self.draw_wires)
        products = self._gi.get_products(_RAW_RAWDIGIT)
        default_products = self._gi.get_default_products(_RAW_RAWDIGIT)
        self._wire_choice = MultiSelectionBox(self, _RAW_RAWDIGIT, products, default_products)
        self._wire_choice.activated.connect(self.draw_wires)

        # opdets
        self._opdet_checkbox = QtWidgets.QCheckBox('OpDets')
        self._opdet_checkbox.toggle()
        self._opdet_checkbox.clicked.connect(self.draw_opdet_waveforms)
        products = self._gi.get_products(_RAW_OPDETWAVEFORM)
        default_products = self._gi.get_default_products(_RAW_OPDETWAVEFORM)
        self._opdet_choice = MultiSelectionBox(self, _RAW_OPDETWAVEFORM, products, default_products)
        self._opdet_choice.activated.connect(self.draw_opdet_waveforms)

        # CRT
        self._crt_checkbox = QtWidgets.QCheckBox('CRT')
        self._crt_checkbox.clicked.connect(self.draw_crts)
        products = self._gi.get_products(_SBND_CRT_FEBDATA)
        default_products = self._gi.get_default_products(_SBND_CRT_FEBDATA)
        self._crt_choice = MultiSelectionBox(self, _SBND_CRT_FEBDATA, products, default_products)
        self._crt_choice.activated.connect(self.draw_crts)

        choice_layout = QtWidgets.QGridLayout()
        choice_layout.addWidget(self._wire_checkbox, 0, 0, 1, 1)
        choice_layout.addWidget(self._opdet_checkbox, 1, 0, 1, 1)
        choice_layout.addWidget(self._crt_checkbox, 2, 0, 1, 1)

        choice_layout.addWidget(self._wire_choice, 0, 1, 1, 1)
        choice_layout.addWidget(self._opdet_choice, 1, 1, 1, 1)
        choice_layout.addWidget(self._crt_choice, 2, 1, 1, 1)
        self._wire_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self._opdet_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self._crt_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        sub_group_box.setLayout(choice_layout)
        main_layout.addWidget(sub_group_box)

        self._dilation_checkbox = QtWidgets.QCheckBox('Wire dilation')
        self._dilation_checkbox.clicked.connect(self.set_dilation)
        main_layout.addWidget(self._dilation_checkbox)

        self._bottom_crt_checkbox = QtWidgets.QCheckBox('Bottom CRTs')
        self._bottom_crt_checkbox.clicked.connect(self.show_bottom_crts)
        main_layout.addWidget(self._bottom_crt_checkbox)

        main_layout.addStretch()

    def set_dilation(self):
        for plane_cryo, view in self._wire_views.items():
            view.dilate(self._dilation_checkbox.isChecked())
            if self._wire_drawer is not None:
                view.drawPlane(self._wire_drawer.getPlane(*plane_cryo))
                view.uniteCathodes(True)

    def show_bottom_crts(self):
        for plane_cryo, view in self._wire_views.items():
            view.draw_bottom_crts(self._bottom_crt_checkbox.isChecked())
            view.draw_crt_planes(self._crt_checkbox.isChecked())
            
            if self._crt_drawer is None:
                return
           
            if self._crt_checkbox.isChecked():
                view.drawCrts(self._crt_drawer.getData())


    def draw_wires(self):
        if not self._wire_checkbox.isChecked():
            self.remove_drawable(self._wire_drawer)
            self._wire_drawer = None
            for plane_cryo, view in self._wire_views.items():
                view.drawBlank()
            return

        all_producers = self._gi.get_producers(_RAW_RAWDIGIT, stage=self._lsm.current_stage)
        if all_producers is None:
            self.remove_drawable(self._wire_drawer)
            self._wire_drawer = None
            return

        if self._gm.current_geom is None:
            self.remove_drawable(self._wire_drawer)
            self._wire_drawer = None
            return

        if self._wire_drawer is None:
            self._wire_drawer = self.register_drawable(
                # drawables.RawDigit(self._gi, self._gm.current_geom)
                drawables.RecoWire(self._gi, self._gm.current_geom)
            )
        producer = self._wire_choice.selected_products()[0]
        self._wire_drawer.set_producer(producer)
        self._gi.process_event(True)

    def draw_opdet_waveforms(self):
        if not self._opdet_checkbox.isChecked():
            self.remove_drawable(self._opdet_waveform_drawer)
            self._opdet_waveform_drawer = None
            self._wire_views[(2, 0)].draw_opdets(False)
            return
        
        self._wire_views[(2, 0)].draw_opdets()
        all_producers = self._gi.get_producers(_RAW_OPDETWAVEFORM, stage=self._lsm.current_stage)
        if all_producers is None:
            print(f"Warning: No {_RAW_OPDETWAVEFORM} data to draw")
            self.remove_drawable(self._opdet_waveform_drawer)
            self._opdet_waveform_drawer = None
            return

        if self._gm.current_geom is None:
            self._opdet_waveform_drawer = None
            self.remove_drawable(self._opdet_waveform_drawer)
            return

        if self._opdet_waveform_drawer is None:
            self._opdet_waveform_drawer = self.register_drawable(
                drawables.OpDetWaveform(self._gi, self._gm.current_geom)
            )
        producer = self._opdet_choice.selected_products()[0]
        self._opdet_waveform_drawer.set_producer(producer)
        self._gi.process_event(True)

    def draw_crts(self):
        view = self._wire_views[(2, 0)]
        if not self._crt_checkbox.isChecked():
            view.clear_crt_hits()
            self.remove_drawable(self._crt_drawer)
            self._crt_drawer = None
            view.draw_crt_planes(False)
            return
        
        view.draw_crt_planes()
        all_producers = self._gi.get_producers(_SBND_CRT_FEBDATA, stage=self._lsm.current_stage)
        if all_producers is None:
            print(f"Warning: No {_SBND_CRT_FEBDATA} data to draw")
            view.clear_crt_hits()
            self.remove_drawable(self._crt_drawer)
            self._crt_drawer = None
            return

        if self._gm.current_geom is None:
            view.clear_crt_hits()
            self.remove_drawable(self._crt_drawer)
            self._crt_drawer = None
            return

        if self._crt_drawer is None:
            self._crt_drawer = self.register_drawable(
                drawables.FEBData(self._gi, self._gm.current_geom)
            )
        producer = self._crt_choice.selected_products()[0]
        self._crt_drawer.set_producer(producer)
        self._gi.process_event(True)

    def update(self):
        if self._wire_drawer is not None:
            for plane_cryo, view in self._wire_views.items():
                view.drawPlane(self._wire_drawer.getPlane(*plane_cryo))
                # TODO there must be a better way
                view.uniteCathodes(True)

        # TODO remove hard-coded cryo/plane
        view = self._wire_views[(2, 0)]
        if self._opdet_waveform_drawer is not None:
            view.drawOpdetWaveforms(self._opdet_waveform_drawer.getData())

        if self._crt_drawer is not None:
            view.drawCrts(self._crt_drawer.getData())


class XZDetectorView(WireView):
    """
    Class for drawing TPC, optical, and CRT systems in XZ view. Inherits from
    WireView from TpcModule, then adds some extra elements from optical and CRT
    systems. We do some tricks with coordinate transformations to make sure all
    systems align with wire coordinates including a cathode gap
    """

    def __init__(self, geometry, plane=-1, cryostat=0, tpc=0):
        super().__init__(geometry, plane, cryostat, tpc)
        self._optical_elements = []
        self._optical_names = ['pmt_coated', 'pmt_uncoated',
                               'arapuca_vuv', 'arapuca_vis',
                               'xarapuca_vuv', 'xarapuca_vis']
        
        self._opdet_circles = {}
        self._opdet_size = 10 # cm
        self._draw_opdets = True
        self.draw_opdets()

        self._crt_strips = {}
        self._crt_strip_bounds_map = {}
        self._drawn_crt_modules = {}
        self._drawn_crt_hits = None
        self._crt_hit_picture_cache = {}

        # global on/off switch for CRTs and on/off switches for each CRT region
        self._draw_crts = True
        self._draw_bot_crts = False
        self._draw_wall_crts = True

        self._init_crt_strips()
        self.init_crt_planes()

        self._view.setAspectLocked(1)
        # self.setColorMap('grayscale')
        self._disable_wrapper = True

        # thicker tracks: useful for nice images when zoomed out to physical scale
        self._dilate = False
        self._dilation_struct = np.ones(shape=(5, 5)) #generate_binary_structure(2, 2)

    def drawPlane(self, image):
        if self._dilate:
            image = grey_dilation(image, structure=self._dilation_struct)

        self._item.setImage(image,autoLevels=False)
        self._item.setLookupTable(self._cmap.getLookupTable(255))
        self.setWrapperVisible(True)
        # Make sure the levels are actually set:
        self.levelChanged()

        if self._geometry.nTPCs() == 2:
            self.drawTPCdivision()

    def mouseClicked(self, event):
        # for now, do nothing
        pass

    def dilate(self, d):
        self._dilate = d

    def draw_bottom_crts(self, d):
        if d != self._draw_bot_crts:
            self._clear_cache()
        self._draw_bot_crts = d
        self.draw_crt_planes(self._draw_crts)

    def uniteCathodes(self, uniteC):
        """ Override to remove anode and cathode parts of the waveform """
        self._uniteCathodes = uniteC
        if not self._uniteCathodes:
            self._removed_entries = 0
            self.drawPlane(self._original_image)
            return

        data = self._item.image
        self._original_image = np.copy(data)

        offset = self._geometry.offset(self._plane)
        t_cathode = (2 * self._geometry.halfwidth() + offset) / self._geometry.time2cm()
        t_anode = offset / self._geometry.time2cm()
        porch = self._geometry.tRange() - t_cathode
        n_removed_entries = 0

        # remove from right TPC anode
        slice_anode_right = slice(0, int(t_anode))
        n_removed_entries += int(t_anode)

        # remove between right and left TPC cathodes
        start_removal = t_cathode
        end_removal = start_removal + 2.0 * porch + self._geometry.cathodeGap()
        slice_cathode = slice(int(start_removal), int(end_removal))
        n_removed_entries += int(end_removal) - int(start_removal)

        # remove from left TPC anode
        start_removal = end_removal + (2.0 * self._geometry.halfwidth()) / self._geometry.time2cm()
        end_removal = 2.0 * self._geometry.tRange() + self._geometry.cathodeGap() - t_anode
        slice_anode_left = slice(int(start_removal), int(end_removal))
        n_removed_entries += int(end_removal) - int(start_removal)

        final_slice = np.r_[slice_cathode, slice_anode_left, slice_anode_right]

        data = np.delete(data, final_slice, axis=1)
        self.drawPlane(data)
        self._removed_entries = n_removed_entries

        # scale the base wire view image into world coordinates
        # this is done by placing the anode/cathode times at the physical
        # anode/cathode positions (unit cm)
        tr = QtGui.QTransform()

        # vertical scaling is done by the number of ticks after uniting the cathodes
        # TODO not sure why this 1.08 fudge factor is needed. Probably a bug in the above slicing!
        tr.scale(self._geometry.wire2cm(), self._geometry.time2cm())

        # translation is in pre-scaled (time tick) units
        ticks = 2.0 * self._geometry.tRange() + self._geometry.cathodeGap() - self._removed_entries
        tr.translate(0, -ticks / 2.)

        self._item.setTransform(tr)


    def draw_opdets(self, draw=True):
        if not self._draw_opdets or not draw:
            for _, info in self._opdet_circles.items():
                self._view.removeItem(info['item'])
                del info['item']
                # self.scene().removeItem(info['item'])
            return

        opdets_x, opdets_y, opdets_z = self._geometry.opdetLoc()
        opdets_name = self._geometry.opdetName()
        
        # only draw one opdet per y coordinate within some small radius
        # we save a list of each opdet skipped so that we can sum the waveforms
        # later
        used_opdets = {}
        for d in range(len(opdets_x)):
            if opdets_name[d] not in self._optical_names:
                continue


            opdet_draw_x = opdets_z[d] - self._opdet_size / 2.
            opdet_draw_y = opdets_x[d] - self._opdet_size / 2.
            
            opdet_draw_pos = (opdet_draw_x, opdet_draw_y)
            if opdet_draw_pos in used_opdets.keys():
                idx = used_opdets[opdet_draw_pos]
                if self._opdet_circles[idx]["name"] == opdets_name[d]:
                    self._opdet_circles[idx]["others"].append(d)
                continue
            
            used_opdets[opdet_draw_pos] = d
            self._opdet_circles[d] = {"name": opdets_name[d], "item": None, "others": []}

            brush = (0, 0, 0, 0)

            ellipse = QtWidgets.QGraphicsEllipseItem(*opdet_draw_pos,
                self._opdet_size, self._opdet_size)
            ellipse.setPen(pg.mkPen((0, 0, 0, 100)))
            ellipse.setBrush(pg.mkBrush((int(255 * np.random.uniform()), 50, 50)))
            self._view.addItem(ellipse)
            self._opdet_circles[d]["item"] = ellipse

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
                bounds = self._crt_strip_world_bounds(strip)
                # # here we cut non-vertically-oriented strips
                # if np.abs(bounds[0][1] - bounds[1][1]) < 20.0:
                #     continue

                self._crt_strips[(local_mac, sv_i)] = strip
                self._crt_strip_bounds_map[strip] = bounds

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
            
            bounds = [to_world_coord(limits_min), to_world_coord(limits_max)]
            bottom = bounds[0][1] <= self._geometry.crt_bot_ymax
            # here we cut non-vertically-oriented planes
            if np.abs(bounds[0][1] - bounds[1][1]) < 2.0 and not bottom:
                continue

            result[plane_number] = bounds
            plane_number += 1

        return result

    def draw_crt_planes(self, show=True):
        self._draw_crts = show
        for region, plane_list in self._drawn_crt_modules.items():
            for item in plane_list:
                if not show or (region == 'bottom' and not self._draw_bot_crts):
                    if item.scene() is None:
                        continue
                    self._view.removeItem(item)
                    continue
                self._view.addItem(item)

    def init_crt_planes(self):
        ''' draw the permanent CRT module outlines '''
        self._drawn_crt_modules = {'bottom': [], 'side': []}
        plane_bounds = self._crt_plane_world_bounds()
        for plane, bounds in plane_bounds.items():
            pt_min, pt_max = bounds
            bottom = False
            if pt_max[1] <= self._geometry.crt_bot_ymax:
                bottom = True

            # use x = Z, y = X coordinates for top-down view
            draw_min = np.array([pt_min[2], pt_min[0]])
            draw_max = np.array([pt_max[2], pt_max[0]])

            # positive widths and heights only!
            draw_min_sort = np.array([min(draw_min[i], draw_max[i]) for i in range(2)])
            draw_max_sort = np.array([max(draw_min[i], draw_max[i]) for i in range(2)])
            w = draw_max_sort[0] - draw_min_sort[0]
            h = draw_max_sort[1] - draw_min_sort[1]

            rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(draw_min_sort[0], draw_min_sort[1], w, h))
            rect.setPen(QtGui.QColor(0, 0, 0))
            key = 'side' if not bottom else 'bottom'
            self._drawn_crt_modules[key].append(rect)

    def drawOpdetWaveforms(self, data):
        """ sums adc counts for opdets """
        if data is None:
            return

        if data.size == 0:
            return

        item_map = {}
        for idx, info in self._opdet_circles.items():
            total_adc = 0
            # flag to indicate there was at least one filled waveform for this group
            filled = False
            for i in [idx] + info["others"]:
                wfm = data[i,:]
                if len(wfm) == 0:
                    continue
                if wfm[0] == self._geometry.opdetDefaultValue():
                    continue
                filled = True
                total_adc += (wfm.max() - wfm.min()) / (len(info["others"]) + 1.)

            if filled:
                item_map[idx] = total_adc
            else:
                self._opdet_circles[idx]["item"].setBrush(_OPDET_COLORMAP.mapToQColor(0))

        min_val = np.min(list(item_map.values()))
        max_val = np.max(list(item_map.values()))
        val_range = max_val - min_val
        for idx, adc in item_map.items():
            color = _OPDET_COLORMAP.mapToQColor((adc - min_val) / val_range)
            self._opdet_circles[idx]["item"].setBrush(color)

    def drawCrts(self, hit_array):
        if hit_array is None:
            return

        if hit_array.size == 0:
            return

        self.clear_crt_hits()
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
                if adc <= 0:
                    continue

                strip = self._crt_strip(mod, sipm)
                if strip is None:
                    continue

                bounds = self._crt_strip_world_bounds(strip)
                bottom = bounds[0][1] <= self._geometry.crt_bot_ymax
                if bottom and not self._draw_bot_crts:
                    continue

                # here we cut non-vertically-oriented strips
                if np.abs(bounds[0][1] - bounds[1][1]) < 20.0 and not bottom:
                    continue

                pt_min, pt_max = self._crt_strip_world_bounds(strip)
                draw_min = np.array([pt_min[2], pt_min[0]])
                draw_max = np.array([pt_max[2], pt_max[0]])

                # order the points so that draw_min_sort is < draw_max_sort in
                # both dimensions
                draw_min_sort = np.array([min(draw_min[i], draw_max[i]) for i in range(2)])
                draw_max_sort = np.array([max(draw_min[i], draw_max[i]) for i in range(2)])

                # extend draw_min and draw_max so that the rectangles are not too thin
                if not bottom:
                    back = draw_min_sort[0] < self._geometry.crt_back_zmax
                    front = draw_max_sort[0] > self._geometry.crt_front_zmin
                    left = draw_min_sort[1] > -381.3
                    right = draw_min_sort[1] < 380.0
                    scale = 60
                    if back:
                        draw_min_sort[0] -= scale
                    elif front:
                        draw_max_sort[0] += scale
                    elif left:
                        draw_max_sort[1] += scale
                    elif right:
                        draw_min_sort[1] -= scale

                draw_coords.append((draw_min_sort, draw_max_sort, time))
            
            picture.add_hits(draw_coords)
            self._crt_hit_picture_cache[key] = picture
        
        self._view.addItem(self._crt_hit_picture_cache[key])
        self._drawn_crt_hits = self._crt_hit_picture_cache[key]
        self._item.update()
    
    def clear_crt_hits(self):
        ''' removes the currently-drawn CrtHitsItem '''
        if self._drawn_crt_hits is not None:
            self._view.removeItem(self._drawn_crt_hits)
        self._drawn_crt_hits = None

    def _clear_cache(self):
        ''' removes all cached CrtHitsItems from this object '''
        self.clear_crt_hits()
        self._crt_hit_picture_cache = {}
