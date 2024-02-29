#/usr/bin/env python3

"""
This module adds the wire plane views and associated controls. Uses the
geometry module to get the currently-selected geometry
"""
import math

import numpy as np
import pyqtgraph as pg 
from pyqtgraph import ViewBox, Point
from PyQt5 import QtWidgets, QtGui, QtCore
import PIL

from titus.modules import Module
from titus.gui.widgets import MultiSelectionBox, recoBox, VerticalLabel, MovablePixmapItem, MovableScaleBar
import titus.drawables as drawables


# place any drawables associated with TPC view here
_DRAWABLE_LIST = {
    'Hit': [drawables.Hit, "recob::Hit"],
    'Cluster': [drawables.Cluster, "recob::Cluster"],
    'Shower': [drawables.Shower, "recob::Shower"],
    'Track': [drawables.Track, "recob::Track"],
    'MCTruth': [drawables.MCTruth, "simb::MCTruth"],
    'MCTrack': [drawables.MCTrack, "sim::MCTrack"],
    'End Point 2D': [drawables.EndPoint2D, "recob::EndPoint2D"],
    'Vertex': [drawables.Vertex, "recob::Vertex"],
    'Space Point': [drawables.SpacePoint, "recob::SpacePoint"],
}


_RECOB_WIRE = 'recob::Wire'
_RAW_RAWDIGIT = 'raw::RawDigit'

class TpcModule(Module):
    def __init__(self, larsoft_module, geom_module):
        super().__init__()
        self._lsm = larsoft_module
        self._gm = geom_module

        self._central_widget = QtWidgets.QSplitter()
        self._central_widget.setOrientation(QtCore.Qt.Vertical)
        self._central_widget.setChildrenCollapsible(False)

        self._draw_dock =  QtWidgets.QDockWidget('TPC Draw Controls', self._gui)
        self._draw_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._view_dock =  QtWidgets.QDockWidget('TPC View Controls', self._gui)
        self._view_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._waveform_dock =  QtWidgets.QDockWidget('Wire Waveform', self._gui)
        self._waveform_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea \
                                            | QtCore.Qt.LeftDockWidgetArea \
                                            | QtCore.Qt.RightDockWidgetArea)

        # don't add waveform dock to this list since its visibility is set by
        # the controls in this gui
        self._dock_widgets = set([self._draw_dock, self._view_dock])

        self._draw_wires = False
        self._wire_drawer = None
        self._product_box_map = {}
        self._wire_views = {}
        self._show_logo = False
        self._show_scale_bar = False
        self._selected_planes = [-1]
        self._selected_cryos = [-1]

    def _initialize(self):
        self._gui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._draw_dock)
        self._gui.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._waveform_dock)
        self._gui.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._view_dock)

        self.init_wire_waveform()

        # main TPC view widget with multiple WireViews and waveform view
        # init once the geometry is selected so that None geometry doesn't do
        # anything
        self._gm.geometryChanged.connect(self.init_tpc_controls)
        self._gm.geometryChanged.connect(self.init_tpc_views)

    def init_tpc_views(self):
        # TODO remove wire views from central widget if they exist (in the case
        # of changing geometry at runtime)
        # self._wire_views = {}
        if self._gm.current_geom is None:
            return

        for c in range(self._gm.current_geom.nCryos()):
            for p in range(self._gm.current_geom.nPlanes()):
                view = WireView(self._gm.current_geom, p, c)
                view.connectWireDrawingFunction(self.drawWireOnPlot)
                view.connectStatusBar(self._gui.statusBar())
                self._wire_views[(p, c)] = view
                self._central_widget.addWidget(view.getWidgetAndLayout()[0])

        self.refresh_draw_list_widget()

    def init_tpc_controls(self):
        # panel with buttons to draw objects on TPC view
        frame = QtWidgets.QWidget(self._draw_dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._draw_dock.setWidget(frame)

        # None/Wire/RawDigit Radio Buttons
        wire_group_box = QtWidgets.QGroupBox('Wire Drawing')
        # None has no drop down
        wire_button_group = QtWidgets.QButtonGroup()

        self._none_wire_button = QtWidgets.QRadioButton("None")
        self._none_wire_button.clicked.connect(self.change_wire_choice)
        wire_button_group.addButton(self._none_wire_button)

        self._wire_button = QtWidgets.QRadioButton("Wire")
        self._wire_button.clicked.connect(self.change_wire_choice)
        wire_button_group.addButton(self._wire_button)

        products = self._gi.get_products(_RECOB_WIRE)
        default_products = self._gi.get_default_products(_RECOB_WIRE)
        self._wire_choice = MultiSelectionBox(_RECOB_WIRE, products, default_products)
        self._wire_choice.activated.connect(self.change_wire_choice)

        # Draw Raw Digit
        self._raw_digit_button = QtWidgets.QRadioButton("Raw Digit")
        self._raw_digit_button.clicked.connect(self.change_wire_choice)
        wire_button_group.addButton(self._raw_digit_button)
        products = self._gi.get_products(_RAW_RAWDIGIT)
        default_products = self._gi.get_default_products(_RAW_RAWDIGIT)
        self._raw_digit_choice = MultiSelectionBox(self, _RAW_RAWDIGIT, products, default_products)
        self._raw_digit_choice.activated.connect(self.change_wire_choice)
        raw_digit_layout = QtWidgets.QHBoxLayout()

        wire_choice_layout = QtWidgets.QGridLayout()
        wire_choice_layout.addWidget(self._none_wire_button, 0, 0, 1, 1)
        wire_choice_layout.addWidget(self._wire_button, 1, 0, 1, 1)
        wire_choice_layout.addWidget(self._wire_choice, 1, 1, 1, 1)
        self._wire_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        wire_choice_layout.addWidget(self._raw_digit_button, 2, 0, 1, 1)
        wire_choice_layout.addWidget(self._raw_digit_choice, 2, 1, 1, 1)
        self._raw_digit_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        wire_group_box.setLayout(wire_choice_layout)
        main_layout.addWidget(wire_group_box)

        # Set the default to be no wires
        self._none_wire_button.toggle()

        # Microboone only:
        # TODO re-enable this & add to geometry
        # if self._geometry.name() == "uboone":
        #     self._noiseFilterBox = QtWidgets.QCheckBox("Noise Filter")
        #     self._noiseFilterBox.stateChanged.connect(self.noiseFilterWorker)

        # Now we get the list of items that are drawable:
        product_group_box = QtWidgets.QGroupBox("Products")
        drawable_layout = QtWidgets.QFormLayout()
        drawable_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        for product_name, product_info in _DRAWABLE_LIST.items():
            prod_class, product = product_info
            producers = self._gi.get_producers(product, self._lsm.current_stage)
            thisBox = recoBox(self, product_name, product, producers)
            self._product_box_map[thisBox] = self.register_drawable(
                prod_class(self._gi, self._gm.current_geom, self)
            )

            thisBox.activated[str].connect(self.reco_box_handler)
            thisBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            drawable_layout.addRow(product_name, thisBox)

        product_group_box.setLayout(drawable_layout)
        main_layout.addWidget(product_group_box)


        # these are the view options
        frame2 = QtWidgets.QWidget(self._view_dock)
        main_layout2 = QtWidgets.QVBoxLayout()
        frame2.setLayout(main_layout2)
        self._view_dock.setWidget(frame2)

        planes_group_box = QtWidgets.QGroupBox("Planes")
        plane_layout = QtWidgets.QGridLayout()
        self._all_views_button = QtWidgets.QPushButton("All Views")
        self._all_views_button.setCheckable(True)
        self._all_views_button.clicked.connect(self.select_views)
        plane_layout.addWidget(self._all_views_button, 0, 0, 1, 4)

        viewport_names = self._gm.current_geom.viewNames()
        self._view_button_array = []
        self._view_labels = []
        for i, v in enumerate(viewport_names):
            label = QtWidgets.QLabel(f"View {v}")
            plane_layout.addWidget(label, i + 1, 1)
            self._view_labels.append(label)
            for c in range(0, self._gm.current_geom.nCryos()):
                text = f"Cryo {c}"
                if self._gm.current_geom.name() == 'sbnd':
                    text = v
                button = QtWidgets.QPushButton(text)
                button.setToolTip(f"Visualize view {v} in cryostat {c}.")
                button.clicked.connect(self.select_views)
                button.setCheckable(True)
                self._view_button_array.append(button)
                plane_layout.addWidget(button, i + 1, c + 2)
        planes_group_box.setLayout(plane_layout)

        # start GUI with all views enabled
        self._all_views_button.click()

        options_group_box = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout()
        self._grayScale = QtWidgets.QCheckBox("Grayscale")
        self._grayScale.setToolTip("Changes the color map to grayscale.")
        self._grayScale.setTristate(False)
        # self._grayScale.stateChanged.connect(self.changeColorMapWorker)

        # Button to set range to max
        self._maxRangeButton = QtWidgets.QPushButton("Max Range")
        self._maxRangeButton.setToolTip("Set the range of the viewers to show the whole event")
        # self._maxRangeButton.clicked.connect(self._view_manager.setRangeToMax)

        # Check box to active autorange
        self._autoRangeBox = QtWidgets.QCheckBox("AutoRange")
        self._autoRangeBox.setToolTip("Set the range of the viewers to the regions of interest")
        self._autoRangeBox.setTristate(False)
        # self._autoRangeBox.stateChanged.connect(self.autoRangeWorker)

        self._lockAspectRatio = QtWidgets.QCheckBox("1:1 Aspect Ratio")
        self._lockAspectRatio.setToolTip("Lock the aspect ratio to 1:1")
        self._lockAspectRatio.stateChanged.connect(self.lock_aspect_ratio)

        self._rangeLayout = QtWidgets.QVBoxLayout()
        self._rangeLayout.addWidget(self._autoRangeBox)
        self._rangeLayout.addWidget(self._lockAspectRatio)

        # check box to toggle the wire drawing
        self._drawWireOption = QtWidgets.QCheckBox("Wire Drawing")
        self._drawWireOption.setToolTip("Draw the wires when clicked on")
        self._drawWireOption.stateChanged.connect(self.draw_wire_waveform)

        self._drawRawOption = QtWidgets.QCheckBox("Draw Raw")
        self._drawRawOption.setToolTip("Draw the raw wire signals in 2D")
        self._drawRawOption.setTristate(False)

        self._subtractPedestal = QtWidgets.QCheckBox("Subtract Pedestal")
        self._subtractPedestal.setToolTip("Subtracts the pedestal from RawDigits. You will need to adjust the range.")
        self._subtractPedestal.setTristate(False)
        self._subtractPedestal.setCheckState(QtCore.Qt.Checked)
        # self._subtractPedestal.stateChanged.connect(self.subtractPedestalWorker)

        # add a box to restore the drawing defaults:
        self._restoreDefaults = QtWidgets.QPushButton("Restore Defaults")
        self._restoreDefaults.setToolTip("Restore the drawing defaults of the views.")
        # self._restoreDefaults.clicked.connect(self.restoreDefaultsWorker)

        self._unitDisplayOption = QtWidgets.QCheckBox("Use cm")
        self._unitDisplayOption.setToolTip("Display the units in cm (checked = true)")
        self._unitDisplayOption.setTristate(False)
        self._unitDisplayOption.stateChanged.connect(self.use_cm)

        self._scaleBarOption = QtWidgets.QCheckBox("Scale bar")
        self._scaleBarOption.setToolTip("Display a scale bar on each view showing the distance")
        self._scaleBarOption.setTristate(False)
        self._scaleBarOption.stateChanged.connect(self.show_scale_bar)

        self._scaleBarLayout = QtWidgets.QVBoxLayout()
        self._scaleBarLayout.addWidget(self._scaleBarOption)
        self._scaleBarLayout.addWidget(self._unitDisplayOption)

        self._logoOption = QtWidgets.QCheckBox("Draw Logo")
        self._logoOption.setToolTip("Display the experiment logo on the window.")
        self._logoOption.setTristate(False)
        self._logoOption.stateChanged.connect(self.draw_logo)

        self._clearPointsButton = QtWidgets.QPushButton("Clear Points")
        self._clearPointsButton.setToolTip("Clear all of the drawn points from the views")
        # self._clearPointsButton.clicked.connect(self.clearPointsWorker)

        self._makePathButton = QtWidgets.QPushButton("Eval. Points")
        self._makePathButton.setToolTip("Compute the ADCs along the path defined by the points")
        # self._makePathButton.clicked.connect(self.drawIonizationWorker)

        # Pack Clear Points and Eval Points in a horizontal layout
        self._clearEvalPointsLayout = QtWidgets.QHBoxLayout()
        self._clearEvalPointsLayout.addWidget(self._clearPointsButton)
        self._clearEvalPointsLayout.addWidget(self._makePathButton)

        self._anodeCathodeOption = QtWidgets.QCheckBox("Draw anode/cathode")
        self._anodeCathodeOption.setToolTip("Shows the anode and cathode position for t0=0.")
        self._anodeCathodeOption.setTristate(False)
        self._anodeCathodeOption.stateChanged.connect(self.show_anode_cathode)

        self._uniteCathodes = QtWidgets.QCheckBox("Unite cathodes")
        self._uniteCathodes.setToolTip("Unites the cathodes waveforms.")
        self._uniteCathodes.setTristate(False)
        # self._uniteCathodes.stateChanged.connect(self.uniteCathodesWorker)

        self._t0sliderLabelIntro = QtWidgets.QLabel("Set t<sub>0</sub>:")
        self._t0slider = QtWidgets.QSlider(0x1)
        self._t0slider.setToolTip("Change the t<sub>0</sub>.")
        if self._gm.current_geom is not None:
            self._t0slider.setMinimum(-self._gm.current_geom.triggerOffset())
            self._t0slider.setMaximum(self._gm.current_geom.triggerOffset())
        self._t0slider.setSingleStep(10)
        # self._t0slider.valueChanged.connect(self.t0sliderWorker)
        self._t0sliderLabel = QtWidgets.QLabel("Current t<sub>0</sub> = 0")

        self._t0sliderLabelIntro.setVisible(False)
        self._t0slider.setVisible(False)
        self._t0sliderLabel.setVisible(False)
        self._uniteCathodes.setVisible(False)

        self._separators = []  
        for i in range(2):
            self._separators.append(QtWidgets.QFrame())
            self._separators[i].setFrameShape(QtWidgets.QFrame.HLine)
            self._separators[i].setFrameShadow(QtWidgets.QFrame.Sunken)
            self._separators[i].setVisible(False)

        self._spliTracksOption = QtWidgets.QCheckBox("Tracks on Both TPCs")
        self._spliTracksOption.setToolTip("Split the MCTracks and Tracks so that they are projected on both TPCs.")
        self._spliTracksOption.setTristate(False)
        # self._spliTracksOption.stateChanged.connect(self.splitTracksWorker)
        self._spliTracksOption.setVisible(False)

        self._plane_frames = QtWidgets.QCheckBox("Plane frames")
        self._plane_frames.setToolTip("Show or hide the scale back frame.")
        self._plane_frames.setTristate(False)
        self._plane_frames.stateChanged.connect(self.show_plane_frames)
        self._plane_frames.setChecked(True)

        options_layout.addWidget(self._grayScale)
        options_layout.addLayout(self._rangeLayout)
        options_layout.addWidget(self._drawWireOption)
        options_layout.addWidget(self._subtractPedestal)
        options_layout.addWidget(self._separators[0])
        options_layout.addWidget(self._anodeCathodeOption)
        options_layout.addWidget(self._t0sliderLabelIntro)
        options_layout.addWidget(self._t0slider)
        options_layout.addWidget(self._t0sliderLabel)
        options_layout.addWidget(self._uniteCathodes)
        options_layout.addWidget(self._separators[1])
        options_layout.addLayout(self._scaleBarLayout)
        options_layout.addWidget(self._logoOption)
        options_layout.addWidget(self._plane_frames)
        options_layout.addWidget(self._spliTracksOption)
        options_layout.addWidget(self._maxRangeButton)
        options_layout.addLayout(self._clearEvalPointsLayout)
        options_layout.addWidget(self._restoreDefaults)
        options_group_box.setLayout(options_layout)

        main_layout2.addWidget(planes_group_box)
        main_layout2.addWidget(options_group_box)

        main_layout.addStretch()
        main_layout2.addStretch()


    def init_wire_waveform(self):
        # panel with buttons to draw objects on TPC view
        self._wire_waveform_widget = QtWidgets.QWidget(self._draw_dock)
        self._waveform_dock.setWidget(self._wire_waveform_widget)

        self._wireDrawerMain = pg.GraphicsLayoutWidget()
        self._wireDrawerMain.setBackground('w')
        self._wirePlot = self._wireDrawerMain.addPlot()
        self._wirePlotItem = pg.PlotDataItem(pen=(0,0,0))
        self._wirePlot.addItem(self._wirePlotItem)
        # self._wireDrawerMain.setMaximumHeight(250)
        self._wireDrawerMain.setMinimumHeight(100)
        self._wireDrawerMain.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, \
                                                 QtWidgets.QSizePolicy.Minimum)

        # self._wireDrawer_name = VerticalLabel("Wire Drawer")
        # self._wireDrawer_name.setMaximumWidth(25)
        # self._wireDrawer_name.setAlignment(QtCore.Qt.AlignCenter)
        # self._wireDrawer_name.setToolTip("Click on a wire to display the waveform.")
        # self._wireDrawer_name.setStyleSheet('color: rgb(169,169,169);')
        self._wireDrawerLayout = QtWidgets.QHBoxLayout()
        # self._wireDrawerLayout.addWidget(self._wireDrawer_name)
        self._wireDrawerLayout.addWidget(self._wireDrawerMain)

        self._fftButton = QtWidgets.QCheckBox("FFT Wire")
        self._fftButton.setToolTip("Compute and show the FFT of the wire currently drawn")
        self._fftButton.stateChanged.connect(self.plotFFT)

        self._left_wire_button = QtWidgets.QPushButton("Previous Wire")
        self._left_wire_button.clicked.connect(self.change_wire)
        self._left_wire_button.setToolTip("Show the previous wire.")
        self._right_wire_button = QtWidgets.QPushButton("Next Wire")
        self._right_wire_button.clicked.connect(self.change_wire)
        self._right_wire_button.setToolTip("Show the next wire.")
        self._wire_drawer_button_layout = QtWidgets.QHBoxLayout()
        self._wire_drawer_button_layout.addWidget(self._fftButton)
        self._wire_drawer_button_layout.addStretch()
        self._wire_drawer_button_layout.addWidget(self._left_wire_button)
        self._wire_drawer_button_layout.addWidget(self._right_wire_button)

        self._wireDrawerVLayout = QtWidgets.QVBoxLayout()
        self._wireDrawerVLayout.addLayout(self._wireDrawerLayout)
        self._wireDrawerVLayout.addLayout(self._wire_drawer_button_layout)
        self._wireDrawerVLayout.addStretch()
        self._wireDrawerVLayout.setContentsMargins(0, 0, 0, 0)

        self._wire_waveform_widget.setLayout(self._wireDrawerVLayout)
        # start closed
        self._waveform_dock.hide()

        # sync check box state with this dock being open
        self._waveform_dock.visibilityChanged.connect(
            lambda: self._drawWireOption.setChecked(self._waveform_dock.isVisible())
        )

    def reco_box_handler(self, text):
        sender = self.sender()
        # Get the full product obj for this:
        product = sender.productObj(text, self._lsm.current_stage)
        print('reco_box_handler', text, sender.name(), product)

        visible = not (text == "--Select--" or text == "--None--" or text == None)
        if not visible:
            self._product_box_map[sender].set_producer(None)
            self._product_box_map[sender].clearDrawnObjects()
            product = None

        if product is not None:
            self._product_box_map[sender].set_producer(product.full_name())
            self._product_box_map[sender].drawObjects()
            return

        # TODO are these next lines needed?
        self._gi.process_event()
        
        # self._gi.redrawProduct(self._gm.current_geom, sender.name(), prod, self)
        self.specialHandles(sender.name(), visible)

    def specialHandles(self, name, visibility):
        '''
        Here we handle all cases specific to 
        the product we are drawing
        '''
        if name == 'MCTrack':
            if visibility:
                self._spliTracksOption.setVisible(False)
            else:
                self._spliTracksOption.setVisible(False)
        if name == 'Track':
            if visibility:
                self._spliTracksOption.setVisible(False)
            else:
                self._spliTracksOption.setVisible(False)

    def change_wire(self):
        if self.sender() == self._left_wire_button:
            wire = max(0, self._current_wire - 1)
        else:
            wire = self._current_wire + 1

        self._current_wire_drawer.show_waveform(wire=wire, tpc=self._current_tpc)


    def change_wire_choice(self):
        if self._none_wire_button.isChecked():
            self.toggle_wires(None)
        if self._wire_button.isChecked():
            self.toggle_wires(_RECOB_WIRE, stage=self._lsm.current_stage, producers=None)
        if self._raw_digit_button.isChecked():
            self.toggle_wires(_RAW_RAWDIGIT, stage=self._lsm.current_stage, producers=None)

        self._gi.process_event(True)
        self.update()

    def update(self):
        if not self._draw_wires:
            for plane_cryo, view in self._wire_views.items():
                view.drawBlank()
            return

        for plane_cryo, view in self._wire_views.items():
            view.drawPlane(self._wire_drawer.getPlane(*plane_cryo))
            # np.save(f'{plane_cryo[0]}_{plane_cryo[1]}', self._wire_drawer.getPlane(*plane_cryo))


    def toggle_wires(self, product, stage=None, subtract_pedestal=True, producers=None):
        ''' get wire data from gallery interface '''
        def clear_wire_drawer():
            self._draw_wires = False
            self.remove_drawable(self._wire_drawer)
            self._wire_drawer = None

        if product is None:
            clear_wire_drawer()
            return

        all_producers = self._gi.get_producers(product, stage)
        if all_producers is None:
            clear_wire_drawer()
            return

        self._draw_wires = True

        if product == _RECOB_WIRE:
            if not self._wire_choice.selected_products():
                clear_wire_drawer()
                return
            self._wire_drawer = self.register_drawable(
                drawables.RecoWire(self._gi, self._gm.current_geom)
            )

            if producers is not None:
                producer = producers
            elif self._gm.current_geom.name() == 'icarus' and len(all_producers) > 3:
                producer = [p.full_name() for p in all_producers[:3]]
            else:
                producer = self._wire_choice.selected_products()[0]

            self._wire_drawer.set_producer(producer)
            # self._gi.processor.add_process(_RECOB_WIRE, self._wire_drawer._process)

        elif product == _RAW_RAWDIGIT:
            if not self._raw_digit_choice.selected_products():
                clear_wire_drawer()
                return
            self._wire_drawer = self.register_drawable(
                drawables.RawDigit(self._gi, self._gm.current_geom)
            )
            self._wire_drawer.setSubtractPedestal(subtract_pedestal)

            if producers is not None:
                producer = producers
            elif self._gm.current_geom.name() == 'icarus' and len(all_producers) > 3:
                producer = [p.full_name() for p in all_producers[:3]]
            else:
                producer = self._raw_digit_choice.selected_products()[0]

            self._wire_drawer.set_producer(producer)


    def lock_aspect_ratio(self, lockstate):
        for view in self._wire_views.values():
            view.lockRatio(lockstate)

    def draw_logo(self, logostate):
        self._show_logo = logostate
        for view in self._wire_views.values():
            view.toggleLogo(logostate)

    def show_scale_bar(self, chkstate):
        self._show_scale_bar = chkstate
        for view in self._wire_views.values():
            view.toggleScale(chkstate)
    
    def use_cm(self, chkstate):
        for view in self._wire_views.values():
            view.useCM(chkstate)

    def show_anode_cathode(self):
        show = self._anodeCathodeOption.isChecked()
        self._separators[0].setVisible(show)
        self._t0slider.setVisible(show)
        self._t0sliderLabel.setVisible(show)
        self._t0sliderLabelIntro.setVisible(show)
        self._uniteCathodes.setVisible(show)
        self._separators[1].setVisible(show)
        for view in self._wire_views.values():
            view.showAnodeCathode(show)

    def show_plane_frames(self):
        show = self._plane_frames.isChecked()
        for p, c in zip(self._selected_planes, self._selected_cryos):
            for key, view in self._wire_views.items():
                # Turn on the requested ones
                view.setWrapperVisible(show)

    def draw_wire_waveform(self):
        """
        Show the wire waveform. Since it's dock-able, add it to the list of
        docked widgets so that it is opened/closed properly
        """
        if self._drawWireOption.isChecked():
            self._waveform_dock.show()
            self._dock_widgets.add(self._waveform_dock)
        else:
            # TODO hack: activating/de-activating this module forces a
            # "stateChange" trigger but we want the checkstate to persist
            # between the module being activated/deactivated so: only remove
            # module if we are active
            if self._active:
                self._waveform_dock.hide()
                self._dock_widgets.remove(self._waveform_dock)

    def select_views(self):
        ''' set the internal state of selected_planes and selected_cryos from buttons '''
        if self.sender() == self._all_views_button:
            # Uncheck all the other buttons
            for btn in self._view_button_array:
                btn.setChecked(False)
            self._selected_planes = [-1]
            self._selected_cryos = [-1]
        else:
            self._all_views_button.setChecked(False)
            self._selected_planes = []
            self._selected_cryos = []
            n_btn_checked = 0
            plane_no = -1
            for i, btn in enumerate(self._view_button_array):
                if i % self._gm.current_geom.nCryos() == 0:
                    plane_no += 1
                if btn.isChecked():
                    n_btn_checked += 1
                    self._selected_planes.append(plane_no)
                    self._selected_cryos.append(i % self._gm.current_geom.nCryos())
            if n_btn_checked == 0:
                # Fall back to the all views
                self._selected_planes = [-1]
                self._selected_cryos = [-1]
                self._all_views_button.setChecked(True)

        self.refresh_draw_list_widget()

    def refresh_draw_list_widget(self):
        for key, view in self._wire_views.items():
            # Turn it off to begin width
            view.toggleLogo(False)
            view.setVisible(False)

        # negative -1 for p and c means all planes are enabled
        for p, c in zip(self._selected_planes, self._selected_cryos):
            for key, view in self._wire_views.items():
                if key != (p, c) and p != -1 and c != -1:
                    continue

                # Turn on the requested ones
                view.setVisible(True)
                view.toggleLogo(self._show_logo)
                view.toggleScale(self._show_scale_bar)

    def drawWireOnPlot(self, wireData, wire=None, plane=None, tpc=None, cryo=None, drawer=None):
        # Need to draw a wire on the wire view
        # Don't bother if the view isn't active:
        if not self._wire_waveform_widget.isVisible():
            return
       
        # set the display to show the wire:
        self._wireData = wireData
        if tpc % 2 != 0:
            self._wireData = np.flip(self._wireData)
        
        self._wirePlotItem.setData(wireData)
        # update the label
        name = f"W: {wire}, P: {plane}, T: {tpc}, C: {cryo}"
        # self._wireDrawer_name.setText(name)
        self._wirePlot.setTitle(name)
        self._wirePlot.setLabel(axis='bottom', text="Time")
        self._wirePlot.autoRange()
        self.plotFFT()

        # Store the viewport that just draw this
        # as we might need it to increase and
        # decrease the displayed wire
        self._current_wire_drawer = drawer
        self._current_wire = wire
        self._current_tpc = tpc
    
    def plotFFT(self):
        '''
        Take the fft of wire data and plot it in place of the wire signal
        '''
        if self._wireData is None:
            return

        if self._fftButton.isChecked():
            fft = np.fft.rfft(self._wireData)
            freqs = np.fft.rfftfreq(len(self._wireData),0.5E-3)
            self._wirePlotItem.setData(freqs,np.absolute(fft))
            self._wirePlot.setLabel(axis='bottom', text="Frequency")
        else:
            self._wirePlotItem.setData(self._wireData)
            self._wirePlot.setLabel(axis='bottom', text="Time")
        
        self._wirePlot.autoRange()


class WireView(pg.GraphicsLayoutWidget):
    """ Class for a single interactive wire plane widget """
    drawHitsRequested = QtCore.pyqtSignal(int, int, int)
    def customMouseDragEvent(self, ev, axis=None):
        '''
        This is a customizaton of ViewBox's mouseDragEvent.
        The default one is here:
        http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
        Here we want:
        - Right click should allow to zoom in the dragged rectangle (ViewBox.RectMode)
        - Left click should allow to move the pic (ViewBox.PanMode)
        '''

        ev.accept()
        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = pos - lastPos
        dif = dif * -1

        ## Ignore axes if mouse is disabled
        mouseEnabled = np.array(self._view.state['mouseEnabled'], dtype=np.float)
        mask = mouseEnabled.copy()
        if axis is not None:
            mask[1-axis] = 0.0

        self._view.state['mouseMode'] = ViewBox.RectMode

        ## Scale with drag and drop a square done with right button
        if ev.button() & (QtCore.Qt.RightButton):
            # RectMode: Zoom in the dragged rectangle
            if ev.isFinish():
                self._view.rbScaleBox.hide()
                ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                ax = self._view.childGroup.mapRectFromParent(ax)
                self._view.showAxRect(ax)
                self._view.axHistoryPointer += 1
                self._view.axHistory = self._view.axHistory[:self._view.axHistoryPointer] + [ax]
            else:
                ## update shape of scale box
                self._view.updateScaleBox(ev.buttonDownPos(), ev.pos())
        elif ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
            # Translation done with left or mid button
            tr = dif*mask
            tr = self._view.mapToView(tr) - self._view.mapToView(Point(0,0))
            x = tr.x() if mask[0] == 1 else None
            y = tr.y() if mask[1] == 1 else None

            self._view._resetTarget()
            if x is not None or y is not None:
                self._view.translateBy(x=x, y=y)
            self._view.sigRangeChangedManually.emit(self._view.state['mouseEnabled'])

    def __init__(self, geometry, plane=-1, cryostat=0, tpc=0):
        super().__init__(border=None)
        # add a view box, which is a widget that allows an image to be shown
        self._view = self.addViewBox(border=None, defaultPadding=0)
        # add an image item which handles drawing (and refreshing) the image
        self._item = pg.ImageItem(useOpenGL=True)
        # self._item._setPen((0,0,0))
        self._view.addItem(self._item)

        self._removed_entries = 0
        self._manual_t0 = 0
        self._showAnodeCathode = False

        self._anode_lines = []
        self._cathode_lines = []
        self._tpc_div_lines = []

        # Overriding the default mouseDragEvent
        self._view.mouseDragEvent = self.customMouseDragEvent

        # connect the scene to click events, used to get wires
        self.scene().sigMouseClicked.connect(self.mouseClicked)
        # connect the views to mouse move events, used to update the info box at the bottom
        self.scene().sigMouseMoved.connect(self.mouseMoved)
        self._plane = plane
        self._tpc = tpc
        self._cryostat = cryostat
        self._cmSpace = False
        self._geometry = geometry
        self._original_image = None

        self._dataPoints = []
        self._drawnPoints = []
        self._polygon = QtGui.QPolygonF()
        self._path = QtGui.QPainterPath()
        self._path.addPolygon(self._polygon)
        self._polyGraphicsItem = QtWidgets.QGraphicsPathItem(self._path)
        self._view.addItem(self._polyGraphicsItem)

        # Connect scale changes to handle the scale bar correctly
        self._view.sigYRangeChanged.connect(self.scaleHandler)
        self._view.sigXRangeChanged.connect(self.scaleHandler)
        self._xBar = None
        self.useScaleBar = False

        self.setBackground('w')

        self._useLogo = False
        self._logo = None

        self._drawingRawDigits = False
        # each drawer contains its own color gradient and levels
        # this class can return a widget containing the right layout for everything
        # Define some color collections:

        self._colorMap = self._geometry.colorMap(self._plane)


        self._cmap = pg.GradientWidget(orientation='right')
        self._cmap.restoreState(self._colorMap)
        self._cmap.sigGradientChanged.connect(self.refreshGradient)
        self._cmap.resize(1,1)

        # These boxes control the levels.
        self._upperLevel = QtWidgets.QLineEdit()
        self._lowerLevel = QtWidgets.QLineEdit()

        self._upperLevel.returnPressed.connect(self.levelChanged)
        self._lowerLevel.returnPressed.connect(self.levelChanged)

        level_lower = self._geometry.getLevels(self._plane)[0]
        level_upper = self._geometry.getLevels(self._plane)[1]

        if self._drawingRawDigits:
            level_lower += self._geometry.getPedestal(self._plane)
            level_upper += self._geometry.getPedestal(self._plane)

        self._lowerLevel.setText(str(level_lower))
        self._upperLevel.setText(str(level_upper))


        # Fix the maximum width of the widgets:
        self._upperLevel.setMaximumWidth(35)
        self._cmap.setMaximumWidth(25)
        self._lowerLevel.setMaximumWidth(35)

        # The name of the viewport with appropriate tooltip
        name = 'View '
        name += self._geometry.viewNames()[plane]
        name += ', Cryo '
        name += str(cryostat)
        self._viewport_name = VerticalLabel(name)
        self._viewport_name.setStyleSheet('color: rgb(169,169,169);')
        tooltip = 'Bottom view is for TPC 0, top view is for TPC 1. '
        tooltip += 'Note that the vaweforms in TPC 1 are flipped in time '
        tooltip += 'so as to retain the same x direction as in TPC 0. '
        if self._geometry.viewNames()[plane] == 'U':
            tooltip += 'NOTE: bottom image is plane 1 for TPC 0 '
            tooltip += 'but top image is plane 2 for TPC 1'
        if self._geometry.viewNames()[plane] == 'V':
            tooltip += 'NOTE: bottom image is plane 2 for TPC 0 '
            tooltip += 'but top image is plane 1 for TPC 1.'
        self._viewport_name.setToolTip(tooltip)
        self._viewport_name.setMaximumWidth(25)

        colors = QtWidgets.QVBoxLayout()
        colors.addWidget(self._upperLevel)
        colors.addWidget(self._cmap)
        colors.addWidget(self._lowerLevel)
        self._totalLayout = QtWidgets.QHBoxLayout()
        self._totalLayout.addWidget(self._viewport_name)
        self._totalLayout.addWidget(self)
        self._totalLayout.addLayout(colors)

        self._widget = QtWidgets.QWidget()
        self._widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._widget.setLayout(self._totalLayout)

        # show_wrapper is false in the case of drawing blank (no data) views,
        # but true when drawing data
        # disable_wrapper overrides show wrapper: Set to true to prevent
        # wrapper drawing even if there is data
        self._show_wrapper = True
        self._disable_wrapper = False

    def setVisible(self, vis):
        """
        override setVisible to include the widget. This allows the wireview to
        be used in other modules without showing the colorbar & label
        """

        self._widget.setVisible(vis)

        # recursive wrapper for sub-layouts
        def __set_visible(w, v):
            if not isinstance(w, QtWidgets.QWidget):
                for c in w.children():
                    __set_visible(c, v)
            else:
                w.setVisible(v)

        # recurse over childen
        for w in self._widget.children():
            # annoyingly, the class's widget holds a reference to the class
            # object skip it to avoid infinite recursion
            if w is self:
                continue
            __set_visible(w, vis and self._show_wrapper and not self._disable_wrapper)

        # finally get the plot
        super().setVisible(vis)

    def setWrapperVisible(self, vis: bool):
        """ Separately hide/show the wrapper widget """
        self._show_wrapper = vis and not self._disable_wrapper
        self.setVisible(self.isVisible())

    def drawingRawDigits(self, status):
        if status != self._drawingRawDigits:
            self._drawingRawDigits = status
            self.restoreDefaults()
        self._drawingRawDigits = status


    def toggleScale(self, scaleBool):
        # If there is a scale, remove it:
        self.useScaleBar = scaleBool
        self.refreshScaleBar()

    def toggleLogo(self, logoBool):
        ''' Toggles the experiment's logo on and off '''

        if self._logo in self.scene().items():
            self.scene().removeItem(self._logo)

        self._useLogo = logoBool
        self.refreshLogo()

    def refreshLogo(self):
        if not self._useLogo:
            return

        self._logo = MovablePixmapItem(QtGui.QPixmap(self._geometry.logo()))
        self._logo.setX(self._geometry.logoPos()[0])
        self._logo.setY(self._geometry.logoPos()[1])
        self._logo.setScale(self._geometry.logoScale())
        self._logo.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.scene().addItem(self._logo)

    def restoreDefaults(self):
        level_lower = self._geometry.getLevels(self._plane)[0]
        level_upper = self._geometry.getLevels(self._plane)[1]

        if self._drawingRawDigits:
            level_lower += self._geometry.getPedestal(self._plane)
            level_upper += self._geometry.getPedestal(self._plane)

        self._lowerLevel.setText(str(level_lower))
        self._upperLevel.setText(str(level_upper))

        self._cmap.restoreState(self._colorMap)

    def mouseDrag(self):
        print("mouse was dragged")

    def getWidgetAndLayout(self):
        return self._widget,self._totalLayout

    def levelChanged(self):
        # First, get the current values of the levels:
        lowerLevel = int(self._lowerLevel.text())
        upperLevel = int(self._upperLevel.text())

        # set the levels as requested:
        levels = (lowerLevel, upperLevel)
        # next, set the levels in the geometry:
        # last, update the levels in the image:
        self._item.setLevels(levels)

    def refreshGradient(self):
        self._item.setLookupTable(self._cmap.getLookupTable(255))

    def useCM(self,useCMBool):
        self._cmSpace = useCMBool
        self.refreshScaleBar()

    def showAnodeCathode(self,showAC):
        self._showAnodeCathode = showAC

        for l in self._cathode_lines:
            if l in self.scene().items():
                self.scene().removeItem(l)

        for l in self._anode_lines:
            if l in self.scene().items():
                self.scene().removeItem(l)

        self.refreshAnodeCathode()

    def refreshAnodeCathode(self):
        '''
        Draws lines corresponding to the cathode and anode positions for t0 = 0 Red
        line = anode Blue line = cathode
        '''

        if not self._showAnodeCathode:
            return

        max_wire = self._geometry._wRange[self._plane]

        for tpc in range(0, int(self._geometry.nTPCs())):
            # Take into account the distance between planes
            offset = self._geometry.triggerOffset() * self._geometry.time2cm() # - delta_plane

            x_cathode = (2 * self._geometry.halfwidth() + offset)/self._geometry.time2cm()
            x_anode   = offset/self._geometry.time2cm()

            # If we are changing the t0, shift the anode and cathode position
            x_cathode += self._manual_t0
            x_anode   += self._manual_t0

            if tpc % 2 == 1:
                # Time is flipped for odd TPC
                x_cathode = self._geometry.tRange() - x_cathode
                x_anode   = self._geometry.tRange() - x_anode


            # Add the ad-hoc gap between TPCs
            x_cathode += tpc * self._geometry.cathodeGap()
            x_anode   += tpc * self._geometry.cathodeGap()

            # Shift up to the appropriate TPC
            x_cathode += tpc * self._geometry.tRange()
            x_anode   += tpc * self._geometry.tRange()

            # If we are deleting entries to see the cathodes together, do it here too
            x_cathode = x_cathode - 2 * tpc * self._removed_entries
            x_anode   = x_anode - 2 * tpc * self._removed_entries


            # Construct the cathode line and append it
            line = QtWidgets.QGraphicsLineItem()
            line.setLine(0, x_cathode, max_wire, x_cathode)
            line.setPen(pg.mkPen(30,144,255, width=2))
            self._cathode_lines.append(line)
            self._view.addItem(line)

            # Construct the anode line and append it
            line = QtWidgets.QGraphicsLineItem()
            line.setLine(0, x_anode, max_wire, x_anode)
            line.setPen(pg.mkPen(250,128,114, width=2))
            self._anode_lines.append(line)
            self._view.addItem(line)


    def uniteCathodes(self,uniteC):
        self._uniteCathodes = uniteC
        if self._uniteCathodes:
            x_cathode = (2 * self._geometry.halfwidth() + self._geometry.offset(self._plane))/self._geometry.time2cm()
            x_anode   = 0 + self._geometry.offset(self._plane)/self._geometry.time2cm()

            x_cathode += self._manual_t0
            x_anode   += self._manual_t0

            data = self._item.image
            self._original_image = np.copy(data)

            n_removable_entries = int(self._geometry.tRange() - x_cathode)

            start_removal = self._geometry.tRange() - n_removable_entries
            end_removal = self._geometry.tRange()
            slice_right = slice(int(start_removal), int(end_removal))

            start_removal = self._geometry.tRange() + self._geometry.cathodeGap()
            end_removal = start_removal + n_removable_entries
            slice_left = slice(int(start_removal), int(end_removal))

            final_slice = np.r_[slice_right, slice_left]

            self._removed_entries = n_removable_entries

            data = np.delete(data, final_slice, axis=1)
            self.drawPlane(data)

            self.showAnodeCathode(self._showAnodeCathode)
        else:
            self._removed_entries = 0
            self.drawPlane(self._original_image)
            self.showAnodeCathode(self._showAnodeCathode)


    def t0slide(self, t0):
        self._manual_t0 = t0
        self.showAnodeCathode(True)

    def restoret0(self):
        self._manual_t0 = 0
        self.showAnodeCathode(False)

    def mouseMoved(self, pos):
        self.q = self._item.mapFromScene(pos)
        self._lastPos = self.q

        offset = 0
        for i in range(self._geometry.nTPCs() * self._geometry.nCryos(), 0, -1):
            if self.q.y() > i * (self._geometry.tRange() + self._geometry.cathodeGap()):
                offset = -i * (self._geometry.tRange() + self._geometry.cathodeGap())
                break

        message= str()

        if self._cmSpace:
            if type(message) != str:
                message.append("X: ")
                message.append("{0:.1f}".format(self.q.x()*self._geometry.wire2cm()))
            else:
                message += "X: "
                message += "{0:.1f}".format(self.q.x()*self._geometry.wire2cm())
        else:
            if type(message) != str:
                message.append("W: ")
                message.append(str(int(self.q.x())))
            else:
                message += "W: "
                message += str(int(self.q.x()))
        
        if self._cmSpace:
            if type(message) != str:
                message.append(", Y: ")
                message.append("{0:.1f}".format((self.q.y()+offset)*self._geometry.time2cm() - self._geometry.offset(self._plane)))
            else:
                message += ", Y: "
                message += "{0:.1f}".format((self.q.y()+offset)*self._geometry.time2cm() - self._geometry.offset(self._plane))
        else:
            if type(message) != str:
                message.append(", T: ")
                message.append(str(int(self.q.y()+offset)))
            else:
                message += ", T: "
                message += str(int(self.q.y()+offset))

        max_trange = self._geometry.tRange() * self._geometry.nTPCs()


        if self.q.x() > 0 and self.q.x() < self._geometry.wRange(self._plane):
            if self.q.y() > 0 and self.q.y() < max_trange:
                self._statusBar.showMessage(message)


    def mouseClicked(self, event):

        if event.modifiers() == QtCore.Qt.ShiftModifier:
            if event.pos() is not  None:
                self.processPoint(self._lastPos)

        # Figure out in which tpc we are, so we can display only the wire for the selected tpc
        self._first_entry = 0
        self._last_entry = self._geometry.tRange()
        tpc = 0
        for i in range(self._geometry.nTPCs(), 0, -1):
            if self.q.y() > i * (self._geometry.tRange() + self._geometry.cathodeGap()):
                tpc = 1
                self._first_entry = int (i * (self._geometry.tRange() + self._geometry.cathodeGap()))
                self._last_entry = int((i+1) * (self._geometry.tRange() + self._geometry.cathodeGap()))
                break

        wire = int(self._lastPos.x())
        self.show_waveform(wire=wire, tpc=tpc)

    def show_waveform(self, wire, tpc):
        '''
        Shows the waveform on the wire drawer.

        Args:
            wire (int): The wire number
            tpc (int): The TPC number where the wire belongs to
        '''

        if self._item.image is not None:
            # get the data from the plot:
            data = self._item.image
            if wire < 0 or wire >= len(data):
                return
            
            self._wireData = data[wire]
            self._wireData = self._wireData[self._first_entry:self._last_entry]

            # Here we want to display the real plane number, not the view.
            # So, make sure that if you are in an odd TPC we display the right number.
            plane = self._plane
            if tpc %2 != 0:
                if self._plane == 0:
                    plane = 1
                elif self._plane == 1:
                    plane = 0

            self._wdf(wireData=self._wireData, wire=wire, plane=plane, tpc=tpc, cryo=self._cryostat, drawer=self)

            # Make a request to draw the hits from this wire:
            self.drawHitsRequested.emit(self._plane, wire, tpc)


    def connectWireDrawingFunction(self, func):
        self._wdf = func

    def connectStatusBar(self, statusBar):
        self._statusBar = statusBar

    def connectMessageBar(self, messageBar):
        self._messageBar = messageBar

    def getMessageBar(self):
        return self._messageBar

    def setColorMap(self, colormaptype='default'):
        self._colorMap = self._geometry.colorMap(self._plane, colormaptype)
        self._cmap.restoreState(self._colorMap)

    def setRangeToMax(self):
        xR = (0, self._geometry.wRange(self._plane))
        n_planes_per_view = self._geometry.nTPCs()
        yR = (0, n_planes_per_view * self._geometry.tRange())
        self._view.setRange(xRange=xR,yRange=yR, padding=0.002)

    def autoRange(self,xR,yR):
        self._view.setRange(xRange=xR,yRange=yR, padding=0.002)


    def scaleHandler(self):
        if self.useScaleBar:
            self.refreshScaleBar()


    def refreshScaleBar(self):
        if not self.useScaleBar:
            if self._xBar is not None:
                self.scene().removeItem(self._xBar)
                self._xBar = None
            return

        nwires = 100
        if self._xBar is None:
            self._xBar = MovableScaleBar(size=nwires, suffix='wires')
            self._xBar.setParentItem(self._view)
            self._xBar.anchor((1, 1), (1, 1), offset=(-20, -20))

        if self._cmSpace:
            self._xBar.setUnits(self._geometry.wire2cm(), suffix='cm')
        else:
            self._xBar.setUnits(1.0, suffix='wires')


    def plane(self):
        return self._plane

    def tpc(self):
        return self._tpc

    def cryostat(self):
        return self._cryostat

    def lockRatio(self, lockAR ):
        ratio = self._geometry.aspectRatio()
        if lockAR:
            self._view.setAspectLocked(True, ratio=self._geometry.aspectRatio())
        else:
            self._view.setAspectLocked(False)

    def drawPlane(self, image):
        self._item.setImage(image,autoLevels=False)
        self._item.setLookupTable(self._cmap.getLookupTable(255))
        self.setWrapperVisible(True)
        # Make sure the levels are actually set:
        self.levelChanged()

        if self._geometry.nTPCs() == 2:
            self.drawTPCdivision()

    def drawTPCdivision(self):

        for l in self._tpc_div_lines:
            if l in self._view.addedItems:
                self._view.removeItem(l)

        max_wire = self._geometry._wRange[self._plane]
        line_width = 1

        for tpc in range(1, self._geometry.nTPCs()):

            x_tpc = tpc * self._geometry.tRange()              # Place it at the end of one TPC
            x_tpc += (tpc - 1) * self._geometry.cathodeGap()   # Add the gap accumulated previously
            x_tpc += self._geometry.cathodeGap() / 2           # Add half the gap between the 2 TPCs
            x_tpc -= tpc * self._removed_entries               # Remove potentially removed entries to unite the cathodes

            # Draw the line and append it
            line = QtWidgets.QGraphicsRectItem()
            line.setPen(pg.mkPen('w')) # pg.mkPen((169,169,169))) # dark grey
            line.setBrush(pg.mkBrush('w')) # pg.mkBrush((169,169,169))) # dark grey
            # Remove half a pixel (line_width/2), that would otherwise cover half a time tick
            line.setRect(0 + line_width/2, x_tpc - self._geometry.cathodeGap() / 2 + line_width/2, max_wire - line_width/2, self._geometry.cathodeGap() - line_width/2)
            self._view.addItem(line)
            self._tpc_div_lines.append(line)

        if self._geometry.splitWire():
            # Draw the line and append it
            line = QtWidgets.QGraphicsRectItem()
            line.setPen(pg.mkPen('w')) # pg.mkPen((169,169,169))) # dark grey
            line.setBrush(pg.mkBrush('w')) # pg.mkBrush((169,169,169))) # dark grey
            # Remove half a pixel (line_width/2), that would otherwise cover half a time tick
            # line.setRect(0 + line_width/2,
            #              x_tpc - self._geometry.cathodeGap() / 2 + line_width/2,
            #              max_wire - line_width/2,
            #              self._geometry.cathodeGap() - line_width/2)
            line.setRect(max_wire / 2 - self._geometry.cathodeGap() / 2 + line_width/2,
                         0 + line_width/2,
                         self._geometry.cathodeGap(),
                         self._geometry.tRange() * 2 + self._geometry.cathodeGap()  - line_width/2)
            self._view.addItem(line)
            self._tpc_div_lines.append(line)



    def drawBlank(self):
        self._item.clear()
        self.setWrapperVisible(False)


    def clearPoints(self):
        for point in self._drawnPoints:
            self._view.removeItem(point)

        self._drawnPoints = []
        self._dataPoints = []
        self._polygon.clear()
        self._path = QtWidgets.QPainterPath()
        self._polyGraphicsItem.setPath(self._path)

    def makeIonizationPath(self):
        if len(self._dataPoints) < 2:
            return None

        if self._item.image is None:
            return None

        data = self._item.image
        totalpath = np.empty(0)

        for p in xrange(len(self._dataPoints) - 1):
            start = int(round(self._dataPoints[p].x())), int(round(self._dataPoints[p].y()))
            end =  int(round(self._dataPoints[p+1].x())), int(round(self._dataPoints[p+1].y()))
            line = self.get_line(start,end)
            path = np.zeros([len(line)])
            for i in xrange(len(line)):
                pt = line[i]
            path[i] = data[pt[0]][pt[1]]
            # print line
            totalpath = np.concatenate((totalpath,path))

        return totalpath


    def processPoint(self,_in_point):
        # Check if this point is close to another point (less than some dist)
        i = 0
        for point in self._drawnPoints:
            if point.contains(_in_point):
                self._dataPoints.pop(i)
                self._polygon.remove(i)
                self._view.removeItem(self._drawnPoints[i])
                self._drawnPoints.pop(i)
                # Refresh the path:
                self._path = QtGui.QPainterPath()
                self._path.addPolygon(self._polygon)
                self._polyGraphicsItem.setPath(self._path)
                return
            i +=1

        # Point wasn't popped, so add it to the list
        self._dataPoints.append(_in_point)
        r = QtWidgets.QGraphicsEllipseItem(_in_point.x()-1, _in_point.y()-10, 2,20)
        r.setBrush(pg.mkColor((0,0,0)))
        self._view.addItem(r)
        self._drawnPoints.append(r)

        # Refresh the polygon and then update the path
        self._polygon.append(_in_point)

        # self._polyGraphicsItem.setPolygon(self._polygon)
        self._path = QtGui.QPainterPath()
        self._path.addPolygon(self._polygon)
        self._polyGraphicsItem.setPath(self._path)


    # Lovingly stolen from wikipedia, this is not my algorithm
    def get_line(self, start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end

        >>> points1 = get_line((0, 0), (3, 4))
        >>> points2 = get_line((3, 4), (0, 0))
        >>> assert(set(points1) == set(points2))
        >>> print points1
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
        >>> print points2
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points


