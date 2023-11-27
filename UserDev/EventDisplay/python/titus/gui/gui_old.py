import sys, signal, datetime
import os, subprocess
import argparse
# import collections
from PyQt5 import Qt, QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from collections import OrderedDict

import gallery_interface

# Import the class that manages the view windows


class view_manager(QtCore.QObject):
  """This class manages a collection of viewports"""

  drawHitsRequested = QtCore.pyqtSignal(int, int, int)

  def __init__(self, geometry):
    super(view_manager, self).__init__()
    self._nviews = 0
    self._drawerList = OrderedDict()
    self._cmapList = []
    self._geometry = geometry

    self._opt_view = opticalviewport(self._geometry)

    self._wireDrawerMain = self.add_wire_drawer_layout()

    self._drawLogo = False
    self._plottedHits = []

    self._selectedPlane = [-1]
    self._selectedCryo = [0]

    self._autoRange = False
    self._wireData = None

    self._drawing_raw_digits = False


  def add_wire_drawer_layout(self):
    self._wireDrawerMain = pg.GraphicsLayoutWidget()
    self._wireDrawerMain.setBackground(None)
    self._wirePlot = self._wireDrawerMain.addPlot()
    self._wirePlotItem = pg.PlotDataItem(pen=(0,0,0))
    self._wirePlot.addItem(self._wirePlotItem)
    # self._wireDrawerMain.setMaximumHeight(250)
    # self._wireDrawerMain.setMinimumHeight(190)

    self._wireDrawer_name = VerticalLabel("Wire Drawer")
    self._wireDrawer_name.setMaximumWidth(25)
    self._wireDrawer_name.setAlignment(QtCore.Qt.AlignCenter)
    self._wireDrawer_name.setToolTip("Click on a wire to display the waveform.")
    self._wireDrawer_name.setStyleSheet('color: rgb(169,169,169);')
    self._wireDrawerLayout = QtWidgets.QHBoxLayout()
    self._wireDrawerLayout.addWidget(self._wireDrawer_name)
    self._wireDrawerLayout.addWidget(self._wireDrawerMain)

    self._fftButton = QtWidgets.QPushButton("FFT Wire")
    self._fftButton.setToolTip("Compute and show the FFT of the wire currently drawn")
    self._fftButton.setCheckable(True)
    self._fftButton.clicked.connect(self.plotFFT)

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

    self._wireDrawer = QtWidgets.QWidget()
    self._wireDrawer.setLayout(self._wireDrawerVLayout)

  def change_wire(self):
    if self.sender() == self._left_wire_button:
      wire = self._current_wire - 1
    else:
      wire = self._current_wire + 1

    if wire > 0:
      self._current_wire_drawer.show_waveform(wire=wire, tpc=self._current_tpc)
    return

  def addEvdDrawer(self,plane,cryostat=0):
    if not self._geometry:
        return
    self._drawerList[(plane, cryostat)] = viewport(self._geometry, plane, cryostat)
    self._drawerList[(plane, cryostat)].connectWireDrawingFunction(self.drawWireOnPlot)
    self._drawerList[(plane, cryostat)].drawHitsRequested.connect(self.hitOnWireHandler)
    # self._drawerList.append(viewport(self._geometry, plane, cryostat))
    # self._drawerList[-1].connectWireDrawingFunction(self.drawWireOnPlot)
    # self._drawerList[-1].drawHitsRequested.connect(self.hitOnWireHandler)
    self._nviews += 1
  
  def selectPlane(self,plane):
    self._selectedPlane = plane

  def selectCryo(self,cryo):
    self._selectedCryo = cryo


  def restoreDefaults(self):
    for view in self._drawerList.values():
      view.restoreDefaults()
    self._opt_view.restoreDefaults()

  def restoret0(self):
    for view in self._drawerList.values():
      view.restoret0()

  def clearPoints(self):
    for view in self._drawerList.values():
      view.clearPoints()

  def hitOnWireHandler(self, plane, wire, tpc):
    if not self._wireDrawer.isVisible():
      return
    # Simply pass the info on to who ever is listening
    # (hint: it's the manager)
    for hit in self._plottedHits:
      self._wirePlot.removeItem(hit)
    self.drawHitsRequested.emit(plane, wire, tpc)

  def getDrawListWidget(self):

    self._widgetList = []

    # loop through the list and add the drawing windows and their scale
    self._widget = QtWidgets.QWidget()
    self._layout = QtWidgets.QVBoxLayout()
    self._layout.setSpacing(0)
    # self._layout.setMargin(0)
    self._layout.setContentsMargins(0,0,0,0)

    self._planeWidgets = OrderedDict()
    for key, view in self._drawerList.items():
      widget,layout = view.getWidget()
      self._layout.addWidget(widget,0)

    self._opt_widget, _ = self._opt_view.getWidget()
    self._layout.addWidget(self._opt_widget, 0)
    self._opt_widget.setVisible(False)

    self._widget.setLayout(self._layout)

    return self._widget


  def refreshDrawListWidget(self):

    # -1 Means the All Views, turn all of them on!
    if self._selectedPlane[0] == -1:
      self.drawOpDets(False)

      for p, c in zip(self._selectedPlane, self._selectedCryo):
        for key, widget in self._planeWidgets.items():
          self._drawerList[key].toggleLogo(False)
          if (p, c) == (0, 0):
            self._drawerList[key].toggleLogo(True)
          widget.setVisible(True)
      return

    # -1 Means the Optical View, turn only that one on!
    if self._selectedPlane[0] == -2:
      self.drawOpDets(True)
      return

    # Otherwise, only draw the selected ones
    self.drawOpDets(False)
    for key, widget in self._planeWidgets.items():
      # Turn it off to begin width
      self._drawerList[key].toggleLogo(False)
      widget.setVisible(False)
    for p, c in zip(self._selectedPlane, self._selectedCryo):
      for key, widget in self._planeWidgets.items():
        if key == (p, c):
          # Turn on the requested ones
          widget.setVisible(True)
          self._drawerList[key].toggleLogo(self._drawLogo)


  def toggleLogo(self,tl):
    # Draw the logo on the top plane OR the selected plane
    # 
    self._drawLogo = tl
    if not tl:
      for view in self._drawerList.values():
        # Turn everything off, just in case.
        view.toggleLogo(tl)
    else:
      # If drawing just one plane, use that one.
      # Else, use plane 0
      plane = self._selectedPlane[0]
      cryo = self._selectedCryo[0]
      if plane == -1:
        plane = 0
      self._drawerList[(plane, cryo)].toggleLogo(tl)


  def connectStatusBar(self,statusBar):
    for view in self._drawerList.values():
      view.connectStatusBar(statusBar)
    self._opt_view.connectStatusBar(statusBar)

  def connectMessageBar(self,messageBar):
    for view in self._drawerList.values():
      view.connectMessageBar(messageBar)
    self._opt_view.connectMessageBar(messageBar)
    self._messageBar = messageBar

  def getMessageBar(self):
    return self._messageBar



  def linkViews(self):
    pass
  #   print "linking views"
  #   self._lockYRange = True
  #   self._drawerList[0]._view.sigYRangeChanged.connect(self.rangeChangeHandler)
  #   self._drawerList[1]._view.sigYRangeChanged.connect(self.rangeChangeHandler)
  #   self._drawerList[2]._view.sigYRangeChanged.connect(self.rangeChangeHandler)
  #   self._wirePlot.sigXRangeChanged.connect(self.rangeChangeHandler)




  # def rangeChangeHandler(self):
  #   range = self.sender().range
  #   if self._lockYRange:
  #     for view in self._drawerList:
  #       if view._view != self.sender():
  #         view._view.setRange
  #   print "range changed by ", self.sender()

  def setDarkMode(self, opt):
    for view in self._drawerList.values():
      view.setDarkMode(opt)

  def changeColorMap(self, colormaptype='default'):
    for view in self._drawerList.values():
      view.setColorMap(colormaptype)

  def toggleScale(self,scaleBool):
    for view in self._drawerList.values():
      view.toggleScale(scaleBool)

  def setRangeToMax(self):
    for view in self._drawerList.values():
      view.setRangeToMax()

  def autoRange(self,event_manager):
    for view in self._drawerList.values():
      xRange,yRange = event_manager.getAutoRange(view.plane())
      view.autoRange(xRange,yRange)

  def lockAR(self, lockRatio):
    for view in self._drawerList.values():
      view.lockRatio(lockRatio)

  def makePath(self):
    for view in self._drawerList.values():
      path = view.makeIonizationPath()
      if path != None:
        self.drawWireOnPlot(path)
        return

  def drawWire(self,wireView):
    if wireView:
      self._layout.addWidget(self._wireDrawer)
      self._wireDrawer.setVisible(True)
    else:
      self._layout.removeWidget(self._wireDrawer)
      self._wireDrawer.setVisible(False)

  def drawOpDets(self,opdetsView):
    if opdetsView:
      # self._layout.addWidget(self._opt_view)
      self._opt_widget.setVisible(True)
      for widget in self._planeWidgets.values():
        widget.setVisible(False)
    else:
      # self._layout.removeWidget(self._opt_view)
      self._opt_widget.setVisible(False)
      for widget in self._planeWidgets.values():
        widget.setVisible(True)

  def useCM(self,useCM):
    for view in self._drawerList.values():
      view.useCM(useCM)

  def showAnodeCathode(self,showAC):
    for view in self._drawerList.values():
      view.showAnodeCathode(showAC)

  def uniteCathodes(self,uniteC):
    for view in self._drawerList.values():
      view.uniteCathodes(uniteC)

  def t0slide(self,t0):
    for view in self._drawerList.values():
      view.t0slide(t0)

  def drawPlanes(self,event_manager):
    for key, viewport in self._drawerList.items():
      if event_manager.hasWireData():
        plane = key[0]
        cryo = key[1]
        viewport.drawPlane(event_manager.getPlane(plane, cryo))
      else:
        viewport.drawBlank()


  def drawOpDetWvf(self, event_manager):
    if event_manager.hasOpDetWvfData():
      self._opt_view.drawOpDetWvf(event_manager.getOpDetWvf())

  def drawWireOnPlot(self, wireData, wire=None, plane=None, tpc=None, cryo=None, drawer=None):
    # Need to draw a wire on the wire view
    # Don't bother if the view isn't active:
    if not self._wireDrawer.isVisible():
      return
    else:
      # set the display to show the wire:
      self._wireData = wireData
      if tpc % 2 != 0:
        self._wireData = np.flip(wireData)
      self._wirePlotItem.setData(self._wireData)
      # update the label
      name = f"W: {wire}, P: {plane}, T: {tpc}, C: {cryo}"
      # self._wireDrawer_name.setText(name)
      self._wireDrawer_name.setToolTip(name)
      self._wirePlot.setLabel(axis='left', text=name)
      self._wirePlot.setLabel(axis='bottom', text="Time")
      self._wirePlot.autoRange()
      self.plotFFT()

      # Store the viewport that just draw this
      # as we might need it to increase and
      # decrease the displayed wire
      self._current_wire_drawer = drawer
      self._current_wire = wire
      self._current_tpc = tpc



  def drawHitsOnPlot(self,hits,flip=False):
    if not self._wireDrawer.isVisible():
      return

    if len(hits) == 0:
      return

    offset = self._geometry.timeOffsetTicks(hits[0].plane())

    for i in range(len(hits)):
      hit = hits[i]

      start_time = hit.start_time() + offset
      end_time   = hit.end_time() + offset
      delta      = hit.end_time() - hit.start_time() + 1
      peak_time  = hit.peak_time() + offset

      xPts = np.linspace(start_time, end_time, int(delta))
      yPts = hit.peak_amplitude() * np.exp( - 0.5 * (xPts - peak_time)**2 / hit.rms()**2  )
      self._plottedHits.append(self._wirePlot.plot(xPts,yPts,pen=pg.mkPen((255,0,0,200),width=2)))



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
      self._wirePlot.autoRange()
    else:
      self._wirePlotItem.setData(self._wireData)
      self._wirePlot.setLabel(axis='bottom', text="Time")
      self._wirePlot.autoRange()


  def setDrawingRawDigits(self, status):
    '''
    Sets True if the viewports are
    currently drawing RawDigits, False
    otherwise.
    '''
    self._drawing_raw_digits = status
    for view in self._drawerList.values():
        view.drawingRawDigits(status)

  def drawingRawDigits(self):
    '''
    Returns True if the viewports are
    currently drawing RawDigits, False
    otherwise.
    '''
    return self._drawing_raw_digits

  def getViewPorts(self):
    '''
    Returns all the viewports
    for the wire data drawing
    '''
    return self._drawerList.values()

  def getOpticalViewport(self):
    '''
    Returns the viewports that 
    shows optical data
    '''
    return self._opt_view


class Gui(QtWidgets.QMainWindow):

  def __init__(self, geometry):
    super().__init__()

    # initUI should not do ANY data handling, it should only get the interface loaded
    self._geometry = geometry
    self._view_manager = view_manager(geometry)
    self._tracksOnBothTPCs = False
    # self.setStyleSheet("background-color:rgb(230,230,230);")
    self._timer = QtCore.QTimer()
    self._timer.timeout.connect(self.fix_a_drink)
    seconds_to_17 = (17 - datetime.datetime.now().hour - 1) * 60 * 60\
                  + (60 - datetime.datetime.now().minute) * 60\
                  + (60 - datetime.datetime.now().second)
    if seconds_to_17 > 0:
      self._timer.start(seconds_to_17*1e3)

  def fix_a_drink(self):
    self._timer.stop()
    choice = QtWidgets.QMessageBox.question(self, 'It is 5 pm!',
                                        "Time to fix yourself a drink!",
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    if choice == QtWidgets.QMessageBox.Yes:
      print("Awesome.")
      sys.exit()
    else:
      print("Boring.")
      pass

  def initManager(self,manager=None):
    if manager is None:
      manager = evdmanager.manager(self._geometry)
    self._event_manager = manager
    self._event_manager.connectGui(self)
    if self._geometry:
        self._event_manager.connectViewManager(self._view_manager)
        self._view_manager.drawHitsRequested.connect(self._event_manager.drawHitsOnWire)

  def closeEvent(self, event):
    self.quit()

  def quit(self):
    QtCore.QCoreApplication.instance().quit()


  def update_event_labels(self):
    ''' Sets the text boxes correctly '''
    self._larlite_event_entry.setText(str(self._event_manager.internalEvent()))

    self._event_label.setText(f'Event: {self._event_manager.event()}')
    self._run_label.setText(f'Run: {self._event_manager.run()}')
    self._subrun_label.setText(f'Subrun: {self._event_manager.subrun()}')
    self.setupEventRunSubrun()


  def update(self):
    # set the text boxes correctly:
    self.update_event_labels()
    self._view_manager.drawPlanes(self._event_manager)
    self._view_manager.drawOpDetWvf(self._event_manager)
    self.autoRangeWorker()

  # This function prepares the buttons such as prev, next, etc and returns a layout
  def getEventControlButtons(self):

    # This is a box to allow users to enter an event (larlite numbering)
    self._goToLabel = QtWidgets.QLabel("Go to: ")
    self._larlite_event_entry = QtWidgets.QLineEdit()
    self._larlite_event_entry.setToolTip("Enter an event to skip to that event.")
    self._larlite_event_entry.returnPressed.connect(self.goToEventWorker)
    # These labels display current events
    self._run_label = QtWidgets.QLabel("Run: 0")
    self._event_label = QtWidgets.QLabel("Event: 0")
    self._subrun_label = QtWidgets.QLabel("Subrun: 0")

    # Add 3 line to edit for run, subrun, event
    self._run_entry = QtWidgets.QLineEdit("run")
    self._subrun_entry = QtWidgets.QLineEdit("subrun")
    self._event_entry = QtWidgets.QLineEdit("event")
    self.setupEventRunSubrun()
    self._goButton = QtWidgets.QPushButton("Go")
    self._goButton.clicked.connect(self.goToEventRunSubrunWorker)

    # Go to the previous event
    self._prev_button = QtWidgets.QPushButton("Previous")
    self._prev_button.clicked.connect(self._event_manager.prev)
    self._prev_button.setToolTip("Move to the previous event.")
    # Jump to the next event
    self._next_button = QtWidgets.QPushButton("Next")
    self._next_button.clicked.connect(self._event_manager.next)
    self._next_button.setToolTip("Move to the next event.")
    # Pack Previous and Next in a horizontal layout
    self._previousNextLayout = QtWidgets.QHBoxLayout()
    self._previousNextLayout.addWidget(self._prev_button)
    self._previousNextLayout.addWidget(self._next_button)

    # Select a file to use
    self._fileSelectButton = QtWidgets.QPushButton("Select File")
    self._fileSelectButton.clicked.connect(self._event_manager.selectFile)

    # pack the buttons into a box
    self._eventControlBox = QtWidgets.QVBoxLayout()

    # Make a horiztontal box for the event entry and label:
    self._eventGrid = QtWidgets.QHBoxLayout()
    self._eventGrid.addWidget(self._goToLabel)
    self._eventGrid.addWidget(self._larlite_event_entry)

    # Add 3 line edit for run, subrun, event
    self._eventRunSubrunGrid = QtWidgets.QHBoxLayout()
    self._eventRunSubrunGrid.addWidget(self._run_entry)
    self._eventRunSubrunGrid.addWidget(self._subrun_entry)
    self._eventRunSubrunGrid.addWidget(self._event_entry)
    self._eventRunSubrunGrid.addWidget(self._goButton)

    # Another horizontal box for the run/subrun
    self._eventControlBox.addLayout(self._eventGrid)
    self._eventControlBox.addLayout(self._eventRunSubrunGrid)
    self._eventControlBox.addWidget(self._event_label)
    self._eventControlBox.addWidget(self._run_label)
    self._eventControlBox.addWidget(self._subrun_label)
    self._eventControlBox.addLayout(self._previousNextLayout)
    self._eventControlBox.addWidget(self._fileSelectButton)

    return self._eventControlBox


  # this function helps pass the entry of the line edit item to the event control
  def goToEventWorker(self):
    try:
      event = int(self._larlite_event_entry.text())
    except:
      print("Error, must enter an integer")
      self._larlite_event_entry.setText(str(self._event_manager.event()))
      return
    self._event_manager.goToEvent(event)

  def goToEventRunSubrunWorker(self):
    try:
      run = int(self._run_entry.text())
      subrun = int(self._subrun_entry.text())
      event = int(self._event_entry.text())
    except:
      print("Error, must enter an integer")
      self._larlite_event_entry.setText(str(self._event_manager.event()))
      self._run_entry.setText("run")
      self._subrun_entry.setText("subrun")
      self._event_entry.setText("event")
      return
    self._event_manager.goToEvent(event, subrun, run)

  def setupEventRunSubrun(self):
    self._run_entry.setMinimumWidth(40)
    self._subrun_entry.setMinimumWidth(40)
    self._event_entry.setMinimumWidth(40)
    runs = self._event_manager.getAvailableRuns()
    subruns = self._event_manager.getAvailableSubruns()
    events = self._event_manager.getAvailableEvents()

    if len(runs) == 1:
      self._run_entry.setText(str(runs[0]))
      self._run_entry.setDisabled(True)
    else:
      tooltip_text = 'Available runs: '
      tooltip_text += ', '.join(map(str, runs))
      self._run_entry.setToolTip(tooltip_text)

    if len(subruns) == 1:
      self._subrun_entry.setText(str(subruns[0]))
      self._subrun_entry.setDisabled(True)
    else:
      tooltip_text = 'Available subruns: '
      tooltip_text += ', '.join(map(str, subruns))
      self._subrun_entry.setToolTip(tooltip_text)

    if len(events) == 1:
      self._event_entry.setText(str(events[0]))
      self._event_entry.setDisabled(True)
    else:
      tooltip_text = 'Available events: '
      tooltip_text += ', '.join(map(str, events))
      self._event_entry.setToolTip(tooltip_text)


  # This function prepares the range controlling options and returns a layout
  def setupViewControls(self):
    dock =  QtWidgets.QDockWidget('View Controls', self)
    dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
    frame = QtWidgets.QWidget(dock)
    main_layout = QtWidgets.QVBoxLayout()
    frame.setLayout(main_layout)
    dock.setWidget(frame)

    self._grayScale = QtWidgets.QCheckBox("Grayscale")
    self._grayScale.setToolTip("Changes the color map to grayscale.")
    self._grayScale.setTristate(False)
    self._grayScale.stateChanged.connect(self.changeColorMapWorker)

    # Button to set range to max
    self._maxRangeButton = QtWidgets.QPushButton("Max Range")
    self._maxRangeButton.setToolTip("Set the range of the viewers to show the whole event")
    self._maxRangeButton.clicked.connect(self._view_manager.setRangeToMax)

    # Check box to active autorange
    self._autoRangeBox = QtWidgets.QCheckBox("AutoRange")
    self._autoRangeBox.setToolTip("Set the range of the viewers to the regions of interest")
    self._autoRangeBox.setTristate(False)
    self._autoRangeBox.stateChanged.connect(self.autoRangeWorker)

    self._lockAspectRatio = QtWidgets.QCheckBox("Lock A.R.")
    self._lockAspectRatio.setToolTip("Lock the aspect ratio to 1:1")
    self._lockAspectRatio.stateChanged.connect(self.lockARWorker)

    self._rangeLayout = QtWidgets.QVBoxLayout()
    self._rangeLayout.addWidget(self._autoRangeBox)
    self._rangeLayout.addWidget(self._lockAspectRatio)

    # check box to toggle the wire drawing
    self._drawWireOption = QtWidgets.QCheckBox("Wire Drawing")
    self._drawWireOption.setToolTip("Draw the wires when clicked on")
    self._drawWireOption.stateChanged.connect(self.drawWireWorker)
    self._drawRawOption = QtWidgets.QCheckBox("Draw Raw")
    self._drawRawOption.setToolTip("Draw the raw wire signals in 2D")
    self._drawRawOption.setTristate(False)

    self._subtractPedestal = QtWidgets.QCheckBox("Subtract Pedestal")
    self._subtractPedestal.setToolTip("Subtracts the pedestal from RawDigits. You will need to adjust the range.")
    self._subtractPedestal.setTristate(False)
    self._subtractPedestal.setCheckState(QtCore.Qt.Checked)
    self._subtractPedestal.stateChanged.connect(self.subtractPedestalWorker)

    # add a box to restore the drawing defaults:
    self._restoreDefaults = QtWidgets.QPushButton("Restore Defaults")
    self._restoreDefaults.setToolTip("Restore the drawing defaults of the views.")
    self._restoreDefaults.clicked.connect(self.restoreDefaultsWorker)

    self._unitDisplayOption = QtWidgets.QCheckBox("Use cm")
    self._unitDisplayOption.setToolTip("Display the units in cm (checked = true)")
    self._unitDisplayOption.setTristate(False)
    self._unitDisplayOption.stateChanged.connect(self.useCMWorker)

    self._scaleBarOption = QtWidgets.QCheckBox("Scale Bar")
    self._scaleBarOption.setToolTip("Display a scale bar on each view showing the distance")
    self._scaleBarOption.setTristate(False)
    self._scaleBarOption.stateChanged.connect(self.scaleBarWorker)

    self._scaleBarLayout = QtWidgets.QVBoxLayout()
    self._scaleBarLayout.addWidget(self._scaleBarOption)
    self._scaleBarLayout.addWidget(self._unitDisplayOption)

    self._logoOption = QtWidgets.QCheckBox("Draw Logo")
    self._logoOption.setToolTip("Display the experiment logo on the window.")
    self._logoOption.setTristate(False)
    self._logoOption.stateChanged.connect(self.logoWorker)


    self._clearPointsButton = QtWidgets.QPushButton("Clear Points")
    self._clearPointsButton.setToolTip("Clear all of the drawn points from the views")
    self._clearPointsButton.clicked.connect(self.clearPointsWorker)

    self._makePathButton = QtWidgets.QPushButton("Eval. Points")
    self._makePathButton.setToolTip("Compute the ADCs along the path defined by the points")
    self._makePathButton.clicked.connect(self.drawIonizationWorker)

    # Pack Clear Points and Eval Points in a horizontal layout
    self._clearEvalPointsLayout = QtWidgets.QHBoxLayout()
    self._clearEvalPointsLayout.addWidget(self._clearPointsButton)
    self._clearEvalPointsLayout.addWidget(self._makePathButton)

    # self._fftButton = QtWidgets.QPushButton("FFT Wire")
    # self._fftButton.setToolTip("Compute and show the FFT of the wire currently drawn")
    # self._fftButton.clicked.connect(self._view_manager.plotFFT)

    self._anodeCathodeOption = QtWidgets.QCheckBox("Draw anode/cathode")
    self._anodeCathodeOption.setToolTip("Shows the anode and cathode position for t0=0.")
    self._anodeCathodeOption.setTristate(False)
    self._anodeCathodeOption.stateChanged.connect(self.showAnodeCathodeWorker)

    self._uniteCathodes = QtWidgets.QCheckBox("Unite cathodes")
    self._uniteCathodes.setToolTip("Unites the cathodes waveforms.")
    self._uniteCathodes.setTristate(False)
    self._uniteCathodes.stateChanged.connect(self.uniteCathodesWorker)

    self._t0sliderLabelIntro = QtWidgets.QLabel("Set t<sub>0</sub>:")
    self._t0slider = QtWidgets.QSlider(0x1)
    self._t0slider.setToolTip("Change the t<sub>0</sub>.")
    if self._geometry:
        self._t0slider.setMinimum(-self._geometry.triggerOffset())
        self._t0slider.setMaximum(self._geometry.triggerOffset())
    self._t0slider.setSingleStep(10)
    self._t0slider.valueChanged.connect(self.t0sliderWorker)
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
    self._spliTracksOption.stateChanged.connect(self.splitTracksWorker)
    self._spliTracksOption.setVisible(False)


    # Pack the stuff into a layout
    main_layout.addWidget(self._restoreDefaults)
    main_layout.addWidget(self._maxRangeButton)
    # main_layout.addWidget(self._clearPointsButton)
    # main_layout.addWidget(self._makePathButton)
    main_layout.addLayout(self._clearEvalPointsLayout)
    # main_layout.addWidget(self._fftButton)
    main_layout.addWidget(self._grayScale)
    # main_layout.addWidget(self._autoRangeBox)
    # main_layout.addWidget(self._lockAspectRatio)
    main_layout.addLayout(self._rangeLayout)
    main_layout.addWidget(self._drawWireOption)
    main_layout.addWidget(self._subtractPedestal)
    main_layout.addWidget(self._separators[0])
    main_layout.addWidget(self._anodeCathodeOption)
    main_layout.addWidget(self._t0sliderLabelIntro)
    main_layout.addWidget(self._t0slider)
    main_layout.addWidget(self._t0sliderLabel)
    main_layout.addWidget(self._uniteCathodes)
    main_layout.addWidget(self._separators[1])
    # main_layout.addWidget(self._unitDisplayOption)
    # main_layout.addWidget(self._scaleBarOption)
    main_layout.addLayout(self._scaleBarLayout)
    main_layout.addWidget(self._logoOption)
    main_layout.addWidget(self._spliTracksOption)

    main_layout.addStretch()

    self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    self.resizeDocks([dock], [350], QtCore.Qt.Horizontal)

  def changeColorMapWorker(self):
    if self._grayScale.isChecked():
      self._view_manager.changeColorMap(colormaptype='grayscale')
    else:
      self._view_manager.changeColorMap(colormaptype='default')


  def autoRangeWorker(self):
    # TODO
    return
    if self._autoRangeBox.isChecked():
      self._view_manager.autoRange(self._event_manager)
    else:
      self._view_manager.setRangeToMax()


  def viewSelectWorker(self):

    # Understand what views the user selected
    if self.sender() == self._all_views_btn:
      self._optical_view_button.setChecked(False)
      # Uncheck all the other buttons
      for btn in self._viewButtonArray:
        btn.setChecked(False)
      self._view_manager.selectPlane([-1])
      self._view_manager.selectCryo([-1])

    elif self.sender() ==  self._optical_view_button:
      self._all_views_btn.setChecked(False)
      for btn in self._viewButtonArray:
        btn.setChecked(False)
      self._view_manager.selectPlane([-2])
      self._view_manager.selectCryo([-2])

    else:
      self._all_views_btn.setChecked(False)
      self._optical_view_button.setChecked(False)
      selected_planes = []
      selected_cryos = []
      n_btn_checked = 0
      plane_no = -1
      for i, btn in enumerate(self._viewButtonArray):
        if i % self._geometry.nCryos() == 0:
          plane_no += 1
        if btn.isChecked():
          n_btn_checked += 1
          selected_planes.append(plane_no)
          selected_cryos.append(i % self._geometry.nCryos())
      if n_btn_checked == 0:
        # Fall back to the all views
        selected_planes = [-1]
        selected_cryos = [-1]
        self._all_views_btn.setChecked(True)
      self._view_manager.selectPlane(selected_planes)
      self._view_manager.selectCryo(selected_cryos)

    self._view_manager.refreshDrawListWidget()

  def subtractPedestalWorker(self):
    # Implemented in evdgui.py
    return

  def scaleBarWorker(self):
    if self._scaleBarOption.isChecked():
      self._view_manager.toggleScale(True)
    else:
      self._view_manager.toggleScale(False)  

  def logoWorker(self):
    if self._logoOption.isChecked():
      self._view_manager.toggleLogo(True)
    else:
      self._view_manager.toggleLogo(False)  

  def splitTracksWorker(self):
    if self._spliTracksOption.isChecked():
      self._tracksOnBothTPCs = True
    else:
      self._tracksOnBothTPCs = False

  def clearPointsWorker(self):
    self._view_manager.clearPoints()
    pass

  def drawIonizationWorker(self):
    self._view_manager.makePath()
    pass

  def lockARWorker(self):
    if self._lockAspectRatio.isChecked():
      self._view_manager.lockAR(True)
    else:
      self._view_manager.lockAR(False)

  def drawWireWorker(self):
    if self._drawWireOption.isChecked():
      self._view_manager.drawWire(True)
    else:
      self._view_manager.drawWire(False)    

  def useCMWorker(self):
    if self._unitDisplayOption.isChecked():
      self._view_manager.useCM(True)
    else:
      self._view_manager.useCM(False)   

  def showAnodeCathodeWorker(self):
    if self._anodeCathodeOption.isChecked():
      self._view_manager.showAnodeCathode(True)
      self._separators[0].setVisible(True)
      self._t0slider.setVisible(True)
      self._t0sliderLabel.setVisible(True)
      self._t0sliderLabelIntro.setVisible(True)
      self._uniteCathodes.setVisible(True)
      self._separators[1].setVisible(True)
    else:
      self._view_manager.showAnodeCathode(False)  
      self._separators[0].setVisible(False)
      self._t0slider.setVisible(False)
      self._t0sliderLabel.setVisible(False)
      self._t0sliderLabelIntro.setVisible(False)
      self._uniteCathodes.setVisible(False)
      self._separators[1].setVisible(False)

    # if self._geometry.name() == 'icarus': 
    #   self._uniteCathodes.setVisible(False)

  def uniteCathodesWorker(self):
    if self._uniteCathodes.isChecked():
      self._view_manager.uniteCathodes(True)
    else:
      self._view_manager.uniteCathodes(False) 

  def t0sliderWorker(self):
    t0 = self._t0slider.value()
    t0_label = t0 * self._geometry.samplingRate() / 1000.
    t0sliderLabel = "Current t<sub>0</sub> = " + str(t0_label) + " &mu;s"
    self._t0sliderLabel.setText(t0sliderLabel)
    self._view_manager.t0slide(t0)

  def restoret0(self):
    self._view_manager.restoret0()
    self._t0slider.setValue(0)

  def restoreDefaultsWorker(self):
    self._view_manager.restoreDefaults()
    self._view_manager.setRangeToMax()
    self._view_manager.uniteCathodes(False)
    self.restoret0()

  def darkModeWorker(self):
    app = QtWidgets.QApplication.instance()
    if app is None:
      raise RuntimeError("No Qt Application found.")

    if self._darkModeButton.isChecked():
      dark_palette = QtWidgets.QPalette()    
      dark_palette.setColor(QtWidgets.QPalette.Window, QtWidgets.QColor(53, 53, 53))
      dark_palette.setColor(QtWidgets.QPalette.WindowText, QtCore.Qt.white)
      dark_palette.setColor(QtWidgets.QPalette.Base, QtWidgets.QColor(25, 25, 25))
      dark_palette.setColor(QtWidgets.QPalette.AlternateBase, QtWidgets.QColor(53, 53, 53))
      dark_palette.setColor(QtWidgets.QPalette.ToolTipBase, QtCore.Qt.white)
      dark_palette.setColor(QtWidgets.QPalette.ToolTipText, QtCore.Qt.white)
      dark_palette.setColor(QtWidgets.QPalette.Text, QtCore.Qt.white)
      dark_palette.setColor(QtWidgets.QPalette.Button, QtWidgets.QColor(53, 53, 53)) # <-
      dark_palette.setColor(QtWidgets.QPalette.ButtonText, QtCore.Qt.white)
      dark_palette.setColor(QtWidgets.QPalette.BrightText, QtCore.Qt.red)
      dark_palette.setColor(QtWidgets.QPalette.Link, QtWidgets.QColor(42, 130, 218))
      dark_palette.setColor(QtWidgets.QPalette.Highlight, QtWidgets.QColor(42, 130, 218))
      dark_palette.setColor(QtWidgets.QPalette.HighlightedText, QtCore.Qt.black)    
      self._app.setPalette(dark_palette)    
      self._app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    else:
      self._app.setPalette(self._app.style().standardPalette())
      self._app.setStyleSheet("")

    # Propagate to viewports:
    self._view_manager.setDarkMode(self._darkModeButton.isChecked())

  # This function prepares the quit buttons layout and returns it
  def getQuitLayout(self):
    self._quitButton = QtWidgets.QPushButton("Quit")
    self._quitButton.setToolTip("Close the viewer.")
    self._quitButton.clicked.connect(self.quit)
    return self._quitButton

  # This function prepares the dark mode buttons layout and returns it
  def getDarkModeLayout(self):
    self._darkModeButton = QtWidgets.QRadioButton("Dark Mode")
    self._darkModeButton.setToolTip("Changes the appearance to dark mode.")
    self._darkModeButton.clicked.connect(self.darkModeWorker)
    self._darkModeButton.setVisible(False)

    return self._darkModeButton

  # This function combines the control button layouts, range layouts, and quit button
  def getWestLayout(self):

    event_control = self.getEventControlButtons()
    draw_control = self.getDrawingControlButtons()


    # Add the quit button?
    quit_control = self.getQuitLayout()

    # Add dark mode button?
    dark_mode_control = self.getDarkModeLayout()
    
    self._westLayout = QtWidgets.QVBoxLayout()
    self._westLayout.addLayout(event_control)
    self._westLayout.addStretch(1)
    self._westLayout.addLayout(draw_control)
    self._westLayout.addStretch(1)


    # Make the view chouce layout    

    self._gridLayout = QtWidgets.QGridLayout()

    self._all_views_btn = QtWidgets.QPushButton("All Views")
    self._all_views_btn.setCheckable(True)
    self._all_views_btn.setChecked(True) # This is the default one, so is checked
    self._all_views_btn.clicked.connect(self.viewSelectWorker)
    self._gridLayout.addWidget(self._all_views_btn, 0, 0, 1, 4)

    viewport_names = self._geometry.viewNames()
    # viewport_names = ['H', 'U', 'V']
    self._viewButtonArray = []
    self._view_labels = []
    
    for i, v in enumerate(viewport_names):
      
      label = QtWidgets.QLabel("View "+v)
      self._gridLayout.addWidget(label, i+1, 1)
      self._view_labels.append(label)

      for c in range(0, self._geometry.nCryos()):

        text = "Cryo "+str(c)
        if self._geometry.name() == 'sbnd': text = v
        button = QtWidgets.QPushButton(text)
        button.setToolTip("Visualize view "+v+" in cryostat "+str(c)+".")
        button.clicked.connect(self.viewSelectWorker)
        button.setCheckable(True)
        self._viewButtonArray.append(button)
        self._gridLayout.addWidget(button, i+1, c+2)


    self._optical_view_button = QtWidgets.QPushButton("Optical")
    self._optical_view_button.setCheckable(True)
    self._optical_view_button.clicked.connect(self.viewSelectWorker)
    self._gridLayout.addWidget(self._optical_view_button, len(viewport_names)+1, 0, 1, 4)

    self._westLayout.addLayout(self._gridLayout)

    self._westLayout.addStretch(1)

    self._westLayout.addWidget(dark_mode_control)
    self._westLayout.addWidget(quit_control)
    self._westWidget = QtWidgets.QWidget()
    self._westWidget.setLayout(self._westLayout)
    self._westWidget.setMaximumWidth(190)
    self._westWidget.setMinimumWidth(140)
    return self._westWidget


  def getSouthLayout(self):
    # This layout contains the status bar, message bar, and the capture screen buttons

    # The screen capture button:
    self._screenCaptureButton = QtWidgets.QPushButton("Capture Screen")
    self._screenCaptureButton.setToolTip("Capture the entire screen to file")
    self._screenCaptureButton.clicked.connect(self.screenCapture)
    self._southWidget = QtWidgets.QWidget()
    self._southLayout = QtWidgets.QHBoxLayout()
    # Add a status bar
    self._statusBar = QtWidgets.QStatusBar()
    self._statusBar.showMessage("Status Bar")
    self._southLayout.addWidget(self._statusBar)
    self._messageBar = QtWidgets.QStatusBar()
    self._messageBar.showMessage("Message Bar")
    self._southLayout.addWidget(self._messageBar)
    # self._southLayout.addStretch(1)
    self._southLayout.addWidget(self._screenCaptureButton)
    self._southWidget.setLayout(self._southLayout)

    return self._southWidget

  def updateMessageBar(self,message):
    # print "Received a message: {msg}".format(msg=message)
    # TODO
    return
    self._messageBar.showMessage(message)

  def getEastLayout(self):
    # This function just makes a dummy eastern layout to use.
    label = QtWidgets.QLabel("Dummy")
    self._eastWidget = QtWidgets.QWidget()
    self._eastLayout = QtWidgets.QVBoxLayout()
    self._eastLayout.addWidget(label)
    self._eastLayout.addStretch(1)
    self._eastWidget.setLayout(self._eastLayout)
    self._eastWidget.setMaximumWidth(200)
    self._eastWidget.setMinimumWidth(100)
    return self._eastWidget

  def refreshEastLayout(self):
    east = getEastLayout()
    self._eastLayout.setVisible(False)
    self._eastLayout.setVisible(True)

  def refreshCenterView(self):

    # for child in self.centerWidget.children():
    #   print type(child)
    #   if type(child) == QtWidgets.QVBoxLayout:
    #     layout = child

    # print layout.children()
    # print layout

    widget = self._view_manager.getDrawListWidget()
    # for child in widget.children():
    #   print child

    # print widget
    # print layout

    # print layout.children()

    # for i in reversed(range(self.centerWidget.layout.count())): 
        # layout.itemAt(i).widget().setParent(None)

    self.centralWidget().setVisible(False)   
    self.centralWidget().setVisible(True)   

  def initUI(self):
    # set up the central widget
    if self._geometry:
        n_viewports = int(self._geometry.nViews() / self._geometry.nTPCs())

        if self._geometry.nTPCs() > 2:
          print('Only 1 or 2 TPCs are supported.')
          exit()
        for c in range(self._geometry.nCryos()):
          for p in range(self._geometry.nPlanes()):
            self._view_manager.addEvdDrawer(p, c)

        self._view_manager.linkViews()

        # self._view_manager.connectStatusBar(self._statusBar)
        # self._view_manager.connectMessageBar(self._messageBar)

        # Put the layout together
    self.setCentralWidget(QtWidgets.QScrollArea())
    self.centralWidget().setWidget(self._view_manager.getDrawListWidget())


    self.setupEventControls()
    self.setupDrawControls()
    self.setupViewControls()
    self.setupMenuBar()
    self.setupStatusBar()

    # ask the view manager to draw the planes:
    if self._geometry:
        self._view_manager.drawPlanes(self._event_manager)
        self._view_manager.drawOpDetWvf(self._event_manager)
        self._view_manager.setRangeToMax()

    # self.setGeometry(0, 0, 2400/2, 1600/2)
    self.setWindowTitle('TITUS Event Display')    
    self.setFocus()
    self.show()


  def setupEventControls(self):
    self._event_dock =  QtWidgets.QDockWidget('Event Controls', self)
    self._event_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
    self._event_dock.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
    frame = QtWidgets.QWidget(self._event_dock)
    main_layout = QtWidgets.QVBoxLayout()
    frame.setLayout(main_layout)
    self._event_dock.setWidget(frame)

    # Run display labels, horizontally centered
    self._run_label = QtWidgets.QLabel("Run: 0")
    self._event_label = QtWidgets.QLabel("Event: 0")
    self._subrun_label = QtWidgets.QLabel("Subrun: 0")
    run_control_layout = QtWidgets.QHBoxLayout()
    run_control_layout.addStretch()
    run_control_layout.addWidget(self._event_label)
    run_control_layout.addWidget(self._run_label)
    run_control_layout.addWidget(self._subrun_label)
    run_control_layout.addStretch()
    main_layout.addLayout(run_control_layout)

    # Next/Previous buttons
    self._prev_button = QtWidgets.QPushButton("Previous")
    self._prev_button.clicked.connect(self._event_manager.prev)
    self._prev_button.setToolTip("Move to the previous event.")
    self._next_button = QtWidgets.QPushButton("Next")
    self._next_button.clicked.connect(self._event_manager.next)
    self._next_button.setToolTip("Move to the next event.")
    prev_next_layout = QtWidgets.QHBoxLayout()
    prev_next_layout.addWidget(self._prev_button)
    prev_next_layout.addWidget(self._next_button)
    main_layout.addLayout(prev_next_layout)
    
    # Go to event
    self._larlite_event_entry = QtWidgets.QLineEdit()
    self._larlite_event_entry.setToolTip("Enter an event to skip to that event.")
    self._larlite_event_entry.returnPressed.connect(self.goToEventWorker)
    event_layout = QtWidgets.QHBoxLayout()
    event_layout.addWidget(QtWidgets.QLabel('Go to:'))
    event_layout.addWidget(self._larlite_event_entry)
    main_layout.addLayout(event_layout)

    self._run_entry = QtWidgets.QLineEdit()
    self._run_entry.setPlaceholderText('Run')
    self._subrun_entry = QtWidgets.QLineEdit()
    self._subrun_entry.setPlaceholderText('Subrun')
    self._event_entry = QtWidgets.QLineEdit()
    self._event_entry.setPlaceholderText('Event')
    self._goto_button = QtWidgets.QPushButton('Go')
    run_entry_layout = QtWidgets.QHBoxLayout()
    run_entry_layout.addWidget(self._run_entry)
    run_entry_layout.addWidget(self._subrun_entry)
    run_entry_layout.addWidget(self._event_entry)
    run_entry_layout.addWidget(self._goto_button)
    main_layout.addLayout(run_entry_layout)

    main_layout.addStretch()
    self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._event_dock)
    self.resizeDocks([self._event_dock], [200], QtCore.Qt.Horizontal)

  def setupDrawControls(self):
    pass

  def setupMenuBar(self):
    pass

  def setupStatusBar(self):
    self.setStatusBar(QtWidgets.QStatusBar())
    self._detector_label = QtWidgets.QLabel()
    self.statusBar().addPermanentWidget(self._detector_label)
    if self._geometry:
        self._view_manager.connectStatusBar(self.statusBar())

    # Get all of the widgets:
    # self.eastWidget  = self.getEastLayout()
    # self.westWidget  = self.getWestLayout()
    # self.southLayout = self.getSouthLayout()
    # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.westWidget)
    # self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.southLayout)

    # Area to hold data:

  def keyPressEvent(self,e):
    if e.key() == QtCore.Qt.Key_P:
      self._event_manager.prev()
      return
    if e.key() == QtCore.Qt.Key_N:
      self._event_manager.next()
      return
    if e.key() == QtCore.Qt.Key_C:
      # print "C was pressed"
      if e.modifiers() and QtCore.Qt.ControlModifier :
        self.quit()
        return

    # if e.key() == QtCore.Qt.Key_C:
  #     self._dataListsAndLabels['Clusters'].setFocus()
    # if e.key() == QtCore.Qt.Key_H:
  #     self._dataListsAndLabels['Hits'].setFocus()

    if e.key() == QtCore.Qt.Key_R:
      self.setRangeToMax()
      return

    super(gui, self).keyPressEvent(e)

  def screenCapture(self):
    print("Screen Capture!")
    dialog = QtWidgets.QFileDialog()
    r = self._event_manager.run()
    e = self._event_manager.event()
    s = self._event_manager.subrun()
    name = "evd_" + self._geometry.name() + "_R" + str(r)
    name = name + "_S" + str(s)
    name = name + "_E" + str(e) + ".png"
    f = dialog.getSaveFileName(self,"Save File",name,
        "PNG (*.png);;JPG (*.jpg);;All Files (*)")

    if pg.Qt.QT_LIB == pg.Qt.PYQT4:
      pixmapImage = QtGui.QPixmap.grabWidget(self)
      pixmapImage.save(f,"PNG")
    else:
      pixmapImage = super(gui, self).grab()
      pixmapImage.save(f[0],"PNG")

  def get_git_version(self):
    '''
    Returns the git version of the repository this file is in.

    Returns a string with the version and a number in parenthesis showing
    the number of commits after that version (if any)
    '''
    version = subprocess.check_output(["git", "describe", "--tags"], cwd=os.path.dirname(__file__))
    version = version.strip()
    version = version.decode('ascii')
    version = version.split('-')
    if len(version) > 1:
      return version[0] + ' (' + version[1] + ')'
    else:
      return version[0]


