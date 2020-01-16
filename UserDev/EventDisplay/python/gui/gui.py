

import sys, signal
import argparse
# import collections
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

import evdmanager

# Import the class that manages the view windows
try:
  from viewport import viewport
  from opticalviewport import opticalviewport
except ImportError:
  from gui.viewport import viewport
  from gui.opticalviewport import opticalviewport

class view_manager(QtCore.QObject):
  """This class manages a collection of viewports"""

  drawHitsRequested = QtCore.pyqtSignal(int, int)

  def __init__(self, geometry):
    super(view_manager, self).__init__()
    self._nviews = 0
    self._drawerList = []
    self._cmapList = []
    self._geometry = geometry

    self._opt_view = opticalviewport(self._geometry)

    self._wireDrawer = pg.GraphicsLayoutWidget()
    self._wireDrawer.setBackground(None)
    self._wirePlot = self._wireDrawer.addPlot()
    # self._wirePlot.setPen((0,0,0))
    self._wirePlotItem = pg.PlotDataItem(pen=(0,0,0))
    # self._wirePlotItem.setBac
    self._wirePlot.addItem(self._wirePlotItem)
    self._wireDrawer.setMaximumHeight(200)
    self._wireDrawer.setMinimumHeight(100)

    self._drawLogo = False
    self._plottedHits = []

    self._selectedPlane = -1

    self._autoRange = False
    self._wireData = None

    self._drawing_raw_digits = False

  def addEvdDrawer(self,plane):
    self._drawerList.append(viewport(self._geometry, plane))
    self._drawerList[-1].connectWireDrawingFunction(self.drawWireOnPlot)
    self._drawerList[-1].drawHitsRequested.connect(self.hitOnWireHandler)
    self._nviews += 1
  
  def selectPlane(self,plane):
    self._selectedPlane = plane


  def restoreDefaults(self):
    for view in self._drawerList:
      view.restoreDefaults()

  def restoret0(self):
    for view in self._drawerList:
      view.restoret0()

  def clearPoints(self):
    for view in self._drawerList:
      view.clearPoints()

  def hitOnWireHandler(self,plane,wire):
    if not self._wireDrawer.isVisible():
      return
    # Simply pass the info on to who ever is listening
    # (hint: it's the manager)
    for hit in self._plottedHits:
      self._wirePlot.removeItem(hit)
    self.drawHitsRequested.emit(plane,wire)

  def getDrawListWidget(self):

    self._widgetList = []

    # loop through the list and add the drawing windows and their scale
    self._widget = QtGui.QWidget()
    self._layout = QtGui.QVBoxLayout()
    self._layout.setSpacing(0)
    # self._layout.setMargin(0)
    self._layout.setContentsMargins(0,0,0,0)

    self._planeWidgets = []
    for view in self._drawerList:
      widget,layout = view.getWidget()
      self._planeWidgets.append(widget)
      self._layout.addWidget(widget,0)

    self._opt_widget, _ = self._opt_view.getWidget()
    self._layout.addWidget(self._opt_widget, 0)
    self._opt_widget.setVisible(False)

    self._widget.setLayout(self._layout)

    return self._widget

  def refreshDrawListWidget(self):

    # The last one is the optical view
    optical_view = self._geometry.nViews() / self._geometry.nTPCs() / self._geometry.nCryos()

    # Draw all planes:
    if self._selectedPlane == -1:
      self.drawOpDets(False)
      i = 0
      for widget in self._planeWidgets:
        widget.setVisible(True)
        if i == 0:
          # Draw the logo on the first plane, maybe
          self._drawerList[i].toggleLogo(self._drawLogo)
        else:
          # Definitely don't draw it on the other planes
          self._drawerList[i].toggleLogo(False)
        i += 1

    elif self._selectedPlane == optical_view:
      self.drawOpDets(True)
    else:
      self.drawOpDets(False)
      i = 0
      for widget in self._planeWidgets:
        if i == self._selectedPlane:
          widget.setVisible(True)
          self._drawerList[i].toggleLogo(self._drawLogo)
        else:
          # If there is a logo, remove it first!
          self._drawerList[i].toggleLogo(False)
          widget.setVisible(False)
        i += 1

  def toggleLogo(self,tl):
    # Draw the logo on the top plane OR the selected plane
    # 
    self._drawLogo = tl
    if not tl:
      for view in self._drawerList:
        # Turn everything off, just in case.
        view.toggleLogo(tl)
    else:
      # If drawing just one plane, use that one.
      # Else, use plane 0
      plane = self._selectedPlane
      if plane == -1:
        plane = 0
      self._drawerList[plane].toggleLogo(tl)


  def connectStatusBar(self,statusBar):
    for view in self._drawerList:
      view.connectStatusBar(statusBar)
    self._opt_view.connectStatusBar(statusBar)



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
    for view in self._drawerList:
      view.setDarkMode(opt)

  def changeColorMap(self, colormaptype='default'):
    for view in self._drawerList:
      view.setColorMap(colormaptype)

  def toggleScale(self,scaleBool):
    for view in self._drawerList:
      view.toggleScale(scaleBool)

  def setRangeToMax(self):
    for view in self._drawerList:
      view.setRangeToMax()

  def autoRange(self,event_manager):
    for view in self._drawerList:
      xRange,yRange = event_manager.getAutoRange(view.plane())
      view.autoRange(xRange,yRange)

  def lockAR(self, lockRatio):
    for view in self._drawerList:
      view.lockRatio(lockRatio)

  def makePath(self):
    for view in self._drawerList:
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
      for widget in self._planeWidgets:
        widget.setVisible(False)
    else:
      # self._layout.removeWidget(self._opt_view)
      self._opt_widget.setVisible(False)
      for widget in self._planeWidgets:
        widget.setVisible(True)

  def useCM(self,useCM):
    for view in self._drawerList:
      view.useCM(useCM)

  def showAnodeCathode(self,showAC):
    for view in self._drawerList:
      view.showAnodeCathode(showAC)

  def uniteCathodes(self,uniteC):
    for view in self._drawerList:
      view.uniteCathodes(uniteC)

  def t0slide(self,t0):
    for view in self._drawerList:
      view.t0slide(t0)

  def drawPlanes(self,event_manager):
    for i in range(len(self._drawerList)):
      if event_manager.hasWireData():
        self._drawerList[i].drawPlane(event_manager.getPlane(i))
      else:
        self._drawerList[i].drawBlank()

  def drawOpDetWvf(self, event_manager):
    if event_manager.hasOpDetWvfData():
      self._opt_view.drawOpDetWvf(event_manager.getOpDetWvf())

 
  def drawWireOnPlot(self, wireData):
    # Need to draw a wire on the wire view
    # Don't bother if the view isn't active:
    if not self._wireDrawer.isVisible():
      return
    else:
      # set the display to show the wire:
      self._wireData = wireData
      self._wirePlotItem.setData(self._wireData)
      # if axisData is not None:
      #   self._wirePlotItem.setData(axisData,wireData)
      # else:

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
    
      # In case of 2 TPCs, also draw the hits on
      # the other plane, but flipping the time
      if flip:
        start_time = 2 * self._geometry.tRange() - start_time + self._geometry.cathodeGap()
        end_time   = 2 * self._geometry.tRange() - end_time   + self._geometry.cathodeGap()
        peak_time  = 2 * self._geometry.tRange() - peak_time  + self._geometry.cathodeGap()

      xPts = np.linspace(start_time, end_time, delta)
      yPts = hit.peak_amplitude() * np.exp( - 0.5 * (xPts - peak_time)**2 / hit.rms()**2  )
      # self._plottedHits.append(self._wirePlot.plot(xPts,yPts))
      self._plottedHits.append(self._wirePlot.plot(xPts,yPts,pen=pg.mkPen((255,0,0,200),width=2)))

      # self._wirePlot.remove


  def plotFFT(self):
    # Take the fft of wire data and plot it in place of the wire signal
    if self._wireData is None:
      return
    fft = np.fft.rfft(self._wireData)
    freqs = np.fft.rfftfreq(len(self._wireData),0.5E-3)
    self._wirePlotItem.setData(freqs,np.absolute(fft))
    return

  def setDrawingRawDigits(self, status):
    '''
    Sets True if the viewports are
    currently drawing RawDigits, False
    otherwise.
    '''
    self._drawing_raw_digits = status
    for view in self._drawerList:
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
    return self._drawerList

  def getOpticalViewport(self):
    '''
    Returns the viewports that 
    shows optical data
    '''
    return self._opt_view


class gui(QtGui.QWidget):

  def __init__(self, geometry):
    super(gui, self).__init__()

    # initUI should not do ANY data handling, it should only get the interface loaded
    self._geometry = geometry
    self._view_manager = view_manager(geometry)
    self._tracksOnBothTPCs = False
    # self.setStyleSheet("background-color:rgb(230,230,230);")

  def initManager(self,manager=None):
    if manager is None:
      manager = evdmanager.manager(self._geometry)
    self._event_manager = manager
    self._event_manager.connectGui(self)
    self._event_manager.connectViewManager(self._view_manager)
    self._view_manager.drawHitsRequested.connect(self._event_manager.drawHitsOnWire)

  def closeEvent(self, event):
    self.quit()  

  def quit(self):
    # if self._running:
      # self.stopRun()
    QtCore.QCoreApplication.instance().quit()


  def update(self):
    # set the text boxes correctly:
    self._larliteEventEntry.setText(str(self._event_manager.internalEvent()))

    eventLabel = "Event: " + str(self._event_manager.event())
    self._eventLabel.setText(eventLabel)
    runLabel = "Run: " + str(self._event_manager.run())
    self._runLabel.setText(runLabel)
    subrunLabel = "Subrun: " + str(self._event_manager.subrun())
    self._subrunLabel.setText(subrunLabel)
    
    self._view_manager.drawPlanes(self._event_manager)
    self.autoRangeWorker()

  # This function prepares the buttons such as prev, next, etc and returns a layout
  def getEventControlButtons(self):

    # This is a box to allow users to enter an event (larlite numbering)
    self._goToLabel = QtGui.QLabel("Go to: ")
    self._larliteEventEntry = QtGui.QLineEdit()
    self._larliteEventEntry.setToolTip("Enter an event to skip to that event.")
    self._larliteEventEntry.returnPressed.connect(self.goToEventWorker)
    # These labels display current events
    self._runLabel = QtGui.QLabel("Run: 0")
    self._eventLabel = QtGui.QLabel("Event: 0")
    self._subrunLabel = QtGui.QLabel("Subrun: 0")

    # Jump to the next event
    self._nextButton = QtGui.QPushButton("Next")
    # self._nextButton.setStyleSheet("background-color: red")
    self._nextButton.clicked.connect(self._event_manager.next)
    self._nextButton.setToolTip("Move to the next event.")
    # Go to the previous event
    self._prevButton = QtGui.QPushButton("Previous")
    self._prevButton.clicked.connect(self._event_manager.prev)
    self._prevButton.setToolTip("Move to the previous event.")
    # Select a file to use
    self._fileSelectButton = QtGui.QPushButton("Select File")
    self._fileSelectButton.clicked.connect(self._event_manager.selectFile)
    
    # pack the buttons into a box
    self._eventControlBox = QtGui.QVBoxLayout()

    # Make a horiztontal box for the event entry and label:
    self._eventGrid = QtGui.QHBoxLayout()
    self._eventGrid.addWidget(self._goToLabel)
    self._eventGrid.addWidget(self._larliteEventEntry)
    # Another horizontal box for the run/subrun
    # self._runSubRunGrid = QtGui.QHBoxLayout()
    # self._runSubRunGrid.addWidget(self._eventLabel)
    # self._runSubRunGrid.addWidget(self._runLabel)
    # Pack it all together
    self._eventControlBox.addLayout(self._eventGrid)
    self._eventControlBox.addWidget(self._eventLabel)
    self._eventControlBox.addWidget(self._runLabel)
    self._eventControlBox.addWidget(self._subrunLabel)
    self._eventControlBox.addWidget(self._nextButton)
    self._eventControlBox.addWidget(self._prevButton)
    self._eventControlBox.addWidget(self._fileSelectButton)

    return self._eventControlBox
  

  # this function helps pass the entry of the line edit item to the event control
  def goToEventWorker(self):
    try:
      event = int(self._larliteEventEntry.text())
    except:
      print("Error, must enter an integer")
      self._larliteEventEntry.setText(str(self._event_manager.event()))
      return
    self._event_manager.goToEvent(event)

  # This function prepares the range controlling options and returns a layout
  def getDrawingControlButtons(self):

    self._grayScale = QtGui.QCheckBox("Gray Scale")
    self._grayScale.setToolTip("Changes the color map to grayscale.")
    self._grayScale.setTristate(False)
    self._grayScale.stateChanged.connect(self.changeColorMapWorker) 

    # Button to set range to max
    self._maxRangeButton = QtGui.QPushButton("Max Range")
    self._maxRangeButton.setToolTip("Set the range of the viewers to show the whole event")
    self._maxRangeButton.clicked.connect(self._view_manager.setRangeToMax)
    # Check box to active autorange
    self._autoRangeBox = QtGui.QCheckBox("Auto Range")
    self._autoRangeBox.setToolTip("Set the range of the viewers to the regions of interest")
    self._autoRangeBox.setTristate(False)
    self._autoRangeBox.stateChanged.connect(self.autoRangeWorker)  

    self._lockAspectRatio = QtGui.QCheckBox("Lock A.R.")
    self._lockAspectRatio.setToolTip("Lock the aspect ratio to 1:1")
    self._lockAspectRatio.stateChanged.connect(self.lockARWorker)

    # check box to toggle the wire drawing
    self._drawWireOption = QtGui.QCheckBox("Wire Drawing")
    self._drawWireOption.setToolTip("Draw the wires when clicked on")
    self._drawWireOption.stateChanged.connect(self.drawWireWorker)
    self._drawRawOption = QtGui.QCheckBox("Draw Raw")
    self._drawRawOption.setToolTip("Draw the raw wire signals in 2D")
    self._drawRawOption.setTristate(False)

    self._subtractPedestal = QtGui.QCheckBox("Subtract Pedestal")
    self._subtractPedestal.setToolTip("Subtracts the pedestal from RawDigits. You will need to adjust the range.")
    self._subtractPedestal.setTristate(False)
    self._subtractPedestal.setCheckState(QtCore.Qt.Checked)
    self._subtractPedestal.stateChanged.connect(self.subtractPedestalWorker)

    # add a box to restore the drawing defaults:
    self._restoreDefaults = QtGui.QPushButton("Restore Defaults")
    self._restoreDefaults.setToolTip("Restore the drawing defaults of the views.")
    self._restoreDefaults.clicked.connect(self.restoreDefaultsWorker)

    self._unitDisplayOption = QtGui.QCheckBox("Use cm")
    self._unitDisplayOption.setToolTip("Display the units in cm (checked = true)")
    self._unitDisplayOption.setTristate(False)
    self._unitDisplayOption.stateChanged.connect(self.useCMWorker)

    self._scaleBarOption = QtGui.QCheckBox("Draw Scale Bar")
    self._scaleBarOption.setToolTip("Display a scale bar on each view showing the distance")
    self._scaleBarOption.setTristate(False)
    self._scaleBarOption.stateChanged.connect(self.scaleBarWorker)

    self._logoOption = QtGui.QCheckBox("Draw Logo")
    self._logoOption.setToolTip("Display the experiment logo on the window.")
    self._logoOption.setTristate(False)
    self._logoOption.stateChanged.connect(self.logoWorker)


    self._clearPointsButton = QtGui.QPushButton("ClearPoints")
    self._clearPointsButton.setToolTip("Clear all of the drawn points from the views")
    self._clearPointsButton.clicked.connect(self.clearPointsWorker)

    self._makePathButton = QtGui.QPushButton("Eval. Points")
    self._makePathButton.setToolTip("Compute the ADCs along the path defined by the points")
    self._makePathButton.clicked.connect(self.drawIonizationWorker)

    self._fftButton = QtGui.QPushButton("FFT Wire")
    self._fftButton.setToolTip("Compute and show the FFT of the wire currently drawn")
    self._fftButton.clicked.connect(self._view_manager.plotFFT)

    self._anodeCathodeOption = QtGui.QCheckBox("Draw anode/cathode")
    self._anodeCathodeOption.setToolTip("Shows the anode and cathode position for t0=0.")
    self._anodeCathodeOption.setTristate(False)
    self._anodeCathodeOption.stateChanged.connect(self.showAnodeCathodeWorker)

    self._uniteCathodes = QtGui.QCheckBox("Unite cathodes")
    self._uniteCathodes.setToolTip("Unites the cathodes waveforms.")
    self._uniteCathodes.setTristate(False)
    self._uniteCathodes.stateChanged.connect(self.uniteCathodesWorker)

    self._t0sliderLabelIntro = QtGui.QLabel("Set t<sub>0</sub>:")
    self._t0slider = QtGui.QSlider(0x1)
    self._t0slider.setToolTip("Change the t<sub>0</sub>.")
    self._t0slider.setMinimum(-self._geometry.triggerOffset())
    self._t0slider.setMaximum(self._geometry.triggerOffset())
    self._t0slider.setSingleStep(10)
    self._t0slider.valueChanged.connect(self.t0sliderWorker)
    self._t0sliderLabel = QtGui.QLabel("Current t<sub>0</sub> = 0")

    self._t0sliderLabelIntro.setVisible(False)
    self._t0slider.setVisible(False)
    self._t0sliderLabel.setVisible(False)
    self._uniteCathodes.setVisible(False)

    self._separators = []  
    for i in range(2):
      self._separators.append(QtGui.QFrame())
      self._separators[i].setFrameShape(QtGui.QFrame.HLine)
      self._separators[i].setFrameShadow(QtGui.QFrame.Sunken)
      self._separators[i].setVisible(False)



    self._spliTracksOption = QtGui.QCheckBox("Tracks on Both TPCs")
    self._spliTracksOption.setToolTip("Split the MCTracks and Tracks so that they are projected on both TPCs.")
    self._spliTracksOption.setTristate(False)
    self._spliTracksOption.stateChanged.connect(self.splitTracksWorker)
    self._spliTracksOption.setVisible(False)


    # Pack the stuff into a layout
    self._drawingControlBox = QtGui.QVBoxLayout()
    self._drawingControlBox.addWidget(self._restoreDefaults)
    self._drawingControlBox.addWidget(self._maxRangeButton)
    self._drawingControlBox.addWidget(self._clearPointsButton)
    self._drawingControlBox.addWidget(self._makePathButton)
    self._drawingControlBox.addWidget(self._fftButton)
    self._drawingControlBox.addWidget(self._grayScale)
    self._drawingControlBox.addWidget(self._autoRangeBox)
    self._drawingControlBox.addWidget(self._lockAspectRatio)
    self._drawingControlBox.addWidget(self._drawWireOption)
    self._drawingControlBox.addWidget(self._subtractPedestal)
    self._drawingControlBox.addWidget(self._separators[0])
    self._drawingControlBox.addWidget(self._anodeCathodeOption)
    self._drawingControlBox.addWidget(self._t0sliderLabelIntro)
    self._drawingControlBox.addWidget(self._t0slider)
    self._drawingControlBox.addWidget(self._t0sliderLabel)
    self._drawingControlBox.addWidget(self._uniteCathodes)
    self._drawingControlBox.addWidget(self._separators[1])
    self._drawingControlBox.addWidget(self._unitDisplayOption)
    self._drawingControlBox.addWidget(self._scaleBarOption)
    self._drawingControlBox.addWidget(self._logoOption)
    self._drawingControlBox.addWidget(self._spliTracksOption)

    return self._drawingControlBox

  def changeColorMapWorker(self):
    if self._grayScale.isChecked():
      self._view_manager.changeColorMap(colormaptype='grayscale')
    else:
      self._view_manager.changeColorMap(colormaptype='default')


  def autoRangeWorker(self):
    if self._autoRangeBox.isChecked():
      self._view_manager.autoRange(self._event_manager)
    else:
      self._view_manager.setRangeToMax()

  def viewSelectWorker(self):

    n_views = int(self._geometry.nViews() / self._geometry.nTPCs() / self._geometry.nCryos())
    n_views += 1 # Add the optical evd

    i = 0
    for i in range(n_views):
      if self.sender() == self._viewButtonArray[i]:
        self._view_manager.selectPlane(i)
        self._view_manager.refreshDrawListWidget()
        return
      else:
        i += 1

    # if there wasn't a match, then it must be the ALL button:
    if self.sender() != None:
      self._view_manager.selectPlane(-1)
      self._view_manager.refreshDrawListWidget()
      return

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

    if self._geometry.name() == 'icarus': 
      self._uniteCathodes.setVisible(False)

  def uniteCathodesWorker(self):
    if self._uniteCathodes.isChecked():
      self._view_manager.uniteCathodes(True)
    else:
      self._view_manager.uniteCathodes(False) 

  def t0sliderWorker(self):
    t0 = self._t0slider.value()
    t0_label = t0 * self._geometry.samplingRate() / 1000.
    t0sliderLabel = "Current t<sub>0</sub> = " + str(t0_label / 1000.) + " &mu;s"
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
    app = QtGui.QApplication.instance()
    if app is None:
      raise RuntimeError("No Qt Application found.")

    if self._darkModeButton.isChecked():
      dark_palette = QtGui.QPalette()    
      dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
      dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
      dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
      dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
      dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
      dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
      dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
      dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53)) # <-
      dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
      dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
      dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
      dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
      dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)    
      self._app.setPalette(dark_palette)    
      self._app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    else:
      self._app.setPalette(self._app.style().standardPalette())
      self._app.setStyleSheet("")

    # Propagate to viewports:
    self._view_manager.setDarkMode(self._darkModeButton.isChecked())

  # This function prepares the quit buttons layout and returns it
  def getQuitLayout(self):
    self._quitButton = QtGui.QPushButton("Quit")
    self._quitButton.setToolTip("Close the viewer.")
    self._quitButton.clicked.connect(self.quit)
    return self._quitButton

  # This function prepares the dark mode buttons layout and returns it
  def getDarkModeLayout(self):
    self._darkModeButton = QtGui.QRadioButton("Dark Mode")
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
    
    self._westLayout = QtGui.QVBoxLayout()
    self._westLayout.addLayout(event_control)
    self._westLayout.addStretch(1)
    self._westLayout.addLayout(draw_control)
    self._westLayout.addStretch(1)

    # Add a section to allow users to just view one window instead of two/three
    self._viewButtonGroup = QtGui.QButtonGroup()
    # Draw all planes:
    self._allViewsButton = QtGui.QRadioButton("All")
    self._allViewsButton.clicked.connect(self.viewSelectWorker)
    self._viewButtonGroup.addButton(self._allViewsButton)

    # Put the buttons in a layout
    self._viewChoiceLayout = QtGui.QVBoxLayout()

    # Make a label for this stuff:
    self._viewChoiceLabel = QtGui.QLabel("View Options")
    self._viewChoiceLayout.addWidget(self._viewChoiceLabel)
    self._viewChoiceLayout.addWidget(self._allViewsButton)

    views = self._geometry.viewNames()
    i = 0
    self._viewButtonArray = []
    n_views = int(self._geometry.nViews() / self._geometry.nTPCs() / self._geometry.nCryos())
    for plane in range(n_views):
      text = "Plane" + str(i)
      # Use U, V and W in the case of SBND and ICARUS
      if self._geometry.nTPCs() == 2: 
        text = "View " + views[i]
      button = QtGui.QRadioButton(text)
      i += 1
      self._viewButtonGroup.addButton(button)
      button.clicked.connect(self.viewSelectWorker)
      self._viewButtonArray.append(button)
      self._viewChoiceLayout.addWidget(button)

    # Add a button for the optical event display
    button = QtGui.QRadioButton("Optical")
    button.clicked.connect(self.viewSelectWorker)
    self._viewButtonGroup.addButton(button)
    self._viewButtonArray.append(button)
    self._viewChoiceLayout.addWidget(button)

    self._westLayout.addLayout(self._viewChoiceLayout)

    self._westLayout.addStretch(1)

    self._westLayout.addWidget(dark_mode_control)
    self._westLayout.addWidget(quit_control)
    self._westWidget = QtGui.QWidget()
    self._westWidget.setLayout(self._westLayout)
    self._westWidget.setMaximumWidth(150)
    self._westWidget.setMinimumWidth(100)
    return self._westWidget


  def getSouthLayout(self):
    # This layout contains the status bar, message bar, and the capture screen buttons

    # The screen capture button:
    self._screenCaptureButton = QtGui.QPushButton("Capture Screen")
    self._screenCaptureButton.setToolTip("Capture the entire screen to file")
    self._screenCaptureButton.clicked.connect(self.screenCapture)
    self._southWidget = QtGui.QWidget()
    self._southLayout = QtGui.QHBoxLayout()
    # Add a status bar
    self._statusBar = QtGui.QStatusBar()
    self._statusBar.showMessage("Test message")
    self._southLayout.addWidget(self._statusBar)
    self._messageBar = QtGui.QStatusBar()
    self._southLayout.addWidget(self._messageBar)
    # self._southLayout.addStretch(1)
    self._southLayout.addWidget(self._screenCaptureButton)
    self._southWidget.setLayout(self._southLayout)

    return self._southWidget

  def updateMessageBar(self,message):
    # print "Received a message: {msg}".format(msg=message)
    self._messageBar.showMessage(message)

  def getEastLayout(self):
    # This function just makes a dummy eastern layout to use.
    label = QtGui.QLabel("Dummy")
    self._eastWidget = QtGui.QWidget()
    self._eastLayout = QtGui.QVBoxLayout()
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
    #   if type(child) == QtGui.QVBoxLayout:
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

    self.centerWidget.setVisible(False)   
    self.centerWidget.setVisible(True)   

  def initUI(self):


    # Get all of the widgets:
    self.eastWidget  = self.getEastLayout()
    self.westWidget  = self.getWestLayout()
    self.southLayout = self.getSouthLayout()

    # Area to hold data:
    nviews = int(self._geometry.nViews() / self._geometry.nTPCs() / self._geometry.nCryos())

    if self._geometry.nTPCs() > 2:
      print('Only 1 or 2 TPCs are supported.')
      exit()

    for i in range(0, nviews):
      # These boxes hold the wire/time views:
      self._view_manager.addEvdDrawer(i)

    self._view_manager.linkViews()

    self._view_manager.connectStatusBar(self._statusBar)

    self.centerWidget = self._view_manager.getDrawListWidget()

    # Put the layout together


    self.master = QtGui.QVBoxLayout()
    self.slave = QtGui.QHBoxLayout()
    self.slave.addWidget(self.westWidget)
    self.slave.addWidget(self.centerWidget)
    self.slave.addWidget(self.eastWidget)
    self.master.addLayout(self.slave)
    self.master.addWidget(self.southLayout)

    self.setLayout(self.master)    

    # ask the view manager to draw the planes:
    self._view_manager.drawPlanes(self._event_manager)


    self.setGeometry(0, 0, 2400/2, 1600/2)
    self.setWindowTitle('Event Display')    
    self.setFocus()
    self.show()
    self._view_manager.setRangeToMax()

  def keyPressEvent(self,e):
    if e.key() == QtCore.Qt.Key_N:
      self._event_manager.next()
      return
    if e.key() == QtCore.Qt.Key_P:
      self._event_manager.prev()
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
    dialog = QtGui.QFileDialog()
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


