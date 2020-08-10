
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import ViewBox, Point
from pyqtgraph import functions as fn
import numpy as np
import math


class VerticalLabel(QtGui.QLabel):

    def __init__(self, *args):
        QtGui.QLabel.__init__(self, *args)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        painter.drawText(0, self.width()/2, self.text())
        painter.end()


class viewport(pg.GraphicsLayoutWidget):

  drawHitsRequested = QtCore.pyqtSignal(int, int)

  def customMouseDragEvent(self, ev, axis=None):
    '''
    This is a customizaton of ViewBox's mouseDragEvent.
    The default one is here: 
    http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
    Here we want:
    - Left click should allow to zoom in the dragged rectangle (ViewBox.RectMode)
    - Right click should allow to move the pic (ViewBox.PanMode)
    '''
    ## if axis is specified, event will only affect that axis.
    ev.accept()  ## we accept all buttons

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

    ## Scale or translate based on mouse button
    if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
        # RectMode: Zoom in the dragged rectangle
        print('Left-Mid button')
        print('self._view.state[\'mouseMode\']', self._view.state['mouseMode'])
        # if self._view.state['mouseMode'] == ViewBox.RectMode:
        print('mouse mode is RectMode')
        if ev.isFinish():  ## This is the final move in the drag; change the view scale now
            self._view.rbScaleBox.hide()
            ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
            ax = self._view.childGroup.mapRectFromParent(ax)
            self._view.showAxRect(ax)
            self._view.axHistoryPointer += 1
            self._view.axHistory = self._view.axHistory[:self._view.axHistoryPointer] + [ax]
        else:
            ## update shape of scale box
            self._view.updateScaleBox(ev.buttonDownPos(), ev.pos())
    elif ev.button() & QtCore.Qt.RightButton:
        # Translation
        tr = dif*mask
        tr = self._view.mapToView(tr) - self._view.mapToView(Point(0,0))
        x = tr.x() if mask[0] == 1 else None
        y = tr.y() if mask[1] == 1 else None
        
        self._view._resetTarget()
        if x is not None or y is not None:
            self._view.translateBy(x=x, y=y)
        self._view.sigRangeChangedManually.emit(self._view.state['mouseEnabled'])


  def __init__(self, geometry, plane=-1, cryostat=0, tpc=0):
    super(viewport, self).__init__(border=None)
    # add a view box, which is a widget that allows an image to be shown
    self._view = self.addViewBox(border=None)
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
    self._polyGraphicsItem = QtGui.QGraphicsPathItem(self._path)
    self._view.addItem(self._polyGraphicsItem)

    # Connect scale changes to handle the scale bar correctly
    self._view.sigYRangeChanged.connect(self.scaleHandler)
    self._view.sigXRangeChanged.connect(self.scaleHandler)
    self._xBar = None
    self._xBarText = None
    # self._yBar = None
    # self._yBarText = None
    self.useScaleBar = False

    # Set up the blank data:
    # self._blankData = np.ones((self._geometry.wRange(self._plane),self._geometry.tRange()))
    self.setBackground('w')
    # self.setBackground(pg.mkColor(0, 0, 0))

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
    self._upperLevel = QtGui.QLineEdit()
    self._lowerLevel = QtGui.QLineEdit()

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


    colors = QtGui.QVBoxLayout()
    colors.addWidget(self._upperLevel)
    colors.addWidget(self._cmap)
    colors.addWidget(self._lowerLevel)
    self._totalLayout = QtGui.QHBoxLayout()
    self._totalLayout.addWidget(self._viewport_name)
    self._totalLayout.addWidget(self)
    self._totalLayout.addLayout(colors)

    self._widget = QtGui.QWidget()
    self._widget.setLayout(self._totalLayout)
    self._widget.setMaximumHeight(1000)
    self._widget.setMinimumHeight(50)

  def setDarkMode(self, opt):
    self._dark_mode = opt
    if self._dark_mode:
      self.setBackground(pg.mkColor(25, 25, 25))
    else:
      self.setBackground('w')

  def drawingRawDigits(self, status):
    if status != self._drawingRawDigits:
      self._drawingRawDigits = status
      self.restoreDefaults()
    self._drawingRawDigits = status


  def toggleScale(self,scaleBool):
    # If there is a scale, remove it:
    if self._xBar in self._view.addedItems:
      self._view.removeItem(self._xBar)
      self._view.removeItem(self._xBarText)
    self.useScaleBar = scaleBool
    self.refreshScaleBar()

  def toggleLogo(self,logoBool):
    '''
    Toggles the experiment's 
    logo on and off
    '''

    if self._logo in self.scene().items():
        self.scene().removeItem(self._logo)

    self._useLogo = logoBool
    self.refreshLogo()

  def refreshLogo(self):
    if not self._useLogo:
      return

    self._logo = QtGui.QGraphicsPixmapItem(QtGui.QPixmap(self._geometry.logo()))
    self._logo.setX(self._geometry.logoPos()[0])
    self._logo.setY(self._geometry.logoPos()[1])
    self._logo.setScale(self._geometry.logoScale())
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

  def getWidget(self):

    return self._widget,self._totalLayout

  def levelChanged(self):
    # First, get the current values of the levels:
    lowerLevel = int(self._lowerLevel.text())
    upperLevel = int(self._upperLevel.text())

    # set the levels as requested:
    levels = (lowerLevel, upperLevel)
    # next, set the levels in the geometry:
    # self._geometry._levels[self._plane] = (lowerLevel,upperLevel)
    # last, update the levels in the image:
    self._item.setLevels(levels)

  def refreshGradient(self):
    self._item.setLookupTable(self._cmap.getLookupTable(255))

  def useCM(self,useCMBool):
    self._cmSpace = useCMBool
    self.refreshScaleBar()

  def showAnodeCathode(self,showAC):
    self._showAnodeCathode = showAC
    # if self._line_a in self.scene().items():
    #     self.scene().removeItem(self._line_a)
    # if self._line_c in self.scene().items():
    #     self.scene().removeItem(self._line_c)

    # if self._line_a_2 in self.scene().items():
    #     self.scene().removeItem(self._line_a_2)
    # if self._line_c_2 in self.scene().items():
    #     self.scene().removeItem(self._line_c_2)

    for l in self._cathode_lines:
        if l in self.scene().items():
            self.scene().removeItem(l)

    for l in self._anode_lines:
        if l in self.scene().items():
            self.scene().removeItem(l)

    self.refreshAnodeCathode()

  def refreshAnodeCathode(self):
    '''
    Draws lines corresponding to the cathode and
    anode positions for t0 = 0
    Red line = anode
    Blue line = cathode
    '''

    # x_pos = 20
    # y_pos = 400
    # x_scale = 2
    # y_scale = -17
    # self._tpcText = QtGui.QGraphicsSimpleTextItem("TPC 0, Plane 0")
    # self._tpcText.setBrush(pg.mkColor(255,255,255))
    # # xScale = 0.015* width
    # # yScale = - 0.5* height
    # self._tpcText.setPos(x_pos, y_pos)
    # self._tpcText.scale(x_scale, y_scale)
    # self._tpcText.font().setPixelSize(15)
    # self._view.addItem(self._tpcText)


    if not self._showAnodeCathode:
      return

    max_wire = self._geometry._wRange[self._plane]

    for tpc in range(0, int(self._geometry.nTPCs())):

        # Take into account the distance between planes
        plane_x_ref = self._geometry.getGeometryCore().Plane(0).GetCenter().X()
        plane_x = self._geometry.getGeometryCore().Plane(self._plane).GetCenter().X()
        delta_plane = abs(plane_x - plane_x_ref)
        
        offset = self._geometry.triggerOffset() * self._geometry.time2cm() - delta_plane

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
        line = QtGui.QGraphicsLineItem()
        line.setLine(0, x_cathode, max_wire, x_cathode)
        line.setPen(pg.mkPen(30,144,255, width=2))
        self._cathode_lines.append(line)
        self._view.addItem(line)

        # Construct the anode line and append it
        line = QtGui.QGraphicsLineItem()
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
        slice_right = slice(start_removal, end_removal)

        start_removal = self._geometry.tRange() + self._geometry.cathodeGap()
        end_removal = start_removal + n_removable_entries
        slice_left = slice(start_removal, end_removal)

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

    if (pg.Qt.QT_LIB == 'PyQt4'):
      message= QtCore.QString()
    else:
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

    # print (message)
    max_trange = self._geometry.tRange() * self._geometry.nTPCs()

    # if self._geometry.nTPCs() == 2: 
    #     max_trange *= 2

    if self.q.x() > 0 and self.q.x() < self._geometry.wRange(self._plane):
      if self.q.y() > 0 and self.q.y() < max_trange:
        self._statusBar.showMessage(message)

  def mouseClicked(self, event):
    # print self
    # print event
    # print event.pos()
    # Get the Mouse position and print it:
    # print "Image position:", self.q.x()
    # use this method to try drawing rectangles
    # self.drawRect()
    # pdi.plot()
    # For this function, a click should get the wire that is
    # being hovered over and draw it at the bottom
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

    if self._item.image is not None:
      # get the data from the plot:
      data = self._item.image
      if wire < len(data):
        self._wireData = data[wire]
        self._wireData = self._wireData[self._first_entry:self._last_entry]
        self._wdf(wireData=self._wireData, wire=wire, plane=self._plane , tpc=tpc, cryo=self._cryostat, drawer=self)

    # Make a request to draw the hits from this wire:
    self.drawHitsRequested.emit(self._plane,wire)


  def connectWireDrawingFunction(self,func):
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
    pass


  def scaleHandler(self):
    # print self.sender()
    if self.useScaleBar:
      self.refreshScaleBar()


  def refreshScaleBar(self):
    if not self.useScaleBar:
      return
    # First, get the range in x and y:
    # [[xmin, xmax], [ymin, ymax]]
    dims = self._view.viewRange()

    # The view bars get drawn on 10% of the lengths
    # if ratio lock is set, only draw X

    # Draw the X bar:
    xMin = dims[0][0]
    xMax = dims[0][1]
    yMin = dims[1][0]
    yMax = dims[1][1]
    width = 0.1*(xMax - xMin)
    height = 0.01*(yMax - yMin)
    xLoc = xMin + 0.1*(xMax - xMin)
    yLoc = yMin + 0.1*(yMax - yMin)


    if self._xBar in self._view.addedItems:
      self._view.removeItem(self._xBar)
      self._view.removeItem(self._xBarText)

    self._xBar = QtGui.QGraphicsRectItem(xLoc,yLoc,width,height)
    self._xBar.setBrush(pg.mkColor(255,255,255))
    self._view.addItem(self._xBar)

    xString = ""
    if self._cmSpace:
      xString = "{0:.0f}".format(round(width*self._geometry.wire2cm()))
      xString = xString + " cm"
    else:
      xString = "{0:.0f}".format(round(width))
      xString = xString + " wires"


    # Add the text:
    self._xBarText = QtGui.QGraphicsSimpleTextItem(xString)
    self._xBarText.setBrush(pg.mkColor(255,255,255))
    xScale = 0.015* width
    yScale = - 0.3* height
    self._xBarText.setPos(xLoc,yLoc)
    self._xBarText.scale(xScale,yScale)
    self._xBarText.font().setPixelSize(15)
    self._view.addItem(self._xBarText)

    # # Now do the y Bar
    # width = 0.01*(xMax - xMin)
    # height = 0.1*(yMax - yMin)
    # xLoc = xMin + 0.1*(xMax - xMin)
    # yLoc = yMin + 0.1*(yMax - yMin)
    # if self._yBar in self._view.addedItems:
    #   self._view.removeItem(self._yBar)
    #   self._view.removeItem(self._yBarText)

    # self._yBar = QtGui.QGraphicsRectItem(xLoc,yLoc,width,height)
    # self._yBar.setBrush(pg.mkColor(0,0,0))
    # self._view.addItem(self._yBar)

    # # Add the text:
    # self._yBarText = QtGui.QGraphicsSimpleTextItem(xString)
    # xScale = 0.015* width
    # yScale = - 0.5* height
    # self._yBarText.setPos(xLoc,yLoc)
    # self._yBarText.scale(xScale,yScale)
    # self._yBarText.setRotation(90)
    # self._yBarText.font().setPixelSize(15)
    # self._view.addItem(self._yBarText)

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
    self._cmap.setVisible(True)
    self._upperLevel.setVisible(True)
    self._lowerLevel.setVisible(True)
    self._item.setVisible(False)
    self._item.setVisible(True)
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
        line = QtGui.QGraphicsRectItem()
        line.setPen(pg.mkPen('w')) # pg.mkPen((169,169,169))) # dark grey
        line.setBrush(pg.mkBrush('w')) # pg.mkBrush((169,169,169))) # dark grey
        # Remove half a pixel (line_width/2), that would otherwise cover half a time tick
        line.setRect(0 + line_width/2, x_tpc - self._geometry.cathodeGap() / 2 + line_width/2, max_wire - line_width/2, self._geometry.cathodeGap() - line_width/2)
        self._view.addItem(line)
        self._tpc_div_lines.append(line)

    if self._geometry.splitWire():
        # Draw the line and append it
        line = QtGui.QGraphicsRectItem()
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



    # self._line_tpc_div = QtGui.QGraphicsRectItem()
    # self._line_tpc_div.setPen(pg.mkPen('w')) # pg.mkPen((169,169,169))) # dark grey
    # self._line_tpc_div.setBrush(pg.mkBrush('w')) # pg.mkBrush((169,169,169))) # dark grey
    # self._line_tpc_div.setRect(0, x_tpc - self._geometry.cathodeGap() / 2, max_wire, self._geometry.cathodeGap())

    # # self._line_tpc_div = QtGui.QGraphicsLineItem()
    # # self._line_tpc_div.setLine(0, x_tpc, max_wire, x_tpc)
    # # self._line_tpc_div.setPen(pg.mkPen(color='r', width=self._geometry.cathodeGap()))


    # self._view.addItem(self._line_tpc_div)



  def drawBlank(self):
    self._item.clear()
    self._cmap.setVisible(False)
    self._upperLevel.setVisible(False)
    self._lowerLevel.setVisible(False)


  def clearPoints(self):
    for point in self._drawnPoints:
      self._view.removeItem(point)

      self._drawnPoints = []
      self._dataPoints = []
      self._polygon.clear()
      self._path = QtGui.QPainterPath()
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
    r = QtGui.QGraphicsEllipseItem(_in_point.x()-1, _in_point.y()-10, 2,20)
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
