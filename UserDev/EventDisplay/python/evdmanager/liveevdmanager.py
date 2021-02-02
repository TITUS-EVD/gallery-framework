import os

from pyqtgraph.Qt import QtCore

from ROOT import gallery
from ROOT import TFile

try:
  from event import manager, event
  from evdmanager import evd_manager_base
except ImportError:
  from evdmanager.event import manager, event
  from evdmanager.evdmanager import evd_manager_base

import datatypes

# import ROOT

class live_evd_manager_2D(evd_manager_base):

    # truthLabelChanged = QtCore.pyqtSignal(str)
    filterNoise = False

    '''
    Class to handle the 2D specific aspects of viewer
    '''

    def __init__(self, geom, file=None):
        super(live_evd_manager_2D, self).__init__(geom, file)
        self._drawableItems = datatypes.drawableItemsLive()

    # this function is meant for the first request to draw an object or
    # when the producer changes
    def redrawProduct(self, informal_type, product, view_manager):
        # print "Received request to redraw ", product, " by ",producer
        # First, determine if there is a drawing process for this product:
        if product is None:
            if informal_type in self._drawnClasses:
                self._drawnClasses[informal_type].clearDrawnObjects(self._view_manager)
                self._drawnClasses.pop(informal_type)
            return
        if informal_type in self._drawnClasses:
            self._drawnClasses[informal_type].setProducer(product.fullName())
            self.processEvent(True)
            self._drawnClasses[informal_type].clearDrawnObjects(self._view_manager)
            if informal_type == 'MCTrack' or informal_type == 'Track':
                self._drawnClasses[informal_type].drawObjects(self._view_manager, self._gui._tracksOnBothTPCs)
            else:
                self._drawnClasses[informal_type].drawObjects(self._view_manager)
            return

        # Now, draw the new product
        if informal_type in self._drawableItems.getListOfTitles():
            # drawable items contains a reference to the class, so instantiate
            # it
            drawingClass = self._drawableItems.getDict()[informal_type][0](self._geom)
            # Special case for clusters, connect it to the signal:
            # if name == 'Cluster':
            #     self.noiseFilterChanged.connect(
            #         drawingClass.setParamsDrawing)
            #     drawingClass.setParamsDrawing(self._drawParams)
            # if name == 'Match':
            #     self.noiseFilterChanged.connect(
            #         drawingClass.setParamsDrawing)
            #     drawingClass.setParamsDrawing(self._drawParams)
            if informal_type == "RawDigit":
                self.noiseFilterChanged.connect(
                    drawingClass.runNoiseFilter)
                drawingClass.SetSubtractPdedestal(True)

            drawingClass.setProducer(product.fullName())
            self._processer.add_process(product, drawingClass._process)
            self._drawnClasses.update({informal_type: drawingClass})
            # Need to process the event
            self.processEvent(True)
            if informal_type == 'MCTrack' or informal_type == 'Track':
                drawingClass.drawObjects(self._view_manager, self._gui._tracksOnBothTPCs)
            else:
                drawingClass.drawObjects(self._view_manager)

    def clearAll(self):
        for recoProduct in self._drawnClasses:
            self._drawnClasses[recoProduct].clearDrawnObjects(
                self._view_manager)
        # self.clearTruth()

    def drawFresh(self):
        # # wires are special:
        if self._drawWires:
          self._view_manager.drawPlanes(self)
        self.clearAll()
        # Draw objects in a specific order defined by drawableItems
        order = self._drawableItems.getListOfTitles()
        # self.drawTruth()
        for item in order:
            if item in self._drawnClasses:
                self._drawnClasses[item].drawObjects(self._view_manager)

    def getAutoRange(self, plane):
        # This gets the max bounds
        xRangeMax, yRangeMax = super(evd_manager_2D, self).getAutoRange(plane)
        xRange = [999,-999]
        yRange = [99999,-99999]
        for process in self._drawnClasses:
            x, y = self._drawnClasses[process].getAutoRange(plane)
            # Check against all four of the parameters:
            if x is not None:
                if x[0] < xRange[0]:
                    xRange[0] = x[0]
                if x[1] > xRange[1]:
                    xRange[1] = x[1]
            if y is not None:
                if y[0] < yRange[0]:
                    yRange[0] = y[0]
                if y[1] > yRange[1]:
                    yRange[1] = y[1]

        # Pad the ranges by 1 cm to accommodate
        padding = 5
        xRange[0] = max(xRangeMax[0], xRange[0] - padding/self._geom.wire2cm())
        xRange[1] = min(xRangeMax[1], xRange[1] + padding/self._geom.wire2cm())
        yRange[0] = max(yRangeMax[0], yRange[0] - padding/self._geom.time2cm())
        yRange[1] = min(yRangeMax[1], yRange[1] + padding/self._geom.time2cm())
        return xRange, yRange

    def get_products(self, name, stage=None):
        '''
        Returns all available products
        '''
        if stage is None:
            stage = 'all'

        if stage not in self._keyTable:
            return None

        if name not in self._keyTable[stage]:
            return None

        return self._keyTable[stage][name]

    def get_default_products(self, name, stage=None):
        '''
        Returns only the products that will be
        drawn by default, unless the user decides what to see
        in the dropdown menu
        '''
        if stage is None:
            stage = 'all'

        if stage not in self._keyTable:
            return None

        if name not in self._keyTable[stage]:
            return None

        if self._geom.name() == 'icarus' and len(self._keyTable[stage][name]) > 3:
            default_products = [self._keyTable[stage][name][0],
                                self._keyTable[stage][name][1],
                                self._keyTable[stage][name][2],
                                self._keyTable[stage][name][3]]
        else:
            default_products = [self._keyTable[stage][name][0]]

        return default_products

    # handle all the wire stuff:
    def toggleWires(self, product, stage=None, subtract_pedestal=True, producers=None):
        # Now, either add the drawing process or remove it:

        if stage is None:
            stage = 'all'

        if stage not in self._keyTable:
            print("No data available to draw")
            return

        if product == 'wire':
            if 'recob::Wire' not in self._keyTable[stage]:
                print("No wire data available to draw")
                self._drawWires = False
                return
            self._drawWires = True
            self._wireDrawer = datatypes.recoWire(self._geom)

            if producers is not None:
                producer = producers
            elif self._geom.name() == 'icarus' and len(self._keyTable[stage]['recob::Wire']) > 3:
                producer = [self._keyTable[stage]['recob::Wire'][0].fullName(),
                            self._keyTable[stage]['recob::Wire'][1].fullName(),
                            self._keyTable[stage]['recob::Wire'][2].fullName(),
                            self._keyTable[stage]['recob::Wire'][3].fullName()]
            else:
                producer = self._keyTable[stage]['recob::Wire'][0].fullName()

            # self._wireDrawer.setProducer(self._keyTable[stage]['recob::Wire'][0].fullName())
            self._wireDrawer.setProducer(producer)
            self._processer.add_process("recob::Wire",self._wireDrawer._process)
            self.processEvent(True)

        elif product == 'rawdigit':
            if 'raw::RawDigit' not in self._keyTable[stage]:
                print("No raw digit data available to draw")
                self._drawWires = False
                return
            self._drawWires = True
            self._wireDrawer = datatypes.rawDigit(self._geom)
            self._wireDrawer.setSubtractPedestal(subtract_pedestal)

            if producers is not None:
                producer = producers
            elif self._geom.name() == 'icarus' and len(self._keyTable[stage]['raw::RawDigit']) > 3:
                producer = [self._keyTable[stage]['raw::RawDigit'][0].fullName(),
                            self._keyTable[stage]['raw::RawDigit'][1].fullName(),
                            self._keyTable[stage]['raw::RawDigit'][2].fullName(),
                            self._keyTable[stage]['raw::RawDigit'][3].fullName()]
            else:
                producer = self._keyTable[stage]['raw::RawDigit'][0].fullName()

            print ('rawdigit, producer', producer)
            self._wireDrawer.setProducer(producer)
            self._processer.add_process("raw::RawDigit", self._wireDrawer._process)
            self._wireDrawer.toggleNoiseFilter(self.filterNoise)

            self.processEvent(True)
        else:
            if 'raw::RawDigit' in self._processer._ana_units.keys():
                self._processer.remove_process('raw::RawDigit')
            if 'recob::Wire' in self._processer._ana_units.keys():
                self._processer.remove_process('recob::Wire')
            self._wireDrawer = None
            self._drawWires = False

    def toggleNoiseFilter(self, filterBool):
        self.filterNoise = filterBool
        if 'raw::RawDigit' in self._processer._ana_units.keys():
            self._wireDrawer.toggleNoiseFilter(self.filterNoise)
            # Rerun the event just for the raw digits:
            self.processEvent(force=True)
            self.drawFresh()

    def toggleOpDetWvf(self, product, stage=None):

        if stage is None:
            stage = 'all'

        if stage not in self._keyTable:
            print("No data available to draw")
            return

        if product == 'opdetwaveform':

            if 'raw::OpDetWaveform' not in self._keyTable[stage]:
                print("No OpDetWaveform data available to draw")
                self._drawWires = False
                return
            self._drawOpDetWvf = True
            self._opDetWvfDrawer = datatypes.opdetwaveform(self._geom)
            self._opDetWvfDrawer.setProducer(self._keyTable[stage]['raw::OpDetWaveform'][0].fullName())
            self._processer.add_process("raw::OpDetWaveform",self._opDetWvfDrawer._process)
            self.processEvent(True)


    def getPlane(self, plane, cryo=0):
        if self._drawWires:
            return self._wireDrawer.getPlane(plane, cryo)

    def getOpDetWvf(self):
        if self._drawOpDetWvf:
            return self._opDetWvfDrawer.getData()

    def hasWireData(self):
        if self._drawWires:
            return True
        else:
            return False

    def hasOpDetWvfData(self):
        if self._drawOpDetWvf:
            return True
        else:
            return False

    def drawHitsOnWire(self, plane, wire):
        if not 'Hit' in self._drawnClasses:
            return
        else:
            # Get the hits:
            hits = self._drawnClasses['Hit'].getHitsOnWire(plane, wire)
            self._view_manager.drawHitsOnPlot(hits)

            if self._geom.nTPCs() == 2:
                hits = self._drawnClasses['Hit'].getHitsOnWire(plane + self._geom.nPlanes(), wire)
                self._view_manager.drawHitsOnPlot(hits, flip=True)



