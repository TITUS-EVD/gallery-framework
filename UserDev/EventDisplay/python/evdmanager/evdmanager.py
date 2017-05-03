from pyqtgraph.Qt import QtCore
from event import manager, event
import datatypes
from ROOT import gallery
import os
from ROOT import TFile
import ROOT



class product(object):
    def __init__(self, name, typeName):
        self._name=name
        self._typeName=typeName
        self._isAssociation=False
        self._associatedProduct=None
        self._producer=None
        self._stage=None
  
        self.parse()
  
    def producer(self):
        return self._producer
  
    def name(self):
        return self._name
  
    def typeName(self):
        return self._typeName
  
    def isAssociation(self):
        return self._isAssociation
  
    # def associationProduct(self):
        # pass
  
    # def reverseAssociationProduct(self):
        # pass
  
    def parse(self):
        tokens=self._name.split('_')
        # Name goes as object_producer_stage
        self._producer=tokens[1]
        self._stage=tokens[-1]
        self._typeName = tokens[0].rstrip('s')

        return


class processer(object):

    def __init__(self):

        # Storing ana units as a map:
        # self._ana_units[data product] -> instance of ana_unit
        self._ana_units = dict()
        pass

    def process_event(self, gallery_event):
        # print "Running ... "
        for key in self._ana_units:
            # print "Processing " + key
            self._ana_units[key].analyze(gallery_event)

    def add_process(self, data_product, ana_unit):
        if data_product in self._ana_units:
            self._ana_units.pop(data_product)
        self._ana_units.update({data_product : ana_unit})
        return

    def remove_process(self, data_product, ana_unit=None):
        if data_product in self._ana_units:
            self._ana_units.pop(data_product)
        return

    def get_process(self, data_product):
        if data_product in self._ana_units:
            return self._ana_units[data_product]
        else:
            return None

    def reset():
        self._ana_units = dict()

class evd_manager_base(manager, QtCore.QObject):
    fileChanged = QtCore.pyqtSignal()
    eventChanged = QtCore.pyqtSignal()

    """docstring for lariat_manager"""

    def __init__(self, geom, file=None):
        super(evd_manager_base, self).__init__(geom, file)
        manager.__init__(self, geom, file)
        QtCore.QObject.__init__(self)
        # For the larlite manager, need both the ana_processor and
        # the storage manager
        self._processer = processer()
        # self._mgr = fmwk.storage_manager()
        self._data_manager = None

        self._keyTable = dict()
        self._drawnClasses = dict()

        if file != None:
            self.setInputFiles(file)

        self._n_entries = 0

        # Toggle whether or not to draw wires:
        self._drawWires = False
        # self._drawParams = False
        # self._drawTruth = False

        self._wireDrawer = None
        # self._truthDrawer = None


    def pingFile(self, file):
        """
        this function opens the file and
        determines what is available to draw
        """
        # This function opens the file to see
        # what data products are available

        # Open the file
        f = TFile(file)
        # Use the larlite_id_tree to find out how many entries are in the file:
        e=f.Get("Events")

        # prepare a dictionary of data products
        lookUpTable = dict()
        products = []
        # Loop over the keys (list of trees)
        for key in e.GetListOfBranches():
            if key.GetTypeName() == 'art::EventAuxiliary':
                continue
            if "Assns" in key.GetTypeName():
                continue

            prod=product(key.GetName(), key.GetTypeName())

            _product = prod._typeName



            # gets three items in thisKeyList, which is a list
            # [dataProduct, producer, 'tree'] (don't care about 'tree')
            # check if the data product is in the dict:
            if _product in lookUpTable:
                # extend the list:
                lookUpTable[_product] += (prod._producer, )
            else:
                lookUpTable.update({_product: (prod._producer,)})

        self._keyTable.update(lookUpTable)
        return 

    def setInputFile(self, file):
        f = [file, ]
        self.setInputFiles(f)

    def setInputFiles(self, files):

        # reset the storage manager and process
        if self._data_manager is not None:
            self._data_manager = None
        # self._process.reset()

        if files == None:
            return

        _file_list = ROOT.vector(ROOT.string)()

        for file in files:
            # First, check that the file exists:
            try:
                if not os.path.exists(file):
                    print "ERROR: requested file does not exist."
                    continue
            except Exception, e:
                print e
                return
            # Next, verify it is a root file:
            if not file.endswith(".root"):
                print "ERROR: must supply a root file."
                continue

            # Finally, ping the file to see what is available to draw
            self.pingFile(file)
            if len(self._keyTable) > 0:
                self._hasFile = True
                _file_list.push_back(file)


        # Create an instance of the data manager:
        if _file_list.size() > 0:
            self._data_manager = gallery.Event(_file_list)
            self._n_entries += self._data_manager.numberOfEventsInFile()

        # Open the manager
        self._lastProcessed = -1
        self.goToEvent(0)
        self.fileChanged.emit()

    # This function will return all producers for the given product
    def getProducers(self, product):
        try:
            return self._keyTable[product]
        except:
            return None

    # This function returns the list of products that can be drawn:
    def getDrawableProducts(self):
        return self._drawableItems.getDict()

    # override the run,event,subrun functions:
    def run(self):
        if self._data_manager is None:
            return 0
        return self._data_manager.eventAuxiliary().run()

    def event(self):
        if self._data_manager is None:
            return 0
        return self._data_manager.eventAuxiliary().event()

    def subrun(self):
        if self._data_manager is None:
            return 0
        return self._data_manager.eventAuxiliary().subRun()

    def internalEvent(self):
        return self._event

    # override the functions from manager as needed here
    def next(self):
        # print "Called next"
        # Check that this isn't the last event:
        if self._event < self._n_entries - 1:
            self.goToEvent(self._event + 1)
        else:
            print "On the last event, can't go to next."

    def prev(self):
        if self._event != 0:
            self.goToEvent(self._event - 1)
        else:
            print "On the first event, can't go to previous."

    def processEvent(self, force=False):
        if self._lastProcessed != self._event or force:
            self._processer.process_event(self._data_manager)
            self._lastProcessed = self._event

    def goToEvent(self, event, force=False):
        # Gallery events don't offer random access
        

        # Loop through until the event is gotten:
        if event < self._n_entries:
            if event == self._event + 1:
                self._data_manager.next()

            else:
                if event > self._event:
                    while event != self._data_manager.eventEntry():
                        self._data_manager.next()
                else:
                    self._data_manager.toBegin()
                    while event != self._data_manager.eventEntry():
                        self._data_manager.next()
        else:
            print "Selected event is too high"
            return

        self.setEvent(self._data_manager.eventEntry())
        self.processEvent()

        # if self._view_manager != None:
            # self._view_manager.drawPlanes(self)
        self.drawFresh()
        self.eventChanged.emit()



class evd_manager_2D(evd_manager_base):

    # truthLabelChanged = QtCore.pyqtSignal(str)
    filterNoise = False
    
    '''
    Class to handle the 2D specific aspects of viewer
    '''

    def __init__(self, geom, file=None):
        super(evd_manager_2D, self).__init__(geom, file)
        self._drawableItems = datatypes.drawableItems()

    # this function is meant for the first request to draw an object or
    # when the producer changes
    def redrawProduct(self, name, product, producer, view_manager):
        # print "Received request to redraw ", product, " by ",producer
        # First, determine if there is a drawing process for this product:
        if producer is None:
            if name in self._drawnClasses:
                self._drawnClasses[name].clearDrawnObjects(self._view_manager)
                self._drawnClasses.pop(name)
            return
        if name in self._drawnClasses:
            self._drawnClasses[name].setProducer(producer)
            self.processEvent(True)
            self._drawnClasses[name].clearDrawnObjects(self._view_manager)
            self._drawnClasses[name].drawObjects(self._view_manager)
            return

        # Now, draw the new product
        if name in self._drawableItems.getListOfTitles():
            # drawable items contains a reference to the class, so instantiate
            # it
            drawingClass = self._drawableItems.getDict()[name][0]()
            # Special case for clusters, connect it to the signal:
            # if name == 'Cluster':
            #     self.noiseFilterChanged.connect(
            #         drawingClass.setParamsDrawing)
            #     drawingClass.setParamsDrawing(self._drawParams)
            # if name == 'Match':
            #     self.noiseFilterChanged.connect(
            #         drawingClass.setParamsDrawing)
            #     drawingClass.setParamsDrawing(self._drawParams)
            if name == "RawDigit":
                self.noiseFilterChanged.connect(
                    drawingClass.runNoiseFilter)

            drawingClass.setProducer(producer)
            self._processer.add_process(product, drawingClass._process)
            self._drawnClasses.update({name: drawingClass})
            # Need to process the event
            self.processEvent(True)
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
        xRangeMax, yRangeMax = super(larlite_manager, self).getAutoRange(plane)
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

    # handle all the wire stuff:
    def toggleWires(self, product):
        # Now, either add the drawing process or remove it:


        if product == 'wire':
            if 'recob::Wire' not in self._keyTable:
                print "No wire data available to draw"
                self._drawWires = False
                return
            self._drawWires = True
            self._wireDrawer = datatypes.recoWire(self._geom)
            self._wireDrawer.setProducer(self._keyTable['recob::Wire'][0])
            self._processer.add_process("recob::Wire",self._wireDrawer._process)
            self.processEvent(True)

        elif product == 'rawdigit':
            if 'raw::RawDigit' not in self._keyTable:
                print "No raw digit data available to draw"
                self._drawWires = False
                return
            self._drawWires = True
            self._wireDrawer = datatypes.rawDigit(self._geom)
            self._wireDrawer.setProducer(self._keyTable['raw::RawDigit'][0])
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

    def getPlane(self, plane):
        if self._drawWires:
            return self._wireDrawer.getPlane(plane)

    def hasWireData(self):
        if self._drawWires:
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

try:
    import pyqtgraph.opengl as gl

    class evd_manager_3D(evd_manager_base):

        """This class handles file I/O and drawing for 3D viewer"""

        showMCCosmic = True

        def __init__(self, geom, file=None):
            super(evd_manager_3D, self).__init__(geom, file)
            self._drawableItems = datatypes.drawableItems3D()

        def getAutoRange(self):
            pass

        # this function is meant for the first request to draw an object or
        # when the producer changes
        def redrawProduct(self, name, product, producer, view_manager):
            # print "Received request to redraw ", product, " by ",producer, " with name ", name
            # First, determine if there is a drawing process for this product:           
            if producer is None:
                if name in self._drawnClasses:
                    self._drawnClasses[name].clearDrawnObjects(self._view_manager)
                    self._drawnClasses.pop(name)
                return
            if name in self._drawnClasses:
                self._drawnClasses[name].setProducer(producer)
                self.processEvent(True)
                self._drawnClasses[name].clearDrawnObjects(self._view_manager)
                self._drawnClasses[name].drawObjects(self._view_manager)



            # Now, draw the new product
            if name in self._drawableItems.getListOfTitles():
                # drawable items contains a reference to the class, so
                # instantiate it
                drawingClass=self._drawableItems.getDict()[name][0]()
                # Special case for clusters, connect it to the signal:
                # if name is 'PFParticle':
                    # self.noiseFilterChanged.connect(
                    #     drawingClass.setParamsDrawing)
                    # drawingClass.setParamsDrawing(self._drawParams)
                # if name == 'Match':
                #     self.noiseFilterChanged.connect(
                #         drawingClass.setParamsDrawing)
                #     drawingClass.setParamsDrawing(self._drawParams)

                drawingClass.setProducer(producer)
                self._processer.add_process(product, drawingClass._process)
                self._drawnClasses.update({name: drawingClass})
                if name == "MCTrack":
                    self._drawnClasses[name].toggleMCCosmic(self.showMCCosmic)
                # Need to process the event
                self.processEvent(True)
                drawingClass.drawObjects(self._view_manager)

        def clearAll(self):
            for recoProduct in self._drawnClasses:
                self._drawnClasses[recoProduct].clearDrawnObjects(
                    self._view_manager)

        # def toggleParams(self, paramsBool):
        #     self._drawParams=paramsBool
        #     self.noiseFilterChanged.emit(paramsBool)
        #     if 'PFParticle' in self._drawnClasses:
        #         self.drawFresh()

        def drawFresh(self):
            # # wires are special:
            # if self._drawWires:
            #   self._view_manager.drawPlanes(self)
            self.clearAll()
            # Draw objects in a specific order defined by drawableItems
            order=self._drawableItems.getListOfTitles()
            for item in order:
                if item in self._drawnClasses:
                    self._drawnClasses[item].drawObjects(self._view_manager)

        def toggleMCCosmic(self, toggleBool):
            self.showMCCosmic = toggleBool
            order=self._drawableItems.getListOfTitles()
            for item in order:
                if item == "MCTrack":
                    if item in self._drawnClasses:
                        self._drawnClasses[item].toggleMCCosmic(toggleBool)
                        self._drawnClasses[item].clearDrawnObjects(self._view_manager)
                        self.processEvent(True)
                        self._drawnClasses[item].drawObjects(self._view_manager)
            #self.drawFresh()

except:
    pass
