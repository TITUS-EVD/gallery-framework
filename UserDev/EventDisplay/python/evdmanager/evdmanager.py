import os

from pyqtgraph.Qt import QtCore

from ROOT import gallery
from ROOT import TFile
from ROOT import vector as ROOTvector
from ROOT import string as ROOTstring

# from memory_profiler import profile

try:
  from event import manager, event
except ImportError:
  from evdmanager.event import manager, event

import datatypes


class product(object):
    def __init__(self, name, typeName):
        self._name=name.rstrip(".")
        self._typeName=typeName
        self._isAssociation=False
        self._associatedProduct=None
        self._producer=None
        self._instance=None
        self._stage=None

        self.parse()

    def append_producer(self, s):
        self._producer += s

    def producer(self):
        return self._producer

    def name(self):
        return self._name

    def fullName(self):
        return "{}:{}:{}".format(self._producer, self._instance, self._stage)

    def typeName(self):
        return self._typeName

    def isAssociation(self):
        return self._isAssociation

    def stage(self):
        return self._stage

    # def associationProduct(self):
        # pass

    # def reverseAssociationProduct(self):
        # pass

    def parse(self):
        tokens=self._name.split('_')
        # Name goes as object_producer_stage
        self._producer=tokens[-3]
        self._instance=tokens[-2]
        self._stage=tokens[-1]
        self._typeName = tokens[0].rstrip('s')

        return


class processer(object):

    def __init__(self):

        # Storing ana units as a map:
        # self._ana_units[data product] -> instance of ana_unit
        self._ana_units = dict()
        pass

    # @profile
    def process_event(self, gallery_event):
        # print "Running ... "
        for key in self._ana_units:
            # print('Processing', key)
            self._ana_units[key].analyze(gallery_event)
            # print('Size of anaunit after processing', asizeof.asized(self._ana_units[key], detail=2).format())

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

    def get_n_processes(self):
        return len(self._ana_units)

    def remove_all_processes(self):
        for data_product in self._ana_units.copy():
            self._ana_units.pop(data_product)

    def reset(self):
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
        self._drawOpDetWvf = False
        # self._drawParams = False
        # self._drawTruth = False

        self._wireDrawer = None
        self._opDetWvfDrawer = None
        # self._truthDrawer = None

        # A list that will contain a dictionary with run, subrun, event keys
        self._run_list = []


    def getAvailableRuns(self):
        '''
        Getter for the available runs

        Returns:
            list: A list of all available runs
        '''
        out = []
        for item in self._run_list:
            if item['run'] in out:
                continue
            else:
                out.append(item['run'])
        return out

    def getAvailableSubruns(self):
        '''
        Getter for the available subruns

        Returns:
            list: A list of all available subruns
        '''
        out = []
        for item in self._run_list:
            if item['subrun'] in out:
                continue
            else:
                out.append(item['subrun'])
        return out

    def getAvailableEvents(self):
        '''
        Getter for the available events

        Returns:
            list: A list of all available events
        '''
        out = []
        for item in self._run_list:
            if item['event'] in out:
                continue
            else:
                out.append(item['event'])
        return out


    def pingFile(self, file):
        """
        this function opens the file and
        determines what is available to draw
        """
        # This function opens the file to see
        # what data products are available

        # Open the file
        f = TFile(file)
        e = f.Get("Events")


        # Get all the (run, subrun, event) IDs
        self._run_list = []
        ev_aux_b = e.GetBranch("EventAuxiliary")
        for i in range(ev_aux_b.GetEntries()):
            ev_aux_b.GetEntry(i)
            ev_aux = e.EventAuxiliary
            self._run_list.append({
                    'run': ev_aux.run(),
                    'subrun': ev_aux.subRun(),
                    'event': ev_aux.event(),
                })


        # prepare a dictionary of data products
        lookUpTable = dict()
        lookUpTable.update({"all" : dict()})

        product_list = []
        # Loop over the keys (list of trees)
        for key in e.GetListOfBranches():

            if key.GetTypeName() == 'art::EventAuxiliary':
                continue

            if "NuMu" in key.GetName() and "Assns" in key.GetTypeName():
                if "PFParticle" in key.GetTypeName():
                    continue
            elif "Assns" in key.GetTypeName():
                continue

            prod=product(key.GetName(), key.GetTypeName())

            # if "NuMu" in key.GetName():
            #     print "NuMu stage is " + str(prod.stage())
            #     print "NuMu name is " +  str(prod.fullName())
            #     print "NuMu type name is " + str(prod.typeName())
            _product = prod._typeName





            # Add the product to the "all" list and
            # also to it's stage list:

            # gets three items in thisKeyList, which is a list
            # [dataProduct, producer, 'tree'] (don't care about 'tree')
            # check if the data product is in the dict:
            if _product in lookUpTable['all']:
                # extend the list:
                lookUpTable['all'][_product] += (prod, )
            else:
                lookUpTable['all'].update({_product: (prod,)})


            if not (prod.stage() in lookUpTable):
                lookUpTable.update({prod.stage() : dict()})
            if _product in lookUpTable[prod.stage()]:
                # extend the list:
                lookUpTable[prod.stage()][_product] += (prod, )
            else:
                lookUpTable[prod.stage()].update({_product: (prod,)})


        self._keyTable.update(lookUpTable)

        f.Close()

        return

    def setInputFile(self, file):
        f = [file, ]
        self.setInputFiles(f)

    # @profile
    def setInputFiles(self, files):

        # reset the storage manager and process
        if self._data_manager is not None:
            del self._data_manager
            self._data_manager = None

        if files == None:
            return

        _file_list = ROOTvector(ROOTstring)()

        for file in files:
            # First, check that the file exists:
            try:
                if not os.path.exists(file):
                    print("\033[91m ERROR: requested file does not exist. \033[0m")
                    continue
            except (Exception, e):
                print(e)
                return
            # Next, verify it is a root file:
            if not file.endswith(".root"):
                print("\033[91m ERROR: must supply a root file. \033[0m")
                continue

            # Finally, ping the file to see what is available to draw
            self.pingFile(file)
            if len(self._keyTable['all']) > 0:
                self._hasFile = True
                _file_list.push_back(file)


        # Have to figure out number of events available
        for _f in _file_list:
            _rf = TFile(str(_f))
            _tree = _rf.Get("Events")
            self._n_entries += _tree.GetEntries()
            _rf.Close()


        # Create an instance of the data manager:
        if _file_list.size() > 0:
            self._data_manager = gallery.Event(_file_list)

        # Open the manager
        self._lastProcessed = -1

        self.goToEvent(0)

        self.fileChanged.emit()


    def getStages(self):
        return self._keyTable.keys()

    # This function will return all producers for the given product
    def getProducers(self, product, stage = None):
        try:
            if stage is not None:
                return self._keyTable[stage][product]
            else:
                return self._keyTable["all"][product]
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
            print("On the last event, can't go to next.")

    def prev(self):
        if self._event != 0:
            self.goToEvent(self._event - 1)
        else:
            print("On the first event, can't go to previous.")

    def processEvent(self, force=False):
        if self._lastProcessed != self._event or force:
            self._processer.process_event(self._data_manager)
            self._lastProcessed = self._event

    def goToEvent(self, event, subrun=None, run=None, force=False):
        # Gallery events don't offer random access

        # if rubrun and run are specified, then we are dealing with real (event, subrun, run)
        # not an event index as usual. So first of all go from (event, subrun, run) to event index
        if subrun is not None and run is not None:
            try:
                item = {'run': run, 'subrun': subrun, 'event': event}
            except:
                print('This combination does not exist:', item)
                return
            event = self._run_list.index(item)


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
            print(f"Selected event is too high. You have requested event {event}, but there is a maximum of {self._n_entries}.")
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

        elif product == 'channelroi':
            if 'recob::ChannelROI' not in self._keyTable[stage]:
                print("No wire data available to draw")
                self._drawWires = False
                return
            self._drawWires = True
            self._wireDrawer = datatypes.recoChannelROI(self._geom)

            if producers is not None:
                producer = producers
            elif self._geom.name() == 'icarus' and len(self._keyTable[stage]['recob::ChannelROI']) > 3:
                producer = [self._keyTable[stage]['recob::ChannelROI'][0].fullName(),
                            self._keyTable[stage]['recob::ChannelROI'][1].fullName(),
                            self._keyTable[stage]['recob::ChannelROI'][2].fullName(),
                            self._keyTable[stage]['recob::ChannelROI'][3].fullName()]
            else:
                producer = self._keyTable[stage]['recob::ChannelROI'][0].fullName()

            # self._wireDrawer.setProducer(self._keyTable[stage]['recob::Wire'][0].fullName())
            self._wireDrawer.setProducer(producer)
            self._processer.add_process("recob::ChannelROI",self._wireDrawer._process)
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

            self._wireDrawer.setProducer(producer)
            self._processer.add_process("raw::RawDigit", self._wireDrawer._process)
            self._wireDrawer.toggleNoiseFilter(self.filterNoise)

            self.processEvent(True)
        else:
            if 'raw::RawDigit' in self._processer._ana_units.keys():
                self._processer.remove_process('raw::RawDigit')
            if 'recob::Wire' in self._processer._ana_units.keys():
                self._processer.remove_process('recob::Wire')
            if 'recob::ChannelROI' in self._processer._ana_units.keys():
                self._processer.remove_process('recob::ChannelROI')
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

    def drawHitsOnWire(self, plane, wire, tpc):
        if not 'Hit' in self._drawnClasses:
            return
        else:
            # Get the right plane number
            this_plane = plane
            if tpc == 1:
                this_plane = self._geom.planeMix()[plane][0]

            # Get the hits:
            hits = self._drawnClasses['Hit'].getHitsOnWire(this_plane, wire)
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
        def redrawProduct(self, name, product, producer, view_manager, stage = None):
            # print "Received request to redraw ", product, " by ",producer, " with name ", name
            # First, determine if there is a drawing process for this product:
            if stage is None:
                stage = 'all'
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
