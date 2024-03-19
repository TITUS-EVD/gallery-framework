import os
import sys

from PyQt5 import QtCore

from ROOT import gallery
from ROOT import TFile
from ROOT import vector as ROOTvector
from ROOT import string as ROOTstring


class Product:
    """ holds information about art products """
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

    def full_name(self):
        return "{}:{}:{}".format(self._producer, self._instance, self._stage)

    def type_name(self):
        return self._typeName

    def is_association(self):
        return self._isAssociation

    def stage(self):
        return self._stage

    def parse(self):
        tokens=self._name.split('_')
        # Name goes as object_producer_stage
        self._producer=tokens[-3]
        self._instance=tokens[-2]
        self._stage=tokens[-1]
        self._typeName = tokens[0].rstrip('s')


class GalleryInterface(QtCore.QObject):
    fileChanged = QtCore.pyqtSignal()
    dirChanged = QtCore.pyqtSignal()
    eventChanged = QtCore.pyqtSignal()

    def __init__(self, file=None):
        super().__init__()

        self._gallery_event_handle = None

        self._keyTable = {}
        self._drawnClasses = {}
        self._lastProcessed = -1

        self._n_entries = 0

        # A list that will contain a dictionary with run, subrun, event keys
        self._run_list = []

        self._run = 0
        self._subrun = 0
        self._entry = 0

        if file != None:
            self.setInputFiles(file)

        self._current_file = ''
        if file is not None:
            self._current_file = file
        self._current_directory = os.getcwd()

    def available_runs(self):
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

    def available_subruns(self):
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

    def available_events(self):
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

    def ping_file(self, file):
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

            prod=Product(key.GetName(), key.GetTypeName())

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

    def set_input_file(self, file):
        f = [file, ]
        self.set_input_files(f)

    @property
    def current_file(self):
        return self._current_file

    @property
    def current_directory(self):
        return self._current_directory

    @current_directory.setter
    def current_directory(self, dirpath):
        self._current_directory = dirpath
        self.dirChanged.emit()

    def set_input_files(self, files):
        # reset the storage manager and process
        if self._gallery_event_handle is not None:
            del self._gallery_event_handle
            self._gallery_event_handle = None

        if files == None:
            return

        _file_list = ROOTvector(ROOTstring)()

        for file in files:
            # First, check that the file exists:
            try:
                if not os.path.exists(file):
                    print(f"\033[91m WARNING\033[0m requested file {file} does not exist, skipping")
                    continue
            except (Exception, e):
                print(e)
                return
            # Next, verify it is a root file:
            if not file.endswith(".root"):
                print(f"\033[91m WARNING\033[0m {file} is not a .root file, skipping")
                continue

            # Finally, ping the file to see what is available to draw
            try:
                self.ping_file(file)
            except (OSError, AttributeError):
                print(f"\033[91m WARNING\033[0m Could not open {file}, skipping")
                continue

            if len(self._keyTable['all']) > 0:
                self._hasFile = True
                _file_list.push_back(file)


        # Have to figure out number of events available
        self._n_entries = 0
        for _f in _file_list:
            _rf = TFile(str(_f))
            _tree = _rf.Get("Events")
            self._n_entries += _tree.GetEntries()
            _rf.Close()


        # Create an instance of the data manager:
        if _file_list.size() == 0:
            return

        # TODO gallery supports vector of files chained together, but for now
        # we only consider one file opened at a time
        self._current_file = str(_file_list[0])
        self._current_directory = os.path.dirname(self._current_file)

        self._gallery_event_handle = gallery.Event(_file_list)

        self._lastProcessed = -1
        self._entry = 0

        # crucial to emit file change before setting event, to give modules
        # a heads-up that they should expect different products in the file
        # before processing the new event
        self.fileChanged.emit()
        self.go_to_event(0)


    def get_stages(self):
        return self._keyTable.keys()

    def get_products(self, name, stage=None):
        '''
        Returns all available products with name at stage
        '''
        if stage is None:
            stage = 'all'

        if stage not in self._keyTable:
            return None

        if name not in self._keyTable[stage]:
            return None

        return self._keyTable[stage][name]

    def get_all_products(self):
        ''' Returns all available products '''
        return self._keyTable

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

        # TODO Remove dependence of geom from gallery_interface
        '''
        if self._geom.name() == 'icarus' and len(self._keyTable[stage][name]) > 3:
            default_products = [self._keyTable[stage][name][0],
                                self._keyTable[stage][name][1],
                                self._keyTable[stage][name][2],
                                self._keyTable[stage][name][3]]
        else:
            default_products = [self._keyTable[stage][name][0]]
        '''
        default_products = self._keyTable[stage][name]

        return default_products

    def get_producers(self, product, stage = None):
        ''' This function will return all producers for the given product '''
        try:
            if stage is not None:
                return self._keyTable[stage][product]
            else:
                return self._keyTable["all"][product]
        except:
            return None

    # This function returns the list of products that can be drawn:
    def getDrawableProducts(self):
        # TODO
        return {}
        # return self._drawableItems.getDict()

    # override the run,event,subrun functions:
    def run(self):
        if self._gallery_event_handle is None:
            return 0
        return self._gallery_event_handle.eventAuxiliary().run()

    def event(self):
        if self._gallery_event_handle is None:
            return 0
        return self._gallery_event_handle.eventAuxiliary().event()

    def event_handle(self):
        return self._gallery_event_handle

    def subrun(self):
        if self._gallery_event_handle is None:
            return 0
        return self._gallery_event_handle.eventAuxiliary().subRun()

    # override the functions from manager as needed here
    def next(self):
        # print "Called next"
        # Check that this isn't the last event:
        if self._entry < self._n_entries - 1:
            self.go_to_event(self._entry + 1)
        else:
            print("On the last event, can't go to next.")

    def prev(self):
        if self._entry != 0:
            self.go_to_event(self._entry - 1)
        else:
            print("On the first event, can't go to previous.")

    def process_event(self, force=False):
        if self._lastProcessed != self._entry or force:
            self._lastProcessed = self._entry
            self.eventChanged.emit()

    def go_to_event(self, event, subrun=None, run=None, force=False):
        '''
        Gallery events don't offer random access
        if rubrun and run are specified, then we are dealing with real (event,
        subrun, run) not an event index as usual. So first of all go from
        (event, subrun, run) to event index
        '''
        if self._gallery_event_handle is None:
            return
        if subrun is not None and run is not None:
            try:
                item = {'run': run, 'subrun': subrun, 'event': event}
            except:
                print('This combination does not exist:', item)
                return
            event = self._run_list.index(item)


        # Loop through until the event is gotten:
        if event < self._n_entries:
            if event == self._entry + 1:
                self._gallery_event_handle.next()

            else:
                if event > self._entry:
                    while event != self._gallery_event_handle.eventEntry():
                        self._gallery_event_handle.next()
                else:
                    self._gallery_event_handle.toBegin()
                    while event != self._gallery_event_handle.eventEntry():
                        self._gallery_event_handle.next()
        else:
            print(f"Selected event is too high. You have requested event {event}, but there is a maximum of {self._n_entries}.")
            return

        self._entry = self._gallery_event_handle.eventEntry()
        self.process_event()
