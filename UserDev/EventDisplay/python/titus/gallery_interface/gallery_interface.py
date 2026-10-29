import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from PyQt5 import QtCore

from titus.readers.base import ProductCache


def _root():
    import ROOT
    return ROOT


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


class OpenError(RuntimeError):
    pass


@dataclass
class OpenedFiles:
    ''' what opening files found: the gallery.Event, its events and products '''
    files: List[str]
    event: object  # gallery.Event, at its first entry
    run_list: List[Dict[str, int]]
    key_table: Dict[str, Dict[str, tuple]]
    warnings: List[str] = field(default_factory=list)
    # the std::vector of file names given to gallery, kept with the event
    file_names: object = None
    # seconds per step: 'TFile' (all files), 'gallery'
    timings: Dict[str, float] = field(default_factory=dict)

    def describe_timings(self):
        return ', '.join(f'{step} {seconds:.1f} s' for step, seconds in self.timings.items())


def readable_files(files):
    ''' (the files worth opening, warnings about the others) '''
    keep, warnings = [], []
    for file in files or []:
        if not os.path.exists(file) and 'root://' not in file:
            warnings.append(f'requested file {file} does not exist, skipping')
        elif not file.endswith('.root'):
            warnings.append(f'{file} is not a .root file, skipping')
        else:
            keep.append(file)
    return keep, warnings


def product_table(branches):
    '''
    {'all' and each stage: {product type: (Product, ...)}} from the
    (name, type name) of a file's Events branches, as the product boxes
    list them. Leaves out the event auxiliary and associations.
    '''
    table = {'all': {}}
    for name, type_name in branches:
        if type_name == 'art::EventAuxiliary':
            continue
        # associations are left out, except the NuMu selection's that aren't to PFParticles
        if 'Assns' in type_name and not ('NuMu' in name and 'PFParticle' not in type_name):
            continue
        try:
            product = Product(name, type_name)
        except IndexError:
            continue  # not a product branch
        for stage in ('all', product.stage()):
            table.setdefault(stage, {}).setdefault(product._typeName, ())
            table[stage][product._typeName] += (product,)
    return table


def _merge_tables(into, table):
    for stage, products in table.items():
        for type_name, found in products.items():
            into.setdefault(stage, {}).setdefault(type_name, ())
            into[stage][type_name] += found


def ping_file(root, file, timings=None):
    '''
    (run list, product table) of a file, read with ROOT's TFile before gallery
    opens it, as TITUS always did: gallery's constructor crashed in the GUI
    when it was the first to open the file (it reads the file's dictionaries).
    timings: {step: seconds} to add this file's steps to.
    '''
    timings = {} if timings is None else timings
    clock = [time.perf_counter()]

    def lap(step):
        now = time.perf_counter()
        timings[step] = timings.get(step, 0.) + now - clock[0]
        clock[0] = now

    f = root.TFile.Open(file)
    if not f or f.IsZombie():
        raise OSError(f'could not open {file}')
    try:
        tree = f.Get('Events')
        lap('TFile.Open')
        aux_branch = tree.GetBranch('EventAuxiliary')
        run_list = []
        n_entries = aux_branch.GetEntries()
        for i in range(n_entries):
            aux_branch.GetEntry(i)
            aux = tree.EventAuxiliary
            run_list.append({'run': aux.run(), 'subrun': aux.subRun(), 'event': aux.event()})
            if i == 0:
                # its dictionary loaded, and PyROOT's cast compiled
                lap('first EventAuxiliary')
        lap(f'run list ({n_entries} events)' if n_entries > 1 else 'run list')
        table = product_table([(b.GetName(), b.GetTypeName()) for b in tree.GetListOfBranches()])
        lap('branches')
    finally:
        f.Close()
    return run_list, table


def scan_files(files, root=None):
    '''
    Open art-root files: each file's (run, subrun, event)s and products read
    with TFile (ping_file), then the gallery.Event to read them, with the
    time each step took. Raises OpenError.
    '''
    root = root or _root()
    start = time.perf_counter()
    files, warnings = readable_files(files)
    if not files:
        raise OpenError('; '.join(warnings) or 'no file given')
    run_list, table, opened, timings = [], {'all': {}}, [], {}
    for file in files:
        try:
            file_runs, file_table = ping_file(root, file, timings)
        except (OSError, AttributeError, ReferenceError) as e:
            warnings.append(f'could not open {file} ({e}), skipping')
            continue
        if not file_table['all']:
            warnings.append(f'no products in {file}, skipping')
            continue
        run_list += file_runs
        _merge_tables(table, file_table)
        opened.append(file)
    if not opened:
        raise OpenError('; '.join(warnings))
    if not run_list:
        raise OpenError(f'no events in {", ".join(opened)}')
    timings['TFile total'] = time.perf_counter() - start
    start = time.perf_counter()
    names = root.std.vector['string']()
    for file in opened:
        names.push_back(file)
    try:
        ev = root.gallery.Event(names)
    except Exception as e:
        raise OpenError(f'gallery could not open {", ".join(opened)}: {e}') from None
    timings['gallery'] = time.perf_counter() - start
    return OpenedFiles(files=opened, event=ev, run_list=run_list, key_table=table, warnings=warnings,
                       file_names=names, timings=timings)


def position_event(ev, entry):
    '''
    Move a gallery.Event to an entry: directly with goToEntry where gallery
    has it, else stepping forward (from the start, to go back)
    '''
    if ev.atEnd():
        # past the end there is no current file, and goToEntry throws
        ev.toBegin()
    if ev.eventEntry() == entry:
        return
    if hasattr(ev, 'goToEntry'):
        ev.goToEntry(entry)
        return
    if ev.eventEntry() > entry:
        ev.toBegin()
    while ev.eventEntry() != entry:
        ev.next()


class GalleryInterface(QtCore.QObject):
    fileChanged = QtCore.pyqtSignal()
    dirChanged = QtCore.pyqtSignal()
    eventChanged = QtCore.pyqtSignal()

    def __init__(self, file=None, scan=None):
        """ scan: replaces scan_files (tests) """
        super().__init__()
        self._scan = scan or scan_files

        self._gallery_event_handle = None
        # products of the current event, shared by the drawables
        self.product_cache = ProductCache()

        self._keyTable = {}
        self._drawnClasses = {}
        self._lastProcessed = -1

        self._n_entries = 0
        self._file_entries = {}

        # A list that will contain a dictionary with run, subrun, event keys
        self._run_list = []

        self._run = 0
        self._subrun = 0
        self._entry = 0

        self._current_file = ''
        self._current_directory = os.getcwd()

        if file is not None:
            self.set_input_files([file] if isinstance(file, str) else file)

    def available_runs(self):
        '''
        Getter for the available runs

        Returns:
            list: A list of all available runs
        '''
        return list(dict.fromkeys(item['run'] for item in self._run_list))

    def available_subruns(self):
        '''
        Getter for the available subruns

        Returns:
            list: A list of all available subruns
        '''
        return list(dict.fromkeys(item['subrun'] for item in self._run_list))

    def available_events(self):
        '''
        Getter for the available events

        Returns:
            list: A list of all available events
        '''
        return list(dict.fromkeys(item['event'] for item in self._run_list))

    def set_input_file(self, file) -> bool:
        f = [file, ]
        return self.set_input_files(f)

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

    def set_input_files(self, files) -> bool:
        """
        Open files: False if none could be opened, and the current file then
        stays open. Prints how long each step took.
        """
        try:
            result = self._scan(files)
        except OpenError as e:
            print(f"\033[91m WARNING\033[0m {e}")
            return False
        if result.timings:
            print(f'Opened {", ".join(result.files)}: {result.describe_timings()}')
        self._install(result)
        return True

    def _install(self, result):
        """ make opened files current """
        for warning in result.warnings:
            print(f"\033[91m WARNING\033[0m {warning}")
        self._gallery_event_handle = result.event
        self._run_list = result.run_list
        self._keyTable = result.key_table
        self._n_entries = len(result.run_list)
        self._file_entries = {}
        self._hasFile = True

        # TODO gallery supports vector of files chained together, but for now
        # we only consider one file opened at a time
        self._current_file = result.files[0]
        self._current_directory = os.path.dirname(self._current_file)
        # if the file has no dirname, assume it is meant to be in the current directory
        if self._current_directory == '':
            self._current_directory = os.getcwd()

        self.product_cache.new_file()
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
        if self.event_handle() is None:
            return 0
        return self.event_handle().eventAuxiliary().run()

    def event(self, qevent=None):
        '''
        The current art event number. Called by Qt with a QEvent (this
        overrides QObject.event), e.g. to deliver a queued signal: that is
        handed to QObject.
        '''
        if qevent is not None:
            return super().event(qevent)
        if self.event_handle() is None:
            return 0
        return self.event_handle().eventAuxiliary().event()

    def event_handle(self):
        return self._gallery_event_handle

    def subrun(self):
        if self.event_handle() is None:
            return 0
        return self.event_handle().eventAuxiliary().subRun()

    def datetime(self):
        if self.event_handle() is None:
            return None
        ts = self.event_handle().eventAuxiliary().time().value()

        # Get the timestamp: https://github.com/LArSoft/lareventdisplay/blob/6cbc95e04f8b8fd1639d14f7ef0ad37841962027/lareventdisplay/EventDisplay/HeaderDrawer.cxx#L41
        mask32 = 0xFFFFFFFF
        timestamp_s = (ts >> 32) & mask32
        timestamp_us = ts & mask32

        return datetime.fromtimestamp(timestamp_s, timezone.utc)

    def date(self, include_time=True):
        dt = self.datetime()
        if dt is None:
            return 'No event selected'

        if include_time:
            return dt.strftime("%B %d, %Y - %H:%M:%S UTC")

        return dt.strftime("%B %d, %Y")

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
            print('No _gallery_event_handle')
            return
        if subrun is not None and run is not None:
            item = {'run': run, 'subrun': subrun, 'event': event}
            try:
                event = self._run_list.index(item)
            except ValueError:
                print('Tried to go to invalid run, subrun, event: ', item)
                return
            print(f'Found event {event}')


        if event >= self._n_entries:
            print(f"Selected event is too high. You have requested event {event}, but there is a maximum of {self._n_entries}.")
            return
        position_event(self._gallery_event_handle, event)

        entry = self._gallery_event_handle.eventEntry()
        if entry != self._entry or self._lastProcessed == -1:
            self.product_cache.new_event()
        self._entry = entry
        self.process_event()
