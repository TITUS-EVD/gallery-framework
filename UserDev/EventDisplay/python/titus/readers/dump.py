"""
Product dumps: a few events' products saved as numpy arrays, written on a
gpvm by tools/dump_products.py and read back without ROOT, for tests and
offline development.

Layout of a dump directory:
    index.json          schema version, source metadata, and per event its
                        ID, file and the {kind: [tags]} it holds
    event_0000.npz      arrays named '<kind>/<tag>/<column>'
"""

import json
import pathlib

import numpy as np

from .base import EventReader, ProductNotFound
from .products import KINDS, EventID

SCHEMA_VERSION = 1
INDEX = 'index.json'


def _key(kind, tag, column):
    return f'{kind}/{tag}/{column}'


class DumpWriter:

    def __init__(self, directory, meta=None):
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta or {})
        self._events = []

    def add_event(self, event_id, products):
        """ products: {(kind, full tag): container} """
        filename = f'event_{len(self._events):04d}.npz'
        arrays, listing = {}, {}
        for (kind, tag), container in products.items():
            if not isinstance(container, KINDS[kind].container):
                raise TypeError(f'{kind} {tag}: expected {KINDS[kind].container.__name__}, '
                                f'got {type(container).__name__}')
            listing.setdefault(kind, []).append(tag)
            for column, array in container.to_arrays().items():
                arrays[_key(kind, tag, column)] = array
        np.savez_compressed(self.directory / filename, **arrays)
        self._events.append({'id': list(map(int, event_id)), 'file': filename, 'products': listing})

    def close(self):
        index = {'schema_version': SCHEMA_VERSION, 'meta': self.meta, 'events': self._events}
        with open(self.directory / INDEX, 'w') as f:
            json.dump(index, f, indent=1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DumpReader(EventReader):

    def __init__(self, directory):
        self.directory = pathlib.Path(directory)
        with open(self.directory / INDEX) as f:
            index = json.load(f)
        if index.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f'unsupported product dump schema version {index.get("schema_version")}')
        self._meta = index['meta']
        self._entries = index['events']
        self._events = [EventID(*e['id']) for e in self._entries]
        self._loaded = (None, None)

    @property
    def events(self):
        return self._events

    @property
    def meta(self):
        return self._meta

    def tags(self, kind, entry=None):
        entries = self._entries if entry is None else [self._entries[entry]]
        found = []
        for e in entries:
            found += [t for t in e['products'].get(kind, []) if t not in found]
        return found

    def _arrays(self, entry):
        """ the npz of one event, kept open for the most recent entry """
        if self._loaded[0] != entry:
            self._loaded = (entry, np.load(self.directory / self._entries[entry]['file']))
        return self._loaded[1]

    def _read(self, entry, kind, tag):
        arrays = self._arrays(entry)
        prefix = _key(kind, tag, '')
        columns = {k[len(prefix):]: arrays[k] for k in arrays.files if k.startswith(prefix)}
        if not columns:
            raise ProductNotFound(f'{kind} {tag} listed but not stored in {self._entries[entry]["file"]}')
        return KINDS[kind].container.from_arrays(columns)
