#!/usr/bin/env python3
"""
Check the compiled reader (titus.readers.GalleryReader + libEventDisplay_Reader)
against the PyROOT extraction of dump_products.py on the same events, and
time both. Run on a gpvm after building gallery-framework, from this directory:

    python check_gallery_reader.py reco1.root --events 2
    python check_gallery_reader.py reco1.root --raw-digits --tags daq

Exits non-zero if any product differs.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import dump_products  # noqa: E402
from titus.readers import GalleryReader  # noqa: E402
from titus.readers.gallery import EXTRACTORS  # noqa: E402
from titus.readers.products import ASSOCIATIONS, association_is_swapped  # noqa: E402


def compare(kind, expected, actual):
    """ list of differing columns """
    differences = []
    for column, want in expected.to_arrays().items():
        got = getattr(actual, column)
        if want.shape != got.shape or not np.array_equal(want, got):
            differences.append(f'{column}: shape {want.shape} vs {got.shape}')
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file')
    parser.add_argument('--events', type=int, default=2)
    parser.add_argument('--raw-digits', action='store_true')
    parser.add_argument('--tags', help='comma-separated module labels (default: all)')
    args = parser.parse_args()

    start = time.perf_counter()
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.ProcessLine('gErrorIgnoreLevel = kError;')
    print(f'import ROOT: {time.perf_counter() - start:.2f} s')

    start = time.perf_counter()
    reader = GalleryReader([args.file], root=ROOT)
    print(f'GalleryReader({args.file}): {len(reader.events)} events, {time.perf_counter() - start:.2f} s')
    print(f'ChannelROI support: {bool(ROOT.titusio.hasChannelROISupport())}')

    # the kinds the titusio library reads so far
    kinds = [k for k in dump_products.DEFAULT_KINDS if k in EXTRACTORS]
    kinds += ['raw_digits'] if args.raw_digits else []
    labels = set(args.tags.split(',')) if args.tags else None
    products = [(k, t) for k in kinds for t in reader.tags(k)
                if labels is None or t.split(':')[0] in labels]
    extractors = {
        'hits': dump_products.extract_hits,
        'wires': dump_products.extract_wire_signals,
        'channel_rois': dump_products.extract_wire_signals,
        'raw_digits': lambda v: dump_products.extract_raw_digits(v, dump_products.make_uncompress(ROOT)),
        'tracks': dump_products.extract_tracks,
        'vertices': dump_products.extract_vertices,
        'spacepoints': dump_products.extract_spacepoints,
        'showers': dump_products.extract_showers,
        'pfparticles': dump_products.extract_pfparticles,
        't0s': dump_products.extract_t0s,
    }
    # C++ product type of each (kind, tag) in the file
    types = reader._file_types[0]

    def product_ids(ev):
        """ ProductID value -> tag of every non-association product in the event """
        ids = {}
        for (k, t), cpp_type in types.items():
            if k in ASSOCIATIONS:
                continue
            try:
                ids[ev.getValidHandle[cpp_type](ROOT.art.InputTag(t)).id().value()] = t
            except Exception:
                pass
        return ids

    failures = 0
    for entry in range(min(args.events, len(reader.events))):
        print(f'Event {reader.events[entry]}:')
        ids = None
        for kind, tag in products:
            try:
                start = time.perf_counter()
                actual = reader.get(entry, kind, tag)
                t_compiled = time.perf_counter() - start

                start = time.perf_counter()
                ev = reader.event_handle(entry)
                cpp_type = types[(kind, tag)]
                handle = ev.getValidHandle[cpp_type](ROOT.art.InputTag(tag))
                if kind in ASSOCIATIONS:
                    ids = product_ids(ev) if ids is None else ids
                    expected = dump_products.extract_associations(
                        handle.product(), lambda pid: ids.get(pid.value(), ''),
                        has_meta='TrackHitMeta' in cpp_type, swapped=association_is_swapped(kind, cpp_type))
                else:
                    expected = extractors[kind](handle.product())
                t_pyroot = time.perf_counter() - start
            except Exception as e:
                failures += 1
                print(f'  {kind:13s} {tag:40s} FAILED {type(e).__name__}: {e}')
                continue
            differences = compare(kind, expected, actual)
            status = 'OK' if not differences else 'DIFFERS ' + '; '.join(differences)
            failures += bool(differences)
            print(f'  {kind:13s} {tag:40s} {len(actual):7d} rows  compiled {t_compiled:6.3f} s  '
                  f'pyroot {t_pyroot:6.2f} s  {status}')

    print('all products match' if not failures else f'{failures} products failed or differ')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
