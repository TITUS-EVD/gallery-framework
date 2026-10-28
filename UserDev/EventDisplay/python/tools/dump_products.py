#!/usr/bin/env python3
"""
Dump the products TITUS draws from a few events of an art-root file into a
product dump (titus/readers/dump.py), readable without ROOT for tests and
offline development.

Needs ROOT, gallery and the product dictionaries only, no LArSoft services.
Run inside the SBN environment (gpvm), from this directory:

    python dump_products.py reco1.root --detector sbnd --outdir dumps/sbnd-reco1
    python dump_products.py reco1.root --detector sbnd --events 2 --raw-digits
    python dump_products.py reco1.root --detector icarus --tags gaushit,roifinder
    python dump_products.py reco2.root --detector sbnd --outdir dumps/sbnd-reco2

By default every product TITUS reads (titus.readers.KINDS) found in the file
is dumped: hits, wires, channel ROIs, tracks with their hit associations,
vertices, spacepoints, showers, and the PFParticles and T0s (with their
track associations) to undo Pandora's T0 shift. Raw digits (large) only with
--raw-digits. Failures are recorded in the dump's index.json and listed at
the end instead of aborting.
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dump_detector import Errors  # noqa: E402
from titus.readers import (ASSOCIATIONS, KINDS, Associations, DumpWriter, Hits, PFParticles,  # noqa: E402
                           Points, RawDigits, Showers, T0s, Tracks, WireSignals)
from titus.readers.products import association_is_swapped, parse_branch  # noqa: E402

# kinds read through associations, after the products they refer to
ASSOCIATION_KINDS = tuple(ASSOCIATIONS)

DEFAULT_KINDS = ('hits', 'wires', 'channel_rois', 'tracks', 'vertices', 'spacepoints', 'showers',
                 'pfparticles', 't0s') + ASSOCIATION_KINDS

# raw::Compress_t
COMPRESSION_NONE = 0


# --- product -> container, on any objects with the LArSoft accessors ---

def extract_hits(hits):
    columns = {name: [] for name in Hits.DTYPES}
    for h in hits:
        wid = h.WireID()
        columns['channel'].append(h.Channel())
        columns['cryostat'].append(wid.Cryostat)
        columns['tpc'].append(wid.TPC)
        columns['plane'].append(wid.Plane)
        columns['wire'].append(wid.Wire)
        columns['peak_time'].append(h.PeakTime())
        columns['rms'].append(h.RMS())
        columns['integral'].append(h.Integral())
        columns['peak_amplitude'].append(h.PeakAmplitude())
        columns['start_tick'].append(h.StartTick())
        columns['end_tick'].append(h.EndTick())
    return Hits(**columns)


def extract_wire_signals(wires):
    """ recob::Wire and recob::ChannelROI: Channel() and a lar::sparse_vector SignalROI() """
    channels, n_ticks, rois = [], [], []
    for w in wires:
        signal = w.SignalROI()
        channels.append(w.Channel())
        n_ticks.append(signal.size())
        rois.append([(r.begin_index(), np.asarray(r.data(), dtype='f4')) for r in signal.get_ranges()])
    return WireSignals.from_rois(channels, n_ticks, rois)


def extract_raw_digits(digits, uncompress=None):
    """ uncompress(adcs, n_samples, compression) -> samples, for compressed digits """
    channels, pedestals, n_samples, adcs = [], [], [], []
    for d in digits:
        samples = d.Samples()
        if d.Compression() == COMPRESSION_NONE:
            adc = np.asarray(d.ADCs(), dtype='i2')
        elif uncompress is None:
            raise ValueError(f'channel {d.Channel()}: compressed raw digit and no uncompress function')
        else:
            adc = np.asarray(uncompress(d.ADCs(), samples, d.Compression()), dtype='i2')
        channels.append(d.Channel())
        pedestals.append(d.GetPedestal())
        n_samples.append(len(adc))
        adcs.append(adc)
    width = max(n_samples, default=0)
    padded = np.zeros((len(adcs), width), dtype='i2')
    for row, adc in enumerate(adcs):
        padded[row, :len(adc)] = adc
    return RawDigits(channel=channels, pedestal=pedestals, n_samples=n_samples, adc=padded)


def extract_tracks(tracks):
    ids, points, valid = [], [], []
    for t in tracks:
        n = t.NumberTrajectoryPoints()
        ids.append(t.ID())
        xyz = np.empty((n, 3), dtype='f4')
        ok = np.empty(n, dtype=bool)
        for i in range(n):
            p = t.LocationAtPoint(i)
            xyz[i] = (p.X(), p.Y(), p.Z())
            ok[i] = bool(t.HasValidPoint(i))
        points.append(xyz)
        valid.append(ok)
    return Tracks.from_points(ids, points, valid)


def _xyz(p):
    return p.X(), p.Y(), p.Z()


def extract_vertices(vertices):
    rows = [(v.ID(),) + _xyz(v.position()) for v in vertices]
    return _points(rows)


def _spacepoint_xyz(sp):
    """
    recob::SpacePoint position. XYZ() returns a Double32_t pointer, which
    PyROOT reads as integers holding the doubles' bits: prefer position(),
    else reinterpret them
    """
    try:
        return _xyz(sp.position())
    except AttributeError:
        xyz = np.array([sp.XYZ()[i] for i in range(3)])
        return tuple(xyz.astype('i8').view('f8') if xyz.dtype.kind in 'iu' else xyz)


def extract_spacepoints(spacepoints):
    return _points([(sp.ID(),) + tuple(_spacepoint_xyz(sp)) for sp in spacepoints])


def _points(rows):
    columns = list(zip(*rows)) if rows else [[], [], [], []]
    return Points(id=columns[0], x=columns[1], y=columns[2], z=columns[3])


def extract_showers(showers):
    columns = {name: [] for name in Showers.DTYPES}
    for s in showers:
        start, direction = s.ShowerStart(), s.Direction()
        columns['id'].append(s.ID())
        columns['start_x'].append(start.X())
        columns['start_y'].append(start.Y())
        columns['start_z'].append(start.Z())
        columns['dir_x'].append(direction.X())
        columns['dir_y'].append(direction.Y())
        columns['dir_z'].append(direction.Z())
        columns['length'].append(s.Length())
        columns['open_angle'].append(s.OpenAngle())
        columns['best_plane'].append(s.best_plane())
    return Showers(**columns)


def extract_pfparticles(pfparticles):
    rows = [(p.Self(), p.PdgCode(), p.Parent()) for p in pfparticles]
    columns = list(zip(*rows)) if rows else [[], [], []]
    # recob::PFParticle::kPFParticlePrimary is the largest size_t
    parent = [-1 if q > 2 ** 31 else q for q in columns[2]]
    return PFParticles(self_index=columns[0], pdg=columns[1], parent=parent)


def extract_t0s(t0s):
    rows = [(t.Time(), t.TriggerType(), t.ID()) for t in t0s]
    columns = list(zip(*rows)) if rows else [[], [], []]
    return T0s(time=columns[0], trigger_type=columns[1], id=columns[2])


def extract_associations(assns, tag_of_product, has_meta, swapped=False):
    """
    art::Assns via its pairs of art::Ptr. tag_of_product(ProductID) -> input
    tag of the product a Ptr points into ('' if unknown). has_meta: the
    association carries recob::TrackHitMeta. swapped: the product has the
    kind's two types in reverse order; left and right are swapped back.
    """
    tags = []

    def product_index(ptr):
        tag = tag_of_product(ptr.id())
        if tag not in tags:
            tags.append(tag)
        return tags.index(tag)

    columns = {name: [] for name in ('left', 'right', 'left_product', 'right_product', 'meta_index')}
    for i in range(assns.size()):
        left, right = assns[i].first, assns[i].second
        if swapped:
            left, right = right, left
        columns['left'].append(left.key())
        columns['right'].append(right.key())
        columns['left_product'].append(product_index(left))
        columns['right_product'].append(product_index(right))
        columns['meta_index'].append(assns.data(i).Index() if has_meta else -1)
    return Associations(product_tags=tags, **columns)


# --- ROOT side ---

def discover_tags(ROOT, path, kinds, labels=None):
    """ [(kind, full tag, C++ product type)] for product branches in the Events tree """
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise OSError(f'cannot open {path}')
    found, other_assns = [], []
    for branch in f.Get('Events').GetListOfBranches():
        parsed = parse_branch(branch.GetName())
        if parsed is not None and parsed[0] is None and 'art::Assns' in branch.GetName():
            other_assns.append(branch.GetName())
        if parsed is None or parsed[0] not in kinds:
            continue
        if labels and parsed[1].split(':')[0] not in labels:
            continue
        found.append(parsed)
    f.Close()
    found = prefer_types(found)
    if other_assns:
        print('Associations not dumped:')
        for name in sorted(other_assns):
            print(f'  {name}')
    # associations last: they refer to the products read before them
    return sorted(found, key=lambda p: (p[0] in ASSOCIATION_KINDS, p[0], p[1]))


def prefer_types(found):
    """
    One (kind, tag, type) per (kind, tag): a producer may store an
    association both with and without metadata (Pandora's track-hit
    associations); keep the type listed first for the kind
    """
    best = {}
    for kind, tag, cpp_type in found:
        order = KINDS[kind].types.index(cpp_type)
        if (kind, tag) not in best or order < best[(kind, tag)][0]:
            best[(kind, tag)] = (order, cpp_type)
    return [(kind, tag, cpp_type) for (kind, tag), (_, cpp_type) in best.items()]


def make_uncompress(ROOT):
    def uncompress(adcs, n_samples, compression):
        out = ROOT.std.vector['short'](n_samples)
        ROOT.raw.Uncompress(adcs, out, compression)
        return out
    return uncompress


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--detector', required=True, help='recorded in the dump, e.g. sbnd or icarus')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--events', type=int, default=3, help='number of events to dump')
    parser.add_argument('--skip', type=int, default=0, help='events to skip first')
    parser.add_argument('--raw-digits', action='store_true', help='also dump raw::RawDigit products')
    parser.add_argument('--tags', help='comma-separated module labels to dump (default: all)')
    args = parser.parse_args()

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.ProcessLine('gErrorIgnoreLevel = kError;')

    kinds = DEFAULT_KINDS + (('raw_digits',) if args.raw_digits else ())
    labels = set(args.tags.split(',')) if args.tags else None
    products = discover_tags(ROOT, args.files[0], kinds, labels)
    print('Dumping:')
    for kind, tag, cpp_type in products:
        print(f'  {kind:13s} {tag:45s} {cpp_type}')
    if not products:
        print('nothing to dump')
        return 1

    extractors = {
        'hits': extract_hits,
        'wires': extract_wire_signals,
        'channel_rois': extract_wire_signals,
        'raw_digits': lambda v, u=make_uncompress(ROOT): extract_raw_digits(v, u),
        'tracks': extract_tracks,
        'vertices': extract_vertices,
        'spacepoints': extract_spacepoints,
        'showers': extract_showers,
        'pfparticles': extract_pfparticles,
        't0s': extract_t0s,
    }
    errors = Errors()
    meta = {
        'detector': args.detector,
        'source_files': [os.path.basename(f) for f in args.files],
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'root_version': ROOT.gROOT.GetVersion(),
        'tool': 'dump_products.py',
        'args': sys.argv[1:],
    }

    ev = ROOT.gallery.Event(ROOT.std.vector['string'](args.files))
    for _ in range(args.skip):
        if ev.atEnd():
            break
        ev.next()

    with DumpWriter(args.outdir, meta) as writer:
        n = 0
        while not ev.atEnd() and n < args.events:
            aux = ev.eventAuxiliary()
            event_id = (aux.run(), aux.subRun(), aux.event())
            print(f'Event {event_id}:', flush=True)
            event_products = {}
            # ProductID value -> tag of the products read so far, to resolve associations
            product_ids = {}

            def tag_of_product(product_id):
                return product_ids.get(product_id.value(), '')

            for kind, tag, cpp_type in products:
                start = time.perf_counter()
                try:
                    handle = ev.getValidHandle[cpp_type](ROOT.art.InputTag(tag))
                    if kind in ASSOCIATION_KINDS:
                        container = extract_associations(handle.product(), tag_of_product,
                                                         has_meta='TrackHitMeta' in cpp_type,
                                                         swapped=association_is_swapped(kind, cpp_type))
                    else:
                        container = extractors[kind](handle.product())
                        try:
                            product_ids[handle.id().value()] = tag
                        except Exception as e:
                            errors.add('ValidHandle.id', e)
                except Exception as e:
                    errors.add(f'{kind} {tag}', e)
                    print(f'  {kind} {tag}: FAILED {type(e).__name__}: {e}')
                    continue
                event_products[(kind, tag)] = container
                print(f'  {kind:13s} {tag:40s} {len(container):7d} rows  {time.perf_counter() - start:6.2f} s')
            writer.add_event(event_id, event_products)
            n += 1
            ev.next()
        writer.meta['errors'] = errors.as_list()

    print(f'Wrote {n} events to {args.outdir}')
    if errors.as_list():
        print(f'\n{len(errors.as_list())} products failed (recorded in index.json):')
        for e in errors.as_list():
            print(f"  {e['label']} (x{e['count']}): {e['first']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
