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

By default every hit, wire and channel ROI product found in the file is
dumped; raw digits (large) only with --raw-digits. Failures are recorded in
the dump's index.json and listed at the end instead of aborting.
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dump_detector import Errors  # noqa: E402
from titus.readers import KINDS, DumpWriter, Hits, RawDigits, WireSignals, parse_branch_name  # noqa: E402

DEFAULT_KINDS = ('hits', 'wires', 'channel_rois')

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


# --- ROOT side ---

def discover_tags(ROOT, path, kinds, labels=None):
    """ [(kind, full tag)] for product branches in the Events tree """
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise OSError(f'cannot open {path}')
    found = []
    for branch in f.Get('Events').GetListOfBranches():
        parsed = parse_branch_name(branch.GetName())
        if parsed is None or parsed[0] not in kinds:
            continue
        if labels and parsed[1].split(':')[0] not in labels:
            continue
        found.append(parsed)
    f.Close()
    return sorted(found)


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
    for kind, tag in products:
        print(f'  {kind:13s} {tag}')
    if not products:
        print('nothing to dump')
        return 1

    extractors = {
        'hits': extract_hits,
        'wires': extract_wire_signals,
        'channel_rois': extract_wire_signals,
        'raw_digits': lambda v, u=make_uncompress(ROOT): extract_raw_digits(v, u),
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
            for kind, tag in products:
                start = time.perf_counter()
                try:
                    handle = ev.getValidHandle[f'std::vector<{KINDS[kind].cpp_type}>'](ROOT.art.InputTag(tag))
                    container = extractors[kind](handle.product())
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
