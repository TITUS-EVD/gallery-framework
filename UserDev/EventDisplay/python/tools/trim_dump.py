#!/usr/bin/env python3
"""
Cut a product dump down to a few events and a wire range, e.g. to make test
data. Keeps the hits and waveforms of channels reading out a wire in the
range, on any plane. Runs locally (no ROOT).

    python trim_dump.py dumps/sbnd-reco1 ../tests/data/sbnd-reco1-trimmed \
        --detector sbnd --events 0 --wires 800:1000
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from titus.geometry import load_detector  # noqa: E402
from titus.readers import KINDS, DumpReader, DumpWriter, Hits, RawDigits, WireSignals  # noqa: E402


def keep_rows(container, keep):
    if isinstance(container, Hits):
        return Hits(**{k: v[keep] for k, v in container.to_arrays().items()})
    if isinstance(container, RawDigits):
        return RawDigits(channel=container.channel[keep], pedestal=container.pedestal[keep],
                         n_samples=container.n_samples[keep], adc=container.adc[keep])
    rows = np.flatnonzero(keep)
    return WireSignals.from_rois(container.channel[rows], container.n_ticks[rows],
                                 [list(container.rois(r)) for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--detector', required=True, help='detector description for the channel map')
    parser.add_argument('--events', default='0', help='comma-separated entries to keep')
    parser.add_argument('--wires', required=True, help='wire range first:last (exclusive)')
    args = parser.parse_args()

    first, last = map(int, args.wires.split(':'))
    cm = load_detector(args.detector).channel_map
    channels = np.unique(cm.channel[(cm.wire >= first) & (cm.wire < last)])

    reader = DumpReader(args.source)
    meta = dict(reader.meta, trimmed_from=os.path.basename(os.path.normpath(args.source)),
                trim={'events': args.events, 'wires': args.wires})
    with DumpWriter(args.destination, meta) as writer:
        for entry in map(int, args.events.split(',')):
            products = {}
            for kind in KINDS:
                for tag in reader.tags(kind, entry):
                    container = reader.get(entry, kind, tag)
                    products[(kind, tag)] = keep_rows(container, np.isin(container.channel, channels))
            writer.add_event(reader.events[entry], products)
    print(f'Wrote {args.destination}')


if __name__ == '__main__':
    sys.exit(main())
