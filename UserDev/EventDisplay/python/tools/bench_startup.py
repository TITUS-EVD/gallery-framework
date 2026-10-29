#!/usr/bin/env python3
"""
Time each stage of TITUS startup, to see where the cost actually is.

Run inside the SBN environment (gpvm). Each stage is timed in order within a
single process, so the numbers are incremental: e.g. 'from ROOT import gallery'
does not include the cost of 'import ROOT'. Run it a few times; the first run
after login is usually the cold-cvmfs-cache case.

    python bench_startup.py --detector sbnd
    python bench_startup.py --detector icarus --file my_art_file.root --titusio --no-services

Pass --json out.json to save the results.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
import traceback

SUPPORTED_SERVICES = {
    'icarus': 'ICARUSservices',
    'sbnd': 'SBNDservices',
}

# order matches titus/modules/larsoft_module.py
SERVICE_NAMES = ['DetectorClocks', 'DetectorProperties', 'Geometry',
                 'WireReadout', 'LArProperties', 'AuxDetGeometry']


PRODUCT_TYPES = [
    'std::vector<recob::Hit>',
    'std::vector<recob::Wire>',
    'std::vector<raw::RawDigit>',
    'std::vector<recob::Track>',
    'std::vector<recob::Shower>',
    'std::vector<recob::Vertex>',
]


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # kilobytes on Linux, bytes on macOS
    return rss / 1024. if sys.platform != 'darwin' else rss / 1024. ** 2


class Timer:
    def __init__(self):
        self.results = []

    def stage(self, label, fn):
        t0 = time.perf_counter()
        error = None
        value = None
        try:
            value = fn()
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
            traceback.print_exc()
        dt = time.perf_counter() - t0
        self.results.append({'stage': label, 'seconds': dt,
                             'peak_rss_mb': peak_rss_mb(), 'error': error})
        status = 'FAILED' if error else ''
        print(f'{label:<45s} {dt:8.2f} s  {peak_rss_mb():8.0f} MB  {status}',
              flush=True)
        return value


def module_version(name):
    try:
        mod = __import__(name)
    except Exception:
        return None
    return getattr(mod, '__version__', getattr(mod, 'PYQT_VERSION_STR', 'unknown'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', choices=SUPPORTED_SERVICES, default='sbnd')
    parser.add_argument('--file', help='art-root file to open with gallery')
    parser.add_argument('--titusio', action='store_true',
                        help='also load the titusio product reader library (what TITUS loads)')
    parser.add_argument('--products', action='store_true',
                        help='time instantiating gallery product access from python '
                             '(run separately from --titusio to measure each in isolation)')
    parser.add_argument('--no-services', action='store_true',
                        help='skip loading the LArSoft services')
    parser.add_argument('--json', help='write results to this file')
    args = parser.parse_args()

    t_start = time.perf_counter()
    timer = Timer()

    ROOT = timer.stage('import ROOT', lambda: __import__('ROOT'))
    if ROOT is None:
        sys.exit('Could not import ROOT; is the SBN environment set up?')
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    timer.stage('from ROOT import gallery', lambda: ROOT.gallery)
    timer.stage('import PyQt5.QtWidgets', lambda: __import__('PyQt5.QtWidgets'))
    timer.stage('import pyqtgraph', lambda: __import__('pyqtgraph'))

    if args.products:
        # cost of reading products directly from python, without the evd
        # library: cling has to instantiate these templates from the headers
        for product in PRODUCT_TYPES:
            timer.stage(f'getValidHandle<{product}>',
                        lambda product=product: ROOT.gallery.Event.getValidHandle[product])
        timer.stage('art::FindMany<recob::Hit>', lambda: ROOT.art.FindMany['recob::Hit'])

    if args.titusio:
        timer.stage('from ROOT import titusio', lambda: ROOT.titusio)

    if args.file:
        def open_file():
            files = ROOT.vector(ROOT.string)()
            files.push_back(args.file)
            return ROOT.gallery.Event(files)
        timer.stage('gallery.Event(file)', open_file)

    services = None
    if not args.no_services:
        services_name = SUPPORTED_SERVICES[args.detector]
        services = timer.stage(f'import {services_name}', lambda: __import__(services_name))
    if services is not None:
        handles = {}
        for name in SERVICE_NAMES:
            handles[name] = timer.stage(f"ServiceManager('{name}')",
                                        lambda name=name: services.ServiceManager(name))

        clocks = handles.get('DetectorClocks')
        detprop = handles.get('DetectorProperties')
        if clocks is not None and detprop is not None:
            timer.stage('DetectorProperties.DataFor(clocks)',
                        lambda: detprop.DataFor(clocks.DataForJob()))

    total = time.perf_counter() - t_start
    print(f'{"TOTAL":<45s} {total:8.2f} s  {peak_rss_mb():8.0f} MB')

    env = {
        'python': sys.version,
        'platform': platform.platform(),
        'host': platform.node(),
        'root': ROOT.gROOT.GetVersion(),
        'numpy': module_version('numpy'),
        'pyqtgraph': module_version('pyqtgraph'),
        'PyQt5': None,
    }
    try:
        from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
        env['PyQt5'] = f'{PYQT_VERSION_STR} (Qt {QT_VERSION_STR})'
    except Exception:
        pass
    # record which software versions were set up, e.g. SBNDCODE_VERSION
    for key, value in sorted(os.environ.items()):
        if key.endswith('_VERSION') and any(s in key for s in
                                            ('LAR', 'SBN', 'ICARUS', 'GALLERY', 'ART', 'ROOT')):
            env[key] = value

    print()
    for key, value in env.items():
        print(f'{key:<20s} {value}')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'detector': args.detector, 'total_seconds': total,
                       'stages': timer.results, 'environment': env}, f, indent=2)
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
