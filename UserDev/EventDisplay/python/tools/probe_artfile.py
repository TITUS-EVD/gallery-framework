#!/usr/bin/env python3
"""
Report what an art-root file carries that TITUS could use directly: the
geometry/clock/detector-property configuration of the jobs that made it, and
the data product branches it contains.

Run inside the SBN environment (gpvm):

    python probe_artfile.py my_file.root --outdir probe/

Writes probe/<file>.keys.txt, probe/<file>.branches.txt and, if art's
config_dumper is available, probe/<file>.config.fcl. Prints the lines of the
configuration that mention geometry, clocks or detector properties.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

INTERESTING = re.compile(r'gdml|geometry|detectorclocks|detectorproperties|'
                         r'larproperties|wirereadout|triggeroffset|numbertimesamples|'
                         r'readoutwindowsize|efield|temperature', re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file')
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.file))[0])

    f = ROOT.TFile.Open(args.file)
    if not f or f.IsZombie():
        sys.exit(f'Could not open {args.file}')

    with open(stem + '.keys.txt', 'w') as out:
        for key in f.GetListOfKeys():
            out.write(f'{key.GetName()}\t{key.GetClassName()}\t'
                      f'cycle={key.GetCycle()}\tobjlen={key.GetObjlen()}\tnbytes={key.GetNbytes()}\n')
    print(f'Wrote {stem}.keys.txt')

    events = f.Get('Events')
    if events:
        with open(stem + '.branches.txt', 'w') as out:
            out.write(f'# {events.GetEntries()} events\n')
            for branch in events.GetListOfBranches():
                out.write(f'{branch.GetName()}\t{branch.GetClassName()}\t'
                          f'zipbytes={branch.GetZipBytes()}\n')
        print(f'Wrote {stem}.branches.txt')

    dumper = shutil.which('config_dumper')
    if dumper is None:
        print('config_dumper not found in PATH; skipping configuration dump')
        return 0

    result = subprocess.run([dumper, '-P', args.file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'config_dumper failed:\n{result.stderr}')
        return 1
    with open(stem + '.config.fcl', 'w') as out:
        out.write(result.stdout)
    print(f'Wrote {stem}.config.fcl')

    print('\nConfiguration lines of interest:')
    for line in result.stdout.splitlines():
        if INTERESTING.search(line):
            print('  ' + line.strip()[:200])
    return 0


if __name__ == '__main__':
    sys.exit(main())
