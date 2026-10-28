#!/usr/bin/env python3
"""
Summarise which files under /cvmfs a process touched, from an strace log.
This is what a cold cvmfs cache has to fetch, independent of whether the
cache happened to be warm when you ran it.

Record a log on the gpvm, e.g.:

    strace -f -e trace=open,openat,execve -o trace-floor.log \\
        python bench_startup.py --no-services

then summarise it (on the gpvm, so file sizes can be looked up):

    python cvmfs_footprint.py trace-floor.log
    python cvmfs_footprint.py trace-floor.log --diff trace-evd.log   # what --evd adds

Sizes are the full file sizes; cvmfs fetches large files in chunks, so the bytes
actually transferred for big libraries can be lower.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# '... openat(AT_FDCWD, "/cvmfs/...", O_RDONLY) = 3' or '= -1 ENOENT (...)'
CALL = re.compile(r'(?:open|openat|execve)\((?:AT_FDCWD, )?"(/cvmfs/[^"]+)".*\)\s+=\s+(-?\d+)')

# spack install dirs look like name-version-hash; ups products like products/name/version
SPACK_DIR = re.compile(r'^[\w.+-]+-[\w.]+-[a-z0-9]{7,}$')


def package_of(path):
    parts = path.split('/')
    for i, part in enumerate(parts):
        if SPACK_DIR.match(part):
            return '/'.join(parts[:i + 1])
        if part == 'products' and i + 2 < len(parts):
            return '/'.join(parts[:i + 3])
    return '/'.join(parts[:4])


def short_name(package):
    last = package.rsplit('/', 1)[-1]
    m = re.match(r'^(.*)-[a-z0-9]{7,}$', last)
    if m:
        return m.group(1)
    if '/products/' in package:
        return ' '.join(package.split('/')[-2:])
    return package


def parse(log):
    opened, missing = set(), set()
    with open(log) as f:
        for line in f:
            m = CALL.search(line)
            if not m:
                continue
            path, ret = m.group(1), int(m.group(2))
            (opened if ret >= 0 else missing).add(path)
    return opened, missing


def summarise(opened, missing, top):
    by_package = defaultdict(lambda: [0, 0])
    total = 0
    for path in opened:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        pkg = by_package[package_of(path)]
        pkg[0] += 1
        pkg[1] += size
        total += size

    print(f'{len(opened)} files opened, {total / 1e6:.1f} MB; '
          f'{len(missing)} failed lookups')
    print(f'\n{"package":<55s} {"files":>6s} {"MB":>9s}')
    rows = sorted(by_package.items(), key=lambda kv: -kv[1][1])
    for pkg, (n, size) in rows[:top]:
        print(f'{short_name(pkg)[:55]:<55s} {n:6d} {size / 1e6:9.1f}')
    if len(rows) > top:
        rest = rows[top:]
        print(f'{"(" + str(len(rest)) + " more)":<55s} {sum(r[1][0] for r in rest):6d} '
              f'{sum(r[1][1] for r in rest) / 1e6:9.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log')
    parser.add_argument('--diff', metavar='LOG',
                        help='show only what this second log touches beyond the first')
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()

    opened, missing = parse(args.log)
    if args.diff:
        opened2, missing2 = parse(args.diff)
        print(f'In {args.diff} but not {args.log}:')
        opened, missing = opened2 - opened, missing2 - missing
    summarise(opened, missing, args.top)
    return 0


if __name__ == '__main__':
    sys.exit(main())
