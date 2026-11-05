#!/usr/bin/env python3
"""
Convert a LArSoft space charge map (voxelized TH3s, as SBND's
SCEoffsets_*_voxelTH3.root) into the map file TITUS draws simulated
particles with (titus/display/space_charge.py): the forward displacement,
from where charge was deposited to where the simulation moved it, per drift
volume. TITUS then needs no ROOT to use it.

Needs PyROOT only (no numpy, no LArSoft), e.g. on a gpvm:

    python dump_sce_map.py SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root --detector sbnd

Writes <outdir>/<map name>.npz (default ../titus/detectors/sce/): arrays
'<side>/d<axis>' (cm, shape (nx, ny, nz), at the voxel centers) for each
side of a dual map ('E': x < 0, 'W': x >= 0) or 'all' for a single map,
'axes' ([n, low edge, high edge] for x, y, z) and 'meta' (JSON bytes).
"""

import argparse
import array
import datetime
import json
import os
import sys
import zipfile

# histogram names: forward displacement along each axis, per side of a dual map
FORWARD = 'TrueFwd_Displacement_{axis}{side}'
SIDES = {'_E': 'E', '_W': 'W', '': 'all'}
# which points each side applies to
SIDE_RANGES = {'E': 'x < 0', 'W': 'x >= 0', 'all': 'everywhere'}


def npy(values, typecode, shape):
    ''' the bytes of an .npy file (format 1.0) of a C-ordered array, written without numpy '''
    data = array.array(typecode, values)
    if sys.byteorder != 'little':
        data.byteswap()
    descr = {'f': '<f4', 'd': '<f8', 'B': '|u1'}[typecode]
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {tuple(shape)!r}, }}"
    # magic, version, header length: the data starts on a multiple of 64 bytes
    pad = 64 - (10 + len(header) + 1) % 64
    header = (header + ' ' * pad + '\n').encode('latin1')
    return b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + data.tobytes()


def axes_of(hist):
    return [[ax.GetNbins(), ax.GetXmin(), ax.GetXmax()] for ax in (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis())]


def voxel_values(hist):
    ''' bin contents without ROOT's under/overflow bins, C order (x, y, z) '''
    (nx, _, _), (ny, _, _), (nz, _, _) = axes_of(hist)
    flat = hist.GetArray()
    flat.reshape((hist.GetNcells(),))
    cells = array.array('f', flat)
    # ROOT's global bin: x fastest, with an under/overflow bin at both ends of each axis
    sx, sy = 1, nx + 2
    sz = (nx + 2) * (ny + 2)
    return [cells[(iz + 1) * sz + (iy + 1) * sy + (ix + 1) * sx]
            for ix in range(nx) for iy in range(ny) for iz in range(nz)]


def read_map(ROOT, path):
    ''' {side: {'dx'|'dy'|'dz': values}}, axes of a map file's forward displacements '''
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise OSError(f'cannot open {path}')
    names = {k.GetName() for k in f.GetListOfKeys()}
    sides, axes = {}, None
    for suffix, side in SIDES.items():
        hists = {f'd{a.lower()}': FORWARD.format(axis=a, side=suffix) for a in 'XYZ'}
        if not all(h in names for h in hists.values()):
            continue
        sides[side] = {}
        for column, name in hists.items():
            hist = f.Get(name)
            if axes is None:
                axes = axes_of(hist)
            elif axes_of(hist) != axes:
                raise ValueError(f'{name}: other binning than {axes}')
            sides[side][column] = voxel_values(hist)
        if side == 'all' or (side in ('E', 'W') and {'E', 'W'} <= set(sides)):
            break
    f.Close()
    if not sides:
        raise ValueError(f'no {FORWARD.format(axis="X", side="[_E|_W]")} histograms in {path}')
    return sides, axes


def write_map(out_path, sides, axes, meta):
    shape = [int(n) for n, _, _ in axes]
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for side, columns in sides.items():
            for column, values in columns.items():
                z.writestr(f'{side}/{column}.npy', npy(values, 'f', shape))
        z.writestr('axes.npy', npy([v for ax in axes for v in ax], 'd', (3, 3)))
        z.writestr('meta.npy', npy(json.dumps(meta).encode(), 'B', (len(json.dumps(meta).encode()),)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('map', help='the space charge map ROOT file')
    parser.add_argument('--detector', required=True, help='recorded in the map file, e.g. sbnd')
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         '..', 'titus', 'detectors', 'sce'))
    args = parser.parse_args()

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    sides, axes = read_map(ROOT, args.map)
    name = os.path.splitext(os.path.basename(args.map))[0]
    meta = {
        'schema_version': 1,
        'detector': args.detector,
        'name': name,
        'source_file': os.path.basename(args.map),
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'root_version': ROOT.gROOT.GetVersion(),
        'tool': 'dump_sce_map.py',
        'quantity': 'forward displacement (cm): deposited -> where the simulation moved the charge',
        'sides': {side: SIDE_RANGES[side] for side in sides},
    }
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f'{name}.npz')
    write_map(out, sides, axes, meta)
    print(f'Wrote {out}: sides {sorted(sides)}, {axes[0][0]}x{axes[1][0]}x{axes[2][0]} voxels, '
          f'{os.path.getsize(out) / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
