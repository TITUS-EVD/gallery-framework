#!/usr/bin/env python3
"""
Dump everything TITUS needs from the LArSoft services into a JSON detector
description, plus a 'golden' table of reference projections computed by the
real services. TITUS can then run (and be tested) without the SBN stack.

Run inside the SBN environment (gpvm):

    python dump_detector.py --detector sbnd
    python dump_detector.py --detector icarus --outdir dumps/

Produces <outdir>/<detector>-<gdml>.json and <outdir>/<detector>-<gdml>.golden.json.
Calls that fail are recorded under "errors" in the output instead of aborting,
so please send the output back even if errors are reported.

Units follow LArSoft: cm, microseconds, ticks.
"""

import argparse
import datetime
import json
import os
import random
import sys
import traceback
from collections import OrderedDict

SCHEMA_VERSION = 1

SUPPORTED_SERVICES = {
    'icarus': 'ICARUSservices',
    'sbnd': 'SBNDservices',
}


class Errors:
    """ Collects failures by label so a loop over channels doesn't flood the report """

    def __init__(self):
        self._errors = OrderedDict()

    def add(self, label, exc):
        entry = self._errors.setdefault(label, {'count': 0, 'first': None})
        entry['count'] += 1
        if entry['first'] is None:
            entry['first'] = f'{type(exc).__name__}: {exc}'

    def as_list(self):
        return [{'label': k, **v} for k, v in self._errors.items()]


def attempt(errors, label, *fns, default=None):
    """ Return the result of the first function that succeeds """
    for fn in fns:
        try:
            return fn()
        except Exception as e:
            errors.add(label, e)
    return default


def xyz(p):
    return [p.X(), p.Y(), p.Z()]


def box(b):
    return {'min': xyz(b.Min()), 'max': xyz(b.Max())}


def load_services(detector):
    services = __import__(SUPPORTED_SERVICES[detector])
    # same calls as titus/modules/larsoft_module.py
    handles = {
        'clocks_service': services.ServiceManager('DetectorClocks'),
        'detprop_service': services.ServiceManager('DetectorProperties'),
        'geom': services.ServiceManager('Geometry'),
        'wire_readout': services.ServiceManager('WireReadout'),
        'lar_prop': services.ServiceManager('LArProperties'),
        'aux_geom': services.ServiceManager('AuxDetGeometry'),
    }
    handles['clocks'] = handles['clocks_service'].DataForJob()
    handles['detprop'] = handles['detprop_service'].DataFor(handles['clocks'])
    return handles


def dump_meta(s, errors, detector):
    geom = s['geom']
    gdml = attempt(errors, 'GDMLFile', lambda: str(geom.GDMLFile()), default='')
    meta = OrderedDict()
    meta['schema_version'] = SCHEMA_VERSION
    meta['detector'] = detector
    meta['detector_name'] = attempt(errors, 'DetectorName', lambda: str(geom.DetectorName()))
    meta['gdml_file'] = gdml
    meta['gdml'] = os.path.splitext(os.path.basename(gdml))[0] if gdml else 'unknown'
    meta['root_file'] = attempt(errors, 'ROOTFile', lambda: str(geom.ROOTFile()))
    meta['created'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta['host'] = os.uname().nodename
    meta['versions'] = {k: v for k, v in sorted(os.environ.items())
                        if k.endswith('_VERSION')
                        and any(t in k for t in ('LAR', 'SBN', 'ICARUS', 'GALLERY', 'ART', 'ROOT'))}
    return meta


def dump_timing(s, errors):
    clocks, detp = s['clocks'], s['detprop']
    t = OrderedDict()
    t['tpc_tick_period'] = attempt(errors, 'TPCClock.TickPeriod', lambda: clocks.TPCClock().TickPeriod())
    t['tpc_frame_ticks'] = attempt(errors, 'TPCClock.FrameTicks', lambda: clocks.TPCClock().FrameTicks())
    t['optical_tick_period'] = attempt(errors, 'OpticalClock.TickPeriod', lambda: clocks.OpticalClock().TickPeriod())
    t['optical_frame_ticks'] = attempt(errors, 'OpticalClock.FrameTicks', lambda: clocks.OpticalClock().FrameTicks())
    t['trigger_offset_tpc'] = attempt(errors, 'TriggerOffsetTPC', lambda: clocks.TriggerOffsetTPC())
    # as computed in Geometry.configure()
    t['trigger_offset_tpc_ticks'] = attempt(errors, 'TriggerOffsetTPC ticks',
                                            lambda: clocks.TPCClock().Ticks(clocks.TriggerOffsetTPC() * -1.))
    t['trigger_time'] = attempt(errors, 'TriggerTime', lambda: clocks.TriggerTime())
    t['beam_gate_time'] = attempt(errors, 'BeamGateTime', lambda: clocks.BeamGateTime())
    t['g4_to_elec_time_at_zero'] = attempt(errors, 'G4ToElecTime', lambda: clocks.G4ToElecTime(0.))

    efield = attempt(errors, 'Efield', lambda: detp.Efield())
    temperature = attempt(errors, 'Temperature', lambda: detp.Temperature())
    t['efield'] = [attempt(errors, f'Efield({i})', lambda i=i: detp.Efield(i)) for i in range(3)]
    t['temperature'] = temperature
    t['drift_velocity'] = attempt(errors, 'DriftVelocity', lambda: detp.DriftVelocity(efield, temperature))
    t['number_time_samples'] = attempt(errors, 'NumberTimeSamples', lambda: detp.NumberTimeSamples())
    t['readout_window_size'] = attempt(errors, 'ReadOutWindowSize', lambda: detp.ReadOutWindowSize())
    if t['tpc_tick_period'] is not None and t['drift_velocity'] is not None:
        t['time_to_cm'] = t['tpc_tick_period'] * t['drift_velocity']
    return t


def get_wire(errors, plane, pid, w):
    import ROOT
    return attempt(errors, 'Plane.Wire', lambda: plane.Wire(w),
                   lambda: plane.Wire(ROOT.geo.WireID(pid, w)))


def wire_record(wire):
    if wire is None:
        return None
    return {'center': xyz(wire.GetCenter()), 'start': xyz(wire.GetStart()),
            'end': xyz(wire.GetEnd()), 'direction': xyz(wire.Direction())}


def dump_plane(s, errors, pid):
    wr, detp = s['wire_readout'], s['detprop']
    plane = wr.Plane(pid)
    p = OrderedDict()
    p['id'] = [pid.Cryostat, pid.TPC, pid.Plane]
    p['view'] = attempt(errors, 'Plane.View', lambda: int(plane.View()))
    p['signal_type'] = attempt(errors, 'SignalType', lambda: int(wr.SignalType(pid)))
    p['center'] = attempt(errors, 'Plane.GetCenter', lambda: xyz(plane.GetCenter()))
    p['n_wires'] = attempt(errors, 'Plane.Nwires', lambda: int(plane.Nwires()))
    p['wire_pitch'] = attempt(errors, 'Plane.WirePitch', lambda: plane.WirePitch())
    p['theta_z'] = attempt(errors, 'Plane.ThetaZ', lambda: plane.ThetaZ())
    p['normal_direction'] = attempt(errors, 'Plane.GetNormalDirection', lambda: xyz(plane.GetNormalDirection()))
    p['increasing_wire_direction'] = attempt(errors, 'Plane.GetIncreasingWireDirection',
                                             lambda: xyz(plane.GetIncreasingWireDirection()))
    if p['n_wires']:
        p['first_wire'] = attempt(errors, 'first wire', lambda: wire_record(get_wire(errors, plane, pid, 0)))
        p['last_wire'] = attempt(errors, 'last wire',
                                 lambda: wire_record(get_wire(errors, plane, pid, p['n_wires'] - 1)))
    # x <-> tick is linear per plane: x = x_at_tick0 + x_per_tick * tick
    x0 = attempt(errors, 'ConvertTicksToX', lambda: detp.ConvertTicksToX(0., pid))
    x1000 = attempt(errors, 'ConvertTicksToX', lambda: detp.ConvertTicksToX(1000., pid))
    p['x_at_tick0'] = x0
    p['x_per_tick'] = (x1000 - x0) / 1000. if x0 is not None and x1000 is not None else None
    return p


def dump_cryostats(s, errors):
    import ROOT
    geom, wr = s['geom'], s['wire_readout']
    cryostats = []
    for c in range(geom.Ncryostats()):
        cid = ROOT.geo.CryostatID(c)
        cryo = geom.Cryostat(cid)
        cr = OrderedDict()
        cr['id'] = c
        cr['bounds'] = attempt(errors, 'Cryostat bounds', lambda: box(cryo.BoundingBox()), lambda: box(cryo))
        cr['half_width'] = attempt(errors, 'Cryostat.HalfWidth', lambda: cryo.HalfWidth())
        cr['half_height'] = attempt(errors, 'Cryostat.HalfHeight', lambda: cryo.HalfHeight())
        cr['length'] = attempt(errors, 'Cryostat.Length', lambda: cryo.Length())
        cr['tpcs'] = []
        for t in range(geom.NTPC(cid)):
            tid = ROOT.geo.TPCID(c, t)
            tpc = geom.TPC(tid)
            tp = OrderedDict()
            tp['id'] = [c, t]
            tp['bounds'] = attempt(errors, 'TPC bounds', lambda: box(tpc), lambda: box(tpc.BoundingBox()))
            tp['active_bounds'] = attempt(errors, 'TPC.ActiveBoundingBox', lambda: box(tpc.ActiveBoundingBox()))
            tp['center'] = attempt(errors, 'TPC.GetCenter', lambda: xyz(tpc.GetCenter()))
            tp['active_center'] = attempt(errors, 'TPC.GetActiveVolumeCenter', lambda: xyz(tpc.GetActiveVolumeCenter()))
            tp['drift_dir'] = attempt(errors, 'TPC.DriftDir', lambda: xyz(tpc.DriftDir()))
            tp['drift_direction_enum'] = attempt(errors, 'TPC.DriftDirection', lambda: int(tpc.DriftDirection()))
            tp['drift_distance'] = attempt(errors, 'TPC.DriftDistance', lambda: tpc.DriftDistance())
            tp['planes'] = [dump_plane(s, errors, ROOT.geo.PlaneID(c, t, p))
                            for p in range(wr.Nplanes(tid))]
            cr['tpcs'].append(tp)
        cryostats.append(cr)
    return cryostats


def dump_channel_map(s, errors):
    """ One row per (channel, wire) pair; a channel can map to several wires """
    wr = s['wire_readout']
    cols = OrderedDict((k, []) for k in ('channel', 'cryostat', 'tpc', 'plane', 'wire'))
    n_channels = wr.Nchannels()
    for ch in range(n_channels):
        try:
            wids = wr.ChannelToWire(ch)
        except Exception as e:
            errors.add('ChannelToWire', e)
            continue
        for wid in wids:
            cols['channel'].append(ch)
            cols['cryostat'].append(int(wid.Cryostat))
            cols['tpc'].append(int(wid.TPC))
            cols['plane'].append(int(wid.Plane))
            cols['wire'].append(int(wid.Wire))
    return {'n_channels': int(n_channels), 'rows': cols}


def dump_opdets(s, errors):
    geom, wr = s['geom'], s['wire_readout']
    n_opdets = int(geom.NOpDets())
    n_channels = attempt(errors, 'NOpChannels', lambda: int(wr.NOpChannels()), default=n_opdets)
    channels = []
    for ch in range(n_channels):
        rec = OrderedDict(channel=ch)
        rec['opdet'] = attempt(errors, 'OpDetFromOpChannel', lambda: int(wr.OpDetFromOpChannel(ch)))
        opdet = attempt(errors, 'OpDetGeoFromOpChannel', lambda: wr.OpDetGeoFromOpChannel(ch))
        if opdet is not None:
            rec['center'] = attempt(errors, 'OpDetGeo.GetCenter', lambda: xyz(opdet.GetCenter()))
            rec['shape'] = attempt(errors, 'OpDetGeo.Shape', lambda: str(opdet.Shape().IsA().GetName()))
            rec['rmax'] = attempt(errors, 'OpDetGeo.RMax', lambda: opdet.RMax())
            rec['half_width'] = attempt(errors, 'OpDetGeo.HalfW', lambda: opdet.HalfW())
            rec['half_height'] = attempt(errors, 'OpDetGeo.HalfH', lambda: opdet.HalfH())
            rec['length'] = attempt(errors, 'OpDetGeo.Length', lambda: opdet.Length())
        channels.append(rec)
    return {'n_opdets': n_opdets, 'n_channels': n_channels, 'channels': channels}


def volume_path(geom, volume_name):
    """ GDML path to a named volume, as done in titus/modules/crt_module.py """
    from ROOT import set as ROOTset, string
    name_set = ROOTset(string)()
    name_set.insert(volume_name)
    geo_paths = geom.FindAllVolumePaths(name_set)
    return '/'.join(p.GetName() for p in geo_paths[0])


def dump_auxdets(s, errors):
    """ CRT modules and strips, following titus/modules/crt_module.py """
    import numpy as np
    import ROOT
    geom, aux = s['geom'], s['aux_geom']
    n = attempt(errors, 'NAuxDets', lambda: int(aux.NAuxDets()), default=0)
    auxdets = []
    for ad_i in range(n):
        try:
            ad = aux.AuxDet(ad_i)
            rec = OrderedDict(index=ad_i)
            rec['name'] = str(ad.TotalVolume().GetName())
            manager = geom.ROOTGeoManager()
            manager.cd(volume_path(geom, rec['name']))
            nodes = [manager.GetCurrentNode()] + [manager.GetMother(i) for i in (1, 2, 3)]
            rec['local_mac'] = int(nodes[1].GetNumber())

            shape = nodes[0].GetVolume().GetShape()
            hw, hh, hl = shape.GetDX(), shape.GetDY(), shape.GetDZ() / 2

            def to_world(limits):
                for node in nodes:
                    out = np.zeros(3)
                    node.LocalToMaster(np.asarray(limits, dtype=float), out)
                    limits = out.copy()
                return limits.tolist()

            rec['array_bounds'] = {'min': to_world([-hw, -hh, -hl]), 'max': to_world([hw, hh, hl])}

            strips = []
            for sv_i in range(ad.NSensitiveVolume()):
                strip = ad.SensitiveVolume(sv_i)
                shw, shh, shl = strip.HalfWidth1(), strip.HalfHeight(), strip.Length() / 2
                lo = strip.toWorldCoords(ROOT.geo.AuxDetSensitiveGeo.LocalPoint_t(-shw, -shh, -shl))
                hi = strip.toWorldCoords(ROOT.geo.AuxDetSensitiveGeo.LocalPoint_t(shw, shh, shl))
                strips.append({'index': sv_i, 'min': xyz(lo), 'max': xyz(hi)})
            rec['strips'] = strips
            auxdets.append(rec)
        except Exception as e:
            errors.add('AuxDet', e)
    return auxdets


def uniform_point(rng, bounds):
    return [rng.uniform(lo, hi) for lo, hi in zip(bounds['min'], bounds['max'])]


def dump_golden(s, errors, cryostats, n_points, seed):
    """
    Reference values from the real services, for testing the lightweight
    implementation: 3D point -> TPC, and per plane -> wire coordinate, nearest
    wire, and tick.
    """
    import ROOT
    geom, wr, detp = s['geom'], s['wire_readout'], s['detprop']
    rng = random.Random(seed)

    def tpc_of(p):
        tid = geom.PositionToTPCID(ROOT.geo.Point_t(*p))
        return [int(tid.Cryostat), int(tid.TPC)] if tid.isValid else None

    projections = []
    for cr in cryostats:
        for tp in cr['tpcs']:
            bounds = tp['active_bounds'] or tp['bounds']
            if bounds is None:
                continue
            for _ in range(n_points):
                p = uniform_point(rng, bounds)
                loc = ROOT.geo.Point_t(*p)
                rec = {'point': p, 'sampled_tpc': tp['id'],
                       'tpc': attempt(errors, 'PositionToTPCID', lambda: tpc_of(p)), 'planes': []}
                for pl in tp['planes']:
                    pid = ROOT.geo.PlaneID(*pl['id'])
                    plane = wr.Plane(pid)
                    rec['planes'].append({
                        'id': pl['id'],
                        'wire_coordinate': attempt(errors, 'WireCoordinate', lambda: plane.WireCoordinate(loc)),
                        'nearest_wire': attempt(errors, 'NearestWireID', lambda: int(plane.NearestWireID(loc).Wire)),
                        'ticks': attempt(errors, 'ConvertXToTicks', lambda: detp.ConvertXToTicks(p[0], pid)),
                    })
                projections.append(rec)

    # points anywhere in each cryostat, to test the point -> TPC lookup
    # including points outside any TPC
    tpc_lookup = []
    for cr in cryostats:
        if cr['bounds'] is None:
            continue
        for _ in range(n_points):
            p = uniform_point(rng, cr['bounds'])
            tpc_lookup.append({'point': p, 'tpc': attempt(errors, 'PositionToTPCID', lambda: tpc_of(p))})

    return {'seed': seed, 'projections': projections, 'tpc_lookup': tpc_lookup}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', choices=SUPPORTED_SERVICES, required=True)
    parser.add_argument('--outdir', default='.')
    parser.add_argument('--golden-points', type=int, default=200,
                        help='random points per TPC (and per cryostat) in the golden table')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.ProcessLine('gErrorIgnoreLevel = kError;')

    print(f'Loading {args.detector} services...', flush=True)
    s = load_services(args.detector)
    errors = Errors()

    sections = OrderedDict()
    sections['meta'] = dump_meta(s, errors, args.detector)
    steps = [('timing', lambda: dump_timing(s, errors)),
             ('cryostats', lambda: dump_cryostats(s, errors)),
             ('channel_map', lambda: dump_channel_map(s, errors)),
             ('opdets', lambda: dump_opdets(s, errors)),
             ('auxdets', lambda: dump_auxdets(s, errors))]
    for name, fn in steps:
        print(f'Dumping {name}...', flush=True)
        try:
            sections[name] = fn()
        except Exception as e:
            traceback.print_exc()
            errors.add(f'section {name}', e)
            sections[name] = None

    golden = None
    if sections['cryostats']:
        print('Computing golden projections...', flush=True)
        try:
            golden = dump_golden(s, errors, sections['cryostats'], args.golden_points, args.seed)
        except Exception as e:
            traceback.print_exc()
            errors.add('section golden', e)

    sections['errors'] = errors.as_list()

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.detector}-{sections['meta']['gdml']}"
    out = os.path.join(args.outdir, stem + '.json')
    with open(out, 'w') as f:
        json.dump(sections, f, indent=1)
    print(f'Wrote {out}')
    if golden is not None:
        golden_out = os.path.join(args.outdir, stem + '.golden.json')
        with open(golden_out, 'w') as f:
            json.dump({'meta': sections['meta'], **golden}, f)
        print(f'Wrote {golden_out}')

    if sections['errors']:
        print(f'\n{len(sections["errors"])} kinds of call failed (recorded in the JSON):')
        for e in sections['errors']:
            print(f"  {e['label']} (x{e['count']}): {e['first']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
