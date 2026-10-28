"""
Product containers, the dump format, and the dumper's product conversion
(tools/dump_products.py) on fake LArSoft objects.
"""

import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from titus.readers import (DumpReader, DumpWriter, EventID, Hits, ProductCache, ProductNotFound, RawDigits,
                           WireSignals, parse_branch_name, resolve_tag)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'tools'))
import dump_products  # noqa: E402


def make_hits(n, plane=0):
    rng = np.random.default_rng(n)
    return Hits(channel=np.arange(n), cryostat=np.zeros(n), tpc=np.arange(n) % 2,
                plane=np.full(n, plane), wire=np.arange(n) * 3, peak_time=rng.uniform(0, 3400, n),
                rms=rng.uniform(1, 5, n), integral=rng.uniform(10, 500, n),
                peak_amplitude=rng.uniform(5, 50, n),
                start_tick=np.zeros(n), end_tick=np.full(n, 10))


def make_wires():
    return WireSignals.from_rois(channels=[5, 9, 12], n_ticks=[100, 100, 100],
                                 rois=[[(10, [1., 2., 3.]), (50, [4.])], [], [(0, [7., 8.])]])


# --- containers ---

def test_hits_columns_and_dtypes():
    hits = make_hits(6)
    assert len(hits) == 6
    assert hits.channel.dtype == np.uint32 and hits.peak_time.dtype == np.float32
    assert hits.on_plane(0, 1, 0).sum() == 3
    assert len(Hits.empty()) == 0


def test_from_arrays_requires_every_column():
    arrays = make_hits(3).to_arrays()
    del arrays['rms']
    with pytest.raises(ValueError, match='rms'):
        Hits.from_arrays(arrays)


def test_wire_signals_rois_and_dense():
    wires = make_wires()
    assert len(wires) == 3
    assert [(s, list(v)) for s, v in wires.rois(0)] == [(10, [1, 2, 3]), (50, [4])]
    assert list(wires.rois(1)) == []
    dense = wires.dense(0)
    assert dense.shape == (100,) and dense[10:13].tolist() == [1, 2, 3] and dense[50] == 4
    assert dense.sum() == 10
    assert wires.dense(2)[:2].tolist() == [7, 8]
    assert len(WireSignals.empty()) == 0 and list(WireSignals.empty().roi_offset) == [0]


def test_raw_digits_reshape():
    digits = RawDigits(channel=[1, 2], pedestal=[400, 2048], n_samples=[3, 3], adc=[1, 2, 3, 4, 5, 6])
    assert digits.adc.shape == (2, 3) and digits.adc.dtype == np.int16


@pytest.mark.parametrize('branch, expected', [
    ('recob::Hits_gaushit__Reco1.', ('hits', 'gaushit::Reco1')),
    ('recob::Wires_simtpc2d_dnnsp_DetSim.', ('wires', 'simtpc2d:dnnsp:DetSim')),
    ('raw::RawDigits_daq__DetSim.', ('raw_digits', 'daq::DetSim')),
    ('recob::OpHits_ophitpmt__Reco1.', (None, 'ophitpmt::Reco1')),
    ('recob::Hitrecob::Wirevoidart::Assns_gaushit__Reco1.', (None, 'gaushit::Reco1')),
    ('EventAuxiliary', None),
])
def test_parse_branch_name(branch, expected):
    assert parse_branch_name(branch) == expected


@pytest.mark.parametrize('tag, expected', [
    ('gaushit::Reco1', 'gaushit::Reco1'),
    ('gaushit', 'gaushit::Reco1'),
    ('simtpc2d:dnnsp', 'simtpc2d:dnnsp:DetSim'),
])
def test_resolve_tag(tag, expected):
    available = ['gaushit::Reco1', 'specialblipgaushit::Reco1', 'simtpc2d:dnnsp:DetSim']
    assert resolve_tag(tag, available) == expected


def test_resolve_tag_errors():
    with pytest.raises(ProductNotFound, match='available'):
        resolve_tag('fasthit', ['gaushit::Reco1'])
    with pytest.raises(ProductNotFound, match='ambiguous'):
        resolve_tag('gaushit', ['gaushit::Reco1', 'gaushit::Reco2'])
    # a bare label means the empty instance
    with pytest.raises(ProductNotFound):
        resolve_tag('simtpc2d', ['simtpc2d:dnnsp:DetSim'])


# --- dump format ---

@pytest.fixture
def dump(tmp_path):
    with DumpWriter(tmp_path, meta={'detector': 'sbnd'}) as w:
        w.add_event((1, 2, 3), {('hits', 'gaushit::Reco1'): make_hits(5),
                                ('wires', 'simtpc2d:dnnsp:DetSim'): make_wires()})
        w.add_event((1, 2, 4), {('hits', 'gaushit::Reco1'): Hits.empty(),
                                ('hits', 'fasthit::Reco1'): make_hits(2)})
    return DumpReader(tmp_path)


def test_dump_roundtrip(dump):
    assert dump.events == [EventID(1, 2, 3), EventID(1, 2, 4)]
    assert dump.meta['detector'] == 'sbnd'
    hits = dump.hits(0, 'gaushit')
    expected = make_hits(5)
    for column, array in expected.to_arrays().items():
        np.testing.assert_array_equal(getattr(hits, column), array)
    wires = dump.wires(0, 'simtpc2d:dnnsp')
    np.testing.assert_array_equal(wires.dense(0), make_wires().dense(0))
    assert len(dump.hits(1, 'gaushit')) == 0


def test_dump_products_per_event(dump):
    assert dump.tags('hits') == ['gaushit::Reco1', 'fasthit::Reco1']
    assert dump.tags('hits', 0) == ['gaushit::Reco1']
    assert dump.tags('raw_digits') == []
    with pytest.raises(ProductNotFound):
        dump.hits(0, 'fasthit')
    assert len(dump.hits(1, 'fasthit')) == 2
    with pytest.raises(ValueError, match='unknown product kind'):
        dump.get(0, 'no_such_kind', 'pandora')
    with pytest.raises(IndexError):
        dump.hits(2, 'gaushit')


def test_dump_rejects_wrong_container(tmp_path):
    with pytest.raises(TypeError):
        DumpWriter(tmp_path).add_event((1, 1, 1), {('wires', 'x::y'): make_hits(1)})


def test_dump_schema_version(tmp_path):
    DumpWriter(tmp_path).close()
    index = json.loads((tmp_path / 'index.json').read_text())
    index['schema_version'] = 99
    (tmp_path / 'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='schema version'):
        DumpReader(tmp_path)


# --- dumper conversion on fake products ---

class FakeRange:
    def __init__(self, start, values):
        self._start, self._values = start, values

    def begin_index(self):
        return self._start

    def data(self):
        return self._values


def fake(**methods):
    """ an object whose methods return the given values: fake(Channel=3).Channel() == 3 """
    return SimpleNamespace(**{k: (lambda v=v: v) for k, v in methods.items()})


def test_extract_hits():
    wid = SimpleNamespace(Cryostat=0, TPC=1, Plane=2, Wire=100)
    hit = fake(WireID=wid, Channel=7, PeakTime=1234.5, RMS=2.5, Integral=300., PeakAmplitude=40.,
               StartTick=1230, EndTick=1240)
    hits = dump_products.extract_hits([hit, hit])
    assert len(hits) == 2
    assert (hits.tpc[0], hits.plane[0], hits.wire[0], hits.peak_time[0]) == (1, 2, 100, 1234.5)
    assert len(dump_products.extract_hits([])) == 0


def test_extract_wire_signals():
    signal = fake(size=200, get_ranges=[FakeRange(5, [1, 2]), FakeRange(100, [3])])
    wires = dump_products.extract_wire_signals([fake(Channel=42, SignalROI=signal),
                                                fake(Channel=43, SignalROI=fake(size=200, get_ranges=[]))])
    assert wires.channel.tolist() == [42, 43] and wires.n_ticks.tolist() == [200, 200]
    dense = wires.dense(0)
    assert dense[5:7].tolist() == [1, 2] and dense[100] == 3 and dense.sum() == 6


def test_extract_raw_digits():
    plain = fake(Channel=1, Samples=4, Compression=0, ADCs=[1, 2, 3, 4], GetPedestal=400.)
    short = fake(Channel=2, Samples=2, Compression=0, ADCs=[5, 6], GetPedestal=401.)
    packed = fake(Channel=3, Samples=3, Compression=1, ADCs=['packed'], GetPedestal=402.)
    digits = dump_products.extract_raw_digits([plain, short, packed],
                                              uncompress=lambda adcs, n, c: [9] * n)
    assert digits.adc.tolist() == [[1, 2, 3, 4], [5, 6, 0, 0], [9, 9, 9, 0]]
    assert digits.n_samples.tolist() == [4, 2, 3]
    with pytest.raises(ValueError, match='compressed'):
        dump_products.extract_raw_digits([packed])


def test_import_readers_without_root():
    code = 'import sys, titus.readers; assert "ROOT" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


# --- reco2 products ---

from titus.readers import Associations, Points, Showers, Tracks  # noqa: E402
from titus.readers.products import friendly_name, parse_branch  # noqa: E402

import trim_dump  # noqa: E402


def make_tracks():
    return Tracks.from_points(ids=[4, 7], points=[[[0, 0, 0], [1, 2, 3], [-999, -999, -999]], [[5, 5, 5]]],
                              valid=[[True, True, False], [True]])


def make_track_hits():
    return Associations(left=[0, 0, 1], right=[2, 4, 1], left_product=[0, 0, 0], right_product=[1, 1, 1],
                        meta_index=[0, 1, 0], product_tags=['pandoraTrack::Reco2', 'gaushit::Reco1'])


def test_tracks():
    tracks = make_tracks()
    assert len(tracks) == 2 and tracks.point_offset.tolist() == [0, 3, 4]
    assert tracks.points(0).tolist() == [[0, 0, 0], [1, 2, 3]]
    assert len(tracks.points(0, valid_only=False)) == 3
    assert tracks.points(1).tolist() == [[5, 5, 5]]
    assert len(Tracks.empty()) == 0 and len(Tracks.from_points([], [])) == 0


def test_associations():
    assns = make_track_hits()
    assert assns.right_of(0).tolist() == [2, 4] and assns.right_of(1).tolist() == [1]
    assert assns.product_tags[assns.right_product[0]] == 'gaushit::Reco1'


def test_reco2_dump_roundtrip(tmp_path):
    points = Points(id=[1], x=[1.5], y=[2.5], z=[3.5])
    showers = Showers(id=[0], start_x=[1], start_y=[2], start_z=[3], dir_x=[0], dir_y=[0], dir_z=[1],
                      length=[50], open_angle=[0.1], best_plane=[2])
    with DumpWriter(tmp_path) as w:
        w.add_event((1, 1, 1), {('tracks', 'pandoraTrack::Reco2'): make_tracks(),
                                ('track_hits', 'pandoraTrack::Reco2'): make_track_hits(),
                                ('vertices', 'pandora::Reco2'): points,
                                ('showers', 'pandoraShower::Reco2'): showers})
    reader = DumpReader(tmp_path)
    assert reader.get(0, 'tracks', 'pandoraTrack').points(0).tolist() == [[0, 0, 0], [1, 2, 3]]
    assns = reader.get(0, 'track_hits', 'pandoraTrack')
    assert assns.product_tags.tolist() == ['pandoraTrack::Reco2', 'gaushit::Reco1']
    assert reader.get(0, 'vertices', 'pandora').xyz.tolist() == [[1.5, 2.5, 3.5]]
    assert reader.get(0, 'showers', 'pandoraShower').best_plane.tolist() == [2]


@pytest.mark.parametrize('branch, expected', [
    ('recob::Tracks_pandoraTrack__Reco2.', ('tracks', 'pandoraTrack::Reco2', 'std::vector<recob::Track>')),
    ('recob::Trackrecob::Hitrecob::TrackHitMetaart::Assns_pandoraTrack__Reco2.',
     ('track_hits', 'pandoraTrack::Reco2', 'art::Assns<recob::Track,recob::Hit,recob::TrackHitMeta>')),
    ('recob::Trackrecob::Hitvoidart::Assns_trajcluster__Reco2.',
     ('track_hits', 'trajcluster::Reco2', 'art::Assns<recob::Track,recob::Hit,void>')),
    ('recob::Hitrecob::Wirevoidart::Assns_gaushit__Reco1.', (None, 'gaushit::Reco1', None)),
])
def test_parse_branch(branch, expected):
    assert parse_branch(branch) == expected


def test_friendly_name():
    assert friendly_name('std::vector<recob::SpacePoint>') == 'recob::SpacePoints'
    assert friendly_name('art::Assns<recob::Track,recob::Hit,void>') == 'recob::Trackrecob::Hitvoidart::Assns'


class FakePoint:
    def __init__(self, x, y, z):
        self._xyz = (x, y, z)

    def X(self):
        return self._xyz[0]

    def Y(self):
        return self._xyz[1]

    def Z(self):
        return self._xyz[2]


def test_extract_tracks():
    points = [FakePoint(0, 0, 0), FakePoint(1, 1, 1), FakePoint(-999, -999, -999)]
    track = SimpleNamespace(ID=lambda: 3, NumberTrajectoryPoints=lambda: 3,
                            LocationAtPoint=lambda i: points[i], HasValidPoint=lambda i: i < 2)
    tracks = dump_products.extract_tracks([track])
    assert tracks.id.tolist() == [3] and tracks.points(0).tolist() == [[0, 0, 0], [1, 1, 1]]


def test_extract_points_and_showers():
    vertices = dump_products.extract_vertices([fake(ID=2, position=FakePoint(1, 2, 3))])
    assert vertices.xyz.tolist() == [[1, 2, 3]] and vertices.id.tolist() == [2]
    spacepoints = dump_products.extract_spacepoints([fake(ID=5, position=FakePoint(4., 5., 6.))])
    assert spacepoints.xyz.tolist() == [[4, 5, 6]]
    # without position(): XYZ() as PyROOT reads the Double32_t pointer, the doubles' bits as integers
    bits = np.array([4., 5., 6.]).view('i8').tolist()
    spacepoints = dump_products.extract_spacepoints([fake(ID=5, XYZ=bits)])
    assert spacepoints.xyz.tolist() == [[4, 5, 6]]
    assert len(dump_products.extract_vertices([])) == 0
    shower = fake(ID=1, ShowerStart=FakePoint(1, 2, 3), Direction=FakePoint(0, 0, 1), Length=40.,
                  OpenAngle=0.2, best_plane=2)
    showers = dump_products.extract_showers([shower])
    assert (showers.start_z[0], showers.dir_z[0], showers.length[0], showers.best_plane[0]) == (3, 1, 40, 2)


class FakeAssns:
    """ art::Assns: pairs of Ptrs (ProductID, key) and optional metadata """

    def __init__(self, pairs, meta=None):
        self._pairs, self._meta = pairs, meta

    def size(self):
        return len(self._pairs)

    def __getitem__(self, i):
        def ptr(product, key):
            return fake(id=product, key=key)
        (lp, lk), (rp, rk) = self._pairs[i]
        return SimpleNamespace(first=ptr(lp, lk), second=ptr(rp, rk))

    def data(self, i):
        return fake(Index=self._meta[i])


def test_extract_associations():
    tags = {10: 'pandoraTrack::Reco2', 20: 'gaushit::Reco1'}
    assns = FakeAssns([((10, 0), (20, 5)), ((10, 0), (20, 6)), ((10, 1), (99, 2))], meta=[0, 1, 0])
    result = dump_products.extract_associations(assns, lambda pid: tags.get(pid, ''), has_meta=True)
    assert result.left.tolist() == [0, 0, 1] and result.right.tolist() == [5, 6, 2]
    assert result.meta_index.tolist() == [0, 1, 0]
    # the hits of an unknown product (not read) have an empty tag
    assert [result.product_tags[i] for i in result.right_product] == ['gaushit::Reco1', 'gaushit::Reco1', '']
    void = dump_products.extract_associations(FakeAssns([((10, 0), (20, 1))]), tags.get, has_meta=False)
    assert void.meta_index.tolist() == [-1]


def test_trim_renumbers_associations():
    kept = {'gaushit::Reco1': np.array([1, 4, 6])}
    trimmed = trim_dump.renumber(make_track_hits(), kept)
    # hits 2 dropped; hit 4 -> 1, hit 1 -> 0
    assert trimmed.left.tolist() == [0, 1] and trimmed.right.tolist() == [1, 0]
    assert trimmed.meta_index.tolist() == [1, 0]
    assert trimmed.product_tags.tolist() == make_track_hits().product_tags.tolist()


def test_extract_pfparticles_and_t0s():
    primary = 2 ** 64 - 1  # recob::PFParticle::kPFParticlePrimary
    pfps = dump_products.extract_pfparticles([fake(Self=0, PdgCode=13, Parent=primary),
                                              fake(Self=1, PdgCode=11, Parent=0)])
    assert pfps.parent.tolist() == [-1, 0] and pfps.pdg.tolist() == [13, 11]
    t0s = dump_products.extract_t0s([fake(Time=-867000.5, TriggerType=3, ID=0)])
    assert t0s.time.tolist() == [-867000.5] and t0s.time.dtype == np.float64


def test_extract_swapped_associations():
    """ stored as Assns<Hit, Track>: returned with tracks on the left """
    tags = {10: 'pandoraTrack::Reco2', 20: 'gaushit::Reco1'}
    assns = FakeAssns([((20, 5), (10, 0)), ((20, 6), (10, 1))], meta=[3, 4])
    result = dump_products.extract_associations(assns, tags.get, has_meta=True, swapped=True)
    assert result.left.tolist() == [0, 1] and result.right.tolist() == [5, 6]
    assert result.product_tags[result.left_product[0]] == 'pandoraTrack::Reco2'


@pytest.mark.parametrize('branch, kind', [
    ('recob::Hitrecob::Trackrecob::TrackHitMetaart::Assns_pandoraTrack__Reco2.', 'track_hits'),
    ('recob::PFParticlerecob::Trackvoidart::Assns_pandoraTrack__Reco2.', 'pfp_tracks'),
    ('anab::T0recob::PFParticlevoidart::Assns_pandora__Reco2.', 'pfp_t0s'),
    ('anab::T0s_pandora__Reco2.', 't0s'),
    ('recob::PFParticles_pandora__Reco2.', 'pfparticles'),
])
def test_parse_reco2_branches(branch, kind):
    assert parse_branch(branch)[0] == kind


def test_prefer_types():
    found = [('track_hits', 'pandoraTrack::Reco2', 'art::Assns<recob::Hit,recob::Track,void>'),
             ('track_hits', 'pandoraTrack::Reco2', 'art::Assns<recob::Hit,recob::Track,recob::TrackHitMeta>'),
             ('hits', 'gaushit::Reco1', 'std::vector<recob::Hit>')]
    assert sorted(dump_products.prefer_types(found)) == sorted(found[1:])


def test_product_cache_reads_once_per_event():
    cache = ProductCache()
    reads = []

    def read(value):
        reads.append(value)
        return value

    assert cache.get(('hits', 'a'), lambda: read(1)) == 1
    assert cache.get(('hits', 'a'), lambda: read(2)) == 1
    assert reads == [1]
    cache.new_event()
    assert cache.get(('hits', 'a'), lambda: read(3)) == 3
    assert reads == [1, 3]


def test_product_cache_remembers_missing_products():
    cache = ProductCache()
    calls = []

    def missing():
        calls.append(1)
        raise ProductNotFound('hits b not in this event')

    for _ in range(2):
        with pytest.raises(ProductNotFound, match='hits b'):
            cache.get(('hits', 'b'), missing)
    assert calls == [1]


def test_product_cache_file_values_last_until_new_file():
    cache = ProductCache()
    assert cache.file_value('products', lambda: {'x': 1}) == {'x': 1}
    cache.new_event()
    assert cache.file_value('products', lambda: {'y': 2}) == {'x': 1}
    cache.new_file()
    assert cache.file_value('products', lambda: {'y': 2}) == {'y': 2}


# --- products for the drawables still on C++ drawers (gpvm step 2) ---

class FakeVector:
    """ a std::vector-like object: size() and [i], no array interface """
    def __init__(self, values):
        self._values = list(values)

    def size(self):
        return len(self._values)

    def __getitem__(self, i):
        return self._values[i]


class FakeWaveform(FakeVector):
    def __init__(self, channel, stamp, values):
        super().__init__(values)
        self._channel, self._stamp = channel, stamp

    def ChannelNumber(self):
        return self._channel

    def TimeStamp(self):
        return self._stamp


class FakeStep(FakePoint):
    def __init__(self, x, y, z, t, e):
        super().__init__(x, y, z)
        self._te = (t, e)

    def T(self):
        return self._te[0]

    def E(self):
        return self._te[1]


class FakeMCTrack(FakeVector):
    def __init__(self, steps, **methods):
        super().__init__(steps)
        for k, v in methods.items():
            setattr(self, k, lambda v=v: v)


def test_extract_clusters_and_endpoints():
    clusters = dump_products.extract_clusters([fake(ID=4, Plane=SimpleNamespace(Cryostat=0, TPC=1, Plane=2), NHits=12,
                                                    StartWire=10., StartTick=500., EndWire=30., EndTick=900.,
                                                    Integral=1e3)])
    assert (clusters.id[0], clusters.tpc[0], clusters.plane[0], clusters.n_hits[0]) == (4, 1, 2, 12)
    points = dump_products.extract_endpoints2d([fake(ID=1, WireID=SimpleNamespace(Cryostat=0, TPC=0, Plane=1, Wire=7),
                                                     DriftTime=321.5, Charge=2., Strength=3)])
    assert (points.plane[0], points.wire[0], points.drift_time[0]) == (1, 7, 321.5)
    assert len(dump_products.extract_clusters([])) == 0


def test_extract_mctruths():
    nu = fake(Nu=fake(PdgCode=14, E=1.5, Vx=1., Vy=2., Vz=3., T=10.), Mode=0, InteractionType=1001, CCNC=0)
    particles = [fake(PdgCode=13, StatusCode=1, E=1.2), fake(PdgCode=2212, StatusCode=1, E=1.0)]
    neutrino = SimpleNamespace(Origin=lambda: 1, NeutrinoSet=lambda: True, GetNeutrino=lambda: nu,
                               NParticles=lambda: 2, GetParticle=lambda i: particles[i])
    cosmic = SimpleNamespace(Origin=lambda: 2, NeutrinoSet=lambda: False, NParticles=lambda: 1,
                             GetParticle=lambda i: fake(PdgCode=13, StatusCode=1, E=5.))
    truths = dump_products.extract_mctruths([neutrino, cosmic])
    assert truths.nu_pdg.tolist() == [14, 0] and truths.has_neutrino.tolist() == [True, False]
    assert (truths.vx[0], truths.vz[0]) == (1, 3) and np.isnan(truths.vx[1])
    pdg, status, energy = truths.particles(0)
    assert pdg.tolist() == [13, 2212] and truths.particles(1)[0].tolist() == [13]
    assert len(dump_products.extract_mctruths([])) == 0


def test_extract_mctracks():
    track = FakeMCTrack([FakeStep(0, 0, 0, 1, 100), FakeStep(1, 2, 3, 2, 90)], TrackID=5, PdgCode=13, Origin=1,
                        Process='primary', Start=FakeStep(0, 0, 0, 1, 100))
    tracks = dump_products.extract_mctracks([track, FakeMCTrack([], TrackID=6, PdgCode=11, Origin=2,
                                                                  Process='eIoni', Start=FakeStep(0, 0, 0, 0, 0))])
    assert tracks.process.tolist() == ['primary', 'eIoni']
    assert tracks.points(0).tolist() == [[0, 0, 0], [1, 2, 3]] and len(tracks.points(1)) == 0


def test_extract_optical():
    flash = fake(Time=1.5, AbsTime=1.6, TimeWidth=0.1, YCenter=10., YWidth=5., ZCenter=250., ZWidth=40.,
                 TotalPE=30., OnBeamTime=1, PEs=FakeVector([0., 10., 20.]))
    flashes = dump_products.extract_opflashes([flash, flash])
    assert flashes.pe_per_channel(1).tolist() == [0, 10, 20] and flashes.total_pe.tolist() == [30, 30]
    waveforms = dump_products.extract_opdet_waveforms([FakeWaveform(3, -1.25, [14000, 14010]),
                                                       FakeWaveform(4, 0.5, [15000])])
    assert waveforms.samples(0).tolist() == [14000, 14010] and waveforms.time_stamp.tolist() == [-1.25, 0.5]


def test_extract_crt():
    feb = fake(Mac5=12, Flags=3, Ts0=1000, Ts1=900, Coinc=5, UnixS=1700000000, ADC=FakeVector(range(32)))
    febs = dump_products.extract_crt_febdata([feb])
    assert febs.adc.shape == (1, 32) and febs.adc[0, 31] == 31 and febs.mac5.tolist() == [12]
    assert dump_products.extract_crt_febdata([]).adc.shape == (0, 32)
    stamps = dump_products.extract_daq_timestamps([fake(Channel=4, Timestamp=2 ** 40)])
    assert stamps.timestamp.tolist() == [2 ** 40]


def test_new_products_roundtrip_through_a_dump(tmp_path):
    products = {
        ('clusters', 'linecluster::Reco1'): dump_products.extract_clusters([]),
        ('mctruths', 'generator::Gen'): dump_products.extract_mctruths([SimpleNamespace(
            Origin=lambda: 2, NeutrinoSet=lambda: False, NParticles=lambda: 0, GetParticle=None)]),
        ('mctracks', 'mcreco::DetSim'): dump_products.extract_mctracks([FakeMCTrack(
            [FakeStep(1, 2, 3, 4, 5)], TrackID=1, PdgCode=13, Origin=1, Process='primary',
            Start=FakeStep(1, 2, 3, 4, 5))]),
        ('crt_febdata', 'crtsim::DetSim'): dump_products.extract_crt_febdata([fake(
            Mac5=1, Flags=1, Ts0=1, Ts1=1, Coinc=0, UnixS=0, ADC=FakeVector([7] * 32))]),
    }
    with DumpWriter(tmp_path, {'detector': 'sbnd'}) as writer:
        writer.add_event((1, 1, 1), products)
    reader = DumpReader(tmp_path)
    for (kind, tag), expected in products.items():
        got = reader.get(0, kind, tag)
        for column, array in expected.to_arrays().items():
            np.testing.assert_array_equal(getattr(got, column), array)
    assert reader.get(0, 'crt_febdata', 'crtsim').adc.shape == (1, 32)
