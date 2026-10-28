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

from titus.readers import (DumpReader, DumpWriter, EventID, Hits, ProductNotFound, RawDigits,
                           WireSignals, parse_branch_name, resolve_tag)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'tools'))
import dump_products  # noqa: E402


def make_hits(n, plane=0):
    rng = np.random.default_rng(n)
    return Hits(channel=np.arange(n), cryostat=np.zeros(n), tpc=np.arange(n) % 2,
                plane=np.full(n, plane), wire=np.arange(n) * 3, peak_time=rng.uniform(0, 3400, n),
                rms=rng.uniform(1, 5, n), integral=rng.uniform(10, 500, n),
                peak_amplitude=rng.uniform(5, 50, n), summed_adc=rng.uniform(10, 500, n),
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
        dump.get(0, 'tracks', 'pandora')
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
               SummedADC=310., StartTick=1230, EndTick=1240)
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
