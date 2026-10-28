"""
TPC view contents from reader products: wire images and hit positions, in
the view coordinates of a ViewLayout.
"""

from dataclasses import dataclass

import numpy as np

from ..readers.products import WireSignals


def plane_index_table(geom):
    """ array [cryostat, tpc, plane] -> TITUS plane index """
    planes = [p.id for p in geom.detector.planes]
    shape = tuple(max(getattr(p, f) for p in planes) + 1 for f in ('cryostat', 'tpc', 'plane'))
    table = np.full(shape, -1, dtype=int)
    for index, p in enumerate(planes):
        table[p.cryostat, p.tpc, p.plane] = index
    return table


def image_n_ticks(geom, *signals):
    """
    Samples per waveform for a view image: the readout window, or longer if the
    data is (SBND v10_14 files hold 3427 ticks, the v10_06 detector description
    says 3400). ChannelROI waveforms end at the last ROI, which would shorten
    the image and misplace flipped TPCs, so these are padded.
    """
    lengths = [int(s.n_ticks.max()) for s in signals if len(s)]
    return max([int(geom.readoutWindowSize())] + lengths)


def raw_digit_signals(digits, subtract_pedestal=True):
    """ RawDigits as WireSignals, one ROI per channel, for wire_image """
    adc = digits.adc.astype('f4')
    if subtract_pedestal:
        adc -= digits.pedestal[:, None]
    in_waveform = np.arange(adc.shape[1]) < digits.n_samples[:, None]
    n = len(digits)
    return WireSignals(channel=digits.channel, n_ticks=digits.n_samples, roi_row=np.arange(n),
                       roi_start=np.zeros(n), roi_offset=np.concatenate([[0], np.cumsum(digits.n_samples)]),
                       values=adc[in_waveform])


class SignalSamples:
    """
    The ROIs of WireSignals (recob::Wire or ChannelROI) on every wire their
    channel reads out (several on ICARUS), as flat arrays per (ROI, wire)
    pair: TITUS plane index, wire, first tick, length and offset into the
    values. Computed once per product and shared by all views (fill()).
    """
    # ROIs at least this long (raw digits, dense wires) are copied one slice
    # each; shorter ones are expanded to samples and written in one assignment
    SLICE_LENGTH = 64

    def __init__(self, signals, channel_map, geom):
        self.values = np.asarray(signals.values)
        rows = np.asarray(signals.roi_row, dtype=np.int64)
        first = np.searchsorted(channel_map.channel, signals.channel, side='left')[rows]
        n_wires = np.searchsorted(channel_map.channel, signals.channel, side='right')[rows] - first

        pair_roi = np.repeat(np.arange(len(rows)), n_wires)
        map_row = np.repeat(first, n_wires) + _ranks(n_wires)
        offsets = np.asarray(signals.roi_offset, dtype=np.int64)
        self.plane_index = plane_index_table(geom)[channel_map.cryostat, channel_map.tpc, channel_map.plane][map_row]
        self.wire = channel_map.wire[map_row].astype(np.int64)
        self.start = np.asarray(signals.roi_start, dtype=np.int64)[pair_roi]
        self.length = (offsets[1:] - offsets[:-1])[pair_roi]
        self.offset = offsets[:-1][pair_roi]

    def __len__(self):
        return len(self.plane_index)

    def fill(self, image, layout):
        """ write the ROIs on the layout's planes into its image; ticks past n_ticks are dropped """
        n = layout.n_ticks
        length = np.clip(np.minimum(self.start + self.length, n) - self.start, 0, None)
        for block in layout.blocks:
            on_block = (self.plane_index == block.plane_index) & (length > 0)
            long_roi = on_block & (length >= self.SLICE_LENGTH)
            short_roi = on_block & ~long_roi
            if short_roi.any():
                counts = length[short_roi]
                in_roi = _ranks(counts)
                ticks = np.repeat(self.start[short_roi], counts) + in_roi
                columns = block.time_offset + (n - 1 - ticks if block.flip else ticks)
                wires = block.wire_offset + np.repeat(self.wire[short_roi], counts)
                image[wires, columns] = self.values[np.repeat(self.offset[short_roi], counts) + in_roi]
            for i in np.flatnonzero(long_roi):
                start, end, offset = self.start[i], self.start[i] + length[i], self.offset[i]
                values = self.values[offset:offset + length[i]]
                row = image[block.wire_offset + self.wire[i]]
                if block.flip:
                    row[block.time_offset + n - end:block.time_offset + n - start] = values[::-1]
                else:
                    row[block.time_offset + start:block.time_offset + end] = values
        return image

    def image(self, layout):
        return self.fill(np.zeros(layout.shape, dtype='f4'), layout)


def _ranks(counts):
    """ 0..count-1 for each count, concatenated """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(counts.sum(), dtype=np.int64) - starts


def wire_image(layout, signals, channel_map):
    """
    View image (wires, ticks) of WireSignals (recob::Wire or ChannelROI).
    Handles channel reading out several wires (ICARUS). Samples beyond the
    layout's n_ticks are dropped. To draw several views of the same product,
    build its SignalSamples once instead.
    """
    if len(signals) == 0:
        return np.zeros(layout.shape, dtype='f4')
    return SignalSamples(signals, channel_map, layout.geom).image(layout)


@dataclass(frozen=True)
class HitPositions:
    """ hits shown in a view: their rows in the Hits container and view coordinates """
    index: np.ndarray
    x: np.ndarray       # wire center
    y: np.ndarray       # peak time
    half_height: np.ndarray  # RMS, in ticks

    def __len__(self):
        return len(self.index)


def hit_positions(layout, hits, time_offset_ticks=0):
    table = plane_index_table(layout.geom)
    planes = table[hits.cryostat, hits.tpc, hits.plane] if len(hits) else np.zeros(0, dtype=int)
    index, x, y = [], [], []
    for block in layout.blocks:
        rows = np.flatnonzero(planes == block.plane_index)
        y_rows = layout.tick_to_view(block.plane_index, hits.peak_time[rows] + time_offset_ticks)
        # hits in ticks cut by "unite cathodes" aren't shown
        rows, y_rows = rows[np.isfinite(y_rows)], y_rows[np.isfinite(y_rows)]
        index.append(rows)
        x.append(layout.wire_to_view(block.plane_index, hits.wire[rows]))
        y.append(y_rows)
    index = np.concatenate(index)
    return HitPositions(index=index, x=np.concatenate(x), y=np.concatenate(y),
                        half_height=hits.rms[index].astype(float))
