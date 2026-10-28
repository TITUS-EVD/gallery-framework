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


def wire_image(layout, signals, channel_map):
    """
    View image (wires, ticks) of WireSignals (recob::Wire or ChannelROI).
    Handles channel reading out several wires (ICARUS). Samples beyond the
    layout's n_ticks are dropped.
    """
    image = np.zeros(layout.shape, dtype='f4')
    if len(signals) == 0:
        return image

    table = plane_index_table(layout.geom)
    map_planes = table[channel_map.cryostat, channel_map.tpc, channel_map.plane]
    in_view = np.isin(map_planes, [b.plane_index for b in layout.blocks])
    first = np.searchsorted(channel_map.channel, signals.channel, side='left')
    last = np.searchsorted(channel_map.channel, signals.channel, side='right')

    n = layout.n_ticks
    for j in range(len(signals.roi_row)):
        row = signals.roi_row[j]
        start = int(signals.roi_start[j])
        values = signals.values[signals.roi_offset[j]:signals.roi_offset[j + 1]]
        end = min(start + len(values), n)
        if end <= start:
            continue
        values = values[:end - start]
        for k in range(first[row], last[row]):
            if not in_view[k]:
                continue
            block = layout.block(map_planes[k])
            wire = block.wire_offset + channel_map.wire[k]
            if block.flip:
                image[wire, block.time_offset + n - end:block.time_offset + n - start] = values[::-1]
            else:
                image[wire, block.time_offset + start:block.time_offset + end] = values
    return image


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
