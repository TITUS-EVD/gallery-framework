"""
Optical waveforms for the optical views.
"""

import numpy as np


def opdet_traces(waveforms, n_channels, tick_period, compression=1, time_offset=0.):
    '''
    raw::OpDetWaveforms as one trace per optical channel:
    {channel: {'time': us, 'adc': ...}}, every compression-th sample of each
    of the channel's waveforms, in time stamp order, each followed by a nan
    so that plotted lines don't join waveforms. Channels past n_channels
    (those not in the detector description, e.g. trigger and timing channels
    in data) are left out. tick_period in us; time_offset added to the time
    stamps.
    '''
    compression = max(int(compression), 1)
    keep = np.flatnonzero(waveforms.channel < n_channels)
    keep = keep[np.lexsort((waveforms.time_stamp[keep], waveforms.channel[keep]))]
    traces = {}
    for row in keep:
        samples = waveforms.samples(row)[::compression].astype(float)
        if len(samples) == 0:
            continue
        time = waveforms.time_stamp[row] + time_offset + np.arange(len(samples)) * compression * tick_period
        traces.setdefault(int(waveforms.channel[row]), []).append((time, samples))
    out = {}
    for channel, pieces in traces.items():
        out[channel] = {'time': np.concatenate([np.append(t, np.nan) for t, _ in pieces]),
                        'adc': np.concatenate([np.append(a, np.nan) for _, a in pieces])}
    return out
