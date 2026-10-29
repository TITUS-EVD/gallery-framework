"""
SBND CRT strip hits from front-end board readouts, for the CRT view.
"""

import numpy as np

# the SPEC-TDC channel holding the event trigger time stamp
TDC_EVENT_TRIGGER_CHANNEL = 4
NS_PER_S = 1_000_000_000


def event_trigger_time(timestamps):
    ''' (s, ns) of the event trigger from sbnd::timing::DAQTimestamps, or None without one '''
    if timestamps is None:
        return None
    rows = np.flatnonzero(timestamps.channel == TDC_EVENT_TRIGGER_CHANNEL)
    if len(rows) == 0:
        return None
    stamp = int(timestamps.timestamp[rows[-1]])
    return stamp // NS_PER_S, stamp % NS_PER_S


def crt_strip_hits(febs, timestamps=None):
    '''
    One row per FEB data readout (flags 1 or 3), for its SiPM with the
    largest ADC: [32 * mac5 + SiPM, t0 (ns), ts1 (ns), ADC]. t0 is Ts0 plus
    the cable delay in Coinc; with the event trigger time stamp (MC has
    none), it is taken relative to the trigger, across a second boundary if
    needed. As DrawFEBData did, in float64 to keep nanoseconds.
    '''
    keep = np.isin(febs.flags, (1, 3))
    adc = febs.adc[keep]
    t0 = febs.ts0[keep].astype(np.int64) + febs.coinc[keep].astype(np.int64)
    trigger = event_trigger_time(timestamps)
    if trigger is not None:
        trigger_s, trigger_ns = trigger
        unix_diff = trigger_s - febs.unix_s[keep].astype(np.int64)
        if np.any(np.abs(unix_diff) > 1):
            raise ValueError(f'CRT unix time stamps differ from the trigger by more than 1 s '
                             f'({unix_diff[np.abs(unix_diff) > 1][0]})')
        t0 = t0 - NS_PER_S * (unix_diff == 1) + NS_PER_S * (unix_diff == -1) - trigger_ns
    sipm = np.argmax(adc, axis=1) if len(adc) else np.zeros(0, dtype=int)
    out = np.empty((len(adc), 4), dtype=np.float64)
    out[:, 0] = 32 * febs.mac5[keep].astype(np.int64) + sipm
    out[:, 1] = t0
    out[:, 2] = febs.ts1[keep]
    out[:, 3] = adc[np.arange(len(adc)), sipm]
    return out
