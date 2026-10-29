import numpy as np

from titus.display import mc_t0
from titus.display.projection import AS_STORED
from titus.drawables.track import Track
from titus.readers import Tracks


def mctracks_as_tracks(mctracks):
    ''' sim::MCTrack steps as Tracks trajectories (every step valid), for the track projection '''
    return Tracks(id=mctracks.track_id, point_offset=mctracks.step_offset, x=mctracks.x, y=mctracks.y,
                  z=mctracks.z, valid=np.ones(len(mctracks.x), dtype=bool))


class MCTrack(Track):
    '''
    sim::MCTrack: simulated particle paths, projected like recob::Track.
    Corrected (the TPC module's "Correct 3D objects" option), a particle's
    time relative to the trigger is its T0, so the path lands on the hits its
    charge made (out-of-time cosmics move), and each step is drawn in the TPC
    whose readout window contains it. Particles from a neutrino are gray,
    others black.
    '''
    KIND = 'mctracks'
    COLOR = (0, 0, 0)
    NEUTRINO_COLOR = (128, 128, 128)
    NEUTRINO_ORIGIN = 1  # simb::kBeamNeutrino

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, geom, tpc_module, *args, **kwargs)
        self._product_name = 'mctrack'
        self._mctracks = []

    def on_data(self):
        self._mctracks = self._data
        self._data = [mctracks_as_tracks(m) for m in self._mctracks]
        self._inputs = [{'t0': mc_t0(self._geom, m.start_t), 'track_hits': None, 'hits': None}
                        for m in self._mctracks]
        self._corrections = [{} for _ in self._data]

    def color(self, producer, row):
        origin = self._mctracks[producer].origin[row]
        return self.NEUTRINO_COLOR if origin == self.NEUTRINO_ORIGIN else self.COLOR

    def genToolTip(self, tracks, row, correction):
        mctracks = next(m for m, t in zip(self._mctracks, self._data) if t is tracks)
        what = correction.describe() if correction is not None else AS_STORED
        return (f'MCTrack {mctracks.track_id[row]}: PDG {mctracks.pdg[row]}, '
                f'{mctracks.start_e[row] / 1e3:.3f} GeV, {mctracks.process[row]}\n'
                f'time {mc_t0(self._geom, mctracks.start_t[row]) / 1e3:.2f} us from the trigger\n{what}')
