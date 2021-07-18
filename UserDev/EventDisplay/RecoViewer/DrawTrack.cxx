#ifndef EVD_DRAWTRACK_CXX
#define EVD_DRAWTRACK_CXX

#include "DrawTrack.h"

namespace evd {

Track2D DrawTrack::getTrack2D(recob::Track track, unsigned int plane) {

  Track2D result;
  result._track.reserve(track.NumberTrajectoryPoints());

  larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop, _det_clock);

  for (unsigned int i = 0; i < track.NumberTrajectoryPoints(); i++) {
    // project a point into 2D:
    if (track.HasValidPoint(i)) {
      auto loc = track.LocationAtPoint(i);
      TVector3 xyz(loc.X(),loc.Y(),loc.Z());
      auto point = geo_helper.Point_3Dto2D(xyz, plane);
      if (point.w != -9999) {
        result._track.push_back(std::make_pair(point.w, point.t));
        result._tpc.push_back(point.tpc);
        result._cryo.push_back(point.cryo);
      }
    }
  }

  return result;
}

DrawTrack::DrawTrack(const geo::GeometryCore&               geometry,
                     const detinfo::DetectorPropertiesData& detectorProperties,
                     const detinfo::DetectorClocksData&     detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawTrack";
  _fout = 0;
}

bool DrawTrack::initialize() {

  _total_plane_number = _geo_service.Nplanes(); // * _geo_service.NTPC() * _geo_service.Ncryostats();

  // Resize data holder
  if (_dataByPlane.size() != _total_plane_number) {
    _dataByPlane.resize(_total_plane_number);
  }
  return true;
}

bool DrawTrack::analyze(const gallery::Event &ev) {

  // get a handle to the tracks
  art::InputTag tracks_tag(_producer);
  auto const &trackHandle =
      ev.getValidHandle<std::vector<recob::Track>>(tracks_tag);

  // Clear out the data but reserve some space for the tracks
  for (unsigned int p = 0; p < _total_plane_number; p++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(trackHandle->size());
    _wireRange.at(p).first = 99999;
    _timeRange.at(p).first = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }

  // Retrieve the hits to infer the TPC the track belongs to
  art::InputTag assn_tag(_producer);
  art::FindMany<recob::Hit> track_to_hits(trackHandle, ev, assn_tag);

  // Populate the track vector:
  size_t index = 0;
  for (auto &track : *trackHandle) {
    std::vector<recob::Hit const*> hits;
    track_to_hits.get(index, hits);

    for (unsigned int p = 0; p < _geo_service.Nplanes(); p++) {

      auto tr = getTrack2D(track, p);

      tr._length = track.Length();
      tr._chi2 = track.Chi2();
      tr._theta = track.Theta();
      tr._phi = track.Phi();

      _dataByPlane.at(p).push_back(tr);

    }

    index++;
  }

  return true;
}

bool DrawTrack::finalize() {

  return true;
}

DrawTrack::~DrawTrack() {}

}

#endif
