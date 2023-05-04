#ifndef EVD_DRAWMCTRACK_CXX
#define EVD_DRAWMCTRACK_CXX

#include "DrawMCTrack.h"

namespace evd {

MCTrack2D DrawMCTrack::getMCTrack2D(sim::MCTrack track, unsigned int plane) {
  larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop, _det_clock);
  MCTrack2D result;
  result._track.reserve(track.size());

  std::cout << "[DrawMCTrack] track.size() " << track.size() << std::endl;

  for (unsigned int i = 0; i < track.size(); i++) {
    // project a point into 2D:
    auto point = geo_helper.Point_3Dto2D(track[i].X(), track[i].Y(), track[i].Z(), plane);
    if (point.w != -9999) {
      result._track.push_back(std::make_pair(point.w, point.t));
      result._tpc.push_back(point.tpc);
      result._cryo.push_back(point.cryo);
    }
  }
  return result;
}

DrawMCTrack::DrawMCTrack(const geo::GeometryCore&               geometry,
                         const detinfo::DetectorPropertiesData& detectorProperties,
                         const detinfo::DetectorClocksData&     detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawMCTrack";
  _fout = 0;
}

bool DrawMCTrack::initialize() {

  // Resize data holder
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  if (_dataByPlane.size() != total_plane_number) {
    _dataByPlane.resize(total_plane_number);
  }
  return true;
}

bool DrawMCTrack::analyze(const gallery::Event & ev) {

  size_t total_plane_number = _geo_service.Nplanes(); // * _geo_service.NTPC() * _geo_service.Ncryostats();

  // get a handle to the tracks
  art::InputTag tracks_tag(_producer);
  auto const &trackHandle =
      ev.getValidHandle<std::vector<sim::MCTrack>>(tracks_tag);

  // Clear out the data but reserve some space for the tracks
  for (unsigned int p = 0; p < total_plane_number; p++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(trackHandle->size());
    _wireRange.at(p).first = 99999;
    _timeRange.at(p).first = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }

  // just a placeholder for now
  unsigned int track_tpc = 0;
  unsigned int track_cryo = 0;

  // Populate the track vector:
  for (auto &track : *trackHandle) {
    for (unsigned int p = 0; p < total_plane_number; p++) {
      auto tr = getMCTrack2D(track, p);

      // Figure out the track time in elec clock
      // (still need to subtract trigger time)
      tr._time = _det_clock.G4ToElecTime(track.Start().T());

      tr._process = track.Process();
      tr._energy = track.Start().E();

      tr._origin = track.Origin();
      tr._pdg = track.PdgCode();

      _dataByPlane.at(p).push_back(tr);
    }
  }

  return true;
}

bool DrawMCTrack::finalize() {

  return true;
}

DrawMCTrack::~DrawMCTrack() {}

} // larlite

#endif
