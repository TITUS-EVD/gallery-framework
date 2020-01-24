#ifndef EVD_DRAWMCTRACK_CXX
#define EVD_DRAWMCTRACK_CXX

#include "DrawMCTrack.h"

namespace evd {

MCTrack2D DrawMCTrack::getMCTrack2D(sim::MCTrack track, unsigned int plane, unsigned int tpc, unsigned int cryostat) {
  larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop);
  MCTrack2D result;
  // auto geoHelper = larutil::GeometryHelper::GetME();
  result._track.reserve(track.size());
  for (unsigned int i = 0; i < track.size(); i++) {
    // project a point into 2D:
    try {
      auto point = geo_helper.Point_3Dto2D(track[i].X(), track[i].Y(), track[i].Z(), 
                                           plane, tpc, cryostat);
      // auto point = geoHelper->Point_3Dto2D(track[i].X(), track[i].Y(),
      //                                      track[i].Z(), plane);
      result._track.push_back(std::make_pair(point.w, point.t));
    } catch (...) {
      continue;
    }
  }

  result._origin = track.Origin();
  result._pdg = track.PdgCode();

  return result;
}

DrawMCTrack::DrawMCTrack(const geo::GeometryCore& geometry, const detinfo::DetectorProperties& detectorProperties) :
    RecoBase(geometry, detectorProperties) 
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

bool DrawMCTrack::analyze(gallery::Event *ev) {

  //
  // Do your event-by-event analysis here. This function is called for
  // each event in the loop. You have "storage" pointer which contains
  // event-wise data. To see what is available, check the "Manual.pdf":
  //
  // http://microboone-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=3183
  //
  // Or you can refer to Base/DataFormatConstants.hh for available data type
  // enum values. Here is one example of getting PMT waveform collection.
  //
  // event_fifo* my_pmtfifo_v = (event_fifo*)(storage->get_data(DATA::PMFIFO));
  //
  // if( event_fifo )
  //
  //   std::cout << "Event ID: " << my_pmtfifo_v->event_id() << std::endl;
  //

  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();


  art::InputTag truth_tag("generator::GenieGen");
  auto const &truthHandle =
      ev->getValidHandle<std::vector<simb::MCTruth>>(truth_tag);
  for (auto &truth : *truthHandle) {
    std::cout << "Neutrino energy: " << truth.GetNeutrino().Nu().E() << std::endl;

  }

  // get a handle to the tracks
  art::InputTag tracks_tag(_producer);
  auto const &trackHandle =
      ev->getValidHandle<std::vector<sim::MCTrack>>(tracks_tag);

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
      auto tr = getMCTrack2D(track, p, track_tpc, track_cryo);
      tr._tpc = track_tpc;
      tr._cryo = track_cryo;
      _dataByPlane.at(p).push_back(tr);
    }
  }

  return true;
}

bool DrawMCTrack::finalize() {

  // This function is called at the end of event loop.
  // Do all variable finalization you wish to do here.
  // If you need, you can store your ROOT class instance in the output
  // file. You have an access to the output file through "_fout" pointer.
  //
  // Say you made a histogram pointer h1 to store. You can do this:
  //
  // if(_fout) { _fout->cd(); h1->Write(); }
  //
  // else
  //   print(MSG::ERROR,__FUNCTION__,"Did not find an output file pointer!!!
  //   File not opened?");
  //
  return true;
}

DrawMCTrack::~DrawMCTrack() {}

} // larlite

#endif
