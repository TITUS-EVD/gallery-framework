#ifndef EVD_DRAWTRACK_CXX
#define EVD_DRAWTRACK_CXX

#include "DrawTrack.h"

namespace evd {

Track2D DrawTrack::getTrack2D(recob::Track track, unsigned int plane, unsigned int tpc, unsigned int cryostat) {
  
  Track2D result;
  result._track.reserve(track.NumberTrajectoryPoints());

  larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop);

  for (unsigned int i = 0; i < track.NumberTrajectoryPoints(); i++) {
    // project a point into 2D:
    try {
      if (track.HasValidPoint(i)) {
          auto loc = track.LocationAtPoint(i);
          TVector3 xyz(loc.X(),loc.Y(),loc.Z());
          if (loc.Z() > 1050 && loc.Z() < 1120) {
            std::cout << " > " << loc.X() << loc.Y() << loc.Z() << std::endl;
          }
          auto point = geo_helper.Point_3Dto2D(xyz, plane, tpc, cryostat);
          result._track.push_back(std::make_pair(point.w, point.t));
      }
    } catch (...) {
      continue;
    }
  }

  return result;
}

DrawTrack::DrawTrack(const geo::GeometryCore& geometry, 
                     const detinfo::DetectorProperties& detectorProperties,
                     const detinfo::DetectorClocks& detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawTrack";
  _fout = 0;
}

bool DrawTrack::initialize() {

  _total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();

  // Resize data holder
  if (_dataByPlane.size() != _total_plane_number) {
    _dataByPlane.resize(_total_plane_number);
  }
  return true;
}

bool DrawTrack::analyze(gallery::Event *ev) {

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

  // get a handle to the tracks
  art::InputTag tracks_tag(_producer);
  auto const &trackHandle =
      ev->getValidHandle<std::vector<recob::Track>>(tracks_tag);

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
  art::FindMany<recob::Hit> track_to_hits(trackHandle, *ev, assn_tag);

  // Populate the track vector:
  size_t index = 0;
  for (auto &track : *trackHandle) {
    std::vector<recob::Hit const*> hits;
    track_to_hits.get(index, hits);
    size_t track_tpc = 0, track_cryo = 0;
    if (hits.size() > 0) {
      track_tpc = hits.at(0)->WireID().TPC;
      track_cryo = hits.at(0)->WireID().Cryostat;
    }

    for (unsigned int p = 0; p < _geo_service.Nplanes(track_tpc); p++) {
          
      int plane = p + track_tpc * _geo_service.Nplanes();
      plane += track_cryo * _geo_service.Nplanes() * _geo_service.NTPC(); 
          
      auto tr = getTrack2D(track, p, track_tpc, track_cryo);
      tr._tpc = track_tpc;
      tr._cryo = track_cryo;
      _dataByPlane.at(plane).push_back(tr);

    }


    // for (unsigned int plane = 0; plane < _total_plane_number; plane++) {
    //   _dataByPlane.at(view).push_back(getTrack2D(track, plane));
    // }

    index++;
  }

  return true;
}

bool DrawTrack::finalize() {

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

DrawTrack::~DrawTrack() {}

} // larlite

#endif
