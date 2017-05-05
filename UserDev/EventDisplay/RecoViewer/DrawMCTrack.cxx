#ifndef EVD_DRAWMCTRACK_CXX
#define EVD_DRAWMCTRACK_CXX

#include "DrawMCTrack.h"

namespace evd {

MCTrack2D getMCTrack2D(sim::MCTrack track, unsigned int plane) {
  MCTrack2D result;
  auto geoHelper = larutil::GeometryHelper::GetME();
  for (unsigned int i = 0; i < track.size(); i++) {
    // project a point into 2D:
    try {
      auto point = geoHelper->Point_3Dto2D(track[i].X(), track[i].Y(), track[i].Z(), plane);
      result._track.push_back(std::make_pair(point.w, point.t));
    }
    catch (...) {
      continue;
    }

  }

  result._origin = track.Origin();

  return result;
}




DrawMCTrack::DrawMCTrack() {
  _name = "DrawMCTrack";
  _fout = 0;
}

bool DrawMCTrack::initialize() {

  // Resize data holder
  if (_dataByPlane.size() != geoService -> Nviews()) {
    _dataByPlane.resize(geoService -> Nviews());
  }
  return true;
}

bool DrawMCTrack::analyze(gallery::Event * ev) {

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
  auto const & trackHandle
        = ev -> getValidHandle<std::vector <sim::MCTrack> >(tracks_tag);



  // Clear out the data but reserve some space for the tracks
  for (unsigned int p = 0; p < geoService -> Nviews(); p ++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(trackHandle -> size());
    _wireRange.at(p).first  = 99999;
    _timeRange.at(p).first  = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }


  // Populate the track vector:
  for (auto & track : *trackHandle) {
    for (unsigned int view = 0; view < geoService -> Nviews(); view++) {
      _dataByPlane.at(view).push_back(getMCTrack2D(track, view));
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
  //   print(MSG::ERROR,__FUNCTION__,"Did not find an output file pointer!!! File not opened?");
  //
  return true;
}

DrawMCTrack::~DrawMCTrack() {}



} // larlite

#endif
