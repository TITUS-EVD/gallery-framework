#ifndef EVD_DRAWNUMUSELECTION_CXX
#define EVD_DRAWNUMUSELECTION_CXX

#include "DrawNumuSelection.h"

namespace evd {

NumuSelection2D DrawNumuSelection::getNumuSelection2D(recob::Vertex vtx, std::vector<recob::Track> tracks, unsigned int plane) {

  NumuSelection2D result;

  // Vertex
  try {
    double pos[3];
    vtx.XYZ(pos);
    auto point = geoHelper->Point_3Dto2D(pos, plane);
    result._vertex = point;
  } catch (...) {
  }

  // Tracks
  double max_length = -1;
  size_t muon_index = 0;
  std::vector<Track2D> track_v;
  track_v.reserve(tracks.size());
  for (size_t t = 0; t < tracks.size(); t++) {
    recob::Track trk = tracks.at(t);
    Track2D trk_out;
    trk_out._track.reserve(trk.NumberTrajectoryPoints());
    for (unsigned int i = 0; i < trk.NumberTrajectoryPoints(); i++) {
      try {
        auto point = geoHelper->Point_3Dto2D(trk.LocationAtPoint(i), plane);
        trk_out._track.push_back(std::make_pair(point.w, point.t));
      } catch (...) {
        continue;
      }
    }
    track_v.emplace_back(trk_out);

    // the longest track is the muon
    double length = trk.Length();
    if (length > max_length){
      muon_index = t;
      max_length = length;
    }
  }

  result._tracks = track_v;
  result._muon_index = muon_index;

  return result;
}

DrawNumuSelection::DrawNumuSelection() {
  _name = "DrawNumuSelection";
  _fout = 0;
}

bool DrawNumuSelection::initialize() {

  // Resize data holder
  if (_dataByPlane.size() != geoService->Nviews()) {
    _dataByPlane.resize(geoService->Nviews());
  }
  return true;
}

bool DrawNumuSelection::analyze(gallery::Event *ev) {

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

  //std::cout << "Producer is " << _producer << std::endl;

  /*
  std::vector<recob::Vertex> vertices;
  std::vector<recob::Track> tracks;
 
  vertices.clear();
  tracks.clear();
  

  auto const& assoc_handle = ev->getValidHandle< art::Assns<recob::Vertex,recob::Track> >(_producer);

  if(assoc_handle->size() == 0) 
    return true; // no selected neutrino in this event

  std::cout << "Ass has size " << assoc_handle->size() << std::endl;

  for (auto &ass : *assoc_handle) {
    art::Ptr<recob::Vertex> v = ass.first;
    vertices.emplace_back(*v);

    art::Ptr<recob::Track>  t = ass.second;
    tracks.emplace_back(*t);
  }
  */
  
  art::InputTag tag("pandoraNu::DataApr2016RecoStage2");
  auto const &trackHandle = ev->getValidHandle<std::vector<recob::Track>>(tag);
  art::FindMany<recob::Vertex> vtx_per_track(trackHandle,*ev,_producer);

  std::vector<recob::Vertex const*> vertices;
  std::vector<recob::Track> tracks;

  for (size_t index = 0; index < vtx_per_track.size(); index++) {
    vtx_per_track.get(index, vertices);
    if (vertices.size() != 0)
      tracks.push_back((*trackHandle)[index]);
  }
  


  // Clear out the data but reserve some space for the tracks
  for (unsigned int p = 0; p < geoService->Nviews(); p++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(vertices.size());
    _wireRange.at(p).first = 99999;
    _timeRange.at(p).first = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }


  for (auto const &vtx : vertices) {
    for (unsigned int view = 0; view < geoService->Nviews(); view++) {
      _dataByPlane.at(view).push_back(this->getNumuSelection2D(*vtx, tracks, view));
    }
  }  

  return true;
}

bool DrawNumuSelection::finalize() {

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

DrawNumuSelection::~DrawNumuSelection() {}

} // larlite

#endif
