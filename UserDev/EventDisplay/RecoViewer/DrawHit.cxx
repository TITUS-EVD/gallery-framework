#ifndef EVD_DRAWHIT_CXX
#define EVD_DRAWHIT_CXX

#include "DrawHit.h"

namespace evd {


DrawHit::DrawHit(const geo::GeometryCore&               geometry,
                 const detinfo::DetectorPropertiesData& detectorProperties,
                 const detinfo::DetectorClocksData&     detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawHit";
  _fout = 0;
}

bool DrawHit::initialize() {
  // Resize data holder to accommodate planes and wires:
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  if (_dataByPlane.size() != total_plane_number) {
    _dataByPlane.resize(total_plane_number);
    _maxCharge.resize(total_plane_number);
  }
  return true;
}

bool DrawHit::analyze(const gallery::Event & ev) {

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


  // get a handle to the hits

  art::InputTag hits_tag(_producer);
  auto const & hitHandle
        = ev.getValidHandle<std::vector <recob::Hit> >(hits_tag);


  // Clear out the hit data but reserve some space for the hits
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  for (unsigned int p = 0; p < total_plane_number; p ++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(hitHandle->size());
    _maxCharge.at(p) = 0.0;
    _wireRange.at(p).first  = 99999;
    _timeRange.at(p).first  = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }




  for (auto & hit : *hitHandle) {

    // unsigned int wire = hit.WireID().Wire;
    unsigned int plane = hit.WireID().Plane;
    unsigned int tpc = hit.WireID().TPC;
    unsigned int cryo = hit.WireID().Cryostat;

    // If a second TPC is present, its planes 0, 1 and 2 are
    // stored consecutively to those of the first TPC.
    // So we have planes 0, 1, 2, 3, 4, 5.
    plane += tpc * _geo_service.Nplanes();
    plane += cryo * _geo_service.Nplanes() * _geo_service.NTPC();

    _dataByPlane.at(plane).emplace_back(
      Hit2D(hit.WireID().Wire,
            hit.PeakTime(),
            hit.Integral(),
            hit.RMS(),
            hit.StartTick(),
            hit.PeakTime(),
            hit.EndTick(),
            hit.PeakAmplitude(),
            plane,
            tpc,
            cryo
           ));
    if (_dataByPlane.at(plane).back()._charge > _maxCharge.at(plane)) {
      if (plane == 3) std::cout << "Increasing maxcharge for plane 3" << std::endl;
      _maxCharge.at(plane) = _dataByPlane.at(plane).back()._charge;
    }
    // Check the auto range values:
    if (_dataByPlane.at(plane).back().wire() < _wireRange.at(plane).first) {
      _wireRange.at(plane).first = _dataByPlane.at(plane).back().wire();
    }
    if (_dataByPlane.at(plane).back().wire() > _wireRange.at(plane).second) {
      _wireRange.at(plane).second = _dataByPlane.at(plane).back().wire();
    }
    if (_dataByPlane.at(plane).back().time() < _timeRange.at(plane).first) {
      _timeRange.at(plane).first = _dataByPlane.at(plane).back().time();
    }
    if (_dataByPlane.at(plane).back().time() > _timeRange.at(plane).second) {
      _timeRange.at(plane).second = _dataByPlane.at(plane).back().time();
    }

  }


  return true;
}

float DrawHit::maxCharge(size_t p) {
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  if (p >= total_plane_number) {
    std::cerr << "ERROR: Request for nonexistent plane " << p << std::endl;
    return 1.0;
  }
  else {
    try {
      return _maxCharge.at(p);
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return 1.0;
    }
  }


}

std::vector<Hit2D> DrawHit::getHitsOnWirePlane(size_t wire, size_t plane) {
  std::vector<Hit2D> result;
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();

  if (plane >= total_plane_number) {
    std::cerr << "ERROR: Request for nonexistent plane " << plane << std::endl;
    return result;
  }
  else {
    for (auto & hit : _dataByPlane.at(plane)) {
      if (hit.wire() == wire)
        result.emplace_back(hit);
    }
  }

  return result;
}


bool DrawHit::finalize() {

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




} // larlite

#endif
