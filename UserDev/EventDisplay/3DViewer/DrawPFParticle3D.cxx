#ifndef DRAWPFPARTICLE3D_CXX
#define DRAWPFPARTICLE3D_CXX

#include "DrawPFParticle3D.h"

namespace evd {


DrawPFParticle3D::DrawPFParticle3D() {
  _name = "DrawPFParticle3D";
  _fout = 0;
}

bool DrawPFParticle3D::initialize() {

  return true;
}

bool DrawPFParticle3D::analyze(gallery::Event * ev) {

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


  // get a handle to the particles

  art::InputTag pfpart_tag(_producer);
  auto const & pfpartHandle
        = ev -> getValidHandle<std::vector <recob::PFParticle> >(pfpart_tag);



  // Get the associated spacepoints for this particle:
  art::InputTag assn_tag(_producer);
  art::FindMany<recob::SpacePoint> sps_for_pfpart(pfpartHandle, *ev, assn_tag);



  // Clear out the data but reserve some space
  _data.clear();
  _data.reserve(pfpartHandle -> size());


  // Populate the particles:
  size_t index = 0;
  for (auto & pfpart : * pfpartHandle) {

    std::vector<recob::SpacePoint const*> sps_v;
    sps_for_pfpart.get(index, sps_v);

    if (sps_v.size() == 0)
      continue;

    _data.push_back(PFPart3D());
    for (auto sps : sps_v){
      auto xyz = sps->XYZ();
      _data.back()._points.push_back(larutil::Point3D(xyz[0],xyz[1],xyz[2]));
    }

    index ++;
  }


  return true;
}

bool DrawPFParticle3D::finalize() {

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

DrawPFParticle3D::~DrawPFParticle3D() {}


} // evd

#endif
