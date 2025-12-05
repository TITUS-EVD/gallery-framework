#ifndef EVD_DRAWSHOWER3D_CXX
#define EVD_DRAWSHOWER3D_CXX

#include "DrawShower3D.h"

namespace evd {


DrawShower3D::DrawShower3D() {
  _name = "DrawShower3D";
  _fout = 0;
}

bool DrawShower3D::initialize() {

  return true;
}

bool DrawShower3D::analyze(gallery::Event * ev) {

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





  // get a handle to the showers
  art::InputTag shower_tag(_producer);
  auto const & showerHandle
        = ev -> getValidHandle<std::vector <recob::Shower > >(shower_tag);

  // Clear out the data but reserve some space
    _data.clear();
    _data.reserve(showerHandle -> size());


  // Populate the shower vector:
  for (auto & shower : *showerHandle) {
    _data.push_back(getShower3d(shower));
  }


  return true;
}

bool DrawShower3D::finalize() {

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

DrawShower3D::~DrawShower3D() {}

Shower3D DrawShower3D::getShower3d(const recob::Shower & shower) {
  Shower3D result;


  result._start_point = shower.ShowerStart();
  result._direction = shower.Direction();
  result._length = shower.Length();
  result._opening_angle = 0.2;

  return result;
}


} // evd

#endif
