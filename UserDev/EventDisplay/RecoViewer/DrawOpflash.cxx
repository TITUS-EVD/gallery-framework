#ifndef EVD_DRAWOPFLASH_CXX
#define EVD_DRAWOPFLASH_CXX

#include "DrawOpflash.h"

namespace evd {


DrawOpflash::DrawOpflash() {
  _name = "DrawOpflash";
  _fout = 0;
}

bool DrawOpflash::initialize() {

  // // Resize data holder
  // if (_dataByPlane.size() != geoService->Nviews()) {
  //   _dataByPlane.resize(geoService->Nviews());
  // }
  return true;
}

bool DrawOpflash::analyze(gallery::Event *ev) {

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


  art::InputTag opflash_tag(_producer);
  auto const & opflashHandle
        = ev -> getValidHandle<std::vector <recob::OpFlash> >(opflash_tag);

  _extraData.clear();
  _extraData.reserve(opflashHandle -> size());

  // Populate the shower vector:
  for (auto & opf : *opflashHandle) {
    Opflash2D this_flash;
    this_flash._y = opf.YCenter();
    this_flash._z = opf.ZCenter();
    this_flash._time = opf.Time();
    this_flash._y_width = opf.YWidth();
    this_flash._z_width = opf.ZWidth();
    this_flash._time_width = opf.TimeWidth();
    this_flash._total_pe = opf.TotalPE();
    this_flash._opdet_pe = opf.PEs();
    _extraData.push_back(this_flash);
  }

  return true;

}

bool DrawOpflash::finalize() {

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

DrawOpflash::~DrawOpflash() {}

} // larlite

#endif
