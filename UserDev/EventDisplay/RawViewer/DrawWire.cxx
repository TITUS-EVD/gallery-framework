
#ifndef EVD_DRAWWIRE_CXX
#define EVD_DRAWWIRE_CXX


#include "DrawWire.h"
#include "LArUtil/DetectorProperties.h"

namespace evd {

DrawWire::DrawWire() {
  _name = "DrawWire";
  _producer = "caldata";

}

void DrawWire::setPadding(size_t padding, size_t plane) {
  if (_padding_by_plane.size() > plane) {
    _padding_by_plane[plane] = padding;
  }
}

bool DrawWire::initialize() {

  //
  // This function is called in the beggining of event loop
  // Do all variable initialization you wish to do here.
  // If you have a histogram to fill in the event loop, for example,
  // here is a good place to create one on the heap (i.e. "new TH1D").
  //
  //

  _padding_by_plane.resize(geoService -> Nviews());

  for (unsigned int p = 0; p < geoService -> Nviews(); p ++) {
    setXDimension(geoService->Nwires(p), p);
    setYDimension(detProp -> ReadOutWindowSize(), p);
  }
  initDataHolder();



  return true;

}

bool DrawWire::analyze(gallery::Event * ev) {

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

  // This is an event viewer.  In particular, this handles raw wire signal drawing.
  // So, obviously, first thing to do is to get the wires.

  art::InputTag wires_tag(_producer);
  auto const & wires
    = ev -> getValidHandle<std::vector <recob::Wire> >(wires_tag);


  _planeData.clear();
  initDataHolder();

  for (auto const& wire : *wires) {
    unsigned int ch = wire.Channel();
    unsigned int detWire = geoService->ChannelToWire(ch);
    unsigned int plane = geoService->ChannelToPlane(ch);
    int offset = detWire * _y_dimensions[plane] + _padding_by_plane[plane];

    for (auto & iROI : wire.SignalROI().get_ranges()) {
      // for (auto iROI = wire.SignalROI().begin_range(); wire.SignalROI().end_range(); ++iROI) {
      const int FirstTick = iROI.begin_index();
      size_t i = 0;
      for (float ADC : iROI) {
        _planeData.at(plane).at(offset + FirstTick + i) = ADC;
        i ++;
      }


    }
  }

  return true;
}

bool DrawWire::finalize() {

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

}

#endif