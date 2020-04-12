#ifndef EVD_DRAWWIRE_CXX
#define EVD_DRAWWIRE_CXX


#include "DrawWire.h"
#include "LArUtil/DetectorProperties.h"
#include "lardataobj/RecoBase/Wire.h"

namespace evd {

DrawWire::DrawWire(const geo::GeometryCore& geometry, const detinfo::DetectorProperties& detectorProperties) :
  RawBase(geometry, detectorProperties)
{
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
  _padding_by_plane.resize(_geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats());
  int counter = 0;
  for (unsigned int c = 0; c < _geo_service.Ncryostats(); c++) {
    for (unsigned int t = 0; t < _geo_service.NTPC(); t++) {
      for (unsigned int p = 0; p < _geo_service.Nplanes(t); p++) {
        setXDimension(_geo_service.Nwires(p, t, c), counter);
        setYDimension(_det_prop.ReadOutWindowSize(), counter);
        counter++;
      }
    }
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

  // art::InputTag wires_tag(_producer);
  // auto const & wires
  //   = ev -> getValidHandle<std::vector <recob::Wire> >(wires_tag);

  std::vector<gallery::ValidHandle<std::vector<recob::Wire>>> wire_v;


  if (_producer != "") {
    std::cout << "Drawing Wires using producer " << _producer << std::endl;
    art::InputTag wires_tag(_producer);
    auto const & wires = ev->getValidHandle<std::vector<recob::Wire>>(wires_tag);
    wire_v.push_back(wires);
  } else {
    for (auto p : _producers) {
      std::cout << "Drawing Wires using producer " << p << std::endl;
      art::InputTag wires_tag(p);
      auto const & wires = ev->getValidHandle<std::vector<recob::Wire>>(wires_tag);
      wire_v.push_back(wires);
    }
  }

  _planeData.clear();
  initDataHolder();

  for (auto const &wires : wire_v) {
    for (auto const& wire : *wires) {
      unsigned int ch = wire.Channel();
      std::vector<geo::WireID> widVec = _geo_service.ChannelToWire(ch);
      for (geo::WireID w_id : widVec) {
        size_t detWire = w_id.Wire;
        size_t plane   = w_id.Plane;
        size_t tpc     = w_id.TPC;
        size_t cryo    = w_id.Cryostat;  
  
        // If a second TPC is present, its planes 0, 1 and 2 are 
        // stored consecutively to those of the first TPC. 
        // So we have planes 0, 1, 2, 3, 4, 5.
        plane += tpc * _geo_service.Nplanes();
        plane += cryo * _geo_service.Nplanes() * _geo_service.NTPC();
      
        int offset = detWire * _y_dimensions[plane] + _padding_by_plane[plane];  

        std::vector<float>&          planeData   = _planeData[plane];
        std::vector<float>::iterator wireDataItr = planeData.begin() + offset;  

        for (auto & iROI : wire.SignalROI().get_ranges()) {
        
          size_t                       firstTick = iROI.begin_index();
          std::vector<float>::iterator adcItr    = wireDataItr + firstTick;  

          std::copy(iROI.begin(),iROI.end(),adcItr);  

          // for (auto iROI = wire.SignalROI().begin_range(); wire.SignalROI().end_range(); ++iROI) {
          // const int firstTick = iROI.begin_index();  

          // size_t i = 0;
          // for (float ADC : iROI) {
          //   _planeData.at(plane).at(offset + firstTick + i) = ADC;
          //   i ++;
          // }  

        }
      } // wire id loop
    } // wires loop
  } // wire producer loop

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