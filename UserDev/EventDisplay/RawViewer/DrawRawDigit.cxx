#ifndef EVD_DRAWRAWDIGIT_CXX
#define EVD_DRAWRAWDIGIT_CXX

#include "DrawRawDigit.h"

namespace evd {

DrawRawDigit::DrawRawDigit(const geo::GeometryCore& geometry, const detinfo::DetectorProperties& detectorProperties) :
  RawBase(geometry, detectorProperties)
{
  _name = "DrawRawDigit";
  _producer = "daq";

  // And whether or not to correct the data:
  _correct_data = false;
}

void DrawRawDigit::setPadding(size_t padding, size_t plane) {
  if (_padding_by_plane.size() > plane) {
    _padding_by_plane[plane] = padding;
  }
}

bool DrawRawDigit::initialize() {
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
    for (unsigned int t = 0; t < _geo_service.NTPC(c); t++) {
      for (unsigned int p = 0; p < _geo_service.Nplanes(t); p++) {
        setXDimension(_geo_service.Nwires(p, t, c), counter);
        setYDimension(_det_prop.ReadOutWindowSize(), counter);
        counter++;
      }
    }
  }
  initDataHolder();

  // if (_geo_service.DetectorName() == "microboone") {
  //   _noise_filter.init();
  // }

  return true;
}

bool DrawRawDigit::analyze(gallery::Event *ev) {
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

  // This is an event viewer.  In particular, this handles raw wire signal
  // drawing.
  // So, obviously, first thing to do is to get the wires.

  // std::cout << " DetectorName " << _geo_service.DetectorName() << std::endl;

  std::vector<gallery::ValidHandle<std::vector<raw::RawDigit>>> raw_digits_v;

  if (_producer != "") {
    std::cout << "Drawing RawDigits using producer " << _producer << std::endl;
    art::InputTag wires_tag(_producer);
    auto const & raw_digits = ev->getValidHandle<std::vector<raw::RawDigit>>(wires_tag);
    raw_digits_v.push_back(raw_digits);
  } else {
    for (auto p : _producers) {
      std::cout << "Drawing RawDigits using producer " << p << std::endl;
      art::InputTag wires_tag(p);
      auto const & raw_digits = ev->getValidHandle<std::vector<raw::RawDigit>>(wires_tag);
      raw_digits_v.push_back(raw_digits);
    }
  }

  // if the tick-length set is different from what is actually stored in the ADC
  // vector -> fix.
  size_t n_views = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();

  if (raw_digits_v[0]->size() > 0) {
    for (size_t pl = 0; pl < n_views; pl++) {
      if (_y_dimensions[pl] < raw_digits_v[0]->at(0).ADCs().size()) {
        _y_dimensions[pl] = raw_digits_v[0]->at(0).ADCs().size();
      }
    }
  }


  _planeData.clear();
  size_t n_ticks = 0;

  for (auto r : raw_digits_v) {
    if ((*r).size()) {
      n_ticks = r->front().ADCs().size();
      break;
    }
  }

  // if (_geo_service.DetectorName() == "microboone") {
  //   _noise_filter.set_n_time_ticks(n_ticks);
  // }
  initDataHolder();

  // If the output data holder is not the same size as RawDigit length,
  // it messes up the noise filter.  Easist thing to do here is to
  // temporarily store the data in it's native format, then copy
  // the data over to the final output.

  std::vector<std::vector<float>> temp_data_holder(n_views);

  for (size_t i_plane = 0; i_plane < n_views; i_plane++) {
    temp_data_holder.at(i_plane).resize(n_ticks * _x_dimensions[i_plane]);
  }

  for (auto const &raw_digits : raw_digits_v) {
    for (auto const &rawdigit : *raw_digits) 
    {
      unsigned int ch  = rawdigit.Channel();
      float        ped = rawdigit.GetPedestal();

      std::vector<geo::WireID> widVec = _geo_service.ChannelToWire(ch);
      for (geo::WireID w_id : widVec) {
        unsigned int wire = w_id.Wire;
        unsigned int plane = w_id.Plane;
        unsigned int tpc = w_id.TPC;
        unsigned int cryo = w_id.Cryostat;  

        std::cout << "RawDigit ch " << ch << ", wire " << wire << ", plane " << plane << ", tpc " << tpc << ", cryo " << cryo << std::endl;

        if (wire > _geo_service.Nwires(plane, tpc, cryo)) continue;  

        if (_geo_service.DetectorName() == "microboone" && ch >= 8254) continue;  

        // If a second TPC is present, its planes 0, 1 and 2 are 
        // stored consecutively to those of the first TPC. 
        // So we have planes 0, 1, 2, 3, 4, 5.
        plane += tpc * _geo_service.Nplanes();
        plane += cryo * _geo_service.Nplanes() * _geo_service.NTPC();  

        int offset = wire * n_ticks;  

        std::vector<float>&          planeRawDigitVec = temp_data_holder[plane];
        std::vector<float>::iterator startItr         = planeRawDigitVec.begin() + offset;  

        float pedestal = 0;
        if (_subtract_pedestal) {
          pedestal = ped;
        }
        // Copy with pedestal subtraction
        for(const auto& adcVal : rawdigit.ADCs()) {
          *startItr++ = adcVal - pedestal;
        }
      }
    }
  }


  // if (_geo_service.DetectorName() == "microboone") {
  //   _noise_filter.set_data(&temp_data_holder);
  //   if (_correct_data && ev->eventAuxiliary().isRealData()) {
  //     _noise_filter.clean_data();
  //   } else {
  //     _noise_filter.pedestal_subtract_only();
  //   }
  // }

  // In some cases, raw digits are truncated and the padding is needed.
  // In other cases, raw digits are not truncated and no padding is needed,
  // even in a truncate file.  What a mess.
  // Hack: wipe out the padding if it's clearly not needed:

  std::vector<int> _temp_padding_by_plane(_padding_by_plane.size(), 0);

  for (size_t i_plane = 0; i_plane < n_views; i_plane++) {
    if (n_ticks + _padding_by_plane[i_plane] > _y_dimensions[i_plane]) {
      _temp_padding_by_plane[i_plane] = 0;
    }
    else{
      _temp_padding_by_plane[i_plane] = _padding_by_plane[i_plane];
    }
  }

  // Now, copy the data from the temp storage to the output storage:
  for (size_t i_plane = 0; i_plane < n_views; i_plane++) {

    size_t n_wires = temp_data_holder.at(i_plane).size() / n_ticks;

    for (size_t i_wire = 0; i_wire < n_wires; i_wire++) {
      int offset_raw   = i_wire * n_ticks;
      int offset_final = i_wire * _y_dimensions[i_plane] + _temp_padding_by_plane[i_plane];
      for (size_t i_tick = 0; i_tick < n_ticks; i_tick++) {
        _planeData.at(i_plane).at(offset_final + i_tick) =
            temp_data_holder.at(i_plane).at(offset_raw + i_tick);
      }
    }
  }

  return true;
}

bool DrawRawDigit::finalize() {
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
}
#endif
