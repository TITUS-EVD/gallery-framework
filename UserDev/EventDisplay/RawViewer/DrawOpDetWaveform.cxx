
#ifndef EVD_DRAWOPDETWAVEFORM_CXX
#define EVD_DRAWOPDETWAVEFORM_CXX


#include "DrawOpDetWaveform.h"
#include "LArUtil/DetectorProperties.h"

namespace evd {

DrawOpDetWaveform::DrawOpDetWaveform() {
  _name = "DrawOpDetWaveform";
  _producer = "opdaq";

  _geoService = larutil::Geometry::GetME();
  _detProp = larutil::DetectorProperties::GetME();
  
  import_array();
}


bool DrawOpDetWaveform::initialize() {

  //
  // This function is called in the beggining of event loop
  // Do all variable initialization you wish to do here.
  // If you have a histogram to fill in the event loop, for example,
  // here is a good place to create one on the heap (i.e. "new TH1D").
  //
  //


  // for (unsigned int p = 0; p < geoService -> Nviews(); p ++) {
  //   setXDimension(geoService->Nwires(p), p);
  //   setYDimension(detProp -> ReadOutWindowSize(), p);
  // }
  initDataHolder();



  return true;

}

bool DrawOpDetWaveform::analyze(gallery::Event * ev) {

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

  art::InputTag op_wvf_tag(_producer);
  auto const & op_wvfs
    = ev -> getValidHandle<std::vector <raw::OpDetWaveform> >(op_wvf_tag);


  _wvf_data.clear();
  initDataHolder();

  for (auto const& op_wvf : *op_wvfs) {
    unsigned int       ch   = op_wvf.ChannelNumber();
    double             time = op_wvf.TimeStamp();


    // time offset is at -1250 mu s
    // time tick period is 0.002
    size_t time_in_ticks = (time + 1250) / 0.002;

    int n_op_channels = 600;
    int n_time_ticks = 625000*3;

    int offset = ch * n_time_ticks;


    size_t i = 0;
    for (short adc : op_wvf) {
      _wvf_data.at(offset + time_in_ticks + i) = (float)adc;
      i++;
    }

  //   unsigned int detWire = geoService->ChannelToWire(ch);
  //   unsigned int plane = geoService->ChannelToPlane(ch);
  //   unsigned int tpc = geoService->ChannelToTPC(ch);

  //   // If a second TPC is present, its planes 0, 1 and 2 are 
  //   // stored consecutively to those of the first TPC. 
  //   // So we have planes 0, 1, 2, 3, 4, 5.
  //   plane += tpc * (geoService->Nplanes() / geoService->NTPC());
    
  //   int offset = detWire * _y_dimensions[plane] + _padding_by_plane[plane];

  //   for (auto & iROI : wire.SignalROI().get_ranges()) {
  //     // for (auto iROI = wire.SignalROI().begin_range(); wire.SignalROI().end_range(); ++iROI) {
  //     const int FirstTick = iROI.begin_index();
  //     if (plane == 2) {
  //     }
  //     size_t i = 0;
  //     for (float ADC : iROI) {
  //       _wvf_data.at(plane).at(offset + FirstTick + i) = ADC;
  //       i ++;
  //     }


  //   }
  }

  return true;
}

bool DrawOpDetWaveform::finalize() {

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

void DrawOpDetWaveform::initDataHolder() {
  // for now 3 windows of 1250 mu s
  // there are 625000 time ticks per window
  // so a tick period of 0.002 mu s
  int n_op_channels = 600;
  int n_time_ticks = 625000*3;
  float default_value = -9999.;
  _wvf_data.clear();
  _wvf_data.resize(n_op_channels * n_time_ticks, default_value);
  // for (size_t i = 0; i < n_op_channels; i++) {
  //   _wvf_data.at(i).resize(n_time_ticks);
  // }
  return;
}


PyObject * DrawOpDetWaveform::getArray() {

  PyObject * returnNull = nullptr;

  int n_op_channels = 600;
  int n_time_ticks = 625000*3;

  try {
    // Convert the wire data to numpy arrays:
    int n_dim = 2;
    int * dims = new int[n_dim];
    dims[0] = n_op_channels;
    dims[1] = n_time_ticks;
    int data_type = PyArray_FLOAT;

    return (PyObject *) PyArray_FromDimsAndData(n_dim, dims, data_type, (char*) & ((_wvf_data)[0]) );
  }
  catch ( ... ) {
    std::cerr << "WARNING: CANNOT GET OP DET WAVEFORM.\n";
    return returnNull;
  }

}

}

#endif