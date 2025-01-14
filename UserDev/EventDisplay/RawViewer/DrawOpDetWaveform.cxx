
#ifndef EVD_DRAWOPDETWAVEFORM_CXX
#define EVD_DRAWOPDETWAVEFORM_CXX

#include <numeric>

#include "DrawOpDetWaveform.h"
#include "LArUtil/DetectorProperties.h"

namespace evd {

DrawOpDetWaveform::DrawOpDetWaveform(const geo::GeometryCore&               geometry,
                                     const detinfo::DetectorPropertiesData& detectorProperties,
                                     const detinfo::DetectorClocksData&     detectorClocks) :
  _geo_service(geometry),
  _det_prop(detectorProperties),
  _det_clocks(detectorClocks)
{
  _name = "DrawOpDetWaveform";
  _producer = "opdaq";

  _import_array();
}


bool DrawOpDetWaveform::initialize() {

  _tick_period = _det_clocks.OpticalClock().TickPeriod();
  _n_op_channels = _geo_service.NOpDets();
  _n_time_ticks = _det_clocks.OpticalClock().FrameTicks() * _n_frames;
  _n_time_ticks /= _n_size_reduction;

  std::cout << "DrawOpDetWaveform::initialize - n_frames: " << _n_frames << " tick_period: " << _tick_period << " # chan: " << _n_op_channels << ", time ticks: " << _n_time_ticks << std::endl;

  initDataHolder(1, 1);

  return true;
}

bool DrawOpDetWaveform::analyze(const gallery::Event & ev) {

  art::InputTag op_wvf_tag(_producer);
  auto const & op_wvfs
    = ev.getValidHandle<std::vector <raw::OpDetWaveform> >(op_wvf_tag);

  std::cout << "OpDetWaveform analyze, op_wvfs size: " << op_wvfs->size() << std::endl;

  // figure out number of valid waveforms & max tick length for output array
  int n_ticks = -1;
  int n_wvfs = 0;
  for (auto const& op_wvf : *op_wvfs) {
    if (n_ticks == -1) {
        n_ticks = op_wvf.size();
    }
    else if (op_wvf.size() != n_ticks) {
        /*
        // warning removed, variable opdetwaveforms are intended
        std::cerr << "WARNING: Different OpDetWaveform sizes " << n_ticks
            << " and " << op_wvf.size() << ". Using the larger one.\n";
        */
        n_ticks = std::max((int)op_wvf.size(), n_ticks);
    }

    // check how many valid waveforms we expect in the output
    unsigned int ch   = op_wvf.ChannelNumber();
    if (ch >= _n_max_chs) {
        std::cout << "got ch=" << ch << " but its too high (max=" << _n_max_chs << "). Skipping!\n";
        continue;
    }
    n_wvfs++;
  }

  int n_ticks_reduced = n_ticks / _n_size_reduction;
  std::cout << "NWaveforms: " << n_wvfs << " Waveform length: " << n_ticks
      << ", in us: " << n_ticks * _tick_period << std::endl;

  initDataHolder(n_ticks_reduced, n_wvfs);
  _wvf_data.at(0) = n_wvfs;
  _wvf_data.at(1) = _n_size_reduction;

  int wvf_count = 0;
  for (auto const& op_wvf : *op_wvfs) {
    unsigned int ch   = op_wvf.ChannelNumber();
    if (ch >= _n_max_chs) {
        std::cout << "got ch=" << ch << " but its too high (max=" << _n_max_chs << "). Skipping!\n";
        continue;
    }

    double time = op_wvf.TimeStamp();
    double time_in_ticks = (time + _time_offset); // / _tick_period;

    // each waveform is n_ticks + 3 elements long
    // first two elements are channel and time offset
    // third element is the size reduction
    int offset = 2 + wvf_count * (n_ticks_reduced + 3);
    _wvf_data.at(offset) = (float)ch;
    _wvf_data.at(offset + 1) = (float)time_in_ticks;

    // start waveform data after first three elements
    // only use every nth element
    size_t this_opwvf_size = op_wvf.size();
    for (int i = 0; i < n_ticks; i++) {
       if ((i - 1) % _n_size_reduction != 0) continue;

       size_t adc_idx = offset + 2 + (i - 1) / _n_size_reduction;
       if (i < this_opwvf_size) {
           _wvf_data.at(adc_idx) = (float)op_wvf.at(i);
       }
       else {
           // pad ends with 0s
           _wvf_data.at(adc_idx) = 0.;
       }
    }

    // NAN at the end to signal the end-of-waveform
    _wvf_data.at(2 + (wvf_count + 1) * (n_ticks_reduced + 3) - 1) = NAN;
    wvf_count++;
  }

  return true;
}

bool DrawOpDetWaveform::finalize() {

  return true;
}

void DrawOpDetWaveform::initDataHolder(int nticks, int nwvfms) {
  float default_value = NAN;
  std::cout << "initializing, # waveforms: " << nwvfms << ", ticks: " << nticks << " default: " << default_value << std::endl;
  _wvf_data.clear();
  // first "tick" in the output waveform will hold the waveform tick offset
  // second "tick" in the output waveform will hold the waveform channel number
  // last tick is NAN to signal end-of-waveform
  _wvf_data.resize(2 + nwvfms * (nticks + 3), default_value);
}


PyObject * DrawOpDetWaveform::getArray() {

  PyObject * returnNull = nullptr;

  // int n_op_channels = 600;
  // int n_time_ticks = 625000*3;

  try {
    // Convert the wire data to numpy arrays:
    // int n_dim = 2;
    // int * dims = new int[n_dim];
    // int dims[2];
    //const npy_intp dims[2] = {_n_op_channels, _n_time_ticks};
    const npy_intp dims[1] = { _wvf_data.size() };
    // const npy_intp dims[2] = { _wvf_data.size() / 3, 3 };
    // dims[0] = _n_op_channels;
    // dims[1] = _n_time_ticks;
    // int data_type = NPY_FLOAT;

    // std::cout << "Returning array, dims: " << dims[0] << ", " << dims[1] << ", n_dim: " << n_dim << ", data_type: " << data_type << std::endl;

    return (PyObject *) PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, _wvf_data.data());
  }
  catch ( ... ) {
    std::cerr << "WARNING: CANNOT GET OP DET WAVEFORM.\n";
    return returnNull;
  }

}

}

#endif
