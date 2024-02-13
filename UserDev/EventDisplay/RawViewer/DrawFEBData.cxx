
#ifndef EVD_DRAWFEBDATA_CXX
#define EVD_DRAWFEBDATA_CXX


#include "DrawFEBData.h"
#include "LArUtil/DetectorProperties.h"

namespace evd {

DrawFEBData::DrawFEBData(const geo::GeometryCore&               geometry,
                         const detinfo::DetectorPropertiesData& detectorProperties,
                         const detinfo::DetectorClocksData&     detectorClocks) :
    _geo_service(geometry),
    _det_prop(detectorProperties),
    _det_clocks(detectorClocks)
{
    _name = "DrawFEBData";
    _producer = "crtsim";
    _n_aux_dets = _geo_service.NAuxDets();

    _import_array();
}


bool DrawFEBData::initialize() {
    _feb_data.clear();
    _feb_data.resize(_n_aux_dets * 32, 0);
    return true;
}


bool DrawFEBData::analyze(const gallery::Event & ev) {
    // each hit will write 4 numbers: index to encode FEB moduble & sipm offset, t0, t1, adc
    _feb_data.clear();

    // not the final size, but saves us a bit since we push_back below
    // this is enough memory for each aux det to register 8 hits before resizing
    _feb_data.reserve(_n_aux_dets * 32);

    art::InputTag feb_data_tag(_producer);

    const auto& febs
        = ev.getValidHandle<std::vector<sbnd::crt::FEBData> >(feb_data_tag);

    // int total_adc = 0;
    for (auto const& feb : *febs) {
        uint32_t mac5 = feb.Mac5();
        // t0 is ???
        // t1 is the offset specific for this FEB
        uint32_t t0 = feb.Ts0();
        uint32_t t1 = feb.Ts1();

        // SiPM index that fired the trigger. Unused
		// uint32_t coinc = feb.Coinc();
        
        const auto& adc_arr = feb.ADC();
        int isipm = 0;
        for (auto adc : adc_arr) {
            uint32_t idx = 32 * mac5 + isipm;
            if (adc >= 0) {
                _feb_data.push_back((float)idx);
                _feb_data.push_back((float)t0);
                _feb_data.push_back((float)t1);
                _feb_data.push_back((float)adc);
            }
            isipm++;
        }
    }

    return true;
}

bool DrawFEBData::finalize() {
    return true;
}

PyObject* DrawFEBData::getArray() {
    PyObject* returnNull = nullptr;

    try {
        // output array: each "row" is a hit with 4 values. There are
        // _feb_data.size() / 4 rows
        const npy_intp dims[2] = { _feb_data.size() / 4, 4 };
        return (PyObject*)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, _feb_data.data());
    }
    catch ( ... ) {
        std::cerr << "WARNING: CANNOT GET FEB DATA.\n";
        return returnNull;
    }

}

}

#endif
