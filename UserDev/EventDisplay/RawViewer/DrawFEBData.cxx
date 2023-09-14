
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

    _feb_data.clear();
    _feb_data.resize(_n_aux_dets * 32, 0);
    art::InputTag feb_data_tag(_producer);

    const auto& febs
        = ev.getValidHandle<std::vector <sbnd::crt::FEBData> >(feb_data_tag);

    int ifeb = 0;
    int total_adc = 0;
    for (auto const& feb : *febs) {
        uint32_t mac5 = feb.Mac5();
        uint32_t t0 = feb.Ts0();
		uint32_t coinc = feb.Coinc();

        const auto& adc_arr = feb.ADC();
        int isipm = 0;
        for (auto adc : adc_arr) {
            uint32_t idx = 32 * ifeb + isipm;
            _feb_data.at(32 * ifeb + isipm++) = (float)adc;
            total_adc += adc;
        }
        ifeb++;
    }

    return true;
}

bool DrawFEBData::finalize() {
    return true;
}

PyObject* DrawFEBData::getArray() {
    PyObject* returnNull = nullptr;

    try {
        const npy_intp dims[2] = { _n_aux_dets, 32 };
        return (PyObject*)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, _feb_data.data());
    }
    catch ( ... ) {
        std::cerr << "WARNING: CANNOT GET FEB DATA.\n";
        return returnNull;
    }

}

}

#endif
