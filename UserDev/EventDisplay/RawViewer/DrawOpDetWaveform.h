/**
 * \file DrawOpDetWaveform.h
 *
 * \ingroup EventViewer
 *
 * \brief Class def header for a class DrawOpDetWaveform
 *
 * @author Marco Del Tutto
 */

/** \addtogroup EventViewer

    @{*/

#ifndef EVD_DRAWOPDETWAVEFORM_H
#define EVD_DRAWOPDETWAVEFORM_H

#include "Analysis/ana_base.h"

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "larcorealg/Geometry/GeometryCore.h"
#include "lardataalg/DetectorInfo/DetectorPropertiesData.h"
#include "lardataalg/DetectorInfo/DetectorClocksData.h"

#include "lardataobj/RawData/OpDetWaveform.h"

struct _object;
typedef _object PyObject;

#ifndef ROOT_TMVA_PyMethodBase
  #ifndef __CINT__
    #include "Python.h"
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include "numpy/arrayobject.h"
  #endif
#endif


namespace evd {
  /**
     \class DrawOpDetWaveform
     User custom analysis class made by SHELL_USER_NAME
   */
  class DrawOpDetWaveform : public galleryfmwk::ana_base {

  public:

    /// Default constructor
    DrawOpDetWaveform(const geo::GeometryCore&               geometry,
                      const detinfo::DetectorPropertiesData& detectorProperties,
                      const detinfo::DetectorClocksData&     detectorClocks);

    /// Default destructor
    virtual ~DrawOpDetWaveform(){}

    /** IMPLEMENT in DrawOpDetWaveform.cc!
        Initialization method to be called before the analysis event loop.
    */
    virtual bool initialize();

    /** IMPLEMENT in DrawOpDetWaveform.cc!
        Analyze a data event-by-event
    */
    virtual bool analyze(const gallery::Event & event);

    /** IMPLEMENT in DrawOpDetWaveform.cc!
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();

    // Function to get the array of data
    PyObject * getArray();

    // This function sets the input target
    // for larlite, this can be used to set the producer
    // for lariat, this can be used to set the file
    void setInput(std::string s){_producer = s;}

    void set_n_frames(int n) {_n_frames = n;}
    void set_time_offset(double t) {_time_offset = t;}
    void set_size_reduction(int n) { _n_size_reduction = std::max(1, n); }


  protected:

    // This section holds the waveforms for the data
    std::vector<float> _wvf_data;

    double _tick_period;
    int _n_op_channels;
    int _n_time_ticks;
    int _n_frames = 1;
    double _time_offset = 0;
    int _n_size_reduction = 10;
    const int _n_max_chs = 360;

    std::string _producer;

    // sets up the _wvf_data data object
    void initDataHolder(int, int);

    const geo::GeometryCore&               _geo_service;
    const detinfo::DetectorPropertiesData& _det_prop;
    const detinfo::DetectorClocksData&     _det_clocks;



  };
}
#endif



/** @} */ // end of doxygen group
