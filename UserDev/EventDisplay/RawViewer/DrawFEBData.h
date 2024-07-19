/**
 * \file DrawFEBData.h
 *
 * \ingroup EventViewer
 *
 * \brief Class def header for a class DrawFEBData
 *
 * @author T. Wester
 */

/** \addtogroup EventViewer

    @{*/

#ifndef EVD_DRAWFEBData_H
#define EVD_DRAWFEBData_H

#include "Analysis/ana_base.h"
#include "RawBase.h"

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "larcorealg/Geometry/GeometryCore.h"
#include "lardataalg/DetectorInfo/DetectorPropertiesData.h"
#include "lardataalg/DetectorInfo/DetectorClocksData.h"

#include "sbnobj/SBND/CRT/FEBData.hh"
#include "sbnobj/SBND/Timing/DAQTimestamp.hh"

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
     \class DrawFEBData
     User custom analysis class made by SHELL_USER_NAME
   */
  class DrawFEBData : public galleryfmwk::ana_base, public RawBase {

  public:

    /// Default constructor
    DrawFEBData(const geo::GeometryCore&               geometry,
                const detinfo::DetectorPropertiesData& detectorProperties);

    /// Default destructor
    virtual ~DrawFEBData(){}

    virtual bool initialize() override;
    virtual bool analyze(const gallery::Event & event);
    virtual bool finalize() override;

    PyObject* getArray();
    void setInput(std::string s){_feb_producer = s;}

  protected:

    std::vector<float> _feb_data;
    unsigned int _n_aux_dets;


    std::string _feb_producer;
    std::string _tdc_producer;

    // const detinfo::DetectorClocksData&     _det_clocks;

  };
}
#endif



/** @} */ // end of doxygen group
