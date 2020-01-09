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
#include "LArUtil/Geometry.h"
#include "LArUtil/DetectorProperties.h"

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "lardataobj/RawData/OpDetWaveform.h"

// #include "TTree.h"
// #include "TGraph.h"

struct _object;
typedef _object PyObject;

#ifndef ROOT_TMVA_PyMethodBase
  #ifndef __CINT__
    #include "Python.h"
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
    DrawOpDetWaveform();

    /// Default destructor
    virtual ~DrawOpDetWaveform(){}

    /** IMPLEMENT in DrawOpDetWaveform.cc!
        Initialization method to be called before the analysis event loop.
    */ 
    virtual bool initialize();

    /** IMPLEMENT in DrawOpDetWaveform.cc! 
        Analyze a data event-by-event  
    */
    virtual bool analyze(gallery::Event * event);

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

  protected:

    // This section holds the waveforms for the data
    std::vector<float> _wvf_data;

    std::string _producer;

    // sets up the _wvf_data data object
    void initDataHolder();

    const larutil::Geometry * _geoService;
    const larutil::DetectorProperties * _detProp;


  };
}
#endif



/** @} */ // end of doxygen group 
