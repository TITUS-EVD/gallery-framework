/**
 * \file ana_base.h
 *
 * \ingroup Analysis
 * 
 * \brief Base class def for all analysis classes
 *
 * @author Kazu - Nevis 2013
 */

/** \addtogroup Analysis

    @{*/

#ifndef GALLERY_FMWK_ANA_BASE_H
#define GALLERY_FMWK_ANA_BASE_H


#include "gallery/Event.h"
#include "Base/messenger.h"
#include "TFile.h"

namespace galleryfmwk {
  /**
     \class ana_base
     A base class for analysis modules to be operated with event_waveform
     data class instance. 
  */
  class ana_base  {
    
  public:
    
    /// Default constructor
    ana_base() {_fout = 0;}
    
    /// Default destructor
    virtual ~ana_base(){}
    
    /// Initialization method to be called before analyze any data
    virtual bool initialize(){return true;}

    /// Function to be called when new run is found
    virtual bool begin_run(gallery::Event * event){return event;}

    /// Function to be called when new sub-run is found
    virtual bool begin_subrun(gallery::Event * event){return event;}

    /// Analyze a data event-by-event  
    virtual bool analyze(gallery::Event * event){return event;}
    
    /// Finalize method to be called after all events processed.
    virtual bool finalize(){return true;}
    
    /// A setter for analysis output file poitner
    void set_output_file(TFile* fout){_fout=fout;}
    
    /// Setter for the verbosity level 
    virtual void set_verbosity(msg::Level level);
    
    inline const std::string name() const {return _name;}

  protected:
    
    TFile* _fout; ///< Analysis output file pointer
    std::string _name;             ///< class name holder
    msg::Level _verbosity_level;   ///< holder for specified verbosity level
  };
}
#endif
  
/** @} */ // end of doxygen group 
  
