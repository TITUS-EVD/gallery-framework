/**
 * \file LArUtilConfig.h
 *
 * \ingroup LArUtil
 * 
 * \brief Class def header for a class LArUtilConfig
 *
 * @author kazuhiro
 */

/** \addtogroup LArUtil

    @{*/
#ifndef GALLERY_FMWK_LARUTILCONFIG_H
#define GALLERY_FMWK_LARUTILCONFIG_H

#include <iostream>
#include "Base/GeoConstants.h"
#include "Base/messenger.h"
#include "TString.h"

namespace larutil {
  /**
     \class LArUtilConfig
     User defined class LArUtilConfig ... these comments are used to generate
     doxygen documentation!
  */
  class LArUtilConfig {
    
  private:
    
    /// Default constructor
    LArUtilConfig(){};
    
    /// Default destructor
    virtual ~LArUtilConfig(){};
    
    static galleryfmwk::geo::DetId_t _detector;
    
  public:
    
    static bool SetDetector(galleryfmwk::geo::DetId_t type);
    
    static galleryfmwk::geo::DetId_t Detector() { return _detector;}
    
  };
}

#endif
/** @} */ // end of doxygen group 

