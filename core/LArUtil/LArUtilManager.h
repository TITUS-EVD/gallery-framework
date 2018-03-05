/**
 * \file LArUtilManager.h
 *
 * \ingroup LArUtil
 * 
 * \brief Class def header for a class LArUtilManager
 *
 * @author kazuhiro
 */

/** \addtogroup LArUtil

    @{*/
#ifndef LARUTILMANAGER_H
#define LARUTILMANAGER_H

#include <iostream>
#include "Geometry.h"
#include "LArProperties.h"
#include "DetectorProperties.h"
#include "GeometryHelper.h"

#include "Base/messenger.h"

namespace larutil {
  /**
     \class LArUtilManager
     User defined class LArUtilManager ... these comments are used to generate
     doxygen documentation!
  */
  class LArUtilManager  {
    
  private:
    
    /// Default constructor
    LArUtilManager(){};
    
    /// Default destructor
    virtual ~LArUtilManager(){};

    /// Method to execute reconfigure for utilitites
    static bool ReconfigureUtilities();
    
  public:

    /// Method to reconfigure utiities for the provided detector type
    static bool Reconfigure(galleryfmwk::geo::DetId_t type);

    /// Method to return number of TPCs in the currently configured geometry
    static bool Ntpcs(galleryfmwk::geo::DetId_t type);

    /// Method to return number of Cryostats in the currently configured geometry
    static bool Ncryostats(galleryfmwk::geo::DetId_t type);
    
  };
}

#endif
/** @} */ // end of doxygen group 

