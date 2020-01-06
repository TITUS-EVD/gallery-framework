/**
 * \file LArUtilServicesHandler.h
 *
 * \ingroup LArUtil
 * 
 * \brief Class def header for a class LArUtilServicesHandler
 *
 * @author Marco Del Tutto
 */

/** \addtogroup LArUtil

    @{*/
#ifndef LARUTILSERVICESHANDLER_H
#define LARUTILSERVICESHANDLER_H

#include <iostream>
#include "Geometry.h"
// #include "LArProperties.h"
// #include "DetectorProperties.h"
// #include "GeometryHelper.h"

#include "Base/messenger.h"


// #include "messagefacility/MessageLogger/MessageLogger.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "fhiclcpp/make_ParameterSet.h"


// - Geometry
#include "larcorealg/Geometry/StandaloneGeometrySetup.h"
#include "larcorealg/Geometry/GeometryCore.h"
#include "larcorealg/Geometry/ChannelMapStandardAlg.h"

// - DetectorClocks
#include "lardataalg/DetectorInfo/DetectorClocksStandardTestHelpers.h"
#include "lardataalg/DetectorInfo/DetectorClocksStandard.h"

// - LArProperties
#include "lardataalg/DetectorInfo/LArPropertiesStandardTestHelpers.h"
#include "lardataalg/DetectorInfo/LArPropertiesStandard.h"

// - DetectorProperties
#include "lardataalg/DetectorInfo/DetectorPropertiesStandardTestHelpers.h"
#include "lardataalg/DetectorInfo/DetectorPropertiesStandard.h"

namespace larutil {
  /**
     \class LArUtilServicesHandler
     User defined class LArUtilServicesHandler ... these comments are used to generate
     doxygen documentation!
  */
  class LArUtilServicesHandler  {
    
  private:
    
    /// Default constructor
    LArUtilServicesHandler(){};
    
    /// Default destructor
    virtual ~LArUtilServicesHandler(){};
    
  public:

    /// Method to get geometry
    static std::unique_ptr<::geo::GeometryCore> GetGeometry(std::string fcl_file_name);
    static std::unique_ptr<detinfo::LArPropertiesStandard>  GetLArProperties(std::string fcl_file_name);
    static std::unique_ptr<detinfo::DetectorPropertiesStandard>  GetDetProperties(std::string fcl_file_name);

    // std::unique_ptr<::geo::GeometryCore> _geom;
    // std::unique_ptr<detinfo::LArPropertiesStandard> _larp;
    // std::unique_ptr<detinfo::DetectorClocksStandard> _detclk;
    // std::unique_ptr<detinfo::DetectorPropertiesStandard> _detp;
    
  };
}

#endif
/** @} */ // end of doxygen group 

