#ifndef LARUTILSERVICESHANDLER_CXX
#define LARUTILSERVICESHANDLER_CXX

#include "LArUtilServicesHandler.h"

namespace larutil {

  std::unique_ptr<::geo::GeometryCore> LArUtilServicesHandler::GetGeometry(std::string fcl_file_name)
  {
    std::string configFile = fcl_file_name;
    fhicl::ParameterSet config;
    cet::filepath_lookup_after1 policy("FHICL_FILE_PATH");
    fhicl::make_ParameterSet(configFile, policy, config);

    // geometry setup (it's special)
    std::unique_ptr<::geo::GeometryCore> _geom = lar::standalone::SetupGeometry<geo::ChannelMapStandardAlg>
            (config.get<fhicl::ParameterSet>("services.Geometry"));

    return _geom;

  }

  std::unique_ptr<detinfo::LArPropertiesStandard> GetLArProperties(std::string fcl_file_name)
  {
    std::string configFile = fcl_file_name;
    fhicl::ParameterSet config;
    cet::filepath_lookup_after1 policy("FHICL_FILE_PATH");
    fhicl::make_ParameterSet(configFile, policy, config);

    // LArProperties setup
    std::unique_ptr<detinfo::LArPropertiesStandard> _larp = testing::setupProvider<detinfo::LArPropertiesStandard>
            (config.get<fhicl::ParameterSet>("services.LArPropertiesService"));

    return _larp;
  }


  std::unique_ptr<detinfo::DetectorPropertiesStandard> LArUtilServicesHandler::GeDetProperties(std::string fcl_file_name)
  {
    std::string configFile = fcl_file_name;
    fhicl::ParameterSet config;
    cet::filepath_lookup_after1 policy("FHICL_FILE_PATH");
    fhicl::make_ParameterSet(configFile, policy, config);

    // geometry setup (it's special)
    std::unique_ptr<::geo::GeometryCore> _geom = lar::standalone::SetupGeometry<geo::ChannelMapStandardAlg>
            (config.get<fhicl::ParameterSet>("services.Geometry"));

    // LArProperties setup
    std::unique_ptr<detinfo::LArPropertiesStandard> _larp = testing::setupProvider<detinfo::LArPropertiesStandard>
            (config.get<fhicl::ParameterSet>("services.LArPropertiesService"));

    // DetectorClocks setup
    std::unique_ptr<detinfo::DetectorClocksStandard> _detclk = testing::setupProvider<detinfo::DetectorClocksStandard>
              (config.get<fhicl::ParameterSet>("services.DetectorClocksService"));

    // DetectorProperties setup
    std::unique_ptr<detinfo::DetectorPropertiesStandard> _detp = testing::setupProvider<detinfo::DetectorPropertiesStandard>(
            config.get<fhicl::ParameterSet>("services.DetectorPropertiesService"),
            detinfo::DetectorPropertiesStandard::providers_type{
              _geom.get(),
              _larp.get(),
              _detclk.get()
           }
    );

    return _detp;
  }

}

#endif
