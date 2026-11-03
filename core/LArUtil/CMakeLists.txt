add_library(gallery_framework_LArUtil SHARED
    DetectorProperties.cxx
    ElecClock.cxx
    Geometry.cxx
    GeometryHelper.cxx
    InvalidWireError.cxx
    LArProperties.cxx
    LArUtilBase.cxx
    LArUtilConfig.cxx
    LArUtilManager.cxx
    LArUtilServicesHandler.cxx
    SimpleGeometryHelper.cxx
    TimeService.cxx
)

target_include_directories(gallery_framework_LArUtil PUBLIC ${PROJECT_SOURCE_DIR}/core)

target_link_libraries(gallery_framework_LArUtil PUBLIC
    gallery::gallery
    canvas::canvas
    cetlib_except::cetlib_except
    fhiclcpp::fhiclcpp
    messagefacility::MF_MessageLogger
    larcoreobj::SummaryData
    larcorealg::Geometry
    lardataobj::RecoBase
    lardataalg::DetectorInfo
    CLHEP::CLHEP
    ROOT::Core
    gallery_framework_Base
)

titus_root_dictionary(gallery_framework_LArUtil
    ClockConstants.h
    DetectorProperties.h
    ElecClock.h
    Geometry.h
    GeometryHelper.h
    InvalidWireError.h
    LArProperties.h
    LArUtilBase.h
    LArUtilConfig.h
    LArUtilConstants.h
    LArUtilException.h
    LArUtilManager.h
    LArUtilServicesHandler.h
    LArUtil-TypeDef.h
    PxUtils.h
    SimpleGeometryHelper.h
    TimeService.h
)
