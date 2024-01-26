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

target_include_directories(gallery_framework_LArUtil PRIVATE ${PROJECT_SOURCE_DIR}/core)

ROOT_GENERATE_DICTIONARY(gallery_framework_LArUtil_Dict
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
    MODULE gallery_framework_LArUtil
    LINKDEF LinkDef.h
)
