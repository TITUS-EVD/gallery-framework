add_library(gallery_framework_Base SHARED
    messenger.cxx
)


ROOT_GENERATE_DICTIONARY(gallery_framework_Base_Dict
    AnalysisConstants.h
    DetectorStatusTypes.h
    FrameworkConstants.h
    GeoConstants.h
    GeoTypes.h
    MCConstants.h
    messenger.h
    RawConstants.h
    MODULE gallery_framework_Base
    LINKDEF LinkDef.h
)

target_link_libraries(gallery_framework_Base
    ${ROOT_LIBRARIES}
)

install(TARGETS gallery_framework_Base DESTINATION lib)
