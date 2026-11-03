add_library(gallery_framework_Base SHARED
    messenger.cxx
)

target_link_libraries(gallery_framework_Base PUBLIC ROOT::Core)

titus_root_dictionary(gallery_framework_Base
    AnalysisConstants.h
    DetectorStatusTypes.h
    FrameworkConstants.h
    GeoConstants.h
    GeoTypes.h
    MCConstants.h
    messenger.h
    RawConstants.h
)
