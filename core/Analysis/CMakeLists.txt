add_library(gallery_framework_Analysis SHARED
    ana_base.cxx
    ana_processor.cxx
)

target_include_directories(gallery_framework_Analysis PUBLIC ${PROJECT_SOURCE_DIR}/core)

target_link_libraries(gallery_framework_Analysis PUBLIC
    gallery::gallery
    ROOT::Core
    gallery_framework_Base
)

titus_root_dictionary(gallery_framework_Analysis
    ana_base.h
    ana_processor.h
)
