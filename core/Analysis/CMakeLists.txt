add_library(gallery_framework_Analysis SHARED
    ana_base.cxx
    ana_processor.cxx
)

target_include_directories(gallery_framework_Analysis PRIVATE ${PROJECT_SOURCE_DIR}/core)

ROOT_GENERATE_DICTIONARY(gallery_framework_Analysis_Dict
    ana_base.h
    ana_processor.h
    MODULE gallery_framework_Analysis
    LINKDEF LinkDef.h
)

target_link_libraries(gallery_framework_Analysis
    ${GALLERY_LIBRARIES}
    ${ROOT_LIBRARIES}
    gallery_framework_Base
)

install(TARGETS gallery_framework_Analysis DESTINATION lib)
