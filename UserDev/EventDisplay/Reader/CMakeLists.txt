# Product extraction for titus.readers.GalleryReader. Depends only on gallery
# and the product libraries: no geometry, services, larcorealg or lardataalg.
# recob::ChannelROI (sbnobj) is optional.

add_library(EventDisplay_Reader SHARED ProductReader.cxx)

find_package(sbnobj)
message(STATUS "EventDisplay_Reader: ChannelROI support (sbnobj) = " ${sbnobj_FOUND})
if (${sbnobj_FOUND})
    target_compile_definitions(EventDisplay_Reader PRIVATE WITH_SBNOBJ)
    target_include_directories(EventDisplay_Reader PRIVATE ${sbnobj_INCLUDE_DIR})
    target_link_libraries(EventDisplay_Reader ${sbnobj_LIBRARY_DIR}/libsbnobj_ICARUS_TPC.so)
endif()

target_link_libraries(EventDisplay_Reader
    ${gallery_LIBRARY_DIR}/libgallery.so
    ${canvas_LIBRARY_DIR}/libcanvas.so
    ${cetlib_except_LIBRARY_DIR}/libcetlib_except.so
    ${lardataobj_LIBRARY_DIR}/liblardataobj_RecoBase.so
    ${lardataobj_LIBRARY_DIR}/liblardataobj_RawData.so
    ${lardataobj_LIBRARY_DIR}/liblardataobj_AnalysisBase.so
    ${larcoreobj_LIBRARY_DIR}/liblarcoreobj_SimpleTypesAndConstants.so
    ${ROOT_LIBRARIES}
)

ROOT_GENERATE_DICTIONARY(EventDisplay_Reader_Dict
    ProductReader.h
    MODULE EventDisplay_Reader
    LINKDEF LinkDef.h
)

install(TARGETS EventDisplay_Reader DESTINATION lib)
