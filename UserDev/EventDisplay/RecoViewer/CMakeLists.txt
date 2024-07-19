add_library(EventDisplay_RecoViewer SHARED
    DrawCluster.cxx
    DrawEndpoint.cxx
    DrawHit.cxx
    DrawMCTrack.cxx
    DrawMCTruth.cxx
    DrawNumuSelection.cxx
    DrawOpflash.cxx
    DrawShower.cxx
    DrawSpacepoint.cxx
    DrawTrack.cxx
    DrawVertex.cxx
)


target_link_libraries(EventDisplay_RawViewer
    ${gallery_LIBRARY_DIR}/libgallery.so
    ${ROOT_LIBRARIES}
    ${CLHEP_LIBRARIES}
    ${canvas_LIBRARY_DIR}/libcanvas.so
    ${larcoreobj_LIBRARY_DIR}/liblarcoreobj_SummaryData.so
    ${larcorealg_LIBRARY_DIR}/liblarcorealg_Geometry.so
    ${lardataobj_LIBRARY_DIR}/liblardataobj_RecoBase.so
    ${lardataalg_LIBRARY_DIR}/liblardataalg_DetectorInfo.so
    ${cetlib_except_LIBRARY_DIR}/libcetlib_except.so 
    gallery_framework_Base
    gallery_framework_Analysis
    gallery_framework_LArUtil
)

target_include_directories(EventDisplay_RecoViewer PRIVATE ${PROJECT_SOURCE_DIR}/core)

find_package(nusimdata REQUIRED)
target_include_directories(EventDisplay_RecoViewer PRIVATE ${nusimdata_INCLUDE_DIRS})

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
target_include_directories(EventDisplay_RecoViewer PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(EventDisplay_RecoViewer Python::NumPy)

ROOT_GENERATE_DICTIONARY(EventDisplay_RecoViewer_Dict
    DrawCluster.h
    DrawEndpoint.h
    DrawHit.h
    DrawMCTrack.h
    DrawMCTruth.h
    DrawNumuSelection.h
    DrawOpflash.h
    DrawShower.h
    DrawSpacepoint.h
    DrawTrack.h
    DrawVertex.h
    RecoBase.h
    MODULE EventDisplay_RecoViewer
    LINKDEF LinkDef.h
)

install(TARGETS EventDisplay_RecoViewer DESTINATION lib)
