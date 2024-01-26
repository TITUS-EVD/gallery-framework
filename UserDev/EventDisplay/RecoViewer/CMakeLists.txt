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
