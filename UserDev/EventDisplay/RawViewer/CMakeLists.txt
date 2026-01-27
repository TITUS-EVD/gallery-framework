# this library optionally depends on sbnobj for FEBData class, default is 
# to compile without

set(EventDisplay_RawViewer_SOURCES
    DrawOpDetWaveform.cxx
    DrawRawDigit.cxx
    DrawWire.cxx
    RawBase.cxx
)
set(EventDisplay_RawViewer_HEADERS
    DrawOpDetWaveform.h
    DrawRawDigit.h
    DrawWire.h
    RawBase.h
)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
set(EventDisplay_RawViewer_INCLUDES
    ${Python_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/core
)

set(EventDisplay_RawViewer_LIBS
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
    Python::NumPy
)

find_package(sbnobj)
message(STATUS "Compile with SBNOBJ=" ${sbnobj_FOUND})
if (${sbnobj_FOUND})
    list(APPEND EventDisplay_RawViewer_SOURCES DrawFEBData.cxx)
    list(APPEND EventDisplay_RawViewer_SOURCES DrawChannelROI.cxx)
    list(APPEND EventDisplay_RawViewer_HEADERS DrawFEBData.h)
    list(APPEND EventDisplay_RawViewer_HEADERS DrawChannelROI.h)
    list(APPEND EventDisplay_RawViewer_INCLUDES ${sbnobj_INCLUDE_DIR})
    list(APPEND EventDisplay_RawViewer_LIBS ${sbnobj_LIBRARY_DIR}/libsbnobj_SBND_CRT.so)
    list(APPEND EventDisplay_RawViewer_LIBS ${sbnobj_LIBRARY_DIR}/libsbnobj_ICARUS_TPC.so)
endif()


find_package(icarusalg)
find_package(icarus_signal_processing)
find_package(icaruscode)
message(STATUS "Compile with ICARUSCODE=" ${icaruscode_FOUND})
if (${icaruscode_FOUND})
    list(APPEND EventDisplay_RawViewer_INCLUDES ${icaruscode_INCLUDE_DIRS})
    list(APPEND EventDisplay_RawViewer_LIBS ${icaruscode_LIBRARY_DIR}/libicaruscode_IcarusObj.so)
endif()

add_library(EventDisplay_RawViewer SHARED ${EventDisplay_RawViewer_SOURCES})

if (${sbnobj_FOUND})
    target_compile_definitions(EventDisplay_RawViewer PUBLIC WITH_SBNOBJ)
endif()

if (${icaruscode_FOUND})
    target_compile_definitions(EventDisplay_RawViewer PUBLIC WITH_ICARUSCODE)
endif()


target_include_directories(EventDisplay_RawViewer PRIVATE ${EventDisplay_RawViewer_INCLUDES})
target_link_libraries(EventDisplay_RawViewer PRIVATE ${EventDisplay_RawViewer_LIBS})


ROOT_GENERATE_DICTIONARY(EventDisplay_RawViewer_Dict
    ${EventDisplay_RawViewer_HEADERS}
    MODULE EventDisplay_RawViewer
    LINKDEF LinkDef.h
)

install(TARGETS EventDisplay_RawViewer DESTINATION lib)
