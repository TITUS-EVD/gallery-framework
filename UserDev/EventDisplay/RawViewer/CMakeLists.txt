# this library optionally depends on sbnobj for FEBData class, default is 
# to compile without

set(EventDisplay_RawViewer_SOURCES
    DrawOpDetWaveform.cxx
    DrawRawDigit.cxx
    DrawWire.cxx
    RawBase.cxx
)

find_package(sbnobj)
message(STATUS "Compile with SBNOBJ=" ${sbnobj_FOUND})
if (${sbnobj_FOUND})
    list(APPEND EventDisplay_RawViewer_SOURCES DrawFEBData.cxx)
endif()

add_library(EventDisplay_RawViewer SHARED ${EventDisplay_RawViewer_SOURCES})

if (${sbnobj_FOUND})
    target_compile_definitions(EventDisplay_RawViewer PUBLIC WITH_SBNOBJ)
    target_link_libraries(EventDisplay_RawViewer ${sbnobj_LIBRARY_DIR}/libsbnobj_SBND_CRT.so)
    target_include_directories(EventDisplay_RawViewer PRIVATE ${sbnobj_INCLUDE_DIRS})
endif()


target_include_directories(EventDisplay_RawViewer PRIVATE ${PROJECT_SOURCE_DIR}/core)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
target_include_directories(EventDisplay_RawViewer PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(EventDisplay_RawViewer Python::NumPy)

set(EventDisplay_RawViewer_HEADERS
    DrawOpDetWaveform.h
    DrawRawDigit.h
    DrawWire.h
    RawBase.h
)
if (${sbnobj_FOUND})
    list(APPEND EventDisplay_RawViewer_HEADERS DrawFEBData.h)
endif()

ROOT_GENERATE_DICTIONARY(EventDisplay_RawViewer_Dict
    ${EventDisplay_RawViewer_HEADERS}
    MODULE EventDisplay_RawViewer
    LINKDEF LinkDef.h
)
