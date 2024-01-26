add_library(EventDisplay_RawViewer SHARED
    DrawFEBData.cxx
    DrawOpDetWaveform.cxx
    DrawRawDigit.cxx
    DrawWire.cxx
    RawBase.cxx
)

target_include_directories(EventDisplay_RawViewer PRIVATE ${PROJECT_SOURCE_DIR}/core)

find_package(sbnobj REQUIRED)
target_include_directories(EventDisplay_RawViewer PRIVATE ${sbnobj_INCLUDE_DIRS})

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
target_include_directories(EventDisplay_RawViewer PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(EventDisplay_RawViewer Python::NumPy)

ROOT_GENERATE_DICTIONARY(EventDisplay_RawViewer_Dict
    DrawFEBData.h
    DrawOpDetWaveform.h
    DrawRawDigit.h
    DrawWire.h
    RawBase.h
    MODULE EventDisplay_RawViewer
    LINKDEF LinkDef.h
)
