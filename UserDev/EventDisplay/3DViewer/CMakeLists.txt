add_library(EventDisplay_3DViewer SHARED
    DrawMCShower3D.cxx
    DrawMCTrack3D.cxx
    DrawOpflash3D.cxx
    DrawPFParticle3D.cxx
    DrawSeed3D.cxx
    DrawShower3D.cxx
    DrawSimChannel3D.cxx
    DrawSpacepoint3D.cxx
    DrawTrack3D.cxx
    DrawVertex3D.cxx
)

target_include_directories(EventDisplay_3DViewer PRIVATE ${PROJECT_SOURCE_DIR}/core)

find_package(nusimdata REQUIRED)
target_include_directories(EventDisplay_3DViewer PRIVATE ${nusimdata_INCLUDE_DIRS})

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
target_include_directories(EventDisplay_3DViewer PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(EventDisplay_3DViewer Python::NumPy)

ROOT_GENERATE_DICTIONARY(EventDisplay_3DViewer_Dict
    DrawMCShower3D.h
    DrawMCTrack3D.h
    DrawOpflash3D.h
    DrawPFParticle3D.h
    DrawSeed3D.h
    DrawShower3D.h
    DrawSimChannel3D.h
    DrawSpacepoint3D.h
    DrawTrack3D.h
    DrawVertex3D.h
    RecoBase3D.h
    MODULE EventDisplay_3DViewer
    LINKDEF LinkDef.h
)
