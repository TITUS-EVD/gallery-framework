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

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

target_link_libraries(EventDisplay_3DViewer PRIVATE
    gallery_framework_Analysis
    gallery_framework_LArUtil
    canvas::canvas
    lardataobj::RecoBase
    lardataobj::MCBase
    lardataobj::Simulation
    nusimdata::SimulationBase
    Python::Python
    Python::NumPy
    ROOT::Core
)

titus_root_dictionary(EventDisplay_3DViewer
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
)
