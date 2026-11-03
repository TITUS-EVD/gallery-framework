# Product extraction for titus.readers.GalleryReader. Depends only on gallery
# and the product libraries: no geometry, services, larcorealg or lardataalg.
# recob::ChannelROI and the SBND CRT/timing products (sbnobj) are optional.

add_library(EventDisplay_Reader SHARED ProductReader.cxx)

target_link_libraries(EventDisplay_Reader PRIVATE
    gallery::gallery
    canvas::canvas
    cetlib_except::cetlib_except
    larcoreobj::SimpleTypesAndConstants
    lardataobj::RecoBase
    lardataobj::RawData
    lardataobj::AnalysisBase
    lardataobj::MCBase
    nusimdata::SimulationBase  # simb::MCTruth
    ROOT::Core
)

set(with_channel_roi OFF)
set(with_sbnd_products OFF)
if (TARGET sbnobj::ICARUS_TPC)
    set(with_channel_roi ON)
    target_compile_definitions(EventDisplay_Reader PRIVATE WITH_SBNOBJ)
    target_link_libraries(EventDisplay_Reader PRIVATE sbnobj::ICARUS_TPC)
endif()
if (TARGET sbnobj::SBND_CRT AND TARGET sbnobj::SBND_Timing)
    set(with_sbnd_products ON)
    target_compile_definitions(EventDisplay_Reader PRIVATE WITH_SBNOBJ_SBND)
    target_link_libraries(EventDisplay_Reader PRIVATE sbnobj::SBND_CRT sbnobj::SBND_Timing)
endif()
message(STATUS "EventDisplay_Reader: sbnobj ${sbnobj_VERSION}: "
    "ChannelROI ${with_channel_roi}, SBND CRT/timing ${with_sbnd_products}")

titus_root_dictionary(EventDisplay_Reader ProductReader.h)
