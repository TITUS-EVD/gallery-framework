#ifndef EVD_DRAWSPACEPOINT3D_CXX
#define EVD_DRAWSPACEPOINT3D_CXX

#include "DrawSpacepoint3D.h"

namespace evd {


DrawSpacepoint3D::DrawSpacepoint3D() {
  _name = "DrawSpacepoint3D";
  _fout = 0;
}

bool DrawSpacepoint3D::initialize() {

  return true;
}

bool DrawSpacepoint3D::analyze(gallery::Event * ev) {


  // get a handle to the tracks
  art::InputTag sps_tag(_producer);
  auto const & spacepointHandle
    = ev -> getValidHandle<std::vector <recob::SpacePoint> >(sps_tag);

  // Clear out the data but reserve some space
  _data.clear();
  _data.reserve(spacepointHandle -> size());


  // Populate the shower vector:
  for (auto & spt : *spacepointHandle) {
    _data.push_back(larutil::Point3D(spt.XYZ()[0],
                                     spt.XYZ()[1],
                                     spt.XYZ()[2]
                                    ));
  }


  return true;
}

bool DrawSpacepoint3D::finalize() {

  return true;
}

DrawSpacepoint3D::~DrawSpacepoint3D() {}


} // evd

#endif
