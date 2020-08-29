#ifndef EVD_DRAWVERTEX_CXX
#define EVD_DRAWVERTEX_CXX

#include "DrawVertex.h"

namespace evd {

DrawVertex::DrawVertex(const geo::GeometryCore&               geometry,
                       const detinfo::DetectorPropertiesData& detectorProperties,
                       const detinfo::DetectorClocksData&     detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawVertex";
  _fout = 0;

}

bool DrawVertex::initialize() {

  //
  // This function is called in the beginning of event loop
  // Do all variable initialization you wish to do here.
  // If you have a histogram to fill in the event loop, for example,
  // here is a good place to create one on the heap (i.e. "new TH1D").
  //
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  if (_dataByPlane.size() != total_plane_number) {
    _dataByPlane.resize(total_plane_number);
  }
  return true;

}

bool DrawVertex::analyze(gallery::Event * ev) {

  larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop, _det_clock);

  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();

  art::InputTag vertex_tag(_producer);
  auto const & vertexHandle
        = ev -> getValidHandle<std::vector <recob::Vertex> >(vertex_tag);



  // Clear out the hit data but reserve some space for the hits
  for (unsigned int p = 0; p < total_plane_number; p ++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(vertexHandle -> size());
    _wireRange.at(p).first  = 99999;
    _timeRange.at(p).first  = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }


  // Loop over the vertices and fill the necessary vectors.
  larutil::Point2D point;
  double * xyz = new double[3];

  for (auto & vertex : * vertexHandle) {
    // A vertex is a 3D object.  So take it and project it into each plane:
    for (unsigned int p = 0; p < total_plane_number; p ++) {

      vertex.XYZ(xyz);


      try {
        point = geo_helper.Point_3Dto2D(xyz, p);
      }
      catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
      }
      _dataByPlane.at(p).push_back(point);


      // Determine if this hit should change the view range:
      if (point.w > _wireRange.at(p).second)
        _wireRange.at(p).second = point.w;
      if (point.w < _wireRange.at(p).first)
        _wireRange.at(p).first = point.w;
      if (point.t > _timeRange.at(p).second)
        _timeRange.at(p).second = point.t;
      if (point.t < _timeRange.at(p).first)
        _timeRange.at(p).first = point.t;
    }
  }


  return true;
}

bool DrawVertex::finalize() {

  // This function is called at the end of event loop.
  // Do all variable finalization you wish to do here.
  // If you need, you can store your ROOT class instance in the output
  // file. You have an access to the output file through "_fout" pointer.
  //
  // Say you made a histogram pointer h1 to store. You can do this:
  //
  // if(_fout) { _fout->cd(); h1->Write(); }
  //
  // else
  //   print(MSG::ERROR,__FUNCTION__,"Did not find an output file pointer!!! File not opened?");
  //
  //

  return true;
}

}
#endif
