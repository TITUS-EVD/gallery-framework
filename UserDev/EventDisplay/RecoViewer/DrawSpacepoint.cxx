#ifndef EVD_DRAWSPACEPOINT_CXX
#define EVD_DRAWSPACEPOINT_CXX

#include "DrawSpacepoint.h"
#include "LArUtil/GeometryHelper.h"

namespace evd {


DrawSpacepoint::DrawSpacepoint(const geo::GeometryCore&               geometry,
                               const detinfo::DetectorPropertiesData& detectorProperties,
                               const detinfo::DetectorClocksData&     detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawSpacepoint";
  _fout = 0;

}

bool DrawSpacepoint::initialize() {
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  if (_dataByPlane.size() != total_plane_number) {
    _dataByPlane.resize(total_plane_number);
  }
  return true;
}

bool DrawSpacepoint::analyze(gallery::Event * ev) {
  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop, _det_clock);



  // get a handle to the tracks
  art::InputTag sps_tag(_producer);
  auto const & spacepointHandle
        = ev -> getValidHandle<std::vector <recob::SpacePoint> >(sps_tag);

  // geoHelper = larutil::GeometryHelper::GetME();

  // Clear out the data but reserve some space
  for (unsigned int p = 0; p < total_plane_number; p ++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(spacepointHandle -> size());
    _wireRange.at(p).first  = 99999;
    _timeRange.at(p).first  = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }

  larutil::Point2D point;

  // Populate the spacepoint vector:
  for (auto & spt : *spacepointHandle) {

    // A spacepoint is a 3D object.  So take it and project it into each plane:
    for (unsigned int p = 0; p < total_plane_number; p ++) {


      try {
        // point = geoHelper -> Point_3Dto2D(spt.XYZ(), p);
        point = geo_helper.Point_3Dto2D(spt.XYZ(), p);
      }
      catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
      }
      _dataByPlane.at(p).push_back(point);


      // Determine if this hit should change the view range:
      if (point.w / geo_helper.WireToCm() > _wireRange.at(p).second)
        _wireRange.at(p).second = point.w / geo_helper.WireToCm();
      if (point.w / geo_helper.WireToCm() < _wireRange.at(p).first)
        _wireRange.at(p).first = point.w / geo_helper.WireToCm();
      if (point.t / geo_helper.TimeToCm() > _timeRange.at(p).second)
        _timeRange.at(p).second = point.t / geo_helper.TimeToCm();
      if (point.t / geo_helper.TimeToCm() < _timeRange.at(p).first)
        _timeRange.at(p).first = point.t / geo_helper.TimeToCm();
    }
  }


  return true;
}

bool DrawSpacepoint::finalize() {

  return true;
}

DrawSpacepoint::~DrawSpacepoint() {}


} // evd

#endif
