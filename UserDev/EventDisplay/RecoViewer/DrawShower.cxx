#ifndef EVD_DRAWSHOWER_CXX
#define EVD_DRAWSHOWER_CXX

#include "DrawShower.h"

namespace evd {


DrawShower::DrawShower(const geo::GeometryCore& geometry,
                       const detinfo::DetectorProperties& detectorProperties,
                       const detinfo::DetectorClocksData& detectorClocks) :
    RecoBase<Shower2D>(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawShower";
  _fout = 0;
  // showerVectorByPlane = new std::vector<std::vector<Shower2D> >;
}

bool DrawShower::initialize() {

  // // Resize data holder to accommodate planes and wires:
  size_t _total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  if (_dataByPlane.size() != _total_plane_number) {
    _dataByPlane.resize(_total_plane_number);
  }
  return true;

}

bool DrawShower::analyze(gallery::Event * ev) {

  size_t total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();


  // get a handle to the showers
  art::InputTag shower_tag(_producer);
  auto const & showerHandle
    = ev -> getValidHandle<std::vector <recob::Shower> >(shower_tag);


  if (showerHandle -> size() == 0) {
    std::cout << "No showers available to draw by producer "
              << _producer
              << std::endl;
    return true;
  }

  // Retrieve the hits to infer the TPC the track belongs to
  art::InputTag assn_tag(_producer);
  art::FindMany<recob::Hit> shower_to_hits(showerHandle, *ev, assn_tag);


  // Clear out the hit data but reserve some space for the showers
  for (unsigned int p = 0; p < total_plane_number; p ++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(showerHandle -> size());
    _wireRange.at(p).first  = 99999;
    _timeRange.at(p).first  = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }


  // Populate the shower vector:
  for (size_t s = 0; s < showerHandle->size(); s++) {

    auto const& shower = showerHandle->at(s);

    std::vector<recob::Hit const*> hits;
    shower_to_hits.get(s, hits);
    size_t shower_tpc = 0, shower_cryo = 0;
    if (hits.size() > 0) {
      shower_tpc = hits.at(0)->WireID().TPC;
      shower_cryo = hits.at(0)->WireID().Cryostat;
    }


    for (unsigned int p = 0; p < _geo_service.Nplanes(shower_tpc); p++) {

      int plane = p + shower_tpc * _geo_service.Nplanes();
      plane += shower_cryo * _geo_service.Nplanes() * _geo_service.NTPC();

      auto sh = getShower2d(shower, p, shower_tpc, shower_cryo);
      sh._tpc = shower_tpc;
      sh._cryo = shower_cryo;
      _dataByPlane.at(plane).push_back(sh);

    }

    // for (unsigned int view = 0; view < total_plane_number; view++) {
    //   // get the reconstructed shower for this plane
    //   auto shr2D = getShower2d(shower, view);
    //   _dataByPlane.at(view).push_back( shr2D );

    // }
  }


  return true;
}

bool DrawShower::finalize() {

  return true;
}



Shower2D DrawShower::getShower2d(recob::Shower shower, unsigned int plane, unsigned int tpc, unsigned int cryostat) {

  larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop);

  Shower2D result;
  result._is_good = false;
  result._plane = plane;
  // Fill out the parameters of the 2d shower
  result._startPoint = geo_helper.Point_3Dto2D(shower.ShowerStart(), plane, tpc, cryostat);
    // = geoHelper -> Point_3Dto2D(shower.ShowerStart(), plane);

  // Next get the direction:
  result._angleInPlane = geo_helper.Slope_3Dto2D(shower.Direction(), plane, tpc, cryostat);

  // Get the opening Angle:
  // result._openingAngle = shower.OpeningAngle();
  result._openingAngle = 0.2;



  auto secondPoint = shower.ShowerStart() + shower.Length() * shower.Direction();


  result._endPoint
    = geo_helper.Point_3Dto2D(secondPoint, plane, tpc, cryostat);

  result._length = sqrt(pow(result.startPoint().w - result.endPoint().w, 2) +
                        pow(result.startPoint().t - result.endPoint().t, 2));

  if (shower.dEdx().size() > plane) {
    result._dedx = shower.dEdx()[plane];
  }
  else {
    result._dedx = 0.0;
  }

  if (shower.Energy().size() > plane) {
    result._energy = shower.Energy()[plane];
  }
  else {
    result._energy = 0.0;
  }

  result._is_good = true;

  return result;
}



} // larlite

#endif
