#ifndef EVD_DRAWSHOWER_CXX
#define EVD_DRAWSHOWER_CXX

#include "DrawShower.h"

namespace evd {


DrawShower::DrawShower()
  : RecoBase<Shower2D>()
{
  _name = "DrawShower";
  _fout = 0;
  // showerVectorByPlane = new std::vector<std::vector<Shower2D> >;
}

bool DrawShower::initialize() {

  // // Resize data holder to accommodate planes and wires:
  if (_dataByPlane.size() != geoService -> Nviews()) {
    _dataByPlane.resize(geoService -> Nviews());
  }
  return true;

}

bool DrawShower::analyze(gallery::Event * ev) {

  // get a handle to the showers
  art::InputTag shower_tag(_producer);
  auto const & showerHandle
        = ev -> getValidHandle<std::vector <recob::Shower> >(shower_tag);


  // Clear out the hit data but reserve some space for the showers
  for (unsigned int p = 0; p < geoService -> Nviews(); p ++) {
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

    for (unsigned int view = 0; view < geoService -> Nviews(); view++) {
      // get the reconstructed shower for this plane
      auto shr2D = getShower2d(shower, view);
      _dataByPlane.at(view).push_back( shr2D );
    }
  }


  return true;
}

bool DrawShower::finalize() {

  return true;
}



Shower2D DrawShower::getShower2d(recob::Shower shower, unsigned int plane) {



  Shower2D result;
  result._is_good = false;
  result._plane = plane;
  // Fill out the parameters of the 2d shower
  result._startPoint
    = geoHelper -> Point_3Dto2D(shower.ShowerStart(), plane);

  // Next get the direction:
  result._angleInPlane = geoHelper->Slope_3Dto2D(shower.Direction(), plane);

  // Get the opening Angle:
  // result._openingAngle = shower.OpeningAngle();
  result._openingAngle = 0.2;


  auto secondPoint = shower.ShowerStart() + shower.Length() * shower.Direction();


  result._endPoint
    = geoHelper -> Point_3Dto2D(secondPoint, plane);

  result._length = sqrt(pow(result.startPoint().w - result.endPoint().w, 2) +
                        pow(result.startPoint().t - result.endPoint().t, 2));

  result._dedx = shower.dEdx()[plane];

  result._energy = shower.Energy()[plane];

  result._is_good = true;
  return result;
}



} // larlite

#endif
