#ifndef EVD_DrawSpacepointHits_CXX
#define EVD_DrawSpacepointHits_CXX

#include "DrawSpacePointHits.h"
#include "LArUtil/GeometryHelper.h"

namespace evd {


DrawSpacepointHits::DrawSpacepointHits(const geo::GeometryCore&               geometry,
                               const detinfo::DetectorPropertiesData& detectorProperties,
                               const geo::WireReadoutGeom&            wireReadout,
                               const detinfo::DetectorClocksData&     detectorClocks) :
    RecoBase(geometry, detectorProperties, wireReadout, detectorClocks)
{
  _name = "DrawSpacepointHits";
  _fout = 0;

}

bool DrawSpacepointHits::initialize() {
  size_t total_plane_number = _wire_readout.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  if (_dataByPlane.size() != total_plane_number) {
    _dataByPlane.resize(total_plane_number);
  }
  return true;
}

bool DrawSpacepointHits::analyze(const gallery::Event & ev) {
  size_t total_plane_number = _wire_readout.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  larutil::SimpleGeometryHelper geo_helper(_geo_service, _wire_readout, _det_prop, _det_clock);



  // get a handle to the tracks
  art::InputTag sps_tag(_producer);
  art::InputTag blip_tag(_producer);
  auto const & spacepointHandle
        = ev.getValidHandle<std::vector <recob::SpacePoint> >(sps_tag);
  gallery::ValidHandle<std::vector<blip::Blip>> blipHandle
    = ev.getValidHandle<std::vector <blip::Blip> >(blip_tag); //why not a handle. Why a vector?
  bool drawingBlips = blipHandle->size()==0;
  art::FindMany<recob::Hit> hits_for_SpacePoints(spacepointHandle, ev, sps_tag);
  // geoHelper = larutil::GeometryHelper::GetME();
  // Clear out the data but reserve some space
  //Space points can be multiple hits so we may need to alter this section
  for (unsigned int p = 0; p < total_plane_number; p ++) {
    _dataByPlane.at(p).clear();
    _dataByPlane.at(p).reserve(spacepointHandle -> size());
    _wireRange.at(p).first  = 99999;
    _timeRange.at(p).first  = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;
  }

  //larutil::Point2D point;

  // Populate the spacepoint vector:
  size_t sp_Index=0;
  for (auto & spt : *spacepointHandle) {
    std::vector<recob::Hit const*> hits; //Vector to hold associated hits
    hits_for_SpacePoints.get(sp_Index, hits);
    //Loop over each hits associated with this space point and fill in _dataByPlane appropriately
    for(size_t hitIndex=0; hitIndex<hits.size(); hitIndex++)
    {
      //point.Clear();
      //Same plane indexing as for Drawhits
      unsigned int plane = hits[hitIndex]->WireID().Plane;
      unsigned int tpc = hits[hitIndex]->WireID().TPC;
      unsigned int cryo = hits[hitIndex]->WireID().Cryostat;
      plane += tpc * _wire_readout.Nplanes();
      plane += cryo * _wire_readout.Nplanes() * _geo_service.NTPC();
      double NSigma = 2;
      int BlipPDG = 0;
      if(drawingBlips)
      {
        BlipPDG = (*blipHandle)[sp_Index].truth.LeadG4PDG;
      }
     _dataByPlane.at(plane).emplace_back(
      HitFromSpacePoint( sp_Index, hits[hitIndex]->WireID().Wire , hits[hitIndex]->PeakTime() - hits[hitIndex]->RMS()*NSigma, plane, tpc, cryo, 
			 hits[hitIndex]->RMS()*NSigma*2, BlipPDG  ) ) ;

    }
    sp_Index++;

    // A spacepoint is a 3D object.  So take it and project it into each plane:
    //for (unsigned int p = 0; p < total_plane_number; p ++) {


    //  try {
        // point = geoHelper -> Point_3Dto2D(spt.XYZ(), p);
    //    point = geo_helper.Point_3Dto2D(spt.XYZ(), p);
    //  }
    //  catch (const std::exception& e) {
    //    std::cerr << e.what() << '\n';
    //  }
    //  _dataByPlane.at(p).push_back(point);


      // Determine if this hit should change the view range:
    //  if (point.w / geo_helper.WireToCm() > _wireRange.at(p).second)
    //    _wireRange.at(p).second = point.w / geo_helper.WireToCm();
    //  if (point.w / geo_helper.WireToCm() < _wireRange.at(p).first)
    //    _wireRange.at(p).first = point.w / geo_helper.WireToCm();
    //  if (point.t / geo_helper.TimeToCm() > _timeRange.at(p).second)
    //    _timeRange.at(p).second = point.t / geo_helper.TimeToCm();
    //  if (point.t / geo_helper.TimeToCm() < _timeRange.at(p).first)
    //    _timeRange.at(p).first = point.t / geo_helper.TimeToCm();
    //}
  }


  return true;
}

bool DrawSpacepointHits::finalize() {

  return true;
}

DrawSpacepointHits::~DrawSpacepointHits() {}


} // evd

#endif
