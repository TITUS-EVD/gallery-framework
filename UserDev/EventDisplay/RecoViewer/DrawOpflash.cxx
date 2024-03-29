#ifndef EVD_DRAWOPFLASH_CXX
#define EVD_DRAWOPFLASH_CXX

#include "DrawOpflash.h"

namespace evd {


DrawOpflash::DrawOpflash(const geo::GeometryCore&               geometry,
                         const detinfo::DetectorPropertiesData& detectorProperties,
                         const detinfo::DetectorClocksData&     detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawOpflash";
  _fout = 0;
}

bool DrawOpflash::initialize() {

  // Resize data holder
  size_t total_plane_number = _geo_service.NTPC() * _geo_service.Ncryostats();

  // Resize data holder
  if (_extraDataByPlane.size() != total_plane_number) {
    _extraDataByPlane.resize(total_plane_number);
  }
  return true;
}

bool DrawOpflash::analyze(const gallery::Event & ev) {

  art::InputTag opflash_tag(_producer);
  auto const & opflashHandle
        = ev.getValidHandle<std::vector <recob::OpFlash> >(opflash_tag);

  // Retrieve the ophits to infer the plane the opflash belongs to
  try {
    art::InputTag assn_tag(_producer);
    art::FindMany<recob::OpHit> flash_to_hits(opflashHandle, ev, assn_tag);

    size_t total_plane_number = _geo_service.NTPC() * _geo_service.Ncryostats();

    for (unsigned int p = 0; p < total_plane_number; p++) {
      _extraDataByPlane[p].clear();
      _extraDataByPlane[p].reserve(opflashHandle -> size());
    }

    size_t index = 0;
    for (auto & opf : *opflashHandle) {
      std::vector<recob::OpHit const*> ophits;
      flash_to_hits.get(index, ophits);
      int opch = 0;
      if (ophits.size() > 0) {
        opch = ophits.at(0)->OpChannel();
      }

      Opflash2D this_flash;
      this_flash._y = opf.YCenter();
      this_flash._z = opf.ZCenter();
      this_flash._time = opf.AbsTime();
      this_flash._y_width = opf.YWidth();
      this_flash._z_width = opf.ZWidth();
      this_flash._time_width = opf.TimeWidth();
      this_flash._total_pe = opf.TotalPE();
      this_flash._opdet_pe = opf.PEs();
      this_flash._plane = find_plane(opch);

      _extraDataByPlane[this_flash._plane].push_back(this_flash);

      index++;
    }

  } catch (...) {
    std::cout << "Failed to retrieve flashes from file." << std::endl;
  }

  return true;

}

int DrawOpflash::find_plane(int opch) {

  auto xyz = _geo_service.OpDetGeoFromOpChannel(opch).GetCenter();
  if (_geo_service.DetectorName().find("icarus") != std::string::npos) {
    if (xyz.X() < -300) return 0;
    if (xyz.X() < 0 && xyz.X() > -300) return 1;
    if (xyz.X() > 0 && xyz.X() < -300) return 2;
    if (xyz.X() > 300) return 3;
  }
  if (_geo_service.DetectorName().find("sbnd") != std::string::npos) {
    if (xyz.X() < 0) return 0;
    if (xyz.X() > 0) return 1;
  }
  return 0;

}

bool DrawOpflash::finalize() {

  return true;
}

DrawOpflash::~DrawOpflash() {}

} // larlite

#endif
