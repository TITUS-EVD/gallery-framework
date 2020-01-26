#ifndef EVD_DRAWOPFLASH_CXX
#define EVD_DRAWOPFLASH_CXX

#include "DrawOpflash.h"

namespace evd {


DrawOpflash::DrawOpflash(const geo::GeometryCore& geometry, const detinfo::DetectorProperties& detectorProperties) :
    RecoBase(geometry, detectorProperties)
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

bool DrawOpflash::analyze(gallery::Event *ev) {

  //
  // Do your event-by-event analysis here. This function is called for
  // each event in the loop. You have "storage" pointer which contains
  // event-wise data. To see what is available, check the "Manual.pdf":
  //
  // http://microboone-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=3183
  //
  // Or you can refer to Base/DataFormatConstants.hh for available data type
  // enum values. Here is one example of getting PMT waveform collection.
  //
  // event_fifo* my_pmtfifo_v = (event_fifo*)(storage->get_data(DATA::PMFIFO));
  //
  // if( event_fifo )
  //
  //   std::cout << "Event ID: " << my_pmtfifo_v->event_id() << std::endl;
  //


  art::InputTag opflash_tag(_producer);
  auto const & opflashHandle
        = ev -> getValidHandle<std::vector <recob::OpFlash> >(opflash_tag);

  // Retrieve the ophits to infer the plane the opflash belongs to
  art::InputTag assn_tag(_producer);
  art::FindMany<recob::OpHit> flash_to_hits(opflashHandle, *ev, assn_tag);

  size_t total_plane_number = _geo_service.NTPC() * _geo_service.Ncryostats();

  for (unsigned int p = 0; p < total_plane_number; p++) {
    _extraDataByPlane[p].clear();
    _extraDataByPlane[p].reserve(opflashHandle -> size());
  }

  // Populate the shower vector:
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
    this_flash._time = opf.Time();
    this_flash._y_width = opf.YWidth();
    this_flash._z_width = opf.ZWidth();
    this_flash._time_width = opf.TimeWidth();
    this_flash._total_pe = opf.TotalPE();
    this_flash._opdet_pe = opf.PEs();
    this_flash._plane = find_plane(opch);

    _extraDataByPlane[this_flash._plane].push_back(this_flash);

    index++;
  }

  return true;

}

int DrawOpflash::find_plane(int opch) {

  auto xyz = _geo_service.OpDetGeoFromOpChannel(opch).GetCenter();
  if (_geo_service.DetectorName() == "icarus") {
    if (xyz.X() < -300) return 0;
    if (xyz.X() < 0 && xyz.X() > -300) return 1;
    if (xyz.X() > 0 && xyz.X() < -300) return 2;
    if (xyz.X() > 300) return 3;
  }
  if (_geo_service.DetectorName() == "sbnd") {
    if (xyz.X() < 0) return 0;
    if (xyz.X() > 0) return 1;
  } 
  return 0;

}

bool DrawOpflash::finalize() {

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
  //   print(MSG::ERROR,__FUNCTION__,"Did not find an output file pointer!!!
  //   File not opened?");
  //
  return true;
}

DrawOpflash::~DrawOpflash() {}

} // larlite

#endif
