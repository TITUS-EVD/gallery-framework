#ifndef EVD_DRAWCLUSTER_CXX
#define EVD_DRAWCLUSTER_CXX

#include "DrawCluster.h"

namespace evd {

DrawCluster::DrawCluster(const geo::GeometryCore& geometry, 
                         const detinfo::DetectorProperties& detectorProperties,
                         const detinfo::DetectorClocks& detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks)
{
  _name = "DrawCluster";
  _fout = 0;

}

bool DrawCluster::initialize() {

  //
  // This function is called in the beginning of event loop
  // Do all variable initialization you wish to do here.
  // If you have a histogram to fill in the event loop, for example,
  // here is a good place to create one on the heap (i.e. "new TH1D").
  //
  _total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();

  // Resize data holder
  if (_dataByPlane.size() != _total_plane_number) {
    _dataByPlane.resize(_total_plane_number);
  }

  return true;
}

bool DrawCluster::analyze(gallery::Event * ev) {

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

  // clear the spots that hold the data:
  // Obtain event-wise data object pointers
  //

  // Clear out the hit data but reserve some space for the hits
  for (unsigned int p = 0; p < _total_plane_number; p ++) {
    _dataByPlane.at(p).clear();
    _wireRange.at(p).first  = 99999;
    _timeRange.at(p).first  = 99999;
    _timeRange.at(p).second = -1.0;
    _wireRange.at(p).second = -1.0;

  }



  // Get all of the clusters from the event:
  art::InputTag clusters_tag(_producer);
  auto const & clusters
    = ev -> getValidHandle<std::vector <recob::Cluster> >(clusters_tag);

  if (clusters->size() == 0) {
    std::cout << "No clusters found." << std::endl;
    return false;
  }

  // Get the associations from clusters (pmtrack)
  // to hits (trajcluster)

  art::InputTag assn_tag(_producer);

  art::FindMany<recob::Hit> hits_for_cluster(clusters, *ev, assn_tag);






  for (unsigned int p = 0; p < _total_plane_number; p ++) {
    _dataByPlane.at(p).reserve(clusters->size());
  }



  // Loop over the clusters and fill the necessary vectors.
  // I don't know how clusters are stored so I'm taking a conservative
  // approach to packaging them for drawing
  std::vector<int> cluster_index;
  cluster_index.resize(_total_plane_number);

  // int view;

  // cluster::DefaultParamsAlg params_alg ;
  // cluster::cluster_params params;
  // params_alg.SetVerbose(false);
  // params_alg.SetDebug(false);
  // params_alg.SetMinHits(10);

  size_t index = 0;
  for (auto const& cluster : * clusters) {
    // view = cluster.View();

    unsigned int plane = cluster.Plane().Plane;
    unsigned int tpc = cluster.Plane().TPC;
    unsigned int cryo = cluster.Plane().Cryostat;
    
    // If a second TPC is present, its planes 0, 1 and 2 are 
    // stored consecutively to those of the first TPC. 
    // So we have planes 0, 1, 2, 3, 4, 5.
    // view = plane + tpc * (geoService->Nplanes() / geoService->NTPC());
    plane += tpc * _geo_service.Nplanes();
    plane += cryo * _geo_service.Nplanes() * _geo_service.NTPC(); 

    // Make a new cluster in the data:
    _dataByPlane.at(plane).push_back(Cluster2D());
    _dataByPlane.at(plane).back()._is_good = true;

    // Fill the cluster params alg
    // _cru_helper.GenerateParams( hit_indices, ev_hit, params);
    // params_alg.FillParams(params);

    // Set the params:
    // _dataByPlane.at(view).back()._params = params;

    std::vector<recob::Hit const*> hits;
    hits_for_cluster.get(index, hits);


    for (auto const& hit : hits) {

      // if (view == 0){
      //   std::cout << "Got a hit, seems to be view " << view
      //             << " and cluster " << cluster_index[view]
      //             << " at " << ev_hit->at(hit_index).WireID().Wire
      //             << ", " << ev_hit->at(hit_index).PeakTime()
      //             << std::endl;
      // }
      // Hit(float w, float t, float c, float r) :

      // Same as above, but for the hit view. Usually hits are in the same
      // plane as the cluster they belong too, although is a particle
      // has been matched across TPCs, the cluster plane is the plane
      // that contains the majority of hits, and the hits are 
      // on two planes, one per TPC.
      unsigned int hit_plane = hit->WireID().Plane;
      unsigned int hit_tpc = hit->WireID().TPC;
      unsigned int hit_cryo = hit->WireID().Cryostat;

      // If a second TPC is present, its planes 0, 1 and 2 are 
      // stored consecutively to those of the first TPC. 
      // So we have planes 0, 1, 2, 3, 4, 5.
      hit_plane += hit_tpc * _geo_service.Nplanes();
      hit_plane += hit_cryo * _geo_service.Nplanes() * _geo_service.NTPC();

      _dataByPlane.at(hit_plane).back().emplace_back(
        Hit2D(hit->WireID().Wire,
              hit->PeakTime(),
              hit->Integral(),
              hit->RMS(),
              hit->StartTick(),
              hit->PeakTime(),
              hit->EndTick(),
              hit->PeakAmplitude(),
              hit_plane,
              hit->WireID().TPC,
              hit->WireID().Cryostat
             ));


      // Determine if this hit should change the view range:
      if (hit->WireID().Wire > _wireRange.at(plane).second)
        _wireRange.at(plane).second = hit->WireID().Wire;
      if (hit->WireID().Wire < _wireRange.at(plane).first)
        _wireRange.at(plane).first = hit->WireID().Wire;
      if (hit->PeakTime() > _timeRange.at(plane).second)
        _timeRange.at(plane).second = hit->PeakTime();
      if (hit->PeakTime() < _timeRange.at(plane).first)
        _timeRange.at(plane).first = hit->PeakTime();

    }

    cluster_index[plane] ++;
    index ++;

  }


  return true;
}

bool DrawCluster::finalize() {

  // This function is called at the end of event loop.
  // Do all variable finalization you wish to do here.
  // If you need, you can store your ROOT class instance in the outputmake
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

DrawCluster::~DrawCluster() {
}

}
#endif
