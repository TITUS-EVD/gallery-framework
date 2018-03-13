#ifndef GALLERY_FMWK_SUPERA_SBND_NEUTRINO_CXX
#define GALLERY_FMWK_SUPERA_SBND_NEUTRINO_CXX

#include "sbnd_neutrino.h"

#include "LArUtil/TimeService.h"

// Larsoft includes:
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv includes:
#include "larcv/core/DataFormat/EventParticle.h"
#include "larcv/core/DataFormat/EventVoxel2D.h"
#include "larcv/core/DataFormat/EventVoxel3D.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"

namespace supera {

void SBNDNeutrino::initialize() {
  plane_meta.clear();
  // parameters for ImageMeta are (xmin, ymin, xmax, ymax, nx, ny, units)
  // Well encode tick in y and wire in x.  Units will be centimeters
  // y (drift direction) goes from -200 to 200 for n_ticks * 2 + spacing
  // x (wire direction) goes from 0
  _max_tick = 2 * n_ticks + n_cathode_ticks;

  plane_meta.push_back(larcv::ImageMeta(
      0, 0, 1986, _max_tick, _max_tick / compression, 1986, 0, larcv::kUnitCM));
  plane_meta.push_back(larcv::ImageMeta(
      0, 0, 1986, _max_tick, _max_tick / compression, 1986, 1, larcv::kUnitCM));
  plane_meta.push_back(larcv::ImageMeta(
      0, 0, 1666, _max_tick, _max_tick / compression, 1666, 2, larcv::kUnitCM));

  voxel_meta.set(-200, -200, 0, 200, 200, 500, 400 / 0.3, 400 / 0.3, 500 / 0.3);
}

void SBNDNeutrino::slice(gallery::Event* ev, larcv::IOManager* io) {
  // Get the neutrino data:
  std::string neutrino_producer = "generator";
  art::InputTag neutrino_tag(neutrino_producer);

  gallery::Handle<std::vector<simb::MCTruth> > mctruth;
  if (!ev->getByLabel(neutrino_tag, mctruth)) {
    return;
  }

  auto neutrino = mctruth->at(0).GetNeutrino().Nu();

  // get the sparse3d objects:
  auto event_cluster3d =
      (larcv::EventClusterVoxel3D*)io->get_data("cluster3d", "sbndvertex");

  auto event_cluster2d =
      (larcv::EventClusterPixel2D*)io->get_data("cluster2d", "sbndvertex");

  std::vector<larcv::ClusterPixel2D> _clusters_by_projection;
  _clusters_by_projection.resize(3);

  int i = 0;
  for (auto& cluster2dSet : _clusters_by_projection) {
    cluster2dSet.resize(1);
    cluster2dSet.meta(plane_meta.at(i));
    i++;
  }

  // std::cout << "Neutrino Vertex is at ("
  //           << neutrino.Vx() << ", "
  //           << neutrino.Vy() << ", "
  //           << neutrino.Vz() << ")\n";

  // larcv::ClusterVoxel3D clusters3d;
  event_cluster3d->resize(1);
  event_cluster3d->meta(voxel_meta);

  // Add voxels in 3D for the neutrino location:
  event_cluster3d->writeable_voxel_set(0).add(larcv::Voxel(
      voxel_meta.id(neutrino.Vx(), neutrino.Vy(), neutrino.Vz()), 1));

  for (int projection_id = 0; projection_id < 3; projection_id++) {
    int tick = (n_ticks +
                neutrino.Vx() / larutil::GeometryHelper::GetME()->TimeToCm()) /
               (1.0 * compression);
    int wire = larutil::Geometry::GetME()->NearestWire(neutrino.Position().Vect(), projection_id);
    // std::cout << "Nearest wire is " << wire << std::endl;
    // std::cout << "X position is " << neutrino.Vx() << std::endl;
    // std::cout << " Tick is " << tick << std::endl;

    // Create a 5x5 of neutrino vertex pixels around the vertex:
    auto & voxel_set =
      _clusters_by_projection.at(projection_id).writeable_voxel_set(0);
    for (int i = -2; i < 3; i ++ ){
      if (tick + i < 0) continue;
      if (tick + i >= n_ticks) continue;
      for (int j = -2; j < 3; j ++ ){
        if (wire + j < 0) continue;
        if (wire + j >= larutil::Geometry::GetME()->Nwires(projection_id) ) continue;
        auto index = plane_meta.at(projection_id).index(tick + i, wire + j);
        voxel_set.add( larcv::Voxel(index,1.0));
      }
    }

  }

  for (auto& cluster_pix_2d : _clusters_by_projection) {
    event_cluster2d->emplace(std::move(cluster_pix_2d));
  }

  return;
}
}

#endif