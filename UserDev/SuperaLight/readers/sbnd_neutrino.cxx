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
    // cosmic_slice(ev, io);
    return;
  }
  else{
    neutrino_slice(ev, io);
  }
  return;

}


void SBNDNeutrino::neutrino_slice(gallery::Event* ev, larcv::IOManager* io){


  std::string neutrino_producer = "generator";
  art::InputTag neutrino_tag(neutrino_producer);

  gallery::Handle<std::vector<simb::MCTruth> > mctruth;

  ev->getByLabel(neutrino_tag, mctruth);

  auto truth = mctruth->at(0);
  auto neutrino = mctruth->at(0).GetNeutrino().Nu();

  // get the sparse3d objects:
  auto event_cluster3d =
      (larcv::EventClusterVoxel3D*)io->get_data("cluster3d", "sbndneutrino");

  auto event_cluster2d =
      (larcv::EventClusterPixel2D*)io->get_data("cluster2d", "sbndneutrino");

  auto event_particle  =
      (larcv::EventParticle*) io->get_data("particle", "sbndneutrino");

  // Start by extracting the particle information:
  larcv::Particle neut_info;
  neut_info.id(0);

  // Info from MCNeutrino:
  neut_info.nu_interaction_type(truth.GetNeutrino().InteractionType());
  neut_info.nu_current_type(truth.GetNeutrino().CCNC());
  neut_info.track_id(neutrino.TrackId());
  neut_info.pdg_code(neutrino.PdgCode());
  neut_info.creation_process(neutrino.Process());
  neut_info.position(neutrino.Vx(),
                     neutrino.Vy(),
                     neutrino.Vz(),
                     neutrino.T());
  neut_info.momentum(neutrino.Px(),
                     neutrino.Py(),
                     neutrino.Pz());
  neut_info.energy_init(neutrino.E());

  event_particle->emplace_back(std::move(neut_info));

  for (size_t i = 0; i < truth.NParticles(); i ++){
    larcv::Particle particle;
    auto larsoft_particle = truth.GetParticle(i);

    particle.id(i+1);
    particle.track_id(larsoft_particle.TrackId());
    particle.pdg_code(larsoft_particle.PdgCode());
    particle.parent_track_id(larsoft_particle.Mother());
    particle.creation_process(larsoft_particle.Process());

    particle.position(larsoft_particle.Vx(),
                       larsoft_particle.Vy(),
                       larsoft_particle.Vz(),
                       larsoft_particle.T());

    particle.momentum(larsoft_particle.Px(),
                       larsoft_particle.Py(),
                       larsoft_particle.Pz());

    particle.energy_init(larsoft_particle.E());
    event_particle->emplace_back(std::move(particle));
  }


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
  int buffer_3d = 1;

  for(int i_x = -buffer_3d; i_x < buffer_3d+1; i_x ++){
    for(int i_y = -buffer_3d; i_y < buffer_3d+1; i_y ++){
      for(int i_z = -buffer_3d; i_z < buffer_3d+1; i_z ++){
        size_t this_id
          = voxel_meta.id(neutrino.Vx() + voxel_meta.size_voxel_x()*i_x,
                          neutrino.Vy() + voxel_meta.size_voxel_y()*i_y,
                          neutrino.Vz() + voxel_meta.size_voxel_z()*i_z);
        event_cluster3d->writeable_voxel_set(0).add(larcv::Voxel(this_id, 1));
      }
    }
  }


  for (int projection_id = 0; projection_id < 3; projection_id++) {

    float tick = tick_position(neutrino.Vx(), 0, projection_id);

    float wire = wire_position(neutrino.Vx(), neutrino.Vy(), neutrino.Vz(), projection_id);


    // std::cout << "Nearest wire is " << wire << std::endl;
    // std::cout << "X position is " << neutrino.Vx() << std::endl;
    // std::cout << " Tick is " << tick << std::endl;

    int buffer = 1;
    // Create a 5x5 of neutrino vertex pixels around the vertex:
    auto & voxel_set =
      _clusters_by_projection.at(projection_id).writeable_voxel_set(0);
    for (int i = -buffer; i < buffer+1; i ++ ){
      if (tick + i < 0) continue;
      if (tick + i >= _max_tick) continue;
      for (int j = -buffer; j < buffer+1; j ++ ){
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

// void SBNDNeutrino::cosmic_slice(gallery::Event* ev, larcv::IOManager* io){

// }

}

#endif