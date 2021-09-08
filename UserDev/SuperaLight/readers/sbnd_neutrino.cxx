#ifndef GALLERY_FMWK_SUPERA_SBND_NEUTRINO_CXX
#define GALLERY_FMWK_SUPERA_SBND_NEUTRINO_CXX

#include "sbnd_neutrino.h"

#include "LArUtil/TimeService.h"

// Larsoft includes:
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv3 includes:
#include "larcv3/core/dataformat/EventParticle.h"
#include "larcv3/core/dataformat/EventSparseCluster.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/SimpleGeometryHelper.h"

#include "larcorealg/Geometry/GeometryCore.h"
#include "lardataalg/DetectorInfo/DetectorProperties.h"
#include "lardataalg/DetectorInfo/DetectorClocksData.h"

namespace supera {

void SBNDNeutrino::initialize() {


}

void SBNDNeutrino::slice(gallery::Event* ev, larcv3::IOManager& io) {
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


void SBNDNeutrino::neutrino_slice(gallery::Event* ev, larcv3::IOManager & io){

  auto geometry = larutil::LArUtilServicesHandler::GetGeometry(_fcl_file_name);

  std::string neutrino_producer = "generator";
  art::InputTag neutrino_tag(neutrino_producer);

  gallery::Handle<std::vector<simb::MCTruth> > mctruth;

  ev->getByLabel(neutrino_tag, mctruth);

  auto truth = mctruth->at(0);
  auto neutrino = mctruth->at(0).GetNeutrino().Nu();

  // get the sparse3d objects:
  // auto event_cluster3d = io.get_data<larcv3::EventSparseCluster3D>("sbndneutrino");
  // auto event_cluster2d = io.get_data<larcv3::EventSparseCluster2D>("sbndneutrino");
  auto & event_particle  = io.get_data<larcv3::EventParticle>("sbndneutrino");

  // Start by extracting the particle information:
  larcv3::Particle neut_info;
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

  event_particle.append(neut_info);

  for (size_t i = 0; i < truth.NParticles(); i ++){
    larcv3::Particle particle;
    auto & larsoft_particle = truth.GetParticle(i);
    if (larsoft_particle.StatusCode() != 1){
      continue;
    }

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
    event_particle.append(particle);
  }

  //
  // std::vector<larcv3::SparseCluster2D> _clusters_by_projection;
  // _clusters_by_projection.resize(3);
  //
  // int i = 0;
  // for (auto& cluster2dSet : _clusters_by_projection) {
  //   cluster2dSet.resize(1);
  //   cluster2dSet.meta(_base_image_meta_2D.at(i));
  //   i++;
  // }
  //
  // std::cout << "Neutrino Vertex is at ("
  //           << neutrino.Vx() << ", "
  //           << neutrino.Vy() << ", "
  //           << neutrino.Vz() << ")\n";
  //
  // larcv3::SparseCluster3D _3d_clusters;
  // _3d_clusters.resize(1);
  // _3d_clusters.meta(_base_image_meta_3D);
  //
  // // Add voxels in 3D for the neutrino location:
  // int buffer_3d = 1;
  // larcv3::VoxelSet vs;
  // std::vector<double> positions; positions.resize(3);
  // for(int i_x = -buffer_3d; i_x < buffer_3d+1; i_x ++){
  //   for(int i_y = -buffer_3d; i_y < buffer_3d+1; i_y ++){
  //     for(int i_z = -buffer_3d; i_z < buffer_3d+1; i_z ++){
  //       positions[0] = neutrino.Vx() + _base_image_meta_3D.number_of_voxels(0)*i_x;
  //       positions[1] = neutrino.Vy() + _base_image_meta_3D.number_of_voxels(1)*i_y;
  //       positions[2] = neutrino.Vz() + _base_image_meta_3D.number_of_voxels(2)*i_z;
  //       size_t this_id = _base_image_meta_3D.position_to_index(positions);
  //       vs.add(larcv3::Voxel(this_id, 1));
  //     }
  //   }
  // }
  // _3d_clusters.emplace(std::move(vs));
  // event_cluster3d.emplace(std::move(_3d_clusters));
  //
  // for (int projection_id = 0; projection_id < 3; projection_id++) {
  //
  //   // For the wire position, we use get
  //
  //   float tick = tick_position(neutrino.Vx(), 0, projection_id);
  //
  //   float wire = wire_position(neutrino.Vx(), neutrino.Vy(), neutrino.Vz(), projection_id);
  //
  //
  //   // std::cout << "Nearest wire is " << wire << std::endl;
  //   // std::cout << "X position is " << neutrino.Vx() << std::endl;
  //   // std::cout << " Tick is " << tick << std::endl;
  //
  //   int buffer = 1;
  //   // Create a 5x5 of neutrino vertex pixels around the vertex:
  //   auto & voxel_set =
  //     _clusters_by_projection.at(projection_id).writeable_voxel_set(0);
  //   std::vector<double> coordinates; coordinates.resize(2);
  //   for (int i = -buffer; i < buffer+1; i ++ ){
  //     if (tick + i < 0) continue;
  //     if (tick + i >= n_ticks_per_chamber) continue;
  //     for (int j = -buffer; j < buffer+1; j ++ ){
  //       if (wire + j < 0) continue;
  //       if (wire + j >= larutil::Geometry::GetME()->Nwires(projection_id) ) continue;
  //       coordinates[0] = wire + j; coordinates[1] = tick + i;
  //       auto index = _base_image_meta_2D.at(projection_id).position_to_index(coordinates);
  //       voxel_set.add( larcv3::Voxel(index,1.0));
  //     }
  //   }
  // }
  //
  //
  // for (auto& cluster_pix_2d : _clusters_by_projection) {
  //   event_cluster2d.emplace(std::move(cluster_pix_2d));
  // }

  return;
}

// void SBNDNeutrino::cosmic_slice(gallery::Event* ev, larcv3::IOManager* io){

// }

}

#endif
