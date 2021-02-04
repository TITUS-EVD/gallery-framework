#ifndef GALLERY_FMWK_SUPERA_SBND_CLUSTER_CXX
#define GALLERY_FMWK_SUPERA_SBND_CLUSTER_CXX

#include "sbnd_cluster.h"

#include "LArUtil/TimeService.h"

// Larsoft includes:
#include "lardataobj/MCBase/MCShower.h"
#include "lardataobj/MCBase/MCTrack.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv3 includes:
#include "larcv3/core/dataformat/EventParticle.h"
#include "larcv3/core/dataformat/EventSparseCluster.h"

namespace supera {

void SBNDCluster::initialize() {

}

void SBNDCluster::build_particle_map(gallery::Event* ev, larcv3::IOManager & io) {
  // This function makes the mapping between geant objects and larcv3 particles

  // It builds the list of particles in larcv3, and populates the maps
  // _particle_to_trackID
  // _trackID_to_particle

  _particle_to_trackID.clear();
  _trackID_to_particle.clear();

  // Get the MCtracks and MCShowers

  std::string producer = "mcreco";
  art::InputTag tag(producer);
  auto const& mctracks = ev->getValidHandle<std::vector<sim::MCTrack> >(tag);
  auto const& mcshowers = ev->getValidHandle<std::vector<sim::MCShower> >(tag);

  // Get the EventParticle from larcv3:
  auto & event_part = io.get_data<larcv3::EventParticle>("sbndseg");

  // std::cout << "Number of mctracks : " << mctracks->size() << std::endl;
  // std::cout << "Number of mcshowers: " << mcshowers->size() << std::endl;

  unsigned int id = 0;

  for (auto& track : *mctracks) {
    larcv3::Particle part;

    part.id(id);
    part.mcst_index(          track.TrackID());

    part.track_id(            track.TrackID());
    part.pdg_code(            track.PdgCode());
    part.nu_interaction_type( track.Origin());
    part.creation_process(    track.Process());

    part.parent_track_id(     track.MotherTrackID());
    part.parent_pdg_code(     track.MotherPdgCode());
    part.ancestor_track_id(   track.AncestorTrackID());
    part.ancestor_pdg_code(   track.AncestorPdgCode());

    part.first_step( track.Start().Position().X(),
                     track.Start().Position().Y(),
                     track.Start().Position().Z(),
                     track.Start().Position().T());

    part.energy_init(track.Start().Momentum().E());
    part.momentum(   track.Start().Momentum().X(),
                     track.Start().Momentum().Y(),
                     track.Start().Momentum().Z());

    _particle_to_trackID.push_back(std::vector<int>());
    _particle_to_trackID.back().push_back(track.TrackID());
    _trackID_to_particle[track.TrackID()] = id;

    event_part.emplace_back(std::move(part));
    id++;
  }

  for (auto& shower : *mcshowers) {
    larcv3::Particle part;

    part.id(id);
    part.mcst_index(          shower.TrackID());

    part.track_id(            shower.TrackID());
    part.pdg_code(            shower.PdgCode());
    part.nu_interaction_type( shower.Origin());
    part.creation_process(    shower.Process());

    part.parent_track_id(     shower.MotherTrackID());
    part.parent_pdg_code(     shower.MotherPdgCode());
    part.ancestor_track_id(   shower.AncestorTrackID());
    part.ancestor_pdg_code(   shower.AncestorPdgCode());

    part.first_step( shower.Start().Position().X(),
                     shower.Start().Position().Y(),
                     shower.Start().Position().Z(),
                     shower.Start().Position().T());

    part.energy_init(shower.Start().Momentum().E());
    part.momentum(   shower.Start().Momentum().X(),
                     shower.Start().Momentum().Y(),
                     shower.Start().Momentum().Z());

    _particle_to_trackID.push_back(std::vector<int>());
    for (auto& daughter_id : shower.DaughterTrackID()) {
      _particle_to_trackID.back().push_back(daughter_id);
      _trackID_to_particle[daughter_id] = id;
    }

    event_part.emplace_back(std::move(part));
    id++;
  }

  return;
}

void SBNDCluster::slice(gallery::Event* ev, larcv3::IOManager & io) {
  //  First, build the particle mapping from geant track ID to
  //  larcv3 particle

  build_particle_map(ev, io);


  // Get the simch data:
  std::string _simch_producer = "largeant";
  art::InputTag simch_tag(_simch_producer);
  auto const& simch =
      ev->getValidHandle<std::vector<sim::SimChannel> >(simch_tag);


  // get the sparse3d objects:
  auto & event_cluster3d = io.get_data<larcv3::EventSparseCluster3D>("sbndseg");

  auto & event_cluster2d = io.get_data<larcv3::EventSparseCluster2D>("sbndseg");

  // Now, loop over the sim channels, and add the depositions to the
  // correct voxels

  int n_particles = _particle_to_trackID.size();

  std::vector<larcv3::SparseCluster2D> _clusters_by_projection;
  _clusters_by_projection.resize(3);

  int i = 0;
  for (auto& cluster2dSet : _clusters_by_projection) {
    cluster2dSet.resize(n_particles + 1);
    cluster2dSet.meta(_base_image_meta_2D.at(i));
    i++;
  }

  std::cout << "number of clusters per projection:" << std::endl;
  std::cout << _clusters_by_projection[0].size() << std::endl;
  std::cout << _clusters_by_projection[1].size() << std::endl;
  std::cout << _clusters_by_projection[2].size() << std::endl;


  // larcv3::ClusterVoxel3D clusters3d;
  larcv3::SparseCluster3D _3d_clusters;
  _3d_clusters.resize(n_particles + 1);
  _3d_clusters.meta(_base_image_meta_3D);

  std::vector<size_t> coord_2d; coord_2d.resize(2);
  std::vector<double> pos_3d; pos_3d.resize(3);

  for (auto& ch : *simch) {
    int this_column = column(ch.Channel());
    int this_projection_id = projection_id(ch.Channel());

    for (auto& TDCIDE : ch.TDCIDEMap()) {
      auto& tdc = TDCIDE.first;
      auto& ides = TDCIDE.second;

      for (auto& ide : ides) {

        if (tdc < tick_offset || tdc > tick_offset + n_ticks_per_chamber){
          continue;
        }
        if (ide.trackID == -1){
          continue;
        }

        // Add this ide to the proper particle in 3D:
        int this_particle = ide.trackID;
        int larcv_particle_id;

        if (_trackID_to_particle.find(this_particle) !=
            _trackID_to_particle.end()) {
          larcv_particle_id = _trackID_to_particle[this_particle];
        } else{
          larcv_particle_id = n_particles;
        }

        pos_3d[0] = ide.x; pos_3d[1] = ide.y; pos_3d[2] = ide.z;
        std::cout << "(x, y, z): (" << ide.x << ", " << ide.y << ", " << ide.z << ")\n";
        auto index = _base_image_meta_3D.position_to_index(pos_3d);
        _3d_clusters.writeable_voxel_set(larcv_particle_id).add(larcv3::Voxel(index, ide.energy));
        std::cout << " Current size, index: " << _3d_clusters.voxel_set(0).size()
                  << ", " << index << std::endl;

        int tick = row(tdc, ch.Channel());

        // if (fabs(ide.x - 182.073) < 0.01) {
        //   std::cout << "(plane " << projection_id(ch.Channel()) << ") "
        //             << "X: " << ide.x
        //             << "\t" << tdc / compression
        //             << "\t" << tick / compression
        //             << "\t" << tick_position(ide.x, 0, this_projection_id)
        //             << "\t" << tick/compression - tick_position(ide.x, 0, this_projection_id)
        //             << std::endl;
        // }


        // map the tdc to a row:
        coord_2d[1] = tick; coord_2d[0] = this_column;
        index = _base_image_meta_2D.at(this_projection_id).index(coord_2d);

        _clusters_by_projection.at(this_projection_id).writeable_voxel_set(larcv_particle_id).add(larcv3::Voxel(index, ide.energy));
      }
      //
    }
  }
  for (size_t i_clust = 0; i_clust < _clusters_by_projection.size(); i_clust ++) {
    event_cluster2d.emplace(std::move(_clusters_by_projection[i_clust]));
  }

  event_cluster3d.emplace(std::move(_3d_clusters));

  // event_cluster3d->set(clusters3d, voxel_meta);
  //   std::cout << ch.TDCIDEMap().size() << std::endl;

  //   //  TDCIDE is std::pair<unsigned short, std::vector<sim::IDE> >
  //   //  TDCIDEMap is std::vector<TDCIDE>
  //   //  So, basically, it's a vector of pairs where the first element of
  //   the
  //   //  pair
  //   //  is the TDC, and the second element of the pair is the
  //   //  list of sim::IDEs that deposited there

  //   // // Loop over the digit and compress it:
  //   // for(size_t row = 0; row < 750; row ++){
  //   //   float val = digit.ADC(row*4);
  //   //   val += digit.ADC(row*4 + 1);
  //   //   val += digit.ADC(row*4 + 2);
  //   //   val += digit.ADC(row*4 + 3);
  //   //   // std::cout << "Setting at (" << column << ", " << row << ")" <<
  //   //   std::endl;
  //   //   _images.at(projection_id).set_pixel(column, row, val * 0.25);
  //   // }
  // }

  // // // Emplace the images:
  // // event_image2d -> emplace(std::move(_images));

  return;
}

}

#endif
