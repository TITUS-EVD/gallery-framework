#ifndef GALLERY_FMWK_SUPERA_SBND_CLUSTER_CXX
#define GALLERY_FMWK_SUPERA_SBND_CLUSTER_CXX

#include "sbnd_cluster.h"

#include "LArUtil/TimeService.h"

// Larsoft includes:
#include "lardataobj/MCBase/MCShower.h"
#include "lardataobj/MCBase/MCTrack.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv includes:
#include "larcv/core/DataFormat/EventParticle.h"
#include "larcv/core/DataFormat/EventVoxel2D.h"
#include "larcv/core/DataFormat/EventVoxel3D.h"

namespace supera {

void SBNDCluster::initialize() {
  plane_meta.clear();
  // parameters for ImageMeta are (xmin, ymin, xmax, ymax, nx, ny, units)
  // Well encode tick in y and wire in x.  Units will be centimeters
  // y (drift direction) goes from -200 to 200 for n_ticks * 2 + spacing
  // x (wire direction) goes from 0
  _max_tick = 2*n_ticks + n_cathode_ticks;
  // plane_meta.push_back(larcv::ImageMeta(-200.0, 0,
  //                                        200.0, 1986,
  //                                       _max_tick / compression,
  //                                       1986,
  //                                       0, larcv::kUnitCM));
  // plane_meta.push_back(larcv::ImageMeta(-200.0, 0,
  //                                        200.0, 1986,
  //                                       _max_tick / compression,
  //                                       1986,
  //                                       1, larcv::kUnitCM));
  // plane_meta.push_back(larcv::ImageMeta(-200.0, 0,
  //                                        200.0, 1986,
  //                                       _max_tick / compression,
  //                                       1666,
  //                                       2, larcv::kUnitCM));
  plane_meta.push_back(larcv::ImageMeta(
      0, 0, 1986, _max_tick, _max_tick / compression, 1986, 0, larcv::kUnitCM));
  plane_meta.push_back(larcv::ImageMeta(
      0, 0, 1986, _max_tick, _max_tick / compression, 1986, 1, larcv::kUnitCM));
  plane_meta.push_back(larcv::ImageMeta(
      0, 0, 1666, _max_tick, _max_tick / compression, 1666, 2, larcv::kUnitCM));


  voxel_meta.set(-200, -200, 0, 200, 200, 500, 400/0.3, 400/0.3, 500/0.3);
}

void SBNDCluster::build_particle_map(gallery::Event* ev, larcv::IOManager* io) {
  // This function makes the mapping between geant objects and larcv particles

  // It builds the list of particles in larcv, and populates the maps
  // _particle_to_trackID
  // _trackID_to_particle

  _particle_to_trackID.clear();
  _trackID_to_particle.clear();

  // Get the MCtracks and MCShowers

  std::string producer = "mcreco";
  art::InputTag tag(producer);
  auto const& mctracks = ev->getValidHandle<std::vector<sim::MCTrack> >(tag);
  auto const& mcshowers = ev->getValidHandle<std::vector<sim::MCShower> >(tag);

  // Get the EventParticle from larcv:
  auto event_part = (larcv::EventParticle*)io->get_data("particle", "sbndseg");

  // std::cout << "Number of mctracks : " << mctracks->size() << std::endl;
  // std::cout << "Number of mcshowers: " << mcshowers->size() << std::endl;

  unsigned int id = 0;

  for (auto& track : *mctracks) {
    larcv::Particle part;

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

    event_part->emplace_back(std::move(part));
    id++;
  }

  for (auto& shower : *mcshowers) {
    larcv::Particle part;

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

    event_part->emplace_back(std::move(part));
    id++;
  }

  return;
}

void SBNDCluster::slice(gallery::Event* ev, larcv::IOManager* io) {
  //  First, build the particle mapping from geant track ID to
  //  larcv particle

  build_particle_map(ev, io);


  // Get the simch data:
  std::string _simch_producer = "largeant";
  art::InputTag digit_tag(_simch_producer);
  auto const& simch =
      ev->getValidHandle<std::vector<sim::SimChannel> >(digit_tag);


  // get the sparse3d objects:
  auto event_cluster3d =
      (larcv::EventClusterVoxel3D*)io->get_data("cluster3d", "sbndseg");

  auto event_cluster2d =
      (larcv::EventClusterPixel2D*)io->get_data("cluster2d", "sbndseg");

  // Now, loop over the sim channels, and add the depositions to the
  // correct voxels

  int n_particles = _particle_to_trackID.size();

  std::vector<larcv::ClusterPixel2D> _clusters_by_projection;
  _clusters_by_projection.resize(3);

  int i = 0;
  for (auto& cluster2dSet : _clusters_by_projection) {
    cluster2dSet.resize(n_particles + 1);
    cluster2dSet.meta(plane_meta.at(i));
    i++;
  }

  // larcv::ClusterVoxel3D clusters3d;
  event_cluster3d->resize(n_particles + 1);
  event_cluster3d->meta(voxel_meta);

  for (auto& ch : *simch) {
    int this_column = column(ch.Channel());
    int this_projection_id = projection_id(ch.Channel());

    for (auto& TDCIDE : ch.TDCIDEMap()) {
      auto& tdc = TDCIDE.first;
      auto& ides = TDCIDE.second;

      for (auto& ide : ides) {

        if (tdc < 0 || tdc > n_ticks){
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
        if (tdc > 0 && tdc <= 3000){
          event_cluster3d->writeable_voxel_set(larcv_particle_id)
              .add(larcv::Voxel(voxel_meta.id(ide.x, ide.y, ide.z), ide.energy));
        }


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

        if (tdc < 3000 && tdc > 0){
          _clusters_by_projection.at(this_projection_id)
              .writeable_voxel_set(larcv_particle_id)
              .add(larcv::Voxel(
                  plane_meta.at(this_projection_id).index(tick / compression, this_column),
                  ide.energy));
        }
      }
      //
    }
  }
  for (auto& cluster_pix_2d : _clusters_by_projection) {
    event_cluster2d->emplace(std::move(cluster_pix_2d));
  }
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