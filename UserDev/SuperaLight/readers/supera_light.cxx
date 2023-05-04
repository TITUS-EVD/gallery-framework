#ifndef GALLERY_FMWK_SUPERA_CXX
#define GALLERY_FMWK_SUPERA_CXX

#include "supera_light.h"


#define PLANE_0_WIRES 1984
#define PLANE_1_WIRES 1984
#define PLANE_2_WIRES 1664

// Larsoft includes:
#include "lardataobj/RawData/RawDigit.h"
#include "lardataobj/RecoBase/Wire.h"
#include "lardataobj/MCBase/MCShower.h"
#include "lardataobj/MCBase/MCTrack.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv3 includes:
#include "larcv3/core/dataformat/ImageMeta.h"
#include "larcv3/core/dataformat/EventTensor.h"
#include "larcv3/core/dataformat/EventParticle.h"
#include "larcv3/core/dataformat/EventSparseCluster.h"
#include "larcv3/core/dataformat/EventBBox.h"

// Gallery Framework includes
#include "LArUtil/Geometry.h"
#include "LArUtil/SimpleGeometryHelper.h"

namespace supera {

bool supera_light::initialize() {


  _base_image_meta_3D.set_dimension(0, 400,  400, -200); // X goes -200 to 200
  _base_image_meta_3D.set_dimension(1, 400, 400, -200);
  _base_image_meta_3D.set_dimension(2, 500, 500, 0 );

  // std::cout << "3d meta: " << _base_image_meta_3D.dump() << std::endl;

  _base_image_meta_2D.resize(3);
  // Set the total ticks per image:
  total_ticks = 2*n_ticks_per_chamber + n_cathode_ticks;
  for (size_t plane = 0; plane < 3; plane ++){
      // For the first dimension, x, we need the number of wires:
      int n_wires = _geo_service->Nwires(plane, 0);
      _base_image_meta_2D[plane].set_dimension(0, 0.3*n_wires, n_wires);
      _base_image_meta_2D[plane].set_dimension(1, 0.078*total_ticks, total_ticks/compression );
      _base_image_meta_2D[plane].set_projection_id(plane);
      // std::cout << "2d meta: " << _base_image_meta_2D[plane].dump() << std::endl;
  }
  return true;
}

void supera_light::set_output_file(std::string outfile){
    _io.set_out_file(outfile);
    _io.initialize();
}

bool supera_light::analyze(gallery::Event* ev) {


  // Get the event ID information for this event:
  int run = ev->eventAuxiliary().run();
  int subrun = ev->eventAuxiliary().subRun();
  int event = ev->eventAuxiliary().event();


  slice_raw_digit(ev, _io);
  slice_wire(ev, _io);
  slice_cluster(ev, _io);
  slice_neutrino(ev, _io);

  // Save the event
  _io.set_id(run, subrun, event);
  _io.save_entry();

  return true;
}


bool supera_light::finalize() {

  _io.finalize();
  return true;
}



void supera_light::slice_raw_digit(gallery::Event* ev, larcv3::IOManager & io) {
  // This module creates raw digit images.  Therefore, it will slice
  // off the TPC images and store them into the larcv3 file.
  //
  // We will need image2D meta for each plane,

  // Get the raw digit data:
  std::string _digit_producer = "daq";
  art::InputTag digit_tag(_digit_producer);
  auto const& raw_digits =
      ev->getValidHandle<std::vector<raw::RawDigit> >(digit_tag);

  // In this module, we only save one side of the tpc (the one where the
  // neutrino
  // interaction happened) so there is an offset
  // We also downsample by a factor of 4 to make wire and time almost line up

  // get the Image2D objects:
  auto & event_image2d = io.get_data<larcv3::EventTensor2D>("sbnddigit");

  std::vector<larcv3::Image2D> images;

  for (size_t i = 0; i < _base_image_meta_2D.size(); i++)
    images.push_back(larcv3::Image2D(_base_image_meta_2D.at(i)));

  std::vector<size_t> coords;
  coords.resize(2);
  int i =0;
  for (auto& digit : *raw_digits) {
    int channel = digit.Channel();

    // std::cout << "Channel: " << channel << std::endl;
    int this_projection_id = projection_id(channel);
    int this_column = column(channel);


    // Loop over the digit and compress it:
    for (size_t i_row = tick_offset; i_row < n_ticks_per_chamber + tick_offset; i_row++) {
      int this_row = row(i_row, channel) / compression;
      float val =
          (digit.ADC(i_row) - digit.GetPedestal()) / (1.0 * compression);
      if (this_row < _base_image_meta_2D.at(this_projection_id).rows() &&
          this_column < _base_image_meta_2D.at(this_projection_id).cols()) {
        coords[1] = this_row; coords[0] = this_column;
        // coords[0] = this_row; coords[1] = this_column;
        val += images.at(this_projection_id).pixel(coords);
        images.at(this_projection_id).set_pixel(coords, val);
      }
      else{
        // std::cout <<" Tried to access at row " << this_row << " and column " << this_column << std::endl;
      }
    }
  }

  // Compress the images with an absolute value threshold:
  for (auto & image : images){
    for (size_t i = 0; i < image.size(); i ++){
      if (fabs(image.as_vector().at(i)) < _threshold)
        image.set_pixel(i, 0.0);
    }
  }

  // Emplace the images:
  event_image2d.emplace(std::move(images));


  return;
}


void supera_light::slice_wire(gallery::Event* ev, larcv3::IOManager & io) {
  // This module creates raw digit images.  Therefore, it will slice
  // off the TPC images and store them into the larcv3 file.
  //
  // We will need image2D meta for each plane,

  // Get the raw digit data:
  std::string _wire_producer = "caldata";
  art::InputTag digit_tag(_wire_producer);
  auto const& wires =
    ev -> getValidHandle<std::vector<recob::Wire> >(digit_tag);

  // In this module, we only save one side of the tpc (the one where the neutrino
  // interaction happened) so there is an offset
  // We also downsample by a factor of 4 to make wire and time almost line up

  // get the Image2D objects:
  auto & event_image2d = io.get_data<larcv3::EventTensor2D>("sbndwire");
  std::vector<larcv3::Image2D> images;

  for (size_t i = 0; i < _base_image_meta_2D.size(); i++)
    images.push_back(larcv3::Image2D(_base_image_meta_2D.at(i)));

  std::vector<size_t> coords;
  coords.resize(2);

  for (auto & wire : * wires)
  {
    int channel = wire.Channel();

    // std::cout << "Channel: " << channel << std::endl;
    int this_projection_id = projection_id(channel);
    int this_column = column(channel);

    // if (this_column != 905) continue;

    for (auto iROI = wire.SignalROI().begin_range();iROI < wire.SignalROI().end_range(); ++iROI) {
      auto ROI = *iROI;
      for (auto tick = ROI.begin_index(); tick < ROI.end_index(); tick ++){

        // CHeck that this tick is strictly within  the TPC active window
        if (tick < tick_offset || tick > tick_offset + n_ticks_per_chamber)
          continue;

        // Convert the tick to the image coordinate:
        int this_row = row(tick, channel) / compression;
        // std::cout << "tick is " << tick << " on channel "
        //           << channel << ", row is " << this_row
        //           << ", column is " << this_column << std::endl;
        coords[1] = this_row; coords[0] = this_column;
        float val = ROI[tick] / (1.0*compression);
        val += images.at(this_projection_id).pixel(coords);

        images.at(this_projection_id).set_pixel(coords, val);
      }
    }

  }

  // Compress the images with an absolute value threshold:
  for (auto & image : images){
    for (size_t i = 0; i < image.size(); i ++){
      if (image.as_vector().at(i) < _threshold)
        image.set_pixel(i, 0.0);
    }
  }

  // for (auto & image : images){
  //   image.threshold(100, 0.0);
  // }

  // Emplace the images:
  event_image2d.emplace(std::move(images));

  return;
}


void supera_light::slice_neutrino(gallery::Event* ev, larcv3::IOManager & io){

  // larutil::SimpleGeometryHelper geo_helper(_geo_service, _det_prop, _det_clock);


  // Get the neutrino data:
  std::string neutrino_producer = "generator";
  art::InputTag neutrino_tag(neutrino_producer);

  gallery::Handle<std::vector<simb::MCTruth> > mctruth;



  bool res = ev->getByLabel(neutrino_tag, mctruth);
  if (!res) return;

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


  larcv3::BBoxCollection3D bbox_set_3d(_base_image_meta_3D);
  std::vector<larcv3::BBoxCollection2D> v_bbox_set_2d;
  v_bbox_set_2d.push_back(larcv3::BBoxCollection2D(_base_image_meta_2D[0]));
  v_bbox_set_2d.push_back(larcv3::BBoxCollection2D(_base_image_meta_2D[1]));
  v_bbox_set_2d.push_back(larcv3::BBoxCollection2D(_base_image_meta_2D[2]));

  std::vector<float> vertex_3d;
  vertex_3d.resize(3);
  bool first = false;
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

    if (! first){

        // What's the vertex of this particle?
        vertex_3d[0] = larsoft_particle.Vx();
        vertex_3d[1] = larsoft_particle.Vy();
        vertex_3d[2] = larsoft_particle.Vz();

        // Create a new BBox3D
        larcv3::BBox3D bb(
            {larsoft_particle.Vx(), larsoft_particle.Vy(), larsoft_particle.Vz()},
            {0.0, 0.0, 0.0}
        );

        // Add to the collection:
        bbox_set_3d.append(bb);

        // for the 2D projections, determine the TPC first:
        int tpc;
        if (larsoft_particle.Vx() < 0){
            tpc = 0;
        }
        else{
            tpc = 1;
        }

        // Go through all three planes and push this to 2d:
        // 3D Location:
        std::cout << "3D Vertex: " << vertex_3d[0] << ", " << vertex_3d[1] << ", " << vertex_3d[2] << std::endl;
        for (int plane = 0; plane < 3; plane ++){
            int align_plane = plane;
            if (plane == 0 && tpc == 1) align_plane = 1;
            if (plane == 1 && tpc == 1) align_plane = 0;
            // Since we flip the TPC, always align with TPC 0
            auto point_2d = wire_time_from_3D(vertex_3d, align_plane, tpc);
            std::cout << "Pre Plane " << plane << ", Point 2d is " << point_2d[0] << ", " << point_2d[1] << std::endl;
            // point_2d[1] *= 3.225;
            // point_2d[1] += 1.55;
            point_2d[0] *= 0.3;

            point_2d[0] = fabs(point_2d[0]);

            // Now, we do some bullshit to account for the fact that the larsoft
            // offset and the one we're using aren't in perfect agreement.
            if (point_2d[1] > 200){
                if (plane == 0){
                    point_2d[1] -= -1.266;
                }
                else if (plane == 1){
                    point_2d[1] -= -1.766;
                }
                else if (plane == 2){
                    point_2d[1] -= -1.018;
                }
            }
            else if (point_2d[1] < 200){
                if (plane == 0){
                    point_2d[1] -= -1.584;
                }
                else if (plane == 1){
                    point_2d[1] -= -1.834;
                }
                else if (plane == 2){
                    point_2d[1] -= -1.774;
                }
            }

            // if (tpc == 1){
            //     // Need to flip the time coordinate to the other side of the view:
            //     point_2d[1] = total_ticks/compression - point_2d[1];
            // }
            std::cout << "-- Post Plane " << plane << ", Point 2d is " << point_2d[0] << ", " << point_2d[1] << std::endl;
            v_bbox_set_2d[plane].append(
                larcv3::BBox2D(
                    {point_2d[0], point_2d[1]},
                    {0.0, 0.0}
                )
            );
        }
        first = true;

    }

  }

  // For each particle, we create a Bbox in 3D and each 2D.
  // We also create it for the neutrino.

  // get the 3D BBox object:
  auto & event_bbox_3d = io.get_data<larcv3::EventBBox3D>("bbox_neutrino");
  auto & event_bbox_2d = io.get_data<larcv3::EventBBox2D>("bbox_neutrino");

  event_bbox_3d.append(bbox_set_3d);

  event_bbox_2d.append(v_bbox_set_2d[0]);
  event_bbox_2d.append(v_bbox_set_2d[1]);
  event_bbox_2d.append(v_bbox_set_2d[2]);

  // Create a BBox collection for 2d and 3d:
  // bbox_2d =

  return;
}

void supera_light::build_particle_map(gallery::Event* ev, larcv3::IOManager & io) {
  // This function makes the mapping between geant objects and larcv3 particles

  // It builds the list of particles in larcv3, and populates the maps
  // _particle_to_trackID
  // _trackID_to_particle

  // // First, we need to get the MC Truth objecst:
  //
  // std::string producer = "largeant";
  // art::InputTag tag(producer);
  // auto const& mcparticles = ev->getValidHandle<std::vector<simb::MCParticle> >(tag);


  _particle_to_trackID.clear();
  _trackID_to_particle.clear();



  std::string producer = "mcreco";
  art::InputTag tag(producer);
  auto const& mctracks = ev->getValidHandle<std::vector<sim::MCTrack> >(tag);
  auto const& mcshowers = ev->getValidHandle<std::vector<sim::MCShower> >(tag);
  // Get the EventParticle from larcv3:
  auto & event_part = io.get_data<larcv3::EventParticle>("sbndseg");

    unsigned int id = 0;

    for (auto& track : *mctracks) {
      // std::cout << "Looking at track with ID " << track.TrackID()
      //           << " and PDG " << track.PdgCode()
      //           << ", parent is " << track.AncestorTrackID()
      //           << " with PDG " << track.AncestorPdgCode() << "\n";

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
      // std::cout << "Looking at shower with ID " << shower.TrackID()
      //           << " and PDG " << shower.PdgCode()
      //           << ", parent is " << shower.AncestorTrackID()
      //           << " with PDG " << shower.AncestorPdgCode() << "\n";

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

void supera_light::slice_cluster(gallery::Event* ev, larcv3::IOManager & io) {
  //  First, build the particle mapping from geant track ID to
  //  larcv3 particle

  build_particle_map(ev, io);


  // Get the simch data:
  std::string _simch_producer = "simdrift";
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

  // larcv3::ClusterVoxel3D clusters3d;
  larcv3::SparseCluster3D _3d_clusters;
  _3d_clusters.resize(n_particles + 1);
  _3d_clusters.meta(_base_image_meta_3D);

  std::vector<size_t> coord_2d; coord_2d.resize(2);
  std::vector<double> pos_3d; pos_3d.resize(3);

  float min_tdc = 999; float max_tdc = -9999;

  size_t cluster_tick_offset = 0;
  // int cluster_tick_offset = 2200 + 69;

  for (auto& ch : *simch) {
    int this_column = column(ch.Channel());
    int this_projection_id = projection_id(ch.Channel());

    // if (this_column != 905) continue;

    for (auto& TDCIDE : ch.TDCIDEMap()) {
      auto& tdc = TDCIDE.first;
      auto& ides = TDCIDE.second;

      // std::cout << "TDC: " << tdc << std::endl;

      // First, we need to figure out if this TDC is valid.
      if (tdc - cluster_tick_offset < tick_offset || tdc - cluster_tick_offset > tick_offset + n_ticks_per_chamber){
        continue;
      }

      int tick = row(tdc  + cluster_tick_offset, ch.Channel()) / compression;

/* FROM SimChannel.h:
/// List of energy deposits at the same time (on this channel)
typedef std::pair<unsigned short, std::vector<sim::IDE> > TDCIDE;

 * @brief Energy deposited on a readout channel by simulated tracks
 *
 * This class stores the list of all energies deposited on a readout channel.
 * The number of electrons is stored as well.
 *
 * The information is organized by time: it is divided by TDC ticks, and
 * each TDC tick where some energy was deposited appears in a separate entry,
 * while the quiet TDC ticks are omitted.
 * For each TDC, the information is stored as a list of energy deposits;
 * each deposit comes from a single Geant4 track and stores the location where
 * the ionization happened according to the simulation (see `sim::IDE` class).
 *
 * Note that there can be multiple energy deposit records (that is `sim::IDE`)
 * for a single track in a single TDC tick.
 */



      if (tdc < min_tdc) min_tdc = tdc;
      if (tdc > max_tdc) max_tdc = tdc;




      for (auto& ide : ides) {


        if (ide.trackID == -1){
          continue;
        }

        // Add this ide to the proper particle in 3D:
        int this_particle = abs(ide.trackID);
        int larcv_particle_id;

        if (_trackID_to_particle.find(this_particle) !=
            _trackID_to_particle.end()) {
          larcv_particle_id = _trackID_to_particle[this_particle];
        } else{
          larcv_particle_id = n_particles;
        }

        pos_3d[0] = ide.x; pos_3d[1] = ide.y; pos_3d[2] = ide.z;
        auto index = _base_image_meta_3D.position_to_index(pos_3d);
        _3d_clusters.writeable_voxel_set(larcv_particle_id).add(larcv3::Voxel(index, ide.energy));


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

  return;
}

int supera_light::projection_id(int channel) {
  // Pretty hacky code here ...

  // In SBND, channels go 0 to 1983 (plane 0), 1984 to 3967, 3968 to 5633
  // Then repeat on the other side with offset of 5634, for a total
  // of 11268 channels

  if (channel < PLANE_0_WIRES)
    return 0;
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES)
    return 1;
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES)
    return 2;
  else if (channel < 2*PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES)
    return 1;
  else if (channel < 2*PLANE_0_WIRES + 2*PLANE_1_WIRES + PLANE_2_WIRES)
    return 0;
  else
    return 2;
}

int supera_light::column(int channel) {
    // In SBND, channels go 0 to 1983 (plane 0), 1984 to 3967, 3968 to 5633
    // Then repeat on the other side with offset of 5634, for a total
    // of 11268 channels

  if (channel < PLANE_0_WIRES){
    return channel;
  }
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES){
    return channel - PLANE_0_WIRES;
  }
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES){
    return channel - (PLANE_0_WIRES + PLANE_1_WIRES);
  }
  else if (channel < 2*PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES) {
    return (channel - (PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES) );
  } else if (channel < 2*PLANE_0_WIRES + 2*PLANE_1_WIRES + PLANE_2_WIRES){
    return (channel - (2*PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES));
  } else {
    return (channel - (2*PLANE_0_WIRES + 2*PLANE_1_WIRES + PLANE_2_WIRES));
  }
}

int supera_light::row(int tick, int channel) {
  if (channel >= PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES) {
    // In this scenario, we need the row to come out higher since it's the inverted
    // TPC, joined to form an image.
    return total_ticks - (tick - tick_offset) - 1;
  } else {
    return tick - tick_offset;
  }
}

std::vector<float> supera_light::wire_time_from_3D(std::vector<float> position_3d, int plane, int tpc){

    std::vector<float> return_vals;
    return_vals.resize(2);


    geo::Point_t loc(position_3d[0],position_3d[1], position_3d[2]);

    // Get the tpc and cryo ids from the 3D point
    unsigned int _tpc = _geo_service->PositionToTPCID(loc).TPC;

    std::cout << "Provided TPC: " << tpc << " vs " << _tpc << std::endl;

    // The wire position can be gotten with Geometry::NearestWire()
    // Convert result to centimeters as part of the unit convention
    // Previously used nearest wire functions, but they are
    // slightly inaccurate
    // If you want the nearest wire, use the nearest wire function!
    const geo::PlaneGeo& planeGeo = _geo_service->Plane(plane, tpc, 0);
    return_vals[0] = planeGeo.WireCoordinate(loc);


    // The time position is the X coordinate, corrected for
    // trigger offset and the offset of the plane
    // auto detp = DetectorProperties::GetME();
    return_vals[1] = position_3d[0];
    // Add in the trigger offset:
    // (Trigger offset is time that the data is recorded
    // before the actual spill.
    // So, it moves the "0" farther away from the actual
    // time and is an addition)
    // returnPoint.t += trigger_offset(clocks)

    //Get the origin point of this plane:
    // Double_t planeOrigin[3];
    // geom -> PlaneOriginVtx(plane, planeOrigin);
    // auto vtx = geom.Plane(plane, 0, cryo).GetCenter();
    // auto vtx = planeGeo.GetCenter();
    // auto vtx = geom.Plane(plane).GetCenter();
    // planeOrigin[0] = vtx.X();
    // planeOrigin[1] = vtx.Y();
    // planeOrigin[2] = vtx.Z();
    // std::cout << "p " << plane << ", t " << tpc << ", c " << cryo << ": plane center x = " << vtx.X() << std::endl;

    // Correct for the origin of the planes
    // X = 0 is at the very first wire plane, so the values
    // of the coords of the planes are either 0 or negative
    // because the planes are negative, the extra distance
    // beyond 0 needs to make the time coordinate larger
    // Therefore, subtract the offest (which is already
    // in centimeters)
    // return_vals[1] += abs(planeOrigin[0]);
    return_vals[1] += 200;

    //  This returns in cm, but I want it in ticks, so divide:
    // return_vals[1] /= 0.078;


    return return_vals;
}


}
#endif
