#ifndef GALLERY_FMWK_SUPERA_DUNE_WIRE_CXX
#define GALLERY_FMWK_SUPERA_DUNE_WIRE_CXX

#include "dune_wire.h"

// Larsoft includes:
#include "lardataobj/RecoBase/Wire.h"

// larcv includes:
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace supera {

void DUNEWire::initialize(){
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
}


void DUNEWire::slice(gallery::Event* ev, larcv::IOManager* io) {
  // This module creates raw digit images.  Therefore, it will slice
  // off the TPC images and store them into the larcv file.
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
  auto event_image2d = (larcv::EventImage2D *) io->get_data("image2d", "dunewire");
  std::vector<larcv::Image2D> images;

  for (size_t i = 0; i < plane_meta.size(); i ++)
    images.push_back(larcv::Image2D(plane_meta.at(i)));

  for (auto & wire : * wires)
  {
    int channel = wire.Channel();

    // std::cout << "Channel: " << channel << std::endl;
    int this_projection_id = projection_id(channel);
    int this_column = column(channel);


    for (auto iROI = wire.SignalROI().begin_range();iROI < wire.SignalROI().end_range(); ++iROI) {
      auto ROI = *iROI;
      for (auto tick = ROI.begin_index(); tick < ROI.end_index(); tick ++){
        if (tick > n_ticks)
          continue;
        // std::cout << "tick is " << tick << std::endl;
        int this_row = row(tick, channel) / compression;
        float val = ROI[tick] / (1.0*compression);
        val += images.at(this_projection_id).pixel(this_row, this_column);
      images.at(this_projection_id).set_pixel(this_row, this_column, val);
      }
    }

  }

  // Compress the images with an absolute value threshold:
  for (auto & image : images){
    for (size_t i = 0; i < image.size(); i ++){
      if (image.as_vector().at(i) < 5)
        image.set_pixel(i, 0.0);
    }
  }

  // for (auto & image : images){
  //   image.threshold(100, 0.0);
  // }

  // Emplace the images:
  event_image2d -> emplace(std::move(images));

  return;
}
}

#endif