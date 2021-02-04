#ifndef GALLERY_FMWK_SUPERA_SBND_WIRE_CXX
#define GALLERY_FMWK_SUPERA_SBND_WIRE_CXX

#include "sbnd_wire.h"

// Larsoft includes:
#include "lardataobj/RecoBase/Wire.h"

// larcv3 includes:
#include "larcv3/core/dataformat/ImageMeta.h"
#include "larcv3/core/dataformat/EventTensor.h"

namespace supera {

void SBNDWire::initialize(){

}


void SBNDWire::slice(gallery::Event* ev, larcv3::IOManager & io) {
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


    for (auto iROI = wire.SignalROI().begin_range();iROI < wire.SignalROI().end_range(); ++iROI) {
      auto ROI = *iROI;
      for (auto tick = ROI.begin_index(); tick < ROI.end_index(); tick ++){

        if (tick < tick_offset || tick > tick_offset + n_ticks_per_chamber)
          continue;
        // std::cout << "tick is " << tick << std::endl;
        int this_row = row(tick, channel) / compression;
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
      if (fabs(image.as_vector().at(i)) < _threshold)
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
}

#endif
