#ifndef GALLERY_FMWK_SUPERA_SBND_RAWDIGIT_CXX
#define GALLERY_FMWK_SUPERA_SBND_RAWDIGIT_CXX

#include "sbnd_rawdigit.h"

// Larsoft includes:
#include "lardataobj/RawData/RawDigit.h"
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv3 includes:
#include "larcv3/core/dataformat/EventTensor.h"
#include "larcv3/core/dataformat/ImageMeta.h"

namespace supera {

void SBNDRawDigit::initialize() {
    std::cout << "Calling initialize in subclass " << std::endl;
}

void SBNDRawDigit::slice(gallery::Event* ev, larcv3::IOManager & io) {
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
}

#endif
