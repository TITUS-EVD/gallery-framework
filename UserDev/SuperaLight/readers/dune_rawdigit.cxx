#ifndef GALLERY_FMWK_SUPERA_DUNE_RAWDIGIT_CXX
#define GALLERY_FMWK_SUPERA_DUNE_RAWDIGIT_CXX

#include "DUNE_rawdigit.h"

// Larsoft includes:
#include "lardataobj/RawData/RawDigit.h"
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv includes:
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"

namespace supera {

void DUNERawDigit::initialize() {
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

void DUNERawDigit::slice(gallery::Event* ev, larcv::IOManager* io) {
  // This module creates raw digit images.  Therefore, it will slice
  // off the TPC images and store them into the larcv file.
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
  auto event_image2d =
      (larcv::EventImage2D*)io->get_data("image2d", "dunedigit");
  std::vector<larcv::Image2D> images;

  for (size_t i = 0; i < plane_meta.size(); i++)
    images.push_back(larcv::Image2D(plane_meta.at(i)));

  for (auto& digit : *raw_digits) {
    int channel = digit.Channel();

    // std::cout << "Channel: " << channel << std::endl;
    int this_projection_id = projection_id(channel);
    int this_column = column(channel);

    // Loop over the digit and compress it:
    for (size_t i_row = 0; i_row < n_ticks; i_row++) {
      int this_row = row(i_row, channel) / compression;
      float val =
          (digit.ADC(i_row) - digit.GetPedestal()) / (1.0 * compression);
      if (this_row < plane_meta.at(this_projection_id).rows() &&
          this_column < plane_meta.at(this_projection_id).cols()) {
        val += images.at(this_projection_id).pixel(this_row, this_column);
        images.at(this_projection_id).set_pixel(this_row, this_column, val);
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
  event_image2d->emplace(std::move(images));

  return;
}
}

#endif