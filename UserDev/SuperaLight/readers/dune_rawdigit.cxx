#ifndef GALLERY_FMWK_SUPERA_DUNE_RAWDIGIT_CXX
#define GALLERY_FMWK_SUPERA_DUNE_RAWDIGIT_CXX

#include "dune_rawdigit.h"

// Larsoft includes:
#include "lardataobj/RawData/RawDigit.h"
#include "lardataobj/RawData/raw.h"
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv includes:
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"

namespace supera {

void DUNERawDigit::initialize() {

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

  for (size_t i = 0; i < _image_meta_2d.size(); i++)
    images.push_back(larcv::Image2D(_image_meta_2d.at(i)));

  for (auto& digit : *raw_digits) {
    int channel = digit.Channel();

    raw::RawDigit::ADCvector_t ADCs(digit.Samples()); // fix the size!
    raw::Uncompress(digit.ADCs(), ADCs, digit.Compression());

    // std::cout << "Channel: " << channel << std::endl;
    // int this_projection_id = projection_id(channel);
    // int this_column = column(channel);
    int this_projection_id = 0;
    int this_column = channel;

    // Loop over the digit and compress it:
    for (size_t i_row = 0; i_row < n_ticks; i_row++) {
      // int this_row = row(i_row, channel) / compression;
      int this_row = i_row;
      float val = (ADCs.at(i_row) - digit.GetPedestal());
      // if (this_row < _image_meta_2d.at(this_projection_id).rows() &&
          // this_column < _image_meta_2d.at(this_projection_id).cols()) {
      val += images.at(this_projection_id).pixel(this_row, this_column);
      images.at(this_projection_id).set_pixel(this_row, this_column, val);
      // }
      // else{
        // std::cout <<" Tried to access at row " << this_row << " and column " << this_column << std::endl;
      // }
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