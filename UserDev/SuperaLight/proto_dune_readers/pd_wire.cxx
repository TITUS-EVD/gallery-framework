#ifndef GALLERY_FMWK_SUPERA_ProtoDune_WIRE_CXX
#define GALLERY_FMWK_SUPERA_ProtoDune_WIRE_CXX

#include "pd_wire.h"

// Larsoft includes:
// #include "lardataobj/RecoBase/Wire.h"
#include "lardataobj/RawData/RawDigit.h"

// larcv includes:
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace supera {

void ProtoDuneWire::initialize(){
  plane_meta.clear();
  // parameters for ImageMeta are (xmin, ymin, xmax, ymax, nx, ny, units)
  // Well encode tick in y and wire in x.  Units will be centimeters
  // y (drift direction) goes from -200 to 200 for n_ticks * 2 + spacing
  // x (wire direction) goes from 0
  _max_tick = 6000;
  plane_meta.push_back(larcv::ImageMeta(
      0, 0, 480, 1000, 6000, 480, 0, larcv::kUnitCM));

}


void ProtoDuneWire::slice(gallery::Event* ev, larcv::IOManager* io) {
  // This module creates raw digit images.  Therefore, it will slice
  // off the TPC images and store them into the larcv file.
  //
  // We will need image2D meta for each plane,

  // Get the raw digit data:
  std::string _wire_producer = "tpcrawdecoder:daq";
  art::InputTag digit_tag(_wire_producer);
  auto const& wires =
    ev -> getValidHandle<std::vector<raw::RawDigit> >(digit_tag);

  std::cout << wires->size() << std::endl;

  // In this module, we only save one side of the tpc (the one where the neutrino
  // interaction happened) so there is an offset
  // We also downsample by a factor of 4 to make wire and time almost line up

  // // get the Image2D objects:
  auto event_image2d = (larcv::EventImage2D *) io->get_data("image2d", "ProtoDunewire");
  auto event_image2d_noise = (larcv::EventImage2D *) io->get_data("image2d", "ProtoDunewireNoise");
  std::vector<larcv::Image2D> images;
  std::vector<larcv::Image2D> images_noise;

  // for (size_t i = 0; i < plane_meta.size(); i ++)
  images.push_back(larcv::Image2D(plane_meta.at(0)));
  images_noise.push_back(larcv::Image2D(plane_meta.at(0)));

  int configuration = 3;

  // Configuration 1: run 4646
  // Configuration 2: run 4696, subrun 1
  // Configuration 3: run 4696, subruns 7 and 9

  int min_channel;
  int max_channel;

  if (configuration == 1){
    min_channel = 4160;
    max_channel = 4639;
  }
  else if (configuration == 2){
    min_channel = 2080;
    max_channel = 2559;
  }
  else if (configuration == 3){
    min_channel = 7200;
    max_channel = 7679;
  }


  for (auto & wire : * wires)
  {
    int channel = wire.Channel();
    if (channel < min_channel | channel > max_channel) continue;
    std::cout << channel << std::endl;

    int wire_index = channel - min_channel;


    if (configuration == 1){
      if (wire_index == 219) continue;
      if (wire_index == 221) continue;
      if (wire_index == 223) continue;
      if (wire_index == 227) continue;
      if (wire_index == 250) continue;
    }
    else if (configuration == 2){
    }
    else if (configuration == 3){
      if (wire_index == 351) continue;
      if (wire_index == 382) continue;

    }

    int i = 0;
    // std::cout << "Number of ADCs: " << wire.NADC() << std::endl;


    for (auto & adc : wire.ADCs() ){
      // std::cout << "i: " << i << ", wire_index: " << wire_index << std::endl;
      images.at(0).set_pixel(i, wire_index, adc - wire.GetPedestal());
      images_noise.at(0).set_pixel(i, wire_index, adc - wire.GetPedestal());
      i ++;
    }
  }

  //   for (auto iROI = wire.SignalROI().begin_range();iROI < wire.SignalROI().end_range(); ++iROI) {
  //     auto ROI = *iROI;
  //     for (auto tick = ROI.begin_index(); tick < ROI.end_index(); tick ++){
  //       if (tick > n_ticks)
  //         continue;
  //       // std::cout << "tick is " << tick << std::endl;
  //       int this_row = row(tick, channel) / compression;
  //       float val = ROI[tick] / (1.0*compression);
  //       val += images.at(this_projection_id).pixel(this_row, this_column);
  //     images.at(this_projection_id).set_pixel(this_row, this_column, val);
  //     }
  //   }

  // }

  // // Compress the images with an absolute value threshold:
  // for (auto & image : images){
  //   for (size_t i = 0; i < image.size(); i ++){
  //     if (image.as_vector().at(i) < 5)
  //       image.set_pixel(i, 0.0);
  //   }
  // }

  // // for (auto & image : images){
  // //   image.threshold(100, 0.0);
  // // }

  event_image2d_noise -> emplace(std::move(images_noise));

  // Here is a weak implementation of the correlated noise filter:
  int block_size = 48;
  std::cout << "Doing correlated noise subtraction." <<std::endl;
  for (int i_block = 0; i_block < 10; i_block ++){
    std::vector<float> correlated_noise_waveform;
    correlated_noise_waveform.resize(6000);

    // Loop over every wire in this block, and calculate the median of those ticks.
    for (int tick = 0; tick < 6000; tick ++){
      std::vector<float> digit_accumulator;
      digit_accumulator.reserve(block_size);

      for (int wire = i_block*block_size; wire < (i_block +1)*block_size; wire++){
        digit_accumulator.push_back(images.at(0).pixel(tick, wire));
      }

      // Calculate the median of this set of digits:
      std::nth_element(digit_accumulator.begin(),
                       digit_accumulator.begin() + digit_accumulator.size()/2,
                       digit_accumulator.end()
                      );
      correlated_noise_waveform.at(tick) = digit_accumulator[digit_accumulator.size()/2];
    }

    // Now subtract the correlated noise from the image:
    for (int wire = i_block*block_size; wire < (i_block +1)*block_size; wire++){
      for (int tick = 0; tick < 6000; tick ++){
        float value = images.at(0).pixel(tick, wire) - correlated_noise_waveform.at(tick);
        images.at(0).set_pixel(tick, wire, value);
      }
    }

  }

  // // Emplace the images:
  event_image2d -> emplace(std::move(images));

  return;
}
}

#endif