#ifndef CORRELATEDNOISEFILTER_CXX
#define CORRELATEDNOISEFILTER_CXX

#include "CorrelatedNoiseFilter.h"

namespace ub_noise_filter {

void CorrelatedNoiseFilter::reset() {
  _correlatedNoiseWaveforms.clear();
  _correlatedNoiseWaveforms.resize(_detector_properties_interface.n_planes());
  for (size_t i = 0; i < _detector_properties_interface.n_planes(); i ++) {
    _correlatedNoiseWaveforms.at(i).resize(_detector_properties_interface.correlated_noise_blocks(i).size() - 1);
  }

  _harmonicNoiseWaveforms.clear();
  _harmonicNoiseWaveforms.resize(_detector_properties_interface.n_planes());


}


void CorrelatedNoiseFilter::remove_correlated_noise(
  float * _data_arr,
  int N,
  unsigned int wire,
  unsigned int plane) {

  if (_wire_status_by_plane->at(plane)[wire] == kDead ||
      _wire_status_by_plane->at(plane)[wire] == kHighNoise) {
    // Make sure all the ticks are zeroed out:
    for (int tick = 0; tick < N; tick ++) {
      _data_arr[tick] = 0.0;
    }
    return;
  }

  // First, need to know what block this wire came from:
  size_t i_block;
  for (i_block = 0;
       i_block < _detector_properties_interface.correlated_noise_blocks(plane).size();
       i_block ++)
  {
    if (_detector_properties_interface.correlated_noise_blocks(plane).at(i_block + 1) > wire) {
      // Then the block is the correct one!
      break;
    }
  }

  // Now subtract the waveform from this wire
  int start_tick = 0;
  int end_tick = N;



  if (_chirp_info_ptr -> at(plane).find(wire) != _chirp_info_ptr -> at(plane).end()) {
    // this wire IS chirping, so only use the good range:
    // Either start or the end of the wire will be one range of the chirping.
    if (_chirp_info_ptr -> at(plane)[wire].chirp_start == 0) {
      start_tick = _chirp_info_ptr -> at(plane)[wire].chirp_stop;
      end_tick = N;
    }
    else {
      start_tick = 0.0;
      end_tick = _chirp_info_ptr -> at(plane)[wire].chirp_start;
    }
  }
  else {
    // Then this wire is not chirping, use the whole range for pedestal subtraction

  }


  for (int tick = start_tick; tick < end_tick; tick ++) {
    // Scale the harmonic noise by wire length:
    _data_arr[tick] -=
      _detector_properties_interface.wire_scale(plane, wire) *
      _harmonicNoiseWaveforms[plane][tick];
    // _data_arr[tick] -= _harmonicNoiseWaveforms[plane][tick];
    _data_arr[tick] -= _correlatedNoiseWaveforms[plane][i_block][tick];
  }

  // std::cout << _correlatedNoiseWaveforms[plane][19][1000] << std::endl;
  // std::cout << _correlatedNoiseWaveforms[plane][20][1000] << std::endl;

  return;

}


void CorrelatedNoiseFilter::build_noise_waveforms(
  float * _plane_data,
  int plane,
  int _n_time_ticks_data)
{

  // Make sure each noise vector is set to zero when this function is called:
  for (size_t i_block = 0; i_block < _detector_properties_interface.correlated_noise_blocks(plane).size() - 1; i_block ++) {
    _correlatedNoiseWaveforms.at(plane).at(i_block).clear();
    _correlatedNoiseWaveforms.at(plane).at(i_block).resize(_n_time_ticks_data);
  }


  _harmonicNoiseWaveforms.at(plane).clear();
  _harmonicNoiseWaveforms.at(plane).resize(_n_time_ticks_data);

  // First build a guess at the harmonic noise waveforms:
  build_harmonic_noise_waveform(_plane_data, plane, _n_time_ticks_data);

  // Build uncorrected correlated noise waveforms
  build_coherent_noise_waveforms(_plane_data, plane, _n_time_ticks_data);


}

void CorrelatedNoiseFilter::build_coherent_noise_waveforms(
  float * _plane_data,
  int plane,
  int _n_time_ticks_data)
{


  // Loop over each block and get the median tick within that block
  for (size_t i_block = 0;
       i_block < _detector_properties_interface.correlated_noise_blocks(plane).size() - 1;
       i_block ++) {
    _correlatedNoiseWaveforms.at(plane).at(i_block).clear();
    _correlatedNoiseWaveforms.at(plane).at(i_block).resize(_n_time_ticks_data);


    int block_wire_start
      = _detector_properties_interface.correlated_noise_blocks(plane).at(i_block);
    int block_wire_end
      = _detector_properties_interface.correlated_noise_blocks(plane).at(i_block + 1);

    std::vector<float> _median_accumulator;

    int offset;
    for (int tick = 0; tick < _n_time_ticks_data; tick ++) {
      _median_accumulator.clear();
      _median_accumulator.reserve(block_wire_end - block_wire_start);
      for (int wire = block_wire_start; wire < block_wire_end ; wire ++) {
        // Only use wires that are Normal in calculating the noise.
        if (_wire_status_by_plane->at(plane)[wire] == kNormal) {
          offset = tick + wire * _n_time_ticks_data;
          float scale = _detector_properties_interface.wire_scale(plane, wire);
          _median_accumulator.push_back(_plane_data[offset] -
                                        scale * _harmonicNoiseWaveforms[plane][tick]);


        }
      }

      // Now find the median of this tick:
      if (_median_accumulator.size() < 8) {
        continue;
      }
      _correlatedNoiseWaveforms.at(plane).at(i_block).at(tick)
        = getMedian(_median_accumulator);


      // if (i_block == 19 and plane == 0) {
      //   std::cout << "Median at tick " << tick << " is "
      //             << _correlatedNoiseWaveforms.at(plane).at(i_block).at(tick)
      //             << std::endl;
      // }

    }




  }



  return;
}



void CorrelatedNoiseFilter::build_harmonic_noise_waveform(
  float * _plane_data,
  int plane,
  int _n_time_ticks_data)
{

  // Only consider wires that are full length wires:
  double _min_length = _max_wire_lengths[plane];

  std::vector<float> harmonic_noise;
  harmonic_noise.resize(_n_time_ticks_data);

  // Loop over the ticks, and then the wires, and get the most probable
  // value for the harmonic noise
  for (int tick = 0; tick < _n_time_ticks_data; tick ++) {

    std::vector<float> values;
    values.reserve(_detector_properties_interface.n_wires(plane));

    // float lowVal = 999;
    // float highVal = -999;

    for (unsigned int wire = 0;
         wire < _detector_properties_interface.n_wires(plane);
         wire += 10)
    {

      // Need to know which correlated noise block this wire is from:
      size_t i_block;
      for (i_block = 0;
           i_block < _detector_properties_interface.correlated_noise_blocks(plane).size();
           i_block ++)
      {
        if (_detector_properties_interface.correlated_noise_blocks(plane).at(i_block + 1) > wire) {
          // Then the block is the correct one!
          break;
        }
      }

      if (_detector_properties_interface.wire_length(plane, wire) >= _min_length) {
        int offset = wire * _n_time_ticks_data;
        auto _data_val = _plane_data[offset + tick] -
                         _correlatedNoiseWaveforms[plane][i_block][tick];
        values.push_back(_data_val);
        // if (_data_val < lowVal) {
        //   lowVal = _data_val;
        // }
        // if (_data_val > highVal) {
        //   highVal = _data_val;
        // }

      }
    }

    // Get the most probable value:
    harmonic_noise.at(tick) = getMedian(values);
  }

  // Merge this information into the best guess for harmonic noise:
  _harmonicNoiseWaveforms.at(plane) = harmonic_noise;

}

void CorrelatedNoiseFilter::fix_medium_angle_tracks(float * _plane_data,
    int plane,
    int _n_time_ticks_data)
{

  // Loop over every same plane pair of correlated waveforms.
  // For the sides of the TPC, do something different.
  //
  // measure the correlation between small blocks of waveforms that *ought*
  // to be very highly correlated
  //
  // If there is a big drop in the correlation, recalculate the correlated noise
  // waveform in that region

  std::vector< int > windows_to_fix;

  int local_windowsize = 200;
  int n_windows = _correlatedNoiseWaveforms[0][0].size() / local_windowsize;


  for (size_t i_block = 0;
       i_block < _detector_properties_interface.correlated_noise_blocks(plane).size();
       i_block ++)
  {

    int current_block = i_block;
    int matched_block = _detector_properties_interface.same_plane_pair(plane, i_block);

    if (matched_block >= _correlatedNoiseWaveforms.at(plane).size()) {
      continue;
    }

    // For each block, get the correlation of this block to the corresponding block
    // on the other cross correlated waveforms

    // keep track of which windows need to be investigated:
    std::vector<int> windows_to_investigate;
    std::vector<float> correlations;

    for (int i_window = 0; i_window < n_windows; i_window ++ ) {
      float * _this_block_data = &(_correlatedNoiseWaveforms[plane][current_block][i_window * local_windowsize]);
      float * _cross_data      = &(_correlatedNoiseWaveforms[plane][matched_block][i_window * local_windowsize]);
      float _corr = getCorrelation(_this_block_data, _cross_data, local_windowsize);

      if (_corr < 0.7) {

        windows_to_investigate.push_back(i_window);
        correlations.push_back(_corr);
      }
    }

    // Now fix the windows that are broken
    // This involves remaking the median estimate, for this window, using
    for (int i_window = 0; i_window < windows_to_investigate.size(); i_window ++ ) {
      int window = windows_to_investigate[i_window];
      // std::cout << "Looking at block of wires from "
      //           << _detector_properties_interface.correlated_noise_blocks(plane)[i_block]
      //           << " to "
      //           << _detector_properties_interface.correlated_noise_blocks(plane)[i_block + 1]
      //           << std::endl;
      // std::cout << "Ticks  " << local_windowsize*window <<  " to " << local_windowsize*(window + 1)
      //           << ", the correlation between "
      //           << "(" << plane << ", " << current_block << ") and "
      //           << "(" << plane << ", " << matched_block << ") is " << correlations[i_window] << std::endl;

      // Have to loop over every tick in the combined range, and recalculate the median.  Naturally,
      // that also involves subtracting harmonic noise too.

      int wire_start = std::min(_detector_properties_interface.correlated_noise_blocks(plane)[current_block],
                                _detector_properties_interface.correlated_noise_blocks(plane)[matched_block]);

      int wire_end = std::max(_detector_properties_interface.correlated_noise_blocks(plane)[current_block + 1],
                              _detector_properties_interface.correlated_noise_blocks(plane)[matched_block + 1]);


      // Actually break this into 4 different medians, and then we'll take the median of THAT
      std::vector<std::vector<float> > _median_accumulator;


      int offset;
      for (int tick = local_windowsize * window; tick < local_windowsize * (window + 1); tick ++) {
        _median_accumulator.clear();
        _median_accumulator.resize(5);
        for (int wire = wire_start; wire < wire_end ; wire ++) {
          int n = (5 * (wire - wire_start)) / (wire_end  - wire_start);


          if (_wire_status_by_plane->at(plane)[wire] == kNormal) {
            offset = tick + wire * _n_time_ticks_data;
            float scale = _detector_properties_interface.wire_scale(plane, wire);
            _median_accumulator.at(n).push_back(_plane_data[offset] -
                                                scale * _harmonicNoiseWaveforms[plane][tick]);


          }

        }

        // std::cout << "_median_accumulator.size() " << _median_accumulator.size() << std::endl;

        // Now find the median of this tick:
        // if (_median_accumulator.size() < 8) {
        // continue;
        // }
        std::vector<float > medians;
        for (auto & vec : _median_accumulator) {
          if (_median_accumulator.size() > 4)
            medians.push_back(getMedian(vec));
        }

        std::sort (medians.begin(), medians.end());

        float _final_median = getMedian(medians);

        // Some care needs to be taken here.  Because there are clearly so many points
        // that have high charge, the median is already sure to be offset.

        // So, we can compute (approximately) the most probable value
        // in this list.  Then, compute the rms, and exclude all points
        // that are more than 1.5 sigma away from the mode.
        //
        // Then, compute the median, and use that.


        if (_final_median != 0.0) {
          _correlatedNoiseWaveforms.at(plane).at(current_block).at(tick)
            = _final_median;
          _correlatedNoiseWaveforms.at(plane).at(matched_block).at(tick)
            = _final_median;
        }

      }



    }

  }




}

void CorrelatedNoiseFilter::fix_correlated_noise_errors() {

  // Loop over the blocks of wires and fix the errors that come up

  for (int i_plane = 0; i_plane < _correlatedNoiseWaveforms.size(); i_plane ++) {
    for (int i_block = 0; i_block < _correlatedNoiseWaveforms.at(i_plane).size(); i_block ++) {
      find_correlated_noise_errors(i_plane, i_block);
    }
  }

}


void CorrelatedNoiseFilter::find_correlated_noise_errors(int target_plane, int target_block) {

  // Here's the strategy:  Look at each correlated noise waveform and
  // its correlation to every other waveform that it has a strong correlation with.
  //
  // Practically, that means every wave form needs it's correlation to 5 other boards:
  //   The other motherboard in the same plane
  //   Both motherboards in both other planes
  //
  // Also, break each correlation into pieces of 200 ticks to help find anomalous correlations

  // For right now, only looking at known locations of problems

  // Let's just look at one block to see how the block making works


  auto correlated_blocks
    = _detector_properties_interface.service_board_block(target_plane, target_block);

  std::vector< int > windows_to_fix;

  std::vector< int > windows_to_investigate;
  int n_windows = _correlatedNoiseWaveforms[target_plane][target_block].size() / windowsize[target_plane];

  windows_to_investigate.resize(n_windows);

  for (size_t i_plane = 0;
       i_plane < correlated_blocks.size();
       i_plane ++)
  {
    for (size_t i_block = 0;
         i_block < correlated_blocks.at(i_plane).size();
         i_block ++)
    {

      int current_block = correlated_blocks.at(i_plane).at(i_block);

      if (i_plane == target_plane && current_block == target_block) {
        continue;
      }

      // Break it into blocks of 195 ticks

      // For each block, get the correlation of this block to the corresponding block
      // on the other cross correlated waveforms

      // keep track of which windows need to be investigated:

      for (int i_window = 0; i_window < n_windows; i_window ++ ) {
        float * _this_block_data = &(_correlatedNoiseWaveforms[target_plane][target_block][i_window * windowsize[target_plane]]);
        float * _cross_data = &(_correlatedNoiseWaveforms[i_plane][current_block][i_window * windowsize[target_plane]]);
        float _corr = getCorrelation(_this_block_data, _cross_data, windowsize[target_plane]);
        // std::cout << "Window " << i_window << ", the correlation between "
        //           << "(" << target_plane << ", " << target_block << ") and "
        //           << "(" << i_plane << ", " << current_block << ") is " << _corr << std::endl;
        if (_corr < 0.5) {
          windows_to_investigate.at(i_window) ++ ;
        }
      }



    }
  }


  // // Print out what regions of this block are under revision:
  // std::cout << "Comparing at plane " << target_plane << " wires "
  //           << _detector_properties_interface.correlated_noise_blocks(target_plane).at(target_block)
  //           << " to "
  //           << _detector_properties_interface.correlated_noise_blocks(target_plane).at(target_block + 1)
  //           << " TO \n plane " << i_plane << " wires "
  //           << _detector_properties_interface.correlated_noise_blocks(i_plane).at(current_block)
  //           << " to "
  //           << _detector_properties_interface.correlated_noise_blocks(i_plane).at(current_block + 1)
  //           << std::endl;



  std::cout << "In plane " << target_plane << " wires "
            << _detector_properties_interface.correlated_noise_blocks(target_plane).at(target_block)
            << " to "
            << _detector_properties_interface.correlated_noise_blocks(target_plane).at(target_block + 1)
            << ": \n";
  for (int i_window = 0; i_window < windows_to_investigate.size(); i_window ++ ) {
    if (windows_to_investigate.at(i_window) > 1) {

      float prev_rms = 0.0;
      float rms = 0.0;
      float next_rms = 0.0;
      for (int tick = 0; tick < windowsize[target_plane]; tick ++) {
        if (i_window != 0) {
          prev_rms
          += _correlatedNoiseWaveforms[target_plane][target_block][windowsize[target_plane] * (i_window - 1) + tick] *
             _correlatedNoiseWaveforms[target_plane][target_block][windowsize[target_plane] * (i_window - 1) + tick];
        }
        rms
        += _correlatedNoiseWaveforms[target_plane][target_block][windowsize[target_plane] * (i_window) + tick] *
           _correlatedNoiseWaveforms[target_plane][target_block][windowsize[target_plane] * (i_window) + tick];

        if (i_window != n_windows - 1) {
          next_rms
          += _correlatedNoiseWaveforms[target_plane][target_block][windowsize[target_plane] * (i_window + 1) + tick] *
             _correlatedNoiseWaveforms[target_plane][target_block][windowsize[target_plane] * (i_window + 1) + tick];
        }
      }
      prev_rms /= (float) windowsize[target_plane];
      rms /= (float) windowsize[target_plane];
      next_rms /= (float) windowsize[target_plane];

      if (rms < rms_minimum[target_plane]) {
        continue;
      }

      // std::cout << "  Examine ticks " << i_window*windowsize[target_plane]
      //           << " to " << (i_window + 1)*(windowsize[target_plane])
      //           << " (strength == " << windows_to_investigate.at(i_window) << ")"
      //           << std::endl;

      // std::cout << "    RMS  of previous window: " << prev_rms << "\n"
      //           << "    RMS  of this window: " << rms << "\n"
      //           << "    RMS  of next window: " << next_rms << "\n";

      if (
        ( i_window != 0 && rms > 3 * prev_rms) ||
        ( i_window != n_windows - 1 && rms > 3 * next_rms)
      ) {
        windows_to_fix.push_back(i_window);
        // std::cout << "----this window tagged to be fixed." << std::endl;
      }

    }
  }

  // Now fix some windows.
  // Measure the correlations between the adjacent windows that need to be fixed and
  // Find the other waveform with the best correlation



  for (int i_window : windows_to_fix) {

    float best_correlation = -999;
    int best_plane = -1;
    int best_block = -1;
    std::vector<float> _this_waveform;
    std::vector<float> _best_waverform;
    // Loop over the correlated regions, and look at the correlation of the outside regions


    int prev_window = i_window - 1;
    int next_window = i_window + 1;

    if (prev_window >= 0) {
      for (int tick = 0; tick < windowsize[target_plane]; tick ++) {
        _this_waveform.push_back(
          _correlatedNoiseWaveforms[target_plane][target_block][prev_window * windowsize[target_plane] + tick]);
      }
    }


    if (next_window !=  n_windows) {
      for (int tick = 0; tick < windowsize[target_plane]; tick ++) {
        _this_waveform.push_back(
          _correlatedNoiseWaveforms[target_plane][target_block][next_window * windowsize[target_plane] + tick]);
      }
    }

    for (size_t i_plane = 0;
         i_plane < correlated_blocks.size();
         i_plane ++)
    {
      for (size_t i_block = 0;
           i_block < correlated_blocks.at(i_plane).size();
           i_block ++)
      {

        int current_block = correlated_blocks.at(i_plane).at(i_block);

        if (i_plane == target_plane) {
          continue;
        }

        float _corr = 0;

        // build up the waveforms to use.
        std::vector<float> _other_waveform;

        if (prev_window >= 0) {
          for (int tick = 0; tick < windowsize[target_plane]; tick ++) {
            _other_waveform.push_back(
              _correlatedNoiseWaveforms[i_plane][current_block][prev_window * windowsize[target_plane] + tick]);
          }
        }

        if (next_window !=  n_windows) {
          for (int tick = 0; tick < windowsize[target_plane]; tick ++) {
            _other_waveform.push_back(
              _correlatedNoiseWaveforms[i_plane][current_block][next_window * windowsize[target_plane] + tick]);
          }
        }

        // get the correlation:
        float correlation = getCorrelation(_this_waveform, _other_waveform);

        if (correlation > best_correlation) {
          best_plane = i_plane;
          best_block = current_block;
          _best_waverform = _other_waveform;
          best_correlation = correlation;
        }

      }
    }

    // If there is no fit, continue:
    if (best_plane == -1) {
      continue;
    }

    // // Now that the best area has been selected, use it.
    // std::cout << "Replacing window " << i_window << " with values from "
    //           << "( " << best_plane << ", " << best_block << ")"
    //           << ", correlation: " << best_correlation
    //           << std::endl;

    float val1 = 0.0;
    float val2 = 0.0;
    for (int tick = 0; tick < windowsize[target_plane]; tick ++) {
      val1 += _correlatedNoiseWaveforms[target_plane][target_block][i_window * windowsize[target_plane] + tick]
              * _correlatedNoiseWaveforms[best_plane][best_block][i_window * windowsize[target_plane] + tick];
      val2 += _correlatedNoiseWaveforms[target_plane][target_block][i_window * windowsize[target_plane] + tick];
    }

    /*  _____ ___  ____   ___
       |_   _/ _ \|  _ \ / _ \
         | || | | | | | | | | |
         | || |_| | |_| | |_| |
         |_| \___/|____/ \___/

    The way the correlations are scaled should really be looked at more carefully.

    Way, way more carefully.

    */

    float alpha = val1 / val2;

    for (int tick = 0; tick < windowsize[target_plane]; tick ++) {
      _correlatedNoiseWaveforms[target_plane][target_block][i_window * windowsize[target_plane] + tick]
        = alpha * _correlatedNoiseWaveforms[best_plane][best_block][i_window * windowsize[target_plane] + tick];
    }

  }

}




}

#endif
