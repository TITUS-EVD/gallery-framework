#ifndef NOISEFILTER_TYPES_H
#define NOISEFILTER_TYPES_H

#include "LArUtil/Geometry.h"


namespace ub_noise_filter {


/**
 * @brief Returns the approximate mode of a list
 * @details Bins the input into a histogram and returns the bin with the
 *          largest number of counts
 *
 * @param _input input vector of data
 * @param lowbin lowest bin to use
 * @param highbin highest bin to use
 * @param n_bins number of bins to use
 * @return Approximate mode of the values
 */
float getMode(const std::vector<float> & _input, float lowbin, float highbin, int n_bins);


/**
 * @brief Returns the median of a vector
 * @details Finds the nth_element of a vector.  This function can and will change the input.
 *
 * @tparam _input input list of values
 * @return The median of the vector
 */
float getMedian( std::vector<float> & _input);

/**
 * @brief Return the mean of a vector
 * @details Calculates the mean of the vector provided in a totally unsurprising way.
 *          WILL CHANGE THE INPUT VECTOR!
 * 
 * @param r vector of floats
 * @return The mean of above vector
 */
float getMean(const std::vector<float> &);


/**
 * @brief Get the correlation of two vectors
 * @details Checks that each vector is of the same size, then uses the overloaded method
 *          of the same name to do the correlation calculation
 * 
 * @param _input1 second vector of floats
 * @param _input2 second vector of floats
 * @return The correlation value, between -1 and 1
 */
float getCorrelation(const std::vector<float> & _input1, const std::vector<float> & _input2);


/**
 * @brief Get the correlation of two arrays
 * @details Compute the correlation of two arrays of length N.  Memory allocation
 *          is not handled here, nor size checking.
 * 
 * @param _input1 second vector of floats
 * @param _input2 second vector of floats
 * @param N Number of elements in each vector
 * @return The correlation value, between -1 and 1
 */
float getCorrelation(const float * _input1, const float * _input2, unsigned int N);


/**
 * @brief Detector Properties interface for the noise filter
 * @details In the entire noise filter framework, this is really the only 
 *          framework (larlite, etc.) dependent class.  This is needed
 *          to get the length of wires, which is important in harmonic noise calculations.
 */
class detPropFetcher {

public:

  detPropFetcher();

  unsigned int n_wires(unsigned int plane);
  unsigned int n_planes();
  double wire_length(unsigned int plane, unsigned int wire);
  double wire_scale(unsigned int plane, unsigned int wire);

  /**
   * @brief Get the start and end wires of mother board blocks on each plane
   * @details Vector of start and end wires for each motherboard block is returned.
   *          In general, use these boundaries in the typical c++ way:  for block i,
   *          use a loop over wires from result[i] to < result[i+1]
   * 
   * @param plane Plane 0, 1, or 2
   */
  const std::vector<float> & correlated_noise_blocks(int plane) const {
    return _correlated_noise_blocks.at(plane);
  }

  /**
   * @brief return the motherboard on the same service board in this plane
   * @details Gives the mother board that should be most highly correlated
   *          to the input motherboard
   * 
   * @param plane 0, 1, or 2
   * @param block Motherboard (or index of block of wires)
   * 
   * @return The same-service-board motherboard (or index of block of wires)
   */
  int same_plane_pair(int plane, int block);

  /**
   * @brief Get the list of (plane, block) within a service board
   * @details Returns a vector of vector of ints, size 3xN.  Input a plane and block
   *          and the return will have all the motherboard blocks on the same service board.
   *          Generally this is 3x2, but the vertical motherboards in uboone are unique.
   * 
   * @param plane 0, 1 or 2
   * @param block Motherboard (or index of block of wires)
   * 
   * @return vector of correlated motherboards to input
   */
  std::vector<std::vector<float> > service_board_block(int plane, int block);

private:

  // The function for wire lengths is called A LOT
  // and so the values of the function are cached
  // (To avoid calls to sqrt)
  std::vector<std::vector<double> > _wire_lengths;
  std::vector<std::vector<double> > _wire_scales;

  // This defines what blocks to use for correlated noise removal.
  // It's every 48 wires in induction, 96 in collection
  // That corresponds to motherboards in the TPC
  const std::vector<std::vector<float> > _correlated_noise_blocks = {
    {
      0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480,
      528, 576, 624, 672, 720, 768, 816, 864, 912, 960,
      1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392,
      1440, 1488, 1536, 1584, 1632, 1680, 1728, 1776, 1824,
      1872, 1920, 1968, 2016, 2064, 2112, 2160, 2208,
      2256, 2304, 2352, 2400
    },
    {
      0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480,
      528, 576, 624, 672, 720, 768, 816, 864, 912, 960,
      1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 
      1440, 1488, 1536, 1584, 1632, 1680, 1728, 1776, 1824, 
      1872, 1920, 1968, 2016, 2064, 2112, 2160, 2208,
      2256, 2304, 2352, 2400
    },
    {
      0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 
      1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728, 1824,
      1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688,
      2784, 2880, 2976, 3072, 3168, 3264, 3360, 3456
    }
  };



};


//Used in classifying which wires are behaving in different ways
enum wireStatus {kNormal, kDead, kHighNoise, kChirping, kNStatus};

// Used to keep track of when chirping starts and stops on a wire
class chirp_info {

public:
  size_t chirp_start;
  size_t chirp_stop;
  float chirp_frac;
};

}

#endif
