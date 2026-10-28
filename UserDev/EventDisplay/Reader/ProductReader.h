/**
 * \file ProductReader.h
 *
 * \brief Fill flat columns from the art products TITUS draws, for
 * titus.readers.GalleryReader.
 *
 * The columns mirror the containers in titus/readers/products.py (same
 * names, row i = element i of the product vector). This header deliberately
 * includes no LArSoft or gallery headers, so loading the dictionary from
 * python doesn't make cling parse them. Depends only on gallery and the
 * product libraries: no geometry, services or larcorealg/lardataalg.
 */

#ifndef TITUSIO_PRODUCTREADER_H
#define TITUSIO_PRODUCTREADER_H

#include <string>
#include <vector>

namespace gallery {
  class Event;
}

namespace titusio {

  /// recob::Hit
  struct HitColumns {
    std::vector<unsigned int> channel;
    std::vector<int> cryostat;
    std::vector<int> tpc;
    std::vector<int> plane;
    std::vector<int> wire;
    std::vector<float> peak_time;
    std::vector<float> rms;
    std::vector<float> integral;
    std::vector<float> peak_amplitude;
    std::vector<int> start_tick;
    std::vector<int> end_tick;

    void clear();
  };

  /// recob::Wire and recob::ChannelROI: ROI j of row roi_row[j] starts at
  /// tick roi_start[j] and holds values[roi_offset[j], roi_offset[j + 1])
  struct WireColumns {
    std::vector<unsigned int> channel;
    std::vector<int> n_ticks;
    std::vector<int> roi_row;
    std::vector<int> roi_start;
    std::vector<long long> roi_offset;
    std::vector<float> values;

    void clear();
  };

  /// raw::RawDigit, uncompressed; adc is row-major (rows x width), each row
  /// padded with zeros to the longest waveform
  struct RawDigitColumns {
    std::vector<unsigned int> channel;
    std::vector<float> pedestal;
    std::vector<int> n_samples;
    std::vector<short> adc;
    unsigned long width = 0;

    void clear();
  };

  /// Each reader clears and fills its columns. Returns false if the event has
  /// no such product; other failures throw.
  bool readHits(gallery::Event const& ev, std::string const& tag, HitColumns& out);
  bool readWires(gallery::Event const& ev, std::string const& tag, WireColumns& out);
  bool readRawDigits(gallery::Event const& ev, std::string const& tag, RawDigitColumns& out);

  /// recob::ChannelROI lives in sbnobj; without it this throws
  bool readChannelROIs(gallery::Event const& ev, std::string const& tag, WireColumns& out);
  bool hasChannelROISupport();

} // namespace titusio

#endif
