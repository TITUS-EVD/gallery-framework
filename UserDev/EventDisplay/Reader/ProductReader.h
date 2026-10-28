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

  /// recob::Track trajectories: track i has points point_offset[i] to
  /// point_offset[i + 1]; valid is 1 for points with a valid position
  struct TrackColumns {
    std::vector<int> id;
    std::vector<long long> point_offset;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<unsigned char> valid;

    void clear();
  };

  /// recob::Vertex and recob::SpacePoint positions
  struct PointColumns {
    std::vector<int> id;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    void clear();
  };

  /// recob::Shower
  struct ShowerColumns {
    std::vector<int> id;
    std::vector<float> start_x;
    std::vector<float> start_y;
    std::vector<float> start_z;
    std::vector<float> dir_x;
    std::vector<float> dir_y;
    std::vector<float> dir_z;
    std::vector<float> length;
    std::vector<float> open_angle;
    std::vector<int> best_plane;

    void clear();
  };

  /// recob::PFParticle; parent is -1 for primaries
  struct PFParticleColumns {
    std::vector<int> self_index;
    std::vector<int> pdg;
    std::vector<int> parent;

    void clear();
  };

  /// anab::T0, time in ns
  struct T0Columns {
    std::vector<double> time;
    std::vector<int> trigger_type;
    std::vector<int> id;

    void clear();
  };

  /// art::Assns: row j links element left[j] of product
  /// product_tags[left_product[j]] to element right[j] of product
  /// product_tags[right_product[j]] ("" if not identified); meta_index is
  /// the recob::TrackHitMeta index, or -1
  struct AssociationColumns {
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> left_product;
    std::vector<int> right_product;
    std::vector<int> meta_index;
    std::vector<std::string> product_tags;

    void clear();
  };

  /// recob::Cluster; its hits come from the cluster_hits association
  struct ClusterColumns {
    std::vector<int> id;
    std::vector<int> cryostat;
    std::vector<int> tpc;
    std::vector<int> plane;
    std::vector<int> n_hits;
    std::vector<float> start_wire;
    std::vector<float> start_tick;
    std::vector<float> end_wire;
    std::vector<float> end_tick;
    std::vector<float> integral;

    void clear();
  };

  /// recob::EndPoint2D
  struct EndPoint2DColumns {
    std::vector<int> id;
    std::vector<int> cryostat;
    std::vector<int> tpc;
    std::vector<int> plane;
    std::vector<int> wire;
    std::vector<float> drift_time;
    std::vector<float> charge;
    std::vector<float> strength;

    void clear();
  };

  /// simb::MCTruth; neutrino columns are set if has_neutrino. Particles of
  /// truth i are particle_offset[i] to particle_offset[i + 1]
  struct MCTruthColumns {
    std::vector<int> origin;
    std::vector<unsigned char> has_neutrino;
    std::vector<int> nu_pdg;
    std::vector<float> nu_energy;
    std::vector<float> vx;
    std::vector<float> vy;
    std::vector<float> vz;
    std::vector<double> vt;
    std::vector<int> mode;
    std::vector<int> interaction_type;
    std::vector<int> ccnc;
    std::vector<long long> particle_offset;
    std::vector<int> particle_pdg;
    std::vector<int> particle_status;
    std::vector<float> particle_energy;

    void clear();
  };

  /// sim::MCTrack; steps of track i are step_offset[i] to step_offset[i + 1]
  struct MCTrackColumns {
    std::vector<int> track_id;
    std::vector<int> pdg;
    std::vector<int> origin;
    std::vector<std::string> process;
    std::vector<double> start_t;
    std::vector<float> start_e;
    std::vector<long long> step_offset;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<double> t;
    std::vector<float> e;

    void clear();
  };

  /// recob::OpFlash; PEs of flash i (by optical channel) are
  /// pe[pe_offset[i], pe_offset[i + 1])
  struct OpFlashColumns {
    std::vector<double> time;
    std::vector<double> abs_time;
    std::vector<float> time_width;
    std::vector<float> y_center;
    std::vector<float> y_width;
    std::vector<float> z_center;
    std::vector<float> z_width;
    std::vector<float> total_pe;
    std::vector<int> on_beam_time;
    std::vector<long long> pe_offset;
    std::vector<float> pe;

    void clear();
  };

  /// raw::OpDetWaveform; samples of row i are adc[adc_offset[i], adc_offset[i + 1])
  struct OpDetWaveformColumns {
    std::vector<unsigned int> channel;
    std::vector<double> time_stamp;
    std::vector<long long> adc_offset;
    std::vector<unsigned short> adc;

    void clear();
  };

  /// sbnd::crt::FEBData; adc is row-major (rows x 32)
  struct CRTFEBDataColumns {
    std::vector<unsigned short> mac5;
    std::vector<unsigned short> flags;
    std::vector<unsigned int> ts0;
    std::vector<unsigned int> ts1;
    std::vector<unsigned int> coinc;
    std::vector<unsigned int> unix_s;
    std::vector<unsigned short> adc;

    void clear();
  };

  /// sbnd::timing::DAQTimestamp
  struct DAQTimestampColumns {
    std::vector<unsigned int> channel;
    std::vector<unsigned long long> timestamp;

    void clear();
  };

  /// Each reader clears and fills its columns. Returns false if the event has
  /// no such product; other failures throw.
  bool readHits(gallery::Event const& ev, std::string const& tag, HitColumns& out);
  bool readWires(gallery::Event const& ev, std::string const& tag, WireColumns& out);
  bool readRawDigits(gallery::Event const& ev, std::string const& tag, RawDigitColumns& out);

  bool readTracks(gallery::Event const& ev, std::string const& tag, TrackColumns& out);
  bool readVertices(gallery::Event const& ev, std::string const& tag, PointColumns& out);
  bool readSpacePoints(gallery::Event const& ev, std::string const& tag, PointColumns& out);
  bool readShowers(gallery::Event const& ev, std::string const& tag, ShowerColumns& out);
  bool readPFParticles(gallery::Event const& ev, std::string const& tag, PFParticleColumns& out);
  bool readT0s(gallery::Event const& ev, std::string const& tag, T0Columns& out);
  bool readClusters(gallery::Event const& ev, std::string const& tag, ClusterColumns& out);
  bool readEndPoints2D(gallery::Event const& ev, std::string const& tag, EndPoint2DColumns& out);

  bool readMCTruths(gallery::Event const& ev, std::string const& tag, MCTruthColumns& out);
  bool readMCTracks(gallery::Event const& ev, std::string const& tag, MCTrackColumns& out);

  bool readOpFlashes(gallery::Event const& ev, std::string const& tag, OpFlashColumns& out);
  bool readOpDetWaveforms(gallery::Event const& ev, std::string const& tag, OpDetWaveformColumns& out);

  /// SBND CRT and timing products live in sbnobj; without it these throw
  bool readCRTFEBData(gallery::Event const& ev, std::string const& tag, CRTFEBDataColumns& out);
  bool readDAQTimestamps(gallery::Event const& ev, std::string const& tag, DAQTimestampColumns& out);
  bool hasSBNDCRTSupport();

  /// An association product of type cpp_type (as in the file, e.g.
  /// "art::Assns<recob::Hit,recob::Track,recob::TrackHitMeta>"), returned
  /// with left = first type of the kind (see titus/readers/products.py
  /// ASSOCIATIONS) whichever order it was stored in. candidate_tags: input
  /// tags of the products the Ptrs may point into, to identify them.
  /// Throws for an unsupported type.
  bool readAssociation(gallery::Event const& ev, std::string const& tag, std::string const& cpp_type,
                       std::vector<std::string> const& candidate_tags, AssociationColumns& out);

  /// recob::ChannelROI lives in sbnobj; without it this throws
  bool readChannelROIs(gallery::Event const& ev, std::string const& tag, WireColumns& out);
  bool hasChannelROISupport();

} // namespace titusio

#endif
