/**
 * \file supera_light.h
 *
 * \ingroup SuperaLight
 *
 * \brief Class def header for a class supera_light
 *
 * @author cadams
 */

/** \addtogroup nuexsec_analysis

    @{*/

#ifndef GALLERY_FMWK_SUPERA_H
#define GALLERY_FMWK_SUPERA_H

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "Analysis/ana_base.h"

#include "larcv3/core/dataformat/IOManager.h"
#include "larcv3/core/dataformat/ImageMeta.h"

#include "larcorealg/Geometry/GeometryCore.h"
#include "lardataalg/DetectorInfo/DetectorPropertiesData.h"
#include "lardataalg/DetectorInfo/DetectorClocksData.h"

#include "LArUtil/SimpleGeometryHelper.h"

#define _fcl_file_name "/home/cadams/Theta/SBND/gallery-framework/core/LArUtil/dat/services_sbnd.fcl"


namespace supera {

/**
   \class supera_light
   User custom analysis class made by SHELL_USER_NAME
 */
class supera_light : galleryfmwk::ana_base {
 public:
  /// Default constructor
  supera_light(
      // const geo::GeometryCore&               geometry,
      // const detinfo::DetectorPropertiesData& detectorProperties,
      // const detinfo::DetectorClocksData&     detectorClocks
  ) :
  // _det_prop(detectorProperties),
  // _det_clock(detectorClocks),
  _io(larcv3::IOManager::kWRITE) {
      _geo_service = larutil::LArUtilServicesHandler::GetGeometry(_fcl_file_name);
      // auto _det_prop_temp = larutil::LArUtilServicesHandler::GetDetProperties(_fcl_file_name);
      _verbose = false;
   }

  /// Default destructor
  // ~supera_light() {}

  bool initialize();

  bool analyze(gallery::Event* ev);

  bool finalize();

  void set_output_file(std::string outfile);

  std::vector<float> wire_time_from_3D(std::vector<float> position_3d, int plane, int tpc);

  /**
 * @brief Add a module to the list of modules that run slicing
 *
 * @param module The module to be added, must at least inherit from
 * SuperaModuleBase
 */
  // void add_supera_module(SuperaModuleBase* module);

  /**
   * @brief set verbosity mode
   */
  void set_verbose(bool b = true) { _verbose = b; }

  void slice_raw_digit(gallery::Event * ev, larcv3::IOManager & io);
  void slice_wire     (gallery::Event * ev, larcv3::IOManager & io);
  void slice_neutrino (gallery::Event * ev, larcv3::IOManager & io);
  void slice_cluster  (gallery::Event * ev, larcv3::IOManager & io);


protected:

  std::unique_ptr<geo::GeometryCore> _geo_service;
  // detinfo::DetectorClocksData      _det_clock;

  int projection_id(int channel);
  int column(int channel);
  int row(int tick, int channel);

  // SBNDRawDigit raw_digit;
  // SBNDWire     wire;
  // SBNDCluster  cluster;
  // SBNDNeutrino neutrino;

  larcv3::IOManager _io;

  bool _verbose;


  float wire_position(float x, float y, float z, int projection_id);
  float tick_position(float x, float time_offset, int projection_id);

  size_t n_ticks_per_chamber = 2560;
  size_t n_cathode_ticks = 50;
  size_t compression = 4;
  size_t tick_offset = 000;

  size_t total_ticks;


  std::vector<larcv3::ImageMeta2D> _base_image_meta_2D;
  larcv3::ImageMeta3D _base_image_meta_3D;

  // Parameters:
  float _threshold = 10.;

  /*
  Builds the map of particles from geant trackIDs to a list of particles in
  the larcv3 world
  */
  void build_particle_map(gallery::Event * ev, larcv3::IOManager & io);

  std::vector< std::vector< int> > _particle_to_trackID;
  std::map< int, int > _trackID_to_particle;

};
}

#endif

//**************************************************************************
//
// For Analysis framework documentation, read Manual.pdf here:
//
// http://microboone-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=3183
//
//**************************************************************************

/** @} */  // end of doxygen group
