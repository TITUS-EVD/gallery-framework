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

#include "supera_module_base.h"
#include "sbnd_rawdigit.h"
#include "sbnd_wire.h"
#include "sbnd_cluster.h"

namespace supera {

/**
   \class supera_light
   User custom analysis class made by SHELL_USER_NAME
 */
class supera_light : galleryfmwk::ana_base {
 public:
  /// Default constructor
  supera_light() : _io(larcv3::IOManager::kWRITE) { _verbose = false; }

  /// Default destructor
  // ~supera_light() {}

  bool initialize();

  bool analyze(gallery::Event* ev);

  bool finalize();

  void set_output_file(std::string outfile);

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

 protected:

  int projection_id(int channel);
  int column(int channel);
  int row(int tick, int channel);

  SBNDRawDigit raw_digit;
  SBNDWire     wire;
  SBNDCluster  cluster;
  // std::vector<SuperaModuleBase*> _modules;

  larcv3::IOManager _io;

  bool _verbose;
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
