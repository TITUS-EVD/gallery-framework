/**
 * \file pd_supera_light.h
 *
 * \ingroup SuperaLight
 *
 * \brief Class def header for a class pd_supera_light
 *
 * @author cadams
 */

/** \addtogroup nuexsec_analysis

    @{*/

#ifndef GALLERY_FMWK_PD_SUPERA_H
#define GALLERY_FMWK_PD_SUPERA_H

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "Analysis/ana_base.h"

#include "pd_module_base.h"

namespace supera {

/**
   \class pd_supera_light
   User custom analysis class made by SHELL_USER_NAME
 */
class pd_supera_light : galleryfmwk::ana_base {
 public:
  /// Default constructor
  pd_supera_light() { _verbose = false; }

  /// Default destructor
  // ~pd_supera_light() {}

  bool initialize();

  bool analyze(gallery::Event* ev);

  bool finalize();

  void set_io_manager(larcv::IOManager * io){_io = io;}

  /**
 * @brief Add a module to the list of modules that run slicing
 *
 * @param module The module to be added, must at least inherit from
 * ProtoDuneModuleBase
 */
  void add_supera_module(ProtoDuneModuleBase* module);

  /**
   * @brief set verbosity mode
   */
  void set_verbose(bool b = true) { _verbose = b; }

 protected:

  int projection_id(int channel);
  int column(int channel);
  int row(int tick, int channel);

  std::vector<ProtoDuneModuleBase*> _modules;

  larcv::IOManager * _io;

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
