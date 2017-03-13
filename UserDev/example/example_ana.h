/**
 * \file example_ana.h
 *
 * \ingroup nuexsec_analysis
 *
 * \brief Class def header for a class example_ana
 *
 * @author cadams
 */

/** \addtogroup nuexsec_analysis

    @{*/

#ifndef GALLERY_FMWK_EXAMPLE_ANA_H
#define GALLERY_FMWK_EXAMPLE_ANA_H

#include "TTree.h"

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "Analysis/ana_base.h"

#include "lardataobj/RecoBase/Track.h"


namespace galleryfmwk {

/**
   \class example_ana
   User custom analysis class made by SHELL_USER_NAME
 */
class example_ana : galleryfmwk::ana_base {

public:

  /// Default constructor
  example_ana() {_verbose = false;}

  /// Default destructor
  // ~example_ana() {}


  bool initialize();


  bool analyze(gallery::Event * ev);


  bool finalize();


  void setTrackProducer(std::string s) {_track_producer = s;}
  void setVerbose(bool b){_verbose = b;}

protected:

  std::string _track_producer;
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

/** @} */ // end of doxygen group
