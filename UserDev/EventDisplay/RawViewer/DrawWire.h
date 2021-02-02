/**
 * \file DrawWire.h
 *
 * \ingroup EventViewer
 *
 * \brief Class def header for a class DrawWire
 *
 * @author cadams
 */

/** \addtogroup EventViewer

    @{*/

#ifndef EVD_DRAWRAWWIRE_H
#define EVD_DRAWRAWWIRE_H

#include "Analysis/ana_base.h"
#include "RawBase.h"

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include "lardataobj/RecoBase/Wire.h"

#include "TTree.h"
#include "TGraph.h"


namespace evd {
  /**
     \class DrawWire
     User custom analysis class made by SHELL_USER_NAME
   */
  class DrawWire : public galleryfmwk::ana_base, public RawBase{

  public:

    /// Default constructor
    DrawWire(const geo::GeometryCore& geometry, const detinfo::DetectorPropertiesData& DetectorProperties);

    /// Default destructor
    virtual ~DrawWire(){}

    /** IMPLEMENT in DrawWire.cc!
        Initialization method to be called before the analysis event loop.
    */
    virtual bool initialize();

    /** IMPLEMENT in DrawWire.cc!
        Analyze a data event-by-event
    */
    virtual bool analyze(const gallery::Event & event);

    /** IMPLEMENT in DrawWire.cc!
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();


    void setPadding(size_t padding, size_t plane);

  private:

    std::vector<size_t> _padding_by_plane;


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
