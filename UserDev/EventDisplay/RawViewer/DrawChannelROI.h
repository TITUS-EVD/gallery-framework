/**
 * \file DrawChannelROI.h
 *
 * \ingroup EventViewer
 *
 * \brief Class def header for a class DrawChannelROI
 *
 * @author cadams
 */

/** \addtogroup EventViewer

    @{*/

#ifndef EVD_DRAWRAWCHANNELROI_H
#define EVD_DRAWRAWCHANNELROI_H

#include "Analysis/ana_base.h"
#include "RawBase.h"

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"

#include  "sbnobj/ICARUS/TPC/ChannelROI.h"

#include "TTree.h"
#include "TGraph.h"


namespace evd {
  /**
     \class DrawChannelROI
     User custom analysis class made by SHELL_USER_NAME
   */
  class DrawChannelROI : public galleryfmwk::ana_base, public RawBase{

  public:

    /// Default constructor
    DrawChannelROI(const geo::GeometryCore& geometry, const detinfo::DetectorPropertiesData& DetectorProperties);

    /// Default destructor
    virtual ~DrawChannelROI(){}

    /** IMPLEMENT in DrawChannelROI.cc!
        Initialization method to be called before the analysis event loop.
    */
    virtual bool initialize();

    /** IMPLEMENT in DrawChannelROI.cc!
        Analyze a data event-by-event
    */
    virtual bool analyze(const gallery::Event & event);

    /** IMPLEMENT in DrawChannelROI.cc!
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
