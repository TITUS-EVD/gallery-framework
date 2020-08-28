/**
 * \file DrawCluster.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawCluster
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/

#ifndef EVD_DRAWCLUSTER_H
#define EVD_DRAWCLUSTER_H

#include "Analysis/ana_base.h"
#include "lardataobj/RecoBase/Cluster.h"
#include "lardataobj/RecoBase/Hit.h"
#include "canvas/Persistency/Common/FindMany.h"

#include "DrawHit.h"



namespace evd {
/**
   \class DrawCluster
   User custom analysis class made by SHELL_USER_NAME
 */

class Cluster2D : public std::vector<evd::Hit2D> {

public:
  Cluster2D() {_is_good = false;}
  // ::cluster::cluster_params _params;
  // ::cluster::cluster_params params() {return _params;}
  bool _is_good;
  bool is_good() {return _is_good;}
};

class DrawCluster : public galleryfmwk::ana_base, public RecoBase<Cluster2D> {

public:

  /// Default constructor
  DrawCluster(const geo::GeometryCore& geometry,
              const detinfo::DetectorProperties& detectorProperties,
              const detinfo::DetectorClocksData& detectorClocks);

  /// Default destructor
  virtual ~DrawCluster();

  /** IMPLEMENT in DrawCluster.cc!
      Initialization method to be called before the analysis event loop.
  */
  virtual bool initialize();

  /** IMPLEMENT in DrawCluster.cc!
      Analyze a data event-by-event
  */
  virtual bool analyze(gallery::Event * event);

  /** IMPLEMENT in DrawCluster.cc!
      Finalize method to be called after all events processed.
  */
  virtual bool finalize();

protected:

  size_t _total_plane_number;

  // ::cluster::CRUHelper _cru_helper;

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
