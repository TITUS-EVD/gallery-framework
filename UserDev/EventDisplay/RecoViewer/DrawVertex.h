/**
 * \file DrawVertex.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawVertex
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/

#ifndef EVD_DRAWVERTEX_H
#define EVD_DRAWVERTEX_H

#include "Analysis/ana_base.h"
#include "RecoBase.h"
#include "LArUtil/Geometry.h"

#include "lardataobj/RecoBase/Vertex.h"


namespace evd {

  typedef larutil::Point2D Vertex2D;

/**
   \class DrawVertex
   User custom analysis class made by SHELL_USER_NAME
 */
class DrawVertex : public galleryfmwk::ana_base, public RecoBase<Vertex2D> {

public:

  /// Default constructor
  DrawVertex(const geo::GeometryCore&               geometry,
             const detinfo::DetectorPropertiesData& detectorProperties,
             const detinfo::DetectorClocksData&     detectorClocks);

  /// Default destructor
  virtual ~DrawVertex() {};

  virtual bool initialize();

  virtual bool analyze(gallery::Event * event);

  virtual bool finalize();



protected:

private:

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
