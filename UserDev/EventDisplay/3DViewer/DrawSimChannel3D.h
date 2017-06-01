/**
 * \file DrawSimChannel3D.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawSimChannel3D
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_DRAWSIMCH_H
#define EVD_DRAWSIMCH_H

#include <iostream>
#include "Analysis/ana_base.h"
#include "lardataobj/Simulation/SimChannel.h"


#include "RecoBase3D.h"
/**
   \class DrawSimChannel3D
   User defined class DrawSimChannel3D ... these comments are used to generate
   doxygen documentation!
 */

// typedef std::vector< std::pair<float,float> > evd::Track2d;

namespace evd {

typedef larutil::Point3D SimChannel3D;

class DrawSimChannel3D : public galleryfmwk::ana_base, public RecoBase3D<SimChannel3D> {

public:

    /// Default constructor
    DrawSimChannel3D();

    /// Default destructor
    ~DrawSimChannel3D();

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


private:


};

} // evd

#endif
/** @} */ // end of doxygen group

