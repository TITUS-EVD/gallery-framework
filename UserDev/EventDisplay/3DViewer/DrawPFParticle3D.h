/**
 * \file DrawPFParticle3D.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawPFParticle3D
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_DRAWPFPARTICLE3D_H
#define EVD_DRAWPFPARTICLE3D_H

#include <iostream>
#include "Analysis/ana_base.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Seed.h"

#include "canvas/Persistency/Common/FindMany.h"

#include "RecoBase3D.h"

/**
   \class DrawPFParticle3D
   User defined class DrawPFParticle3D ... these comments are used to generate
   doxygen documentation!
 */

// typedef std::vector< std::pair<float,float> > evd::Track2d;

namespace evd {


class PFPart3D {

friend class DrawPFParticle3D;

public:

    // cluster3D::cluster3D_params params(){return _params;}
    const std::vector<larutil::Point3D> points(){return _points;}

private:

    std::vector<larutil::Point3D> _points;

};


class DrawPFParticle3D : public galleryfmwk::ana_base, public RecoBase3D<PFPart3D> {

public:

    /// Default constructor
    DrawPFParticle3D();

    /// Default destructor
    ~DrawPFParticle3D();

    /** IMPLEMENT in DrawCluster.cc!
        Initialization method to be called before the analysis event loop.
    */
    virtual bool initialize();

    /** IMPLEMENT in DrawCluster.cc!
        Analyze a data event-by-event
    */
    virtual bool analyze(gallery::Event* storage);

    /** IMPLEMENT in DrawCluster.cc!
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();


private:

    // cluster3D::CRU3DHelper _cru3d_helper;
    // cluster3D::Default3DParamsAlg _params_alg;

};

} // evd

#endif
/** @} */ // end of doxygen group

