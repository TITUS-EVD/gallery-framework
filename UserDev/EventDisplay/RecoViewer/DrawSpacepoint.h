/**
 * \file DrawSpacepoint.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawSpacepoint
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_DRAWSPACEPOINT_H
#define EVD_DRAWSPACEPOINT_H

#include <iostream>
#include "Analysis/ana_base.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "canvas/Persistency/Common/FindMany.h"
#include "lardataobj/RecoBase/Hit.h"

#include "RecoBase.h"
#include "LArUtil/PxUtils.h"
/**
   \class DrawSpacepoint
   User defined class DrawSpacepoint ... these comments are used to generate
   doxygen documentation!
 */

// typedef std::vector< std::pair<float,float> > evd::Track2d;

namespace evd {
  class HitFromSpacePoint {
    public:
        HitFromSpacePoint() {}
        HitFromSpacePoint(int SPID, float w, float t, int p, int __tpc, int __cryo) :
        _SpacePointID(SPID),
        _wire(w),
        _time(t),
        _plane(p),
        _tpc( __tpc ),
        _cryo(__cryo)
        {}
        ~HitFromSpacePoint() {}
        float _wire;
        float _time;
        int   _SpacePointID;
        int   _plane;
        int   _tpc;
        int   _cryo;

        float wire()   {return _wire;}
        float time()   {return _time;}
        int   plane()  {return _plane;}
        int   tpc()  {return _tpc;}
        int   cryo()  {return _cryo;}
        int   SpacePointID() {return _SpacePointID;}

        //Conversion functions
        //operator larutil::PxPoint() const { return larutil::PxPoint(_plane, _wire, _time,_SpacePointID, _tpc, _cryo); }
  };


  class DrawSpacepoint : public galleryfmwk::ana_base, public RecoBase<HitFromSpacePoint> {

public:

    /// Default constructor
    DrawSpacepoint(const geo::GeometryCore&               geometry,
                   const detinfo::DetectorPropertiesData& detectorProperties,
                   const detinfo::DetectorClocksData&     detectorClocks);

    /// Default destructor
    ~DrawSpacepoint();

    /** IMPLEMENT in DrawCluster.cc!
        Initialization method to be called before the analysis event loop.
    */
    virtual bool initialize();

    /** IMPLEMENT in DrawCluster.cc!
        Analyze a data event-by-event
    */
    virtual bool analyze(const gallery::Event &event);

    /** IMPLEMENT in DrawCluster.cc!
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();


private:


};

} // evd

#endif
/** @} */ // end of doxygen group

