/**
 * \file DrawHit.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawHit
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_DRAWHIT_H
#define EVD_DRAWHIT_H

#include <iostream>
#include "Analysis/ana_base.h"
#include "RecoBase.h"

#include "lardataobj/RecoBase/Hit.h"


/**
   \class DrawHit
   User defined class DrawHit ... these comments are used to generate
   doxygen documentation!
 */

namespace evd {

class Hit2D {

public:
    Hit2D() {}
    Hit2D(float w, float t, float c, float r, float st, float pt, float et, float pa, int pl, int tpc, int cryo) :
        _wire(w),
        _time(t),
        _charge(c),
        _rms(r),
        _start_time(st),
        _peak_time(pt),
        _end_time(et),
        _peak_amplitude(pa),
        _plane(pl),
        _tpc(tpc),
        _cryo(cryo)
    {}
    ~Hit2D() {}
    float _wire;
    float _time;
    float _charge;
    float _rms;
    float _start_time;
    float _peak_time;
    float _end_time;
    float _peak_amplitude;
    int   _plane;
    int   _tpc;
    int   _cryo;

    float wire()   {return _wire;}
    float time()   {return _time;}
    float charge() {return _charge;}
    float rms()    {return _rms;}
    float start_time()  {return _start_time;}
    float peak_time()   {return _peak_time;}
    float end_time()    {return _end_time;}
    float peak_amplitude()  {return _peak_amplitude;}
    int   plane()  {return _plane;}
    int   tpc()  {return _tpc;}
    int   cryo()  {return _cryo;}
};

class DrawHit : public galleryfmwk::ana_base, public RecoBase<Hit2D> {

public:

    /// Default constructor
    DrawHit(const geo::GeometryCore& geometry, 
            const detinfo::DetectorProperties& detectorProperties,
            const detinfo::DetectorClocks& detectorClocks);

    /// Default destructor
    // ~DrawHit(){}

    /**
    Initialization method to be called before the analysis event loop.
    */
    virtual bool initialize();

    /** IMPLEMENT in DrawCluster.cc!
        Analyze a data event-by-event
    */
    virtual bool analyze(gallery::Event* event);

    /** IMPLEMENT in DrawCluster.cc!
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();

    float maxCharge(size_t plane);

    std::vector<Hit2D> getHitsOnWirePlane(size_t wire, size_t plane);

private:

    std::vector <float> _maxCharge;

};

} // evd

#endif
/** @} */ // end of doxygen group

