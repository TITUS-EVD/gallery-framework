/**
 * \file DrawOpflash.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawOpflash
 *
 * @author Marco Del Tutto
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_DRAWOPFLASH_H
#define EVD_DRAWOPFLASH_H

#include "Analysis/ana_base.h"
#include "lardataobj/RecoBase/OpFlash.h"
#include "lardataobj/RecoBase/OpHit.h"
#include "canvas/Persistency/Common/FindMany.h"
#include <iostream>

#include "RecoBase.h"

/**
   \class DrawOpflash
   User defined class DrawOpflash ... these comments are used to generate
   doxygen documentation!
 */

namespace evd {

class Opflash2D {

friend class DrawOpflash;

public:

    float y(){return _y;}
    float y_width(){return _y_width;}
    float z(){return _z;}
    float z_width(){return _z_width;}
    float time(){return _time;}
    float time_width(){return _time_width;}
    float total_pe(){return _total_pe;}
    float plane(){return _plane;}
    std::vector<double> pe_per_opdet(){return _opdet_pe;}

private:
    float _y;
    float _y_width;
    float _z;
    float _z_width;
    float _time;
    float _time_width;
    float _total_pe;
    float _plane;
    std::vector<double> _opdet_pe;
};


class DrawOpflash : public galleryfmwk::ana_base, public RecoBase<Opflash2D> {

public:
  /// Default constructor
  DrawOpflash(const geo::GeometryCore&               geometry,
              const detinfo::DetectorPropertiesData& detectorProperties,
              const detinfo::DetectorClocksData&     detectorClocks);

  /// Default destructor
  ~DrawOpflash();

  virtual bool initialize();

  virtual bool analyze(const gallery::Event & event);

  virtual bool finalize();

private:

  int find_plane(int opch);

};

} // evd

#endif
/** @} */ // end of doxygen group
