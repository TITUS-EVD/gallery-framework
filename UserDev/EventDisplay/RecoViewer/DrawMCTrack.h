/**
 * \file DrawMCTrack.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawMCTrack
 *
 * @author cadams, mdeltutt
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_LARLITE_DRAWMCTRACK_H
#define EVD_LARLITE_DRAWMCTRACK_H

#include <iostream>
#include "Analysis/ana_base.h"
#include "lardataobj/MCBase/MCTrack.h"
#include "lardataobj/MCBase/MCStep.h"


#include "RecoBase.h"
/**
   \class DrawMCTrack
   User defined class DrawTrack ... these comments are used to generate
   doxygen documentation!
 */


namespace evd {



class MCTrack2D {
public:
  std::vector<std::pair<float, float> > _track;
  const std::vector<std::pair<float, float> > & track() {return _track;}
  const std::vector<std::pair<float, float> > & direction() {return _track;}
  int _origin; //mc origin type: 0=unknown, 1=beam neutrino, 2=cosmic, 3=supernova neutrino, 4=single particle
  const int & origin() {return _origin;}
};

MCTrack2D getMCTrack2D(sim::MCTrack track, unsigned int plane);

class DrawMCTrack : public galleryfmwk::ana_base, public RecoBase<MCTrack2D> {

public:

  /// Default constructor
  DrawMCTrack();

  /// Default destructor
  ~DrawMCTrack();

  virtual bool initialize();

  virtual bool analyze(gallery::Event * event);

  virtual bool finalize();



private:


};

} // evd

#endif
/** @} */ // end of doxygen group

