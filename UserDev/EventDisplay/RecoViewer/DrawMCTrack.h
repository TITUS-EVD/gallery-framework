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
#ifndef EVD_DRAWMCTRACK_H
#define EVD_DRAWMCTRACK_H

#include "Analysis/ana_base.h"
#include "lardataobj/MCBase/MCStep.h"
#include "lardataobj/MCBase/MCTrack.h"
#include "nusimdata/SimulationBase/MCTruth.h"
#include <iostream>

#include "DrawTrack.h"
#include "RecoBase.h"
/**
   \class DrawMCTrack
   User defined class DrawTrack ... these comments are used to generate
   doxygen documentation!
 */

namespace evd {

class MCTrack2D : public Track2D {

  friend class DrawMCTrack;

public:
  const int         &origin() { return _origin; }
  const int         &pdg() { return _pdg; }
  const float       &time() { return _time; }
  const float       &energy() { return _energy; }
  const std::string &process() { return _process; }
  const int         &tpc() { return _tpc; }

protected:
  int _origin;          // mc origin type: 0=unknown, 1=beam neutrino, 2=cosmic,
                        // 3=supernova neutrino, 4=single particle
  int _pdg;             // pdg number
  float _time = -9999;  // track time
  float _energy = -9999;// Track energy
  std::string _process; // Start G4 process
  int _tpc;             // TPC ID where the track starts
};

class DrawMCTrack : public galleryfmwk::ana_base, public RecoBase<MCTrack2D> {

public:
  /// Default constructor
  DrawMCTrack(const geo::GeometryCore&               geometry,
              const detinfo::DetectorPropertiesData& detectorProperties,
              const detinfo::DetectorClocksData&     detectorClocks);

  /// Default destructor
  ~DrawMCTrack();

  virtual bool initialize();

  virtual bool analyze(gallery::Event *event);

  virtual bool finalize();

private:
  MCTrack2D getMCTrack2D(sim::MCTrack track, unsigned int plane, unsigned int tpc = 0, unsigned int cryostat = 0);
};

} // evd

#endif
/** @} */ // end of doxygen group
