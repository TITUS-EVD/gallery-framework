/**
 * \file DrawMCTruth.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawMCTruth
 *
 * @author mdeltutt
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_DRAWMCTRUTH_H
#define EVD_DRAWMCTRUTH_H

#include "Analysis/ana_base.h"
#include "nusimdata/SimulationBase/MCTruth.h"
#include <iostream>

#include "RecoBase.h"
/**
   \class DrawMCTruth
   User defined class DrawTrack ... these comments are used to generate
   doxygen documentation!
 */

namespace evd {

class MCTruth {

  friend class DrawMCTruth;

public:
  const int                 &origin() { return _origin; }
  const int                 &nu_pdg() { return _nu_pdg; }
  const float               &nu_energy() { return _nu_energy; }
  const int                 &int_mode() { return _int_mode; }
  const std::vector<double> &vertex() { return _vertex; }
  const float               &nu_time() { return _nu_time; }

  const std::vector<int>    &finalstate_pdg() { return _finalstate_pdg; }
  const std::vector<float>  &finalstate_energy() { return _finalstate_energy; }

protected:
  int _origin;                      // mc origin type: 0=unknown, 1=beam neutrino, 2=cosmic,
                                    // 3=supernova neutrino, 4=single particle
  int _nu_pdg = -9999;              // pdg number of the neutrino (if neutrino origin)
  float _nu_energy = -9999;         // energy of the neutrino (if neutrino origin)
  std::vector<double> _vertex;      // Vertex of neutrino (if neutrino origin)
  float _nu_time = -9999;           // time of the neutrino (if neutrino origin)
  int _int_mode = -9999;            // interaction (if neutrino origin)

  std::vector<int> _finalstate_pdg;      // final state particle pdg (if neutrino origin)
  std::vector<float> _finalstate_energy; // final state particle energy (if neutrino origin)

};

class DrawMCTruth : public galleryfmwk::ana_base, public RecoBase<MCTruth> {

public:
  /// Default constructor
  DrawMCTruth(const geo::GeometryCore&               geometry,
              const detinfo::DetectorPropertiesData& detectorProperties,
              const detinfo::DetectorClocksData&     detectorClocks);

  /// Default destructor
  ~DrawMCTruth();

  virtual bool initialize();

  virtual bool analyze(const gallery::Event & event);

  virtual bool finalize();

};

} // evd

#endif
/** @} */ // end of doxygen group
