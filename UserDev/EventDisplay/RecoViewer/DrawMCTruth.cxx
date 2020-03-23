#ifndef EVD_DRAWMCTRUTH_CXX
#define EVD_DRAWMCTRUTH_CXX

#include "DrawMCTruth.h"

namespace evd {

DrawMCTruth::DrawMCTruth(const geo::GeometryCore& geometry, 
                         const detinfo::DetectorProperties& detectorProperties,
                         const detinfo::DetectorClocks& detectorClocks) :
    RecoBase(geometry, detectorProperties, detectorClocks) 
{
  _name = "DrawMCTruth";
  _fout = 0;
}

bool DrawMCTruth::initialize() {

  return true;
}

bool DrawMCTruth::analyze(gallery::Event *ev) {



  art::InputTag truth_tag(_producer);
  auto const &truthHandle =
      ev->getValidHandle<std::vector<simb::MCTruth>>(truth_tag);

  for (auto &truth : *truthHandle) {

    MCTruth mct;

    mct._origin = truth.Origin();

    // Continue if not neutrino origin
    if (truth.Origin() != 1) continue;

    mct._nu_pdg = truth.GetNeutrino().Nu().PdgCode();
    mct._nu_energy = truth.GetNeutrino().Nu().E();
    std::vector<double> vtx = {truth.GetNeutrino().Nu().Vx(), 
                               truth.GetNeutrino().Nu().Vy(), 
                               truth.GetNeutrino().Nu().Vz()};
    mct._vertex = vtx;
    mct._nu_pdg = truth.GetNeutrino().Nu().PdgCode();
    mct._int_mode = truth.GetNeutrino().Mode();


    std::vector<int> pdgs;
    std::vector<float> energies;

    for (int i = 0; i < truth.NParticles(); i++) {
      auto mcp = truth.GetParticle(i);
      if (mcp.StatusCode() != 1) continue;
      pdgs.push_back(mcp.PdgCode());
      energies.push_back(mcp.E());
    }
    mct._finalstate_pdg = pdgs;
    mct._finalstate_energy = energies;


    _data.push_back(mct);

  }

  

  return true;
}

bool DrawMCTruth::finalize() {

  return true;
}

DrawMCTruth::~DrawMCTruth() {}

} 

#endif
