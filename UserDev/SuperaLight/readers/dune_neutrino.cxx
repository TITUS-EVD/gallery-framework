#ifndef GALLERY_FMWK_SUPERA_DUNE_NEUTRINO_CXX
#define GALLERY_FMWK_SUPERA_DUNE_NEUTRINO_CXX

#include "dune_neutrino.h"

#include "LArUtil/TimeService.h"

// Larsoft includes:
#include "nusimdata/SimulationBase/MCTruth.h"

// larcv includes:
#include "larcv/core/DataFormat/EventParticle.h"
#include "larcv/core/DataFormat/EventVoxel2D.h"
#include "larcv/core/DataFormat/EventVoxel3D.h"

#include "LArUtil/Geometry.h"

namespace supera {

void DUNENeutrino::initialize() {

}

void DUNENeutrino::slice(gallery::Event* ev, larcv::IOManager* io) {
  // Get the neutrino data:
  std::string neutrino_producer = "generator";
  art::InputTag neutrino_tag(neutrino_producer);

  gallery::Handle<std::vector<simb::MCTruth> > mctruth;
  if (!ev->getByLabel(neutrino_tag, mctruth)) {
    // cosmic_slice(ev, io);
    return;
  }
  else{
    neutrino_slice(ev, io);
  }
  return;

}


void DUNENeutrino::neutrino_slice(gallery::Event* ev, larcv::IOManager* io){


  std::string neutrino_producer = "generator";
  art::InputTag neutrino_tag(neutrino_producer);

  gallery::Handle<std::vector<simb::MCTruth> > mctruth;

  ev->getByLabel(neutrino_tag, mctruth);

  auto truth = mctruth->at(0);
  auto neutrino = mctruth->at(0).GetNeutrino().Nu();

  auto event_particle  =
      (larcv::EventParticle*) io->get_data("particle", "duneneutrino");

  // Start by extracting the particle information:
  larcv::Particle neut_info;
  neut_info.id(0);

  // Info from MCNeutrino:
  neut_info.nu_interaction_type(truth.GetNeutrino().InteractionType());
  neut_info.nu_current_type(truth.GetNeutrino().CCNC());
  neut_info.track_id(neutrino.TrackId());
  neut_info.pdg_code(neutrino.PdgCode());
  neut_info.creation_process(neutrino.Process());
  neut_info.position(neutrino.Vx(),
                     neutrino.Vy(),
                     neutrino.Vz(),
                     neutrino.T());
  neut_info.momentum(neutrino.Px(),
                     neutrino.Py(),
                     neutrino.Pz());
  neut_info.energy_init(neutrino.E());

  event_particle->emplace_back(std::move(neut_info));

  for (size_t i = 0; i < truth.NParticles(); i ++){
    larcv::Particle particle;
    auto & larsoft_particle = truth.GetParticle(i);
    if (larsoft_particle.StatusCode() != 1){
      continue;
    }

    particle.id(i+1);
    particle.track_id(larsoft_particle.TrackId());
    particle.pdg_code(larsoft_particle.PdgCode());
    particle.parent_track_id(larsoft_particle.Mother());
    particle.creation_process(larsoft_particle.Process());

    particle.position(larsoft_particle.Vx(),
                       larsoft_particle.Vy(),
                       larsoft_particle.Vz(),
                       larsoft_particle.T());

    particle.momentum(larsoft_particle.Px(),
                       larsoft_particle.Py(),
                       larsoft_particle.Pz());

    particle.energy_init(larsoft_particle.E());
    event_particle->emplace_back(std::move(particle));
  }


  return;
}

// void DUNENeutrino::cosmic_slice(gallery::Event* ev, larcv::IOManager* io){

// }

}

#endif