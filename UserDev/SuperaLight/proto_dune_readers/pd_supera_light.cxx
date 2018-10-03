#ifndef GALLERY_FMWK_PD_SUPERA_CXX
#define GALLERY_FMWK_PD_SUPERA_CXX

#include "pd_supera_light.h"


namespace supera {

bool pd_supera_light::initialize() {

  if (!_io) {
    std::cout << "Must set io manager before initializing!" << std::endl;
    throw std::exception();
  }

  // init all modules:
  for (size_t n = 0; n < _modules.size(); n++) {
    _modules[n]->initialize();
  }

  _io->initialize();
  return true;
}

bool pd_supera_light::analyze(gallery::Event* ev) {


  // Get the event ID information for this event:
  int run = ev->eventAuxiliary().run();
  int subrun = ev->eventAuxiliary().subRun();
  int event = ev->eventAuxiliary().event();

  // Run all modules:
  for (size_t n = 0; n < _modules.size(); n++) {
    _modules[n]->slice(ev, _io);
  }

  // Save the event
  _io->set_id(run, subrun, event);
  _io->save_entry();

  return true;
}

void pd_supera_light::add_supera_module(ProtoDuneModuleBase* module){
  _modules.push_back(module);
}

bool pd_supera_light::finalize() {

  _io->finalize();
  return true;
}
}
#endif
