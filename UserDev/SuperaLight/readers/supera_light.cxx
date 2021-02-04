#ifndef GALLERY_FMWK_SUPERA_CXX
#define GALLERY_FMWK_SUPERA_CXX

#include "supera_light.h"

#include "nusimdata/SimulationBase/MCTruth.h"

#include "sbnd_rawdigit.h"

namespace supera {

bool supera_light::initialize() {


  // // Hard code modules:
  // _modules.push_back(new SBNDRawDigit() );
  //
  // std::cout << "Just added " << _modules[0] -> name() << std::endl;

  // init all modules:
  // for (size_t n = 0; n < _modules.size(); n++) {
  //   _modules[n]->initialize();
  // }

  raw_digit.initialize();
  wire.initialize();
  cluster.initialize();
  
  return true;
}

void supera_light::set_output_file(std::string outfile){
    _io.set_out_file(outfile);
    _io.initialize();
}

bool supera_light::analyze(gallery::Event* ev) {


  // Get the event ID information for this event:
  int run = ev->eventAuxiliary().run();
  int subrun = ev->eventAuxiliary().subRun();
  int event = ev->eventAuxiliary().event();
  //
  // // Run all modules:
  // std::cout << "Running over " << _modules.size() << " modules" << std::endl;
  // for (size_t n = 0; n < _modules.size(); n++) {
  //   std::cout << "Slice with " << _modules[n]->name() << std::endl;
  //   _modules[n]->slice(ev, _io);
  // }
  raw_digit.slice(ev, _io);
  wire.slice(ev, _io);
  cluster.slice(ev, _io);

  // Save the event
  _io.set_id(run, subrun, event);
  _io.save_entry();

  return true;
}
//
// void supera_light::add_supera_module(SuperaModuleBase* module){
//   std::cout << "Adding module with name " << module -> name() << std::endl;
//   _modules.push_back(module);
// }

bool supera_light::finalize() {

  _io.finalize();
  return true;
}
}
#endif
