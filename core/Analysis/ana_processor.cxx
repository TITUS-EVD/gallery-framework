#ifndef GALLERY_FMWK_ANA_PROCESSOR_CXX
#define GALLERY_FMWK_ANA_PROCESSOR_CXX

#include "ana_processor.h"

namespace galleryfmwk {

ana_processor::ana_processor() {
  _ofile_name = "";
  _event = nullptr;
  //_storage=storage_manager::get();=
  _fout = 0;
  reset();
  _filter_enable = false;
  _ana_unit_status = true;
}

void ana_processor::set_verbosity(msg::Level level) {

  _verbosity_level = level;

  for (std::vector<ana_base*>::iterator iter(_analyzers.begin());
       iter != _analyzers.end();
       ++iter)

    (*iter)->set_verbosity(level);

}

void ana_processor::reset() {

  // if (_verbosity[msg::kDEBUG])
  //   Message::send(msg::kDEBUG, __PRETTY_FUNCTION__, "called...");

  if (_fout) {
    _fout->Close();
    _fout = 0;
  }

  if (_event && _event -> isValid()) {
    _event->toBegin();
  }

  _last_run_id = -1;
  _last_subrun_id = -1;

  _analyzers.clear();
  _ana_status.clear();
  _ana_index.clear();
  _nevents = 0;
  _index = 0;

  _process = kINIT;
}

void ana_processor::add_input_file(std::string name) {

  _input_files.push_back(name);
}

bool ana_processor::initialize() {

  set_verbosity(_verbosity_level);

  if (_process != kINIT) {
    Message::send(msg::kERROR, __FUNCTION__,
                  "Logic error: the function should not be called.");
    return false;
  }

  //_storage->set_io_mode(storage_manager::READ);
  _event = new gallery::Event(_input_files);
  if (!_event->isValid()) {

    Message::send(msg::kERROR, __FUNCTION__, "File I/O failure...");

    return false;

  }

  if (_ofile_name.size() == 0)

    Message::send(msg::kWARNING, __FUNCTION__,
                  "Analysis output file will not be created for this time...");

  else

    _fout = TFile::Open(_ofile_name.c_str(), "RECREATE");


  bool status = true;

  _ana_status.resize(_analyzers.size(), false);

  for (size_t i = 0; i < _analyzers.size(); ++i) {

    _analyzers[i]->set_output_file(_fout);

    _ana_status[i] = _analyzers[i]->initialize();

    if (!_ana_status[i]) {

      Message::send(msg::kERROR, __PRETTY_FUNCTION__,
                    Form("Failed to initialize: %s", _analyzers[i]->name().c_str()));

      status = false;
    }

  }
  _ana_unit_status = true;
  _process = kREADY;
  _index = 0;
  _nevents = 0;
  if (_verbosity_level == msg::kDEBUG )
    Message::send(msg::kDEBUG, __PRETTY_FUNCTION__, "ends...");
  return status;
}

bool ana_processor::process_event() {


  if (_process == kINIT) {

    if (!initialize()) {
      Message::send(msg::kERROR, __FUNCTION__, "Aborting.");
      return false;
    }
  }


  _ana_unit_status = true;

  if (_event->eventAuxiliary().run() != _last_run_id) {

    for (size_t i = 0; i < _analyzers.size(); ++i) {
      auto& au = _analyzers[i];
      if (_process != kPROCESSING) {
        au->begin_run(_event);
        au->begin_subrun(_event);
      } else {
        au->begin_run(_event);
        au->begin_subrun(_event);
      }
    }
    _last_run_id = _event->eventAuxiliary().run();
  } else if (_event->eventAuxiliary().subRun() != _last_subrun_id) {

    for (size_t i = 0; i < _analyzers.size(); ++i) {
      auto& au = _analyzers[i];
      if (_process != kPROCESSING)
        au->begin_subrun(_event);
      else
        au->begin_subrun(_event);
    }
    _last_subrun_id = _event->eventAuxiliary().subRun();

  }

  for (size_t i = 0; i < _analyzers.size(); ++i) {

    _ana_status[i] = false;

    _ana_status[i] = _analyzers[i]->analyze(_event);

    _ana_unit_status = _ana_unit_status && _ana_status[i];

    if (!_ana_unit_status) break;


  }

  _index++;
  _nevents++;

  if (_process != kPROCESSING)
    _process = kPROCESSING;
  return true;

}

bool ana_processor::run(unsigned int nevents, unsigned int nskip) {



  int nfiles = _input_files.size();

  if (_verbosity_level == msg::kDEBUG)
    Message::send(msg::kDEBUG, __PRETTY_FUNCTION__, "called...");

  bool status = true;

  if (_process == kINIT) status = initialize();

  if (!status) {

    Message::send(msg::kERROR, __PRETTY_FUNCTION__, "Aborting.");

    return false;
  }


  int ten_percent_ctr = 0;

  if (!nevents){
    if (nfiles == 1){
      nevents = _event -> numberOfEventsInFile();
    }
    else{
      nevents = -1;
    }
  }


  if (nskip){
      if (nfiles == 1 && nskip < _event -> numberOfEventsInFile()){
          _event -> goToEntry(nskip);
      }
      else{
          nskip = 0;
      }
  }

  char _buf[200];
  sprintf(_buf, "Processing %d events from entry %d...", nevents, nskip);
  Message::send(msg::kNORMAL, __FUNCTION__, _buf);

  while (status && _event->fileEntry() < nfiles) {

    status = process_event();

    if ( nevents >= 10 && (_nevents >= ten_percent_ctr * nevents / 10.) ) {

      if (ten_percent_ctr) {
        sprintf(_buf, " ... %3d%% done ...", ten_percent_ctr * 10);
        Message::send(msg::kNORMAL, __FUNCTION__, _buf);
      }
      ten_percent_ctr++;
    }

    if (nevents && nevents == _nevents) {
      Message::send(msg::kNORMAL, __FUNCTION__, Form("Processed %d/%d events! Aborting...", _nevents, nevents));
      break;
    }

    if (_event -> atEnd()) break;

    _event->next();


    if (_process != kPROCESSING) break;

  }

  if (_process != kFINISHED)
    status = finalize();

  return status;

}

bool ana_processor::finalize() {

  if (_verbosity_level == msg::kDEBUG)
    Message::send(msg::kDEBUG, __PRETTY_FUNCTION__, "called...");

  if (_process != kPROCESSING && _process != kREADY) {
    Message::send(msg::kERROR, __FUNCTION__,
                  "Logic error: the function should not be called.");
    return false;
  }

  bool status = true;

  for (size_t i = 0; i < _analyzers.size(); ++i) {

    _ana_status[i] = _analyzers[i]->finalize() && _ana_status[i];

    status = status && _ana_status[i];
  }

  _process = kFINISHED;
  reset();
  return status;
}

bool ana_processor::get_ana_status(ana_base* ptr) const {

  std::map<ana_base*, size_t>::const_iterator iter(_ana_index.find(ptr));
  if (iter == _ana_index.end()) return false;

  else return _ana_status[(*iter).second];

}
}
#endif
