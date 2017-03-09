/**
 * \file ana_processor.h
 *
 * \ingroup Analysis
 *
 * \brief A batch processor class for arbitrary number of analysis modules
 *
 * @author Kazu - Nevis 2013
 */

/** \addtogroup Analysis

    @{*/
#ifndef GALLERY_FMWK_ANA_PROCESSOR_H
#define GALLERY_FMWK_ANA_PROCESSOR_H

#include <vector>
#include "ana_base.h"
#include "TFile.h"

namespace galleryfmwk {
/**
   \class ana_processor
   A class to be loaded with arbitrary number of ana_base inherited analysis
   modules. This class process data by applying loaded analysis modules in
   consecutive order.
*/
class ana_processor {

public:

  /// Process flag enum
  enum ProcessFlag_t {
    kINIT,       ///< Process is @ the initialization stage
    kREADY,      ///< Process is ready to start data processing
    kPROCESSING, ///< Process is processing data
    kFINISHED    ///< Process has finished processing allevents.
  };

  /// Default constructor
  ana_processor();

  /// Default destructor
  virtual ~ana_processor() {if (_event) delete _event;}

  /// Override a method to set verbosity
  /**
     It changes the verbosity level of not just this class but also owned
     other class instances.
  */
  virtual void set_verbosity(msg::Level level);

  /// Setter for an input data file name
  void add_input_file(std::string name);

  /// Setter for an analysis output root file name
  void set_ana_output_file(std::string name) {_ofile_name = name;}

  /// Getter of running analysis status
  bool get_ana_status(ana_base* ptr) const;

  /// A method to run a batch process
  bool run(unsigned int nevents = 0);

  /// A method to process just one event.
  bool process_event();

  /// A method to append analysis class instance. Returns index number.
  size_t add_process(ana_base* ana, bool filter = false)
  {
    _ana_index.insert(std::make_pair(ana, _analyzers.size()));
    _analyzers.push_back(ana);
    _filter_marker_v.push_back(filter);
    return (*_ana_index.find(ana)).second;
  }

  bool remove_process(size_t, bool finalize = true);

  /// A method to inquire attached analysis class instance.
  ana_base* get_process(size_t loc) {return (_analyzers.size() > loc) ? _analyzers[loc] : 0;}

  /// A method to inquire the process status
  ProcessFlag_t get_process_status() {return _process;}

  /// Setter to enable filtering mode
  void enable_filter(bool doit = true) { _filter_enable = doit; }

  /// A method to reset members
  void reset();


private:

  /// A method to initialize and prepare for running analysis
  bool initialize();

  /// A method to finalize data processing
  bool finalize();

  std::vector<ana_base*>   _analyzers;  ///< A vector of analysis modules
  std::vector<bool>        _ana_status; ///< A vector of analysis modules' status
  std::vector<bool>   _filter_marker_v; ///< A vector to mark specific analysis unit as a filter
  std::map<ana_base*, size_t> _ana_index; ///< A map of analysis module status

  std::vector<std::string> _input_files;

  ProcessFlag_t _process;    ///< Processing status flag
  unsigned int _nevents;     ///< Number of events being processed
  unsigned int _index;       ///< Index of currently processing event
  std::string _ofile_name;   ///< Output file name
  TFile*   _fout;            ///< Output file pointer

  bool _filter_enable;
  bool _ana_unit_status;

  std::string _name;             ///< class name holder

  msg::Level _verbosity_level;   ///< holder for specified verbosity level

  size_t _last_run_id;
  size_t _last_subrun_id;


  gallery::Event * _event;

};
}
#endif

/** @} */ // end of doxygen group
