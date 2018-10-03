/**
 * \file ShowerRecoModuleBase.h
 *
 * \ingroup SuperaLight
 *
 * \brief Class def header for a class ShowerRecoModuleBase
 *
 * @author cadams
 */

/** \addtogroup ModularAlgo

    @{*/
#ifndef GALLERY_FMWK_SUPERA_MODULE_BASE_PD_H
#define GALLERY_FMWK_SUPERA_MODULE_BASE_PD_H

#include <iostream>

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"
#include "canvas/Persistency/Common/FindMany.h"

#include "Analysis/ana_base.h"

#include "larcv/core/DataFormat/IOManager.h"

/**
   \class ShowerRecoModuleBase
   User defined class ShowerRecoModuleBase ... these comments are used to generate
   doxygen documentation!
 */
namespace supera {

class ProtoDuneModuleBase {

public:

    /// Default constructor
    ProtoDuneModuleBase();

    /// Default destructor
    virtual ~ProtoDuneModuleBase() {}

    /**
     * @brief get the name of this module, used in helping organize order of modules and insertion/removal
     * @return name
     */
    std::string name() {return _name;}

    /**
     * @brief Virtual function that is overridden in child class,
     * does the slicing work.
     *
     */
    virtual void slice(gallery::Event * ev, larcv::IOManager * io) = 0;

    /**
     * @brief Verbosity setter function for each Modular Algo
     */
    void setVerbosity(bool on) { _verbose = on; }

    /**
     * @brief Function to initialize the algorithm (such as setting up tree)
     */
    virtual void initialize() {};



protected:

    int projection_id(int channel);
    int column(int channel);

    int n_ticks = 6000;
    int compression = 1;

    int _max_tick;

    std::string _name;


    bool _verbose;



};

} // showerreco

#endif
/** @} */ // end of doxygen group

