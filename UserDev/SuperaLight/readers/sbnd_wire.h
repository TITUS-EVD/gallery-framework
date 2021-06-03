/**
 * \file ShowerRecoModuleBase.h
 *
 * \ingroup SuperaLight
 *
 * \brief Class def header for a class ShowerRecoModuleBase
 *
 * @author cadams
 **/

#ifndef GALLERY_FMWK_SUPERA_SBND_WIRE_H
#define GALLERY_FMWK_SUPERA_SBND_WIRE_H

#include "supera_module_base.h"

#include "larcv3/core/dataformat/Tensor.h"


/**
   \class ShowerRecoModuleBase
   User defined class ShowerRecoModuleBase ... these comments are used to generate
   doxygen documentation!
**/
namespace supera {

class SBNDWire : SuperaModuleBase {

public:

    /// Default constructor
    SBNDWire(){_name = "SBNDWire";_verbose=false;}

    /// Default destructor
    ~SBNDWire() {}

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
    void slice(gallery::Event * ev, larcv3::IOManager & io);

    /**
     * @brief Function to initialize the algorithm (such as setting up tree)
     */
    void initialize();



protected:


    std::string _name;

    float _threshold = 10.;

    bool _verbose;



};

} // showerreco

#endif
/** @} */ // end of doxygen group
