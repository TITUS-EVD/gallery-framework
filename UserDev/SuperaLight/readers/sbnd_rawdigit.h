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
#ifndef GALLERY_FMWK_SUPERA_SBND_RAWDIGIT_H
#define GALLERY_FMWK_SUPERA_SBND_RAWDIGIT_H

#include "supera_module_base.h"

#include "larcv3/core/dataformat/Tensor.h"


/**
   \class ShowerRecoModuleBase
   User defined class ShowerRecoModuleBase ... these comments are used to generate
   doxygen documentation!
 */
namespace supera {

class SBNDRawDigit : public SuperaModuleBase {

public:

    /// Default constructor
    SBNDRawDigit(){_name = "SBNDRawDigit";_verbose=false;}

    /// Default destructor
    ~SBNDRawDigit() {}

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


    float _threshold = 10;


    std::string _name;

    bool _verbose;



};

} // showerreco

#endif
/** @} */ // end of doxygen group
