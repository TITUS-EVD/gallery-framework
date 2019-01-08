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
#ifndef GALLERY_FMWK_SUPERA_DUNE_RAWDIGIT_H
#define GALLERY_FMWK_SUPERA_DUNE_RAWDIGIT_H

#include "supera_module_base.h"

#include "larcv/core/DataFormat/Image2D.h"


/**
   \class ShowerRecoModuleBase
   User defined class ShowerRecoModuleBase ... these comments are used to generate
   doxygen documentation!
 */
namespace supera {

class DUNERawDigit : SuperaModuleBase {

public:

    /// Default constructor
    DUNERawDigit(){_name = "DUNERawDigit";_verbose=false;}

    /// Default destructor
    ~DUNERawDigit() {}

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
    void slice(gallery::Event * ev, larcv::IOManager * io);

    /**
     * @brief Function to initialize the algorithm (such as setting up tree)
     */
    void initialize();



protected:


    float _threshold = 5;


    std::string _name;
    std::vector<larcv::ImageMeta> plane_meta;

    bool _verbose;



};

} // showerreco

#endif
/** @} */ // end of doxygen group

