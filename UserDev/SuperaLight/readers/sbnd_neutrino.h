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
#ifndef GALLERY_FMWK_SUPERA_SBND_NEUTRINO_H
#define GALLERY_FMWK_SUPERA_SBND_NEUTRINO_H

#include "supera_module_base.h"

#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/Voxel3DMeta.h"


/**
   \class ShowerRecoModuleBase
   User defined class ShowerRecoModuleBase ... these comments are used to generate
   doxygen documentation!
 */
namespace supera {

class SBNDNeutrino : SuperaModuleBase {

public:

    /// Default constructor
    SBNDNeutrino(){_name = "SBNDNeutrino";_verbose=false;}

    /// Default destructor
    ~SBNDNeutrino() {}

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
    void neutrino_slice(gallery::Event* ev, larcv::IOManager* io);



    std::string _name;
    std::vector<larcv::ImageMeta> plane_meta;
    larcv::Voxel3DMeta voxel_meta;

    bool _verbose;



};

} // showerreco

#endif
/** @} */ // end of doxygen group

