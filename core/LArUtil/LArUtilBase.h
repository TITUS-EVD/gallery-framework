/**
 * \file LArUtilBase.h
 *
 * \ingroup LArUtil
 * 
 * \brief Class def header for a class LArUtilBase
 *
 * @author kazuhiro
 */

/** \addtogroup LArUtil

    @{*/
#ifndef GALLERY_FMWK_LARUTILBASE_H
#define GALLERY_FMWK_LARUTILBASE_H

#include <map>
#include <sstream>
#include <TChain.h>
#include "Base/FrameworkConstants.h"
#include "LArUtilConstants.h"
#include "LArUtilConfig.h"
#include "LArUtilException.h"

#include "Base/messenger.h"

#include <cstdio>
#include "Base/GeoTypes.h"

namespace larutil {
  /**
     \class LArUtilBase
     Base class for utility classes. Currently we take a simple approach of reading-in parameter
     values from TTree. This base class implements a way to read TTree and also store output TTree.
     DB access may be implemented here in future.
  */
  class LArUtilBase {
    
  public:
    
    /// Default constructor
    LArUtilBase();
    
    /// Default destructor
    virtual ~LArUtilBase(){};

    /// Method to set data file name
    void SetFileName(const std::string filename)
    { _file_name = filename; }

    /// Method to set data TTree name
    void SetTreeName(const std::string treename)
    { _tree_name = treename; }

    /// Method to load data TTree
    virtual bool LoadData(bool force_reload=false);

  protected:

    virtual bool ReadTree()=0;

    virtual void ClearData()=0;

    std::string _name;

    std::string _file_name;
    std::string _tree_name;
    bool _loaded;
  };
}

#endif
/** @} */ // end of doxygen group 

