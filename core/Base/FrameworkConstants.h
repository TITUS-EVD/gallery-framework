/**
 * \file FrameworkConstants.h
 *
 * \ingroup Base
 *
 * \brief defines basic constants used in this framework
 *
 * @author Kazu - Nevis 2013
 */

/** \addtogroup Base

    @{*/

#ifndef GALLERY_FMWK_FRAMEWORKCONSTANTS_H
#define GALLERY_FMWK_FRAMEWORKCONSTANTS_H
#include <string>
#include <limits>

/// Namespace of everything in this framework
namespace galleryfmwk {

namespace data {
const unsigned char  kINVALID_UCHAR  = std::numeric_limits<unsigned char>::max();
const char           kINVALID_CHAR   = std::numeric_limits<char>::max();
const unsigned short kINVALID_USHORT = std::numeric_limits<unsigned short>::max();
const short          kINVALID_SHORT  = std::numeric_limits<short>::max();
const unsigned int   kINVALID_UINT   = std::numeric_limits<unsigned int>::max();
const int            kINVALID_INT    = std::numeric_limits<int>::max();
const size_t         kINVALID_SIZE   = std::numeric_limits<size_t>::max();

const double kINVALID_DOUBLE = std::numeric_limits<double>::max();
const float  kINVALID_FLOAT  = std::numeric_limits<float>::max();
}

/// Defines constants for Message utility
namespace msg {

/// Defines message level
enum Level {
  kDEBUG = 0,    ///< Message level ... useful to debug a crash
  kINFO,         ///< Debug info but not the lowest level
  kNORMAL,       ///< Normal stdout
  kWARNING,      ///< notify a user in the standard operation mode for an important finding.
  kERROR,        ///< notify a user when something is clearly wrong
  kMSG_TYPE_MAX
};

const std::string ColorPrefix[kMSG_TYPE_MAX] =
{
  "\033[94m", ///< blue ... DEBUG
  "\033[92m", ///< green ... INFO
  "\033[95m", ///< magenta ... NORMAL
  "\033[93m", ///< yellow ... WARNING
  "\033[91m"  ///< red ... ERROR
};
///< Color coding of message

const std::string StringPrefix[kMSG_TYPE_MAX] =
{
  "     [DEBUG]  ", ///< DEBUG message prefix
  "      [INFO]  ", ///< INFO message prefix
  "    [NORMAL]  ", ///< NORMAL message prefix
  "   [WARNING]  ", ///< WARNING message prefix
  "     [ERROR]  "  ///< ERROR message prefix
};
///< Prefix of message
}

}
#endif
/** @} */ // end of doxygen group
