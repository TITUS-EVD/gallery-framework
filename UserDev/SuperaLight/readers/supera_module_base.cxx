#ifndef GALLERY_FMWK_SUPERA_MODULE_BASE_CXX
#define GALLERY_FMWK_SUPERA_MODULE_BASE_CXX

#include "supera_module_base.h"

#include "LArUtil/Geometry.h"

namespace supera {
SuperaModuleBase::SuperaModuleBase() {}

int SuperaModuleBase::projection_id(int channel) {
  // Pretty hacky code here ...

  // In SBND, channels go 0 to 1985 (plane 0), 1986 to 3971, 3972 to 5637
  // Then repeat on the other side with offset of 5638, for a total
  // of 11276 channels

  if (channel < 1986)
    return 0;
  else if (channel < 3972)
    return 1;
  else if (channel < 5638)
    return 2;
  else if (channel < 7624)
    return 1;
  else if (channel < 9610)
    return 0;
  else
    return 2;
}

int SuperaModuleBase::column(int channel) {
  // Pretty hacky code here ...

  // In SBND, channels go 0 to 1985 (plane 0), 1986 to 3971, 3972 to 5637
  // Then repeat on the other side with offset of 5638, for a total
  // of 11276 channels

  if (channel < 1986)
    return channel;
  else if (channel < 3972)
    return channel - 1986;
  else if (channel < 5638)
    return channel - 3972;
  else if (channel < 7624) {
    return (channel - 5638);
  } else if (channel < 9610) {
    return (channel - 7624);
  } else {
    return (channel - 9610);
  }
}

int SuperaModuleBase::row(int tick, int channel) {
  if (channel > 5638) {
    return _max_tick - tick - 1;
  } else {
    return tick;
  }
}

}
#endif