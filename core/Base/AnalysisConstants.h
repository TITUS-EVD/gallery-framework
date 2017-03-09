#ifndef GALLERY_FMWK_ANALYSISCONSTANTS_H
#define GALLERY_FMWK_ANALYSISCONSTANTS_H

namespace galleryfmwk{

  namespace anab{

    typedef enum cosmic_tag_id{
      kUnknown=-1,
      kNotTagged=0,
      kGeometry_YY=1,
      kGeometry_YZ,
      kGeometry_ZZ,
      kGeometry_XX,
      kGeometry_XY,
      kGeometry_XZ,
      kGeometry_Y=21,
      kGeometry_Z,
      kGeometry_X,
      kOutsideDrift_Partial=100,
      kOutsideDrift_Complete,
      kFlash_BeamIncompatible=200,
      kFlash_Match=300
    } CosmicTagID_t;
  }
}
#endif
