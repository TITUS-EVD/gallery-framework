#ifndef GALLERY_FMWK_SUPERA_MODULE_BASE_CXX
#define GALLERY_FMWK_SUPERA_MODULE_BASE_CXX

#include "supera_module_base.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"

namespace supera {
SuperaModuleBase::SuperaModuleBase() {
  
  //  In this function, the image meta is created for each plane and for 3D.
  // Dune has many TPCs and read out planes, but this assumes disambiguation
  // to map channel to a plane.  The tick and channel combination 
  // determine the row and column in a 2D projection.


  _voxel_meta = larcv::Voxel3DMeta();
  
  float voxel_3d_size = 1.0;
  _voxel_meta.set(-1000, -1000, 0, 1000, 1000, 2000, 2000/voxel_3d_size, 2000/voxel_3d_size, 2000/voxel_3d_size);

  _image_meta_2d.clear();

  // parameters for ImageMeta are (xmin, ymin, xmax, ymax, nx, ny, units)
  // We'll encode tick in y and wire in x.  Units will be centimeters
  // y (drift direction) goes from -200 to 200 for n_ticks * 2 + spacing
  // x (wire direction) goes from 0
  _max_tick = 4*n_ticks;

  // int _readout_length = 4492;
  // int _n_channels = 30720;

  _image_meta_2d.push_back(larcv::ImageMeta(
      0, 0, 30720, _max_tick, _max_tick / compression, 30720, 0, larcv::kUnitCM));
  _image_meta_2d.push_back(larcv::ImageMeta(
      0, 0, 30720, _max_tick, _max_tick / compression, 30720, 1, larcv::kUnitCM));
  _image_meta_2d.push_back(larcv::ImageMeta(
      0, 0, 30720, _max_tick, _max_tick / compression, 30720, 2, larcv::kUnitCM));

}

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

int SuperaModuleBase::column(int tick, int channel) {
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

float SuperaModuleBase::wire_position(float x, float y, float z, int projection_id){
    double vtx[3];
    vtx[0] = x;
    vtx[1] = y;
    vtx[2] = z;
    try{
      return larutil::Geometry::GetME()->WireCoordinate(vtx, projection_id);
    }
    catch(...){
      return -999.;
    }
}
float SuperaModuleBase::tick_position(float x, float time_offset, int projection_id){
    // Convert an x coordinate to a tick position

    // First, convert the tick into the plane with the drift velocity:
    // (Add an offset for the distance between planes)
    float tick = x / larutil::GeometryHelper::GetME()->TimeToCm();

    if (x > 0){
      tick -= 7;
      if (projection_id == 0){
        tick -= 0.48;
      }
      if (projection_id == 1){
        tick -= -3.035;
      }
      if (projection_id == 2){
        tick -= 3.646;
      }
    }
    else{
      if (projection_id == 0){
        tick += -3.035;
      }
      if (projection_id == 1){
        tick += 0.48;
      }
      if (projection_id == 2){
        tick += 3.646;
      }
    }



    // if there is a time offset, add it:
    if(time_offset != 0){
      tick += time_offset;
    }


    // if (x < 0){

    // }

    // Accomodate the central x=0 position:
    tick += n_ticks;

    // Apply compression:
    tick /= compression;

    return tick;
}


}
#endif