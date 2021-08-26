#ifndef GALLERY_FMWK_SUPERA_MODULE_BASE_CXX
#define GALLERY_FMWK_SUPERA_MODULE_BASE_CXX

#include "supera_module_base.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"

#include "LArUtil/LArUtilServicesHandler.h"

#define PLANE_0_WIRES 1984
#define PLANE_1_WIRES 1984
#define PLANE_2_WIRES 1664

namespace supera {
SuperaModuleBase::SuperaModuleBase() {

    auto _geo_service = larutil::LArUtilServicesHandler::GetGeometry(_fcl_file_name);
    auto _det_service = larutil::LArUtilServicesHandler::GetDetProperties(_fcl_file_name);

    // Use GeoService to build the meta information:

    // auto fDetLength     = _geo_service->DetLength(0);
    // auto fDetHalfWidth  = _geo_service->DetHalfWidth(0); // 2x for sbnd dual drift
    // auto fDetHalfHeight = _geo_service->DetHalfHeight(0);

    // In 3D, using z = length, y = height, x = width

    // _base_image_meta_2D
    // void set_dimension(size_t axis, double image_size, size_t number_of_voxels, double origin = 0);

    _base_image_meta_3D.set_dimension(0, 400,  400, -200); // X goes -200 to 200
    _base_image_meta_3D.set_dimension(1, 400, 400, -200);
    _base_image_meta_3D.set_dimension(2, 500, 500, 0 );

    // std::cout << "3d meta: " << _base_image_meta_3D.dump() << std::endl;

    _base_image_meta_2D.resize(3);
    // Set the total ticks per image:
    total_ticks = 2*n_ticks_per_chamber + n_cathode_ticks;
    for (size_t plane = 0; plane < 3; plane ++){
        // For the first dimension, x, we need the number of wires:
        int n_wires = _geo_service->Nwires(plane, 0);
        _base_image_meta_2D[plane].set_dimension(0, 0.3*n_wires, n_wires);
        _base_image_meta_2D[plane].set_dimension(1, 0.078*total_ticks, total_ticks/compression );
        _base_image_meta_2D[plane].set_projection_id(plane);
        // std::cout << "2d meta: " << _base_image_meta_2D[plane].dump() << std::endl;
    }


}

int SuperaModuleBase::projection_id(int channel) {
  // Pretty hacky code here ...

  // In SBND, channels go 0 to 1983 (plane 0), 1984 to 3967, 3968 to 5633
  // Then repeat on the other side with offset of 5634, for a total
  // of 11268 channels

  if (channel < PLANE_0_WIRES)
    return 0;
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES)
    return 1;
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES)
    return 2;
  else if (channel < 2*PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES)
    return 0;
  else if (channel < 2*PLANE_0_WIRES + 2*PLANE_1_WIRES + PLANE_2_WIRES)
    return 1;
  else
    return 2;
}

int SuperaModuleBase::column(int channel) {
    // In SBND, channels go 0 to 1983 (plane 0), 1984 to 3967, 3968 to 5633
    // Then repeat on the other side with offset of 5634, for a total
    // of 11268 channels

  if (channel < PLANE_0_WIRES){
    return channel;
  }
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES){
    return channel - PLANE_0_WIRES;
  }
  else if (channel < PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES){
    return channel - (PLANE_0_WIRES + PLANE_1_WIRES);
  }
  else if (channel < 2*PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES) {
    return (channel - (PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES) );
  } else if (channel < 2*PLANE_0_WIRES + 2*PLANE_1_WIRES + PLANE_2_WIRES){
    return (channel - (2*PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES));
  } else {
    return (channel - (2*PLANE_0_WIRES + 2*PLANE_1_WIRES + PLANE_2_WIRES));
  }
}

int SuperaModuleBase::row(int tick, int channel) {
  if (channel >= PLANE_0_WIRES + PLANE_1_WIRES + PLANE_2_WIRES) {
    // In this scenario, we need the row to come out higher since it's the inverted
    // TPC, joined to form an image.
    return total_ticks - (tick - tick_offset) - 1;
  } else {
    return tick - tick_offset;
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
// float SuperaModuleBase::tick_position(float x, float time_offset, int projection_id){
//     // Convert an x coordinate to a tick position
//
//     // First, convert the tick into the plane with the drift velocity:
//     // (Add an offset for the distance between planes)
//     float tick = x / larutil::GeometryHelper::GetME()->TimeToCm();
//
//     if (x > 0){
//       tick -= 7;
//       if (projection_id == 0){
//         tick -= 0.48;
//       }
//       if (projection_id == 1){
//         tick -= -3.035;
//       }
//       if (projection_id == 2){
//         tick -= 3.646;
//       }
//     }
//     else{
//       if (projection_id == 0){
//         tick += -3.035;
//       }
//       if (projection_id == 1){
//         tick += 0.48;
//       }
//       if (projection_id == 2){
//         tick += 3.646;
//       }
//     }
//
//
//
//     // if there is a time offset, add it:
//     if(time_offset != 0){
//       tick += time_offset;
//     }
//
//
//     // if (x < 0){
//
//     // }
//
//     // Accomodate the central x=0 position:
//     tick += n_ticks_per_chamber;
//
//     // Apply compression:
//     tick /= compression;
//
//     return tick;
// }


}
#endif
