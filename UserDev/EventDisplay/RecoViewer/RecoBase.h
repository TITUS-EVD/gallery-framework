/**
 * \file RecoBase.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class RecoBase
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef RECOBASE_H
#define RECOBASE_H

#include <iostream>

struct _object;
typedef _object PyObject;

// #ifndef __CINT__
// #include "Python.h"
// #include "numpy/arrayobject.h"
// #endif

// #include "LArUtil/Geometry.h"
// #include "LArUtil/GeometryHelper.h"
// #include "LArUtil/DetectorProperties.h"

#include "larcorealg/Geometry/GeometryCore.h"
#include "lardataalg/DetectorInfo/DetectorProperties.h"
#include "lardataalg/DetectorInfo/DetectorClocksData.h"

#include "LArUtil/SimpleGeometryHelper.h"


/**
   \class RecoBase
   User defined class RecoBase ... these comments are used to generate
   doxygen documentation!
 */
namespace evd {

template <class DATA_TYPE >
class RecoBase {

public:

  /// Default constructor
  RecoBase(const geo::GeometryCore& geometry,
           const detinfo::DetectorProperties& detectorProperties,
           const detinfo::DetectorClocksData& detectorClocks);

  /// Default destructor
  virtual ~RecoBase() {}

  std::pair<float, float> getWireRange(size_t p);
  std::pair<float, float> getTimeRange(size_t p);

  void setProducer(std::string s);

  /// Returns the data on the selected plane
  const std::vector<DATA_TYPE> & getDataByPlane(size_t p);
  // PyObject * getDataByPlane(size_t p);

  /// Returns a vector of data, what is stored depends on the implementation
  const std::vector<DATA_TYPE> & getExtraData(size_t p);

  /// Returns a vector of data (not plane dependent)
  const std::vector<DATA_TYPE> & getData();

protected:

  void _init_base();

  // const larutil::Geometry * geoService;
  // const larutil::GeometryHelper * geoHelper;
  // const larutil::DetectorProperties * detProp;

  const geo::GeometryCore&           _geo_service;
  const detinfo::DetectorProperties& _det_prop;
  const detinfo::DetectorClocksData& _det_clock;

  std::string _producer;

  // Store the reco data to draw;
  std::vector <std::vector < DATA_TYPE > > _dataByPlane;
  // Extra reco data to draw (in case we need this)
  std::vector <std::vector < DATA_TYPE > > _extraDataByPlane;
  // A vector of data (for stuff not plane dependent)
  std::vector < DATA_TYPE > _data;

  // Store the bounding parameters of interest:
  // highest and lowest wire, highest and lowest time
  // Have to sort by plane still

  std::vector<std::pair<float, float> > _wireRange;
  std::vector<std::pair<float, float> > _timeRange;
};


template <class DATA_TYPE>
RecoBase <DATA_TYPE>::RecoBase(const geo::GeometryCore& geometry,
                               const detinfo::DetectorProperties& detectorProperties,
                               const detinfo::DetectorClocksData& detectorClocks) :
  _geo_service(geometry),
  _det_prop(detectorProperties),
  _det_clock(detectorClocks)
{
  // geoService = larutil::Geometry::GetME();
  // geoHelper = larutil::GeometryHelper::GetME();
  // detProp = larutil::DetectorProperties::GetME();

  // Set up default values of the _wire and _time range
  int total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  _wireRange.resize(total_plane_number);
  _timeRange.resize(total_plane_number);

  size_t counter = 0;
  for (unsigned int c = 0; c < _geo_service.Ncryostats(); c++) {
    for (unsigned int t = 0; t < _geo_service.NTPC(c); t++) {
      for (unsigned int p = 0; p < _geo_service.Nplanes(t); p++) {
        _wireRange.at(counter).first  = 0;
        _wireRange.at(counter).second = _geo_service.Nwires(p, t, c);
        _timeRange.at(counter).first  = 0;
        _timeRange.at(counter).second = _det_prop.ReadOutWindowSize();
        counter++;
      }
    }
  }
  // for (size_t total_plane_number = 0; total_plane_number < total_plane_number; total_plane_number ++) {
  //   _wireRange.at(total_plane_number).first  = 0;
  //   _wireRange.at(total_plane_number).second = geoService -> Nwires(view);
  //   _timeRange.at(total_plane_number).first  = 0;
  //   _timeRange.at(total_plane_number).second = detProp -> NumberTimeSamples();
  // }
  // import_array();
}

template <class DATA_TYPE>
void RecoBase <DATA_TYPE>::setProducer(std::string s) {
  _producer = s;
}

template <class DATA_TYPE>
std::pair<float, float> RecoBase<DATA_TYPE>::getWireRange(size_t p) {
  int total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  static std::pair<float, float> returnNull;
  if (p >= total_plane_number) {
    std::cerr << "ERROR: Request for nonexistent plane " << p << std::endl;
    return returnNull;
  }
  else {
    try {
      return _wireRange.at(p);
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return returnNull;
    }
  }
}


template <class DATA_TYPE>
std::pair<float, float> RecoBase<DATA_TYPE>::getTimeRange(size_t p) {
  int total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  static std::pair<float, float> returnNull;
  if (p >= total_plane_number) {
    std::cerr << "ERROR: Request for nonexistent plane " << p << std::endl;
    return returnNull;
  }
  else {
    try {
      return _timeRange.at(p);
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return returnNull;
    }
  }
}

template <class DATA_TYPE>
const std::vector<DATA_TYPE> & RecoBase<DATA_TYPE>::getDataByPlane(size_t p) {
  int total_plane_number = _geo_service.Nplanes() * _geo_service.NTPC() * _geo_service.Ncryostats();
  static std::vector<DATA_TYPE> returnNull;
  if (p >= total_plane_number) {
    std::cerr << "ERROR: Request for nonexistent plane " << p << std::endl;
    return returnNull;
  }
  else {
    try {
      return _dataByPlane.at(p);
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return returnNull;
    }
  }
}


template <class DATA_TYPE>
const std::vector<DATA_TYPE> & RecoBase<DATA_TYPE>::getExtraData(size_t p) {
  int total_plane_number = _geo_service.NTPC() * _geo_service.Ncryostats();
  static std::vector<DATA_TYPE> returnNull;
  if (p >= total_plane_number) {
    std::cerr << "ERROR: Request for nonexistent plane " << p << std::endl;
    return returnNull;
  }
  else {
    try {
      return _extraDataByPlane.at(p);
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return returnNull;
    }
  }
}

template <class DATA_TYPE>
const std::vector<DATA_TYPE> & RecoBase<DATA_TYPE>::getData() {
  static std::vector<DATA_TYPE> returnNull;
  try {
    return _data;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return returnNull;
  }
}

} // evd

#endif
/** @} */ // end of doxygen group

