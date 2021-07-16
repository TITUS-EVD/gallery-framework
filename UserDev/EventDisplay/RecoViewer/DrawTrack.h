/**
 * \file DrawTrack.h
 *
 * \ingroup RecoViewer
 *
 * \brief Class def header for a class DrawTrack
 *
 * @author cadams
 */

/** \addtogroup RecoViewer

    @{*/
#ifndef EVD_DRAWTRACK_H
#define EVD_DRAWTRACK_H

#include "Analysis/ana_base.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/Hit.h"
#include "canvas/Persistency/Common/FindMany.h"
#include <iostream>

#include "RecoBase.h"

/**
   \class DrawTrack
   User defined class DrawTrack ... these comments are used to generate
   doxygen documentation!
 */

namespace evd {

class Track2D {
public:
  friend class DrawTrack;
  friend class DrawNumuSelection;

  Track2D(){}
  Track2D(std::vector<std::pair<float, float>> _track){
    this->_track = _track;
  }

  const std::vector<std::pair<float, float>> &track() { return _track; }
  const std::vector<std::pair<float, float>> &direction() { return _track; }
  const std::vector<unsigned int> &tpc() { return _tpc; }
  const std::vector<unsigned int> &cryo() { return _cryo; }
  const double &length() { return _length; }
  const float &chi2() { return _chi2; }
  const double &theta() { return _theta; }
  const double &phi() { return _phi; }

protected:
  std::vector<std::pair<float, float>> _track; ///< Trajectory points
  std::vector<unsigned int> _tpc; ///< TPC of each trajectory point
  std::vector<unsigned int> _cryo; ///< Cryo of each trajectory point
  double _length;
  float _chi2;
  double _theta;
  double _phi;
};

// typedef std::vector<std::pair<float, float> > Track2D;

class DrawTrack : public galleryfmwk::ana_base, public RecoBase<Track2D> {

public:
  /// Default constructor
  DrawTrack(const geo::GeometryCore&               geometry,
            const detinfo::DetectorPropertiesData& detectorProperties,
            const detinfo::DetectorClocksData&     detectorClocks);

  /// Default destructor
  ~DrawTrack();

  virtual bool initialize();

  virtual bool analyze(const gallery::Event &event);

  virtual bool finalize();

private:
  Track2D getTrack2D(recob::Track track, unsigned int plane);

  size_t _total_plane_number;
};

} // evd

#endif
/** @} */ // end of doxygen group
