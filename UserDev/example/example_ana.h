/**
 * \file example_ana.h
 *
 * \ingroup nuexsec_analysis
 *
 * \brief Class def header for a class example_ana
 *
 * @author cadams
 */

/** \addtogroup nuexsec_analysis

    @{*/

#ifndef GALLERY_FMWK_EXAMPLE_ANA_H
#define GALLERY_FMWK_EXAMPLE_ANA_H

#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"
#include "canvas/Persistency/Common/FindMany.h"

#include "Analysis/ana_base.h"

#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/Shower.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "lardataobj/RecoBase/Hit.h"

#include "GeoAlgo/GeoAlgo.h"
#include "GeoAlgo/GeoVector.h"
#include "GeoAlgo/GeoTrajectory.h"

namespace galleryfmwk {

/**
   \class example_ana
   User custom analysis class made by SHELL_USER_NAME
 */
class example_ana : galleryfmwk::ana_base {

geoalgo::GeoAlgo const _geo_algo_instance;

public:

bool inFV(double x_vtx, double y_vtx, double z_vtx, double fromWall);

/// Default constructor
example_ana() {
	_verbose = false;
}

/// Default destructor
// ~example_ana() {}


bool initialize();


bool analyze(gallery::Event * ev);


bool finalize();


void setTrackProducer(std::string s) {
	_track_producer = s;
}
void setShowerProducer(std::string s) {
	_shower_producer = s;
}
void setVerbose(bool b){
	_verbose = b;
}

protected:

std::string _track_producer;
std::string _shower_producer;
std::string _pfp_producer;
int num_cosmic;
int num_primary_pfp;
int num_nue;
int num_numu;

double x_boundary1;
double x_boundary2;
double y_boundary1;
double y_boundary2;
double z_boundary1;
double z_boundary2;
double fromWall;

double fv_cut_max;

bool _verbose;

TH1D * h_nue_fv_cuts;
TH1D * h_numu_fv_cuts;
TH2D * h_nue_like_daughters;
TH1D * h_nue_like_trk_daughters;
TH2D * h_numu_like_daughters;

TH2D * h_nue_like_shwr_daughters_xy;
TH2D * h_nue_like_shwr_daughters_yz;
TH2D * h_nue_like_trk_daughters_xy;
TH2D * h_nue_like_trk_daughters_yz;
TH2D * h_nue_like_vtx_xy;
TH2D * h_nue_like_vtx_yz;

TH2D * h_numu_like_shwr_daughters_xy;
TH2D * h_numu_like_shwr_daughters_yz;
TH2D * h_numu_like_trk_daughters_xy;
TH2D * h_numu_like_trk_daughters_yz;
TH2D * h_numu_like_vtx_xy;
TH2D * h_numu_like_vtx_yz;

TH1D * h_nue_cosmic_closest;
TH1D * h_nue_shwr_cosmic_closest;
TH1D * h_nue_shwr_vtx_dist;

TH1D * h_nue_shwr_E;
TH2D * h_nue_shwr_cosmic_closest_vs_E;

TH1D * h_cosmic_trk_length;
TH1D * h_nue_trk_length;

};

}

#endif

//**************************************************************************
//
// For Analysis framework documentation, read Manual.pdf here:
//
// http://microboone-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=3183
//
//**************************************************************************

/** @} */ // end of doxygen group
