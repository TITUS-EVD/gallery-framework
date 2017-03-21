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
#include "TSystem.h"
#include "TROOT.h"

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

bool inFV(double x_vtx, double y_vtx, double z_vtx,
          double x1, double x2, double y1, double y2, double z1, double z2);

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
void setNearestCutDist(double cut){
	_cut = cut;
}
void fiducial_volume_x_right(double right){
	_right = right;
}
void fiducial_volume_x_left(double left){
	_left = left;
}
void fiducial_volume_y_up(double up){
	_up = up;
}
void fiducial_volume_y_down(double down){
	_down = down;
}
void fiducial_volume_z_back(double back){
	_back = back;
}
void fiducial_volume_z_front(double front){
	_front = front;
}

protected:

bool _verbose;
std::string _track_producer;
std::string _shower_producer;
std::string _pfp_producer;
double _cut;
double _right;
double _left;
double _up;
double _down;
double _back;
double _front;


int num_cosmic;
int num_primary_pfp;
int num_nue;
int num_numu;
int num_nue_per_event;
int cosmic_vertex_cut_pass;
int cosmic_vertex_shower_cut_pass;

double x_boundary1;
double x_boundary2;
double y_boundary1;
double y_boundary2;
double z_boundary1;
double z_boundary2;
double fromWall;

double fv_cut_max;

TH1D * h_nue_fv_cuts;
TH1D * h_nue_fv_top_cuts;
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

TH2D * h_nue_like_shwr_lrgDist_vtx_xy;
TH2D * h_nue_like_shwr_lrgDist_vtx_zy;
TH1D * h_nue_like_shwr_lrgDist_dist_to_track;
TH1D * h_nue_like_shwr_lrgDist_num_trks;
TH1D * h_nue_like_shwr_lrgDist_dist_to_cosmic;

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
TH2D * h_nue_shwr_cosmic_closest_vs_y;

TH1D * h_cosmic_trk_length;
TH1D * h_nue_trk_length;

TH1D * h_nue_trk_closest;
TH1D * h_nue_shwr_trk_closest;

TH1D * h_num_trks_nearby;

TH2D * h_nue_shwr_cut_vtx_xy;
TH2D * h_nue_shwr_cut_vtx_zy;

TH1D * h_num_nue_per_event;


TCanvas * c1;
TCanvas * c1b;
TCanvas * c2;
TCanvas * c3;
TCanvas * c3b;
TCanvas * c4;
TCanvas * c5;
TCanvas * c6;
TCanvas * c7;
TCanvas * c8;
TCanvas * c9;
TCanvas * c10;
TCanvas * c11;
TCanvas * c12;
TCanvas * c13;
TCanvas * c14;
TCanvas * c15;
TCanvas * c16;
TCanvas * c17;
TCanvas * c17b;
TCanvas * c18;
TCanvas * c19;
TCanvas * c19b;
TCanvas * c19c;
TCanvas * c20a;
TCanvas * c20b;
TCanvas * c21a;
TCanvas * c21b;
TCanvas * c22;
TCanvas * c23a;
TCanvas * c23b;
TCanvas * c24;
TCanvas * c25a;
TCanvas * c25b;
TCanvas * c25c;
TCanvas * c25d;
TCanvas * c25e;

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
