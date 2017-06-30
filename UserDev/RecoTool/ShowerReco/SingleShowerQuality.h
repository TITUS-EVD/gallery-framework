/**
 * \file SingleShowerQuality.h
 *
 * \ingroup showerreco_analysis
 *
 * \brief Class def header for a class SingleShowerQuality
 *
 * @author Yun-Tse Tsai
 */

/** \addtogroup showerreco_analysis

    @{*/

#ifndef GALLERY_FMWK_SINGLESHOWERQUALITY_H
#define GALLERY_FMWK_SINGLESHOWERQUALITY_H

#include "TTree.h"

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Persistency/Common/FindOneP.h"
#include "cetlib/maybe_ref.h"

#include "Analysis/ana_base.h"

#include "lardataobj/MCBase/MCShower.h"
#include "lardataobj/RecoBase/Shower.h"
#include "lardataobj/RecoBase/Hit.h"


namespace galleryfmwk {

/**
   \class SingleShowerQuality
   User custom analysis class made by SHELL_USER_NAME
 */
class SingleShowerQuality : galleryfmwk::ana_base {

public:

    /// Default constructor
    SingleShowerQuality();
  
    /// Default destructor
    ~SingleShowerQuality() {}
  
  
    bool initialize();
  
  
    bool analyze(gallery::Event * ev);
  
  
    bool finalize();
  
  
    void setMCShowerProducer( std::string s ) { fMCShowerProducer = s; }
    void setShowerProducer( std::string s ) { fShowerProducer = s; }
    void setHitProducer( std::string s ) { fHitProducer = s; }
    void setVerbose( bool b ){ _verbose = b; }

protected:

    std::string fMCShowerProducer;
    std::string fShowerProducer;
    std::string fHitProducer;
    bool _verbose;
  
    /// Analysis TTree. Filled once per reconstructed shower.
    TTree *fShowerTree;
  
    /// Analysis TTree. Filled once per event.
    TTree *fEventTree;
  
    int event, subrun, run;
    std::map< int, double > fTotalHitQs;
    std::map< int, double > fShowerHitQs;
  
    /// For convenience: struct to define a set of parameters per shower to be stored in per-reconstructed-shower TTree
    struct ShowerTreeParams_t {

        double reco_x, reco_y, reco_z;
        double reco_dcosx, reco_dcosy, reco_dcosz;
        double reco_energy_U;
        double reco_energy_V;
        double reco_energy_Y;
        double reco_dedx;
        double reco_dedx_U;
        double reco_dedx_V;
        double reco_dedx_Y;
        double reco_dqdx;
        double reco_dqdx_U;
        double reco_dqdx_V;
        double reco_dqdx_Y;
        double mc_x, mc_y, mc_z;
        double mc_dcosx, mc_dcosy, mc_dcosz;
        double mc_energy;
        double mc_reco_anglediff;
        double mc_reco_dist;
        double cluster_eff_U;
        double cluster_eff_V;
        double cluster_eff_Y;
        double cluster_pur_U;
        double cluster_pur_V;
        double cluster_pur_Y;
        double mc_containment;
        double reco_length;
        double reco_width1;
        double reco_width2;
        double mc_length;
        double mc_wildlength;
        int    match;
    
    } fShowerTreeParams;


    struct EventTreeParams_t {

        int n_mcshowers;
        int n_recoshowers;
        //Detprofile energy of the FIRST (for now, only) mcshower
        double mcs_E;
        double mc_containment;
    
    } fEventTreeParams;

    /// Function to fill the variables related to the performance of shower reconstruction
    void FillQualityInfo( const recob::Shower& reco_shower, const sim::MCShower& mc_shower, size_t ishower );

    /// Function to calculate the total charges of each cluster
    std::map< int, double > ClusterCharges( std::vector< recob::Hit > const& hits );

    /// Function to fill the clustering efficiecny and purity
    void FillClusterEP();

    /// Function to set all of once-per-shower tree parameters to default values
    void ResetShowerTreeParams();
  
    /// Function to set all of once-per-event tree parameters to default values
    void ResetEventTreeParams();
  
    /// Function to prepare TTrees
    void InitializeAnaTrees();

    /// Function to calculate MC shower length from the MC deposited energy
    double Length( double energy );
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
