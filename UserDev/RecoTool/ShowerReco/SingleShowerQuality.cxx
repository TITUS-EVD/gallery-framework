#ifndef GALLERY_FMWK_SINGLESHOWERQUALITY_CXX
#define GALLERY_FMWK_SINGLESHOWERQUALITY_CXX

#include "SingleShowerQuality.h"

#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Cluster.h"
#include "lardataobj/RecoBase/Hit.h"


namespace galleryfmwk {

SingleShowerQuality::SingleShowerQuality() {

    _name = "SingleShowerQuality";
  
    fMCShowerProducer = "";
    fShowerProducer = "";
    fHitProducer = "";
    fShowerTree = nullptr;
    fEventTree = nullptr;
  
}


bool SingleShowerQuality::initialize() {

    if ( fShowerProducer.empty() || fMCShowerProducer.empty() ) {
       std::cerr << "Shower producer's name is not set!" << std::endl;
    }

    if ( fHitProducer.empty() )
       std::cerr << "Hit producer's name is not set!" << std::endl;

    InitializeAnaTrees();

    return true;
}


bool SingleShowerQuality::analyze(gallery::Event * ev) {

    ResetEventTreeParams();

    // Get all of the tracks from the event:
    art::InputTag mcshowerTag( fMCShowerProducer );
    auto const & mcshowers
        = ev -> getValidHandle< std::vector < sim::MCShower > >( mcshowerTag );

    if ( mcshowers->size() == 0 ) {
        if ( _verbose )
            std::cout << "No MC showers found." << std::endl;
        return false;
    }

    if ( mcshowers->size() > 1 ) {
        std::cout << "More than one MC shower, use the first MC shower in the analysis!" << std::endl;
        for ( size_t imcshower = 0; imcshower < mcshowers->size(); ++imcshower ) {
            std::cout << "PDG code: " << mcshowers->at(imcshower).PdgCode() 
                      << ", TrackID: " << mcshowers->at(imcshower).TrackID() 
                      << ", Process: " << mcshowers->at(imcshower).Process()
                      << ", Parent TrackID: " << mcshowers->at(imcshower).MotherTrackID() << std::endl;
        }
    }

    run = ev->eventAuxiliary().run();
    event = ev->eventAuxiliary().event();

    // Before getting the reconstructed showers, we store some true (mcshower) information
    // to be used as the denominator in efficiency calculations (n reco showers / n true showers, etc)
    fEventTreeParams.n_mcshowers = mcshowers->size();
    fEventTreeParams.mcs_E = mcshowers->at(0).DetProfile().E();
    fEventTreeParams.mc_containment = mcshowers->at(0).DetProfile().E() / mcshowers->at(0).Start().E();

    art::InputTag showerTag( fShowerProducer );
    gallery::Handle< std::vector< recob::Shower > > showers;
    if ( !ev->getByLabel( showerTag, showers ) ) {
        std::cout << "Run " << run << " event " << event << " has no shower data product "
                  << showerTag.encode() << std::endl;
        return false;
    }
    // std::vector< recob::Shower > const* showers = shower_handle.product();

    if ( showers->size() == 0 ) {
        if ( _verbose )
            std::cout << "No showers found." << std::endl;
        fEventTreeParams.n_recoshowers = 0;
        fEventTree->Fill();
        return false;
    }

    if ( _verbose ) {
        std::cout << "Number of showers by " << fShowerProducer
                  << ": " << showers->size() << std::endl;
    }

    //Fill event TTree number of reconstructed showers
    fEventTreeParams.n_recoshowers = showers->size();

    // Get pfparticles, clusters, hits
    auto const & ev_pfparts
        = ev -> getValidHandle< std::vector < recob::PFParticle > >( showerTag );
    auto const & ev_clusters
        = ev -> getValidHandle< std::vector < recob::Cluster > >( showerTag );
    art::InputTag hitTag( fHitProducer );
    auto const & ev_hits
        = ev -> getValidHandle< std::vector < recob::Hit > >( hitTag );

    // Calculate total charges
    fTotalHitQs = ClusterCharges( *ev_hits );
    std::cout << "fTotalHitQs: ";
    for ( auto it = fTotalHitQs.begin(); it != fTotalHitQs.end(); ++it ) {
        std::cout << "View: " << it->first << ", Charges: " << it->second << " ,";
    }
    std::cout << std::endl;

    // Get associated pfparticles, clusters, hits, etc.
    std::vector< std::vector< art::Ptr< recob::Cluster > > > cluster_per_shower( showers->size() );
    art::FindOneP< recob::PFParticle > shower_pfpart( showers, *ev, showerTag );

    art::InputTag pfpartTag( fShowerProducer );
    art::FindManyP< recob::Cluster > pfpart_cluster( ev_pfparts, *ev, pfpartTag );

    art::InputTag clusterTag( fShowerProducer );
    art::FindManyP< recob::Hit > cluster_hit( ev_clusters, *ev, clusterTag );

    for ( size_t ishower = 0; ishower < showers->size(); ++ishower ) {
        fShowerHitQs.clear();
        auto const & shower = showers->at( ishower );
        auto const & mcshower = mcshowers->at( 0 );
        FillQualityInfo( shower, mcshower, ishower );

        art::Ptr< recob::PFParticle > pfparts = shower_pfpart.at( ishower );
        if ( !pfparts ) continue;
        std::vector< recob::Hit > Hits;
        std::vector< art::Ptr< recob::Cluster > >& clusters = cluster_per_shower[ishower];
        size_t ipfpart = pfparts.key();
        auto const& PfoClusters = pfpart_cluster.at( ipfpart );
        if ( PfoClusters.size() == 0 ) continue;
        for ( art::Ptr< recob::Cluster > const& PfoCluster: PfoClusters ) {
            clusters.push_back( PfoCluster );
            size_t icluster = PfoCluster.key();
            auto const& clusterHits = cluster_hit.at( icluster );
            if ( clusterHits.size() == 0 ) continue;
            for ( auto const& clusterHit : clusterHits ) {
                Hits.push_back( *clusterHit );
            }
        }
        fShowerHitQs = ClusterCharges( Hits );
        std::cout << "Shower " << ishower << " fShowerHitQs: ";
        for ( auto it = fShowerHitQs.begin(); it != fShowerHitQs.end(); ++it ) {
            std::cout << "View: " << it->first << ", Charges: " << it->second << ", ";
        }
        std::cout << std::endl;
        FillClusterEP();
        std::cout << "fShowerTreeParams.cluster_eff_Y = " << fShowerTreeParams.cluster_eff_Y << std::endl;
        // Fill Tree
        fShowerTree->Fill();
    }

    fEventTree->Fill();

    return true;
}


bool SingleShowerQuality::finalize() {

    // This function is called at the end of event loop.
    // Do all variable finalization you wish to do here.
    // If you need, you can store your ROOT class instance in the output
    // file. You have an access to the output file through "_fout" pointer.
    //
    // Say you made a histogram pointer h1 to store. You can do this:
    //
    // if (_fout) { _fout->cd(); _tree->Write(); }
    if ( fShowerTree ) fShowerTree->Write();
    if ( fEventTree )  fEventTree->Write();

    return true;
}

void SingleShowerQuality::FillQualityInfo( const recob::Shower& reco_shower, const sim::MCShower& mc_shower, size_t ishower ) {

    ResetShowerTreeParams();

    // MC Info
    fShowerTreeParams.mc_x = mc_shower.DetProfile().X();
    fShowerTreeParams.mc_y = mc_shower.DetProfile().Y();
    fShowerTreeParams.mc_z = mc_shower.DetProfile().Z();
  
    fShowerTreeParams.mc_energy = mc_shower.DetProfile().E();
    fShowerTreeParams.mc_containment = mc_shower.DetProfile().E() / mc_shower.Start().E();
  
    fShowerTreeParams.mc_dcosx = mc_shower.Start().Px() / mc_shower.Start().E();
    fShowerTreeParams.mc_dcosy = mc_shower.Start().Py() / mc_shower.Start().E();
    fShowerTreeParams.mc_dcosz = mc_shower.Start().Pz() / mc_shower.Start().E();
  
    fShowerTreeParams.mc_length = Length( mc_shower.DetProfile().E() );
    TVector3 WildShowerDir( mc_shower.End().X() - mc_shower.Start().X(), mc_shower.End().Y() - mc_shower.Start().Y(), mc_shower.End().Z() - mc_shower.Start().Z() );
    fShowerTreeParams.mc_wildlength = WildShowerDir.Mag();

    // Reco vtx
    // fShowerTreeParams.reco_x = reco_shower.ShowerStart()[0] + 125.4;
    fShowerTreeParams.reco_x = reco_shower.ShowerStart()[0];
    fShowerTreeParams.reco_y = reco_shower.ShowerStart()[1];
    fShowerTreeParams.reco_z = reco_shower.ShowerStart()[2];
  
    // Reco angle
    fShowerTreeParams.reco_dcosx = reco_shower.Direction()[0];
    fShowerTreeParams.reco_dcosy = reco_shower.Direction()[1];
    fShowerTreeParams.reco_dcosz = reco_shower.Direction()[2];

    // Reco - MC angle diff
    fShowerTreeParams.mc_reco_anglediff = acos( fShowerTreeParams.reco_dcosx * fShowerTreeParams.mc_dcosx +
                                          fShowerTreeParams.reco_dcosy * fShowerTreeParams.mc_dcosy +
                                          fShowerTreeParams.reco_dcosz * fShowerTreeParams.mc_dcosz ) / 3.14159265359 * 180.;
    // Reco - MC vtx distance
    fShowerTreeParams.mc_reco_dist = sqrt( pow(fShowerTreeParams.reco_x - fShowerTreeParams.mc_x, 2) +
                                           pow(fShowerTreeParams.reco_y - fShowerTreeParams.mc_y, 2) +
                                           pow(fShowerTreeParams.reco_z - fShowerTreeParams.mc_z, 2) );
  

    fShowerTreeParams.reco_energy_U = reco_shower.Energy().at(0);
    fShowerTreeParams.reco_energy_V = reco_shower.Energy().at(1);
    fShowerTreeParams.reco_energy_Y = reco_shower.Energy().at(2);
  
    // fShowerTreeParams.reco_dedx     = reco_shower.dEdx().at(2);
    // fShowerTreeParams.reco_dedx_U   = reco_shower.dEdx().at(0);
    // fShowerTreeParams.reco_dedx_V   = reco_shower.dEdx().at(1);
    // fShowerTreeParams.reco_dedx_Y   = reco_shower.dEdx().at(2);

    // fShowerTreeParams.reco_dqdx   = reco_shower.dQdx();
    // fShowerTreeParams.reco_dqdx_U = reco_shower.dQdx_v().at(0);
    // fShowerTreeParams.reco_dqdx_V = reco_shower.dQdx_v().at(1);
    // fShowerTreeParams.reco_dqdx_Y = reco_shower.dQdx_v().at(2);
  
    fShowerTreeParams.reco_length = reco_shower.Length();
    // fShowerTreeParams.reco_width1 = reco_shower.Width()[0];
    // fShowerTreeParams.reco_width2 = reco_shower.Width()[1];
  
    fShowerTreeParams.match = 1;
  
  
}

double SingleShowerQuality::Length( double energy ) {

    /// This formula taken from Andrzej
    double shower_length = 13.8874 + 0.121734*energy - (3.75571e-05)*energy*energy;
  
    if(shower_length < 5) return 5.;
  
    return shower_length;
}

std::map< int, double > SingleShowerQuality::ClusterCharges( std::vector< recob::Hit > const& hits ) {
    std::map< int, double > totalCharges;

    for ( auto const & hit : hits ) {
        int iplane = (int)hit.View();
        double charges = hit.Integral();
        if ( !totalCharges.count( iplane ) ) totalCharges[iplane] = charges;
        else totalCharges[iplane] += charges;
    }
    return totalCharges;
}

void SingleShowerQuality::FillClusterEP() {

    std::cout << "fShowerHitQs[2]/fTotalHitQs[2] = " << fShowerHitQs[2] << "/" << fTotalHitQs[2] << std::endl;
    // Reco cluster efficiency & purity
    fShowerTreeParams.cluster_eff_U = fShowerHitQs[0] / fTotalHitQs[0];
    fShowerTreeParams.cluster_eff_V = fShowerHitQs[1] / fTotalHitQs[1];
    fShowerTreeParams.cluster_eff_Y = fShowerHitQs[2] / fTotalHitQs[2];
    fShowerTreeParams.cluster_pur_U = 1.;
    fShowerTreeParams.cluster_pur_V = 1.;
    fShowerTreeParams.cluster_pur_Y = 1.;

}

void SingleShowerQuality::InitializeAnaTrees() {
    //////////////////////////////////////////////////////
    // This tree is filled once per reconstructed shower
    //////////////////////////////////////////////////////
    if ( fShowerTree ) delete fShowerTree;
    fShowerTree = new TTree( "fShowerTree", "" );

    fShowerTree->Branch("event", &event, "event/I");
    fShowerTree->Branch("run", &run, "run/I");
    fShowerTree->Branch("subrun", &subrun, "subrun/I");
    fShowerTree->Branch("reco_x", &fShowerTreeParams.reco_x, "reco_x/D");
    fShowerTree->Branch("reco_y", &fShowerTreeParams.reco_y, "reco_y/D");
    fShowerTree->Branch("reco_z", &fShowerTreeParams.reco_z, "reco_z/D");
    fShowerTree->Branch("reco_dcosx", &fShowerTreeParams.reco_dcosx, "reco_dcosx/D");
    fShowerTree->Branch("reco_dcosy", &fShowerTreeParams.reco_dcosy, "reco_dcosy/D");
    fShowerTree->Branch("reco_dcosz", &fShowerTreeParams.reco_dcosz, "reco_dcosz/D");
    fShowerTree->Branch("reco_energy_U", &fShowerTreeParams.reco_energy_U, "reco_energy_U/D");
    fShowerTree->Branch("reco_energy_V", &fShowerTreeParams.reco_energy_V, "reco_energy_V/D");
    fShowerTree->Branch("reco_energy_Y", &fShowerTreeParams.reco_energy_Y, "reco_energy_Y/D");
    fShowerTree->Branch("mc_x", &fShowerTreeParams.mc_x, "mc_x/D");
    fShowerTree->Branch("mc_y", &fShowerTreeParams.mc_y, "mc_y/D");
    fShowerTree->Branch("mc_z", &fShowerTreeParams.mc_z, "mc_z/D");
    fShowerTree->Branch("mc_dcosx", &fShowerTreeParams.mc_dcosx, "mc_dcosx/D");
    fShowerTree->Branch("mc_dcosy", &fShowerTreeParams.mc_dcosy, "mc_dcosy/D");
    fShowerTree->Branch("mc_dcosz", &fShowerTreeParams.mc_dcosz, "mc_dcosz/D");
    fShowerTree->Branch("reco_dqdx", &fShowerTreeParams.reco_dqdx, "reco_dqdx/D");
    fShowerTree->Branch("reco_dqdx_U", &fShowerTreeParams.reco_dqdx_U, "reco_dqdx_U/D");
    fShowerTree->Branch("reco_dqdx_V", &fShowerTreeParams.reco_dqdx_V, "reco_dqdx_V/D");
    fShowerTree->Branch("reco_dqdx_Y", &fShowerTreeParams.reco_dqdx_Y, "reco_dqdx_Y/D");
    fShowerTree->Branch("mc_energy", &fShowerTreeParams.mc_energy, "mc_energy/D");
    fShowerTree->Branch("reco_dedx", &fShowerTreeParams.reco_dedx, "reco_dedx/D");
    fShowerTree->Branch("reco_dedx_U", &fShowerTreeParams.reco_dedx_U, "reco_dedx_U/D");
    fShowerTree->Branch("reco_dedx_V", &fShowerTreeParams.reco_dedx_V, "reco_dedx_V/D");
    fShowerTree->Branch("reco_dedx_Y", &fShowerTreeParams.reco_dedx_Y, "reco_dedx_Y/D");
    fShowerTree->Branch("mc_reco_anglediff", &fShowerTreeParams.mc_reco_anglediff, "mc_reco_anglediff/D");
    fShowerTree->Branch("mc_reco_dist", &fShowerTreeParams.mc_reco_dist, "mc_reco_dist/D");
    fShowerTree->Branch("cluster_eff_U", &fShowerTreeParams.cluster_eff_U, "cluster_eff_U/D");
    fShowerTree->Branch("cluster_eff_V", &fShowerTreeParams.cluster_eff_V, "cluster_eff_V/D");
    fShowerTree->Branch("cluster_eff_Y", &fShowerTreeParams.cluster_eff_Y, "cluster_eff_Y/D");
    fShowerTree->Branch("cluster_pur_U", &fShowerTreeParams.cluster_pur_U, "cluster_pur_U/D");
    fShowerTree->Branch("cluster_pur_V", &fShowerTreeParams.cluster_pur_V, "cluster_pur_V/D");
    fShowerTree->Branch("cluster_pur_Y", &fShowerTreeParams.cluster_pur_Y, "cluster_pur_Y/D");
    fShowerTree->Branch("mc_containment", &fShowerTreeParams.mc_containment, "mc_containment/D");
    fShowerTree->Branch("reco_length",&fShowerTreeParams.reco_length,"reco_length/D");
    fShowerTree->Branch("reco_width1",&fShowerTreeParams.reco_width1,"reco_width1/D");
    fShowerTree->Branch("reco_width2",&fShowerTreeParams.reco_width2,"reco_width2/D");
    fShowerTree->Branch("mc_length",  &fShowerTreeParams.mc_length,  "mc_length/D");
    fShowerTree->Branch("mc_wildlength", &fShowerTreeParams.mc_wildlength, "mc_wildlength/D");
    fShowerTree->Branch("match", &fShowerTreeParams.match, "match/I");

    //////////////////////////////////////////////////////
    // This tree is filled once per event
    //////////////////////////////////////////////////////
    if ( fEventTree ) delete fEventTree;
    fEventTree = new TTree( "fEventTree", "" );
  
    fEventTree->Branch("n_recoshowers", &fEventTreeParams.n_recoshowers, "n_recoshowers/I");
    fEventTree->Branch("n_mcshowers", &fEventTreeParams.n_mcshowers, "n_mcshowers/I");
    fEventTree->Branch("mcs_E", &fEventTreeParams.mcs_E, "mcs_E/D");
    fEventTree->Branch("mc_containment", &fEventTreeParams.mc_containment, "mc_containment/D");
}

void SingleShowerQuality::ResetShowerTreeParams() {

    fShowerTreeParams.reco_x = -1.; fShowerTreeParams.reco_y = -1.; fShowerTreeParams.reco_z = -1.;
    fShowerTreeParams.reco_dcosx = -1.; fShowerTreeParams.reco_dcosy = -1.; fShowerTreeParams.reco_dcosz = -1.;
    fShowerTreeParams.reco_energy_U = -1.;
    fShowerTreeParams.reco_energy_V = -1.;
    fShowerTreeParams.reco_energy_Y = -1.;
    fShowerTreeParams.reco_dedx_U = -1.;
    fShowerTreeParams.reco_dedx_V = -1.;
    fShowerTreeParams.reco_dedx_Y = -1.;
    fShowerTreeParams.reco_dqdx_U =-1.;
    fShowerTreeParams.reco_dqdx_V =-1.;
    fShowerTreeParams.reco_dqdx_Y =-1.;
    fShowerTreeParams.mc_x = -1.; fShowerTreeParams.mc_y = -1.; fShowerTreeParams.mc_z = -1.;
    fShowerTreeParams.mc_dcosx = -1.; fShowerTreeParams.mc_dcosy = -1.; fShowerTreeParams.mc_dcosz = -1.;
    fShowerTreeParams.mc_energy = -1.;
    fShowerTreeParams.mc_reco_anglediff = -1.;
    fShowerTreeParams.mc_reco_dist = -1.;
    fShowerTreeParams.cluster_eff_U = -1.234;
    fShowerTreeParams.cluster_eff_V = -1.234;
    fShowerTreeParams.cluster_eff_Y = -1.234;
    fShowerTreeParams.cluster_pur_U = -1.234;
    fShowerTreeParams.cluster_pur_V = -1.234;
    fShowerTreeParams.cluster_pur_Y = -1.234;
    fShowerTreeParams.mc_containment = -1.;
    fShowerTreeParams.reco_length = -1.;
    fShowerTreeParams.reco_width1 = -1.;
    fShowerTreeParams.reco_width2 = -1.;
    fShowerTreeParams.mc_length   = -1.;
    fShowerTreeParams.mc_wildlength  = -1.;
}

void SingleShowerQuality::ResetEventTreeParams() {
    fEventTreeParams.n_mcshowers = 0;
    fEventTreeParams.n_recoshowers = 0;
    fEventTreeParams.mcs_E = -1.;
    fEventTreeParams.mc_containment = -1.;
}

}
#endif
