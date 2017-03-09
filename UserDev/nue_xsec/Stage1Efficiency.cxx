#ifndef LARLITE_STAGE1EFFICIENCY_CXX
#define LARLITE_STAGE1EFFICIENCY_CXX

#include "Stage1Efficiency.h"


namespace argoana {

bool Stage1Efficiency::initialize() {

    //
    // This function is called in the beggining of event loop
    // Do all variable initialization you wish to do here.
    // If you have a histogram to fill in the event loop, for example,
    // here is a good place to create one on the heap (i.e. "new TH1D").
    //

    _tree = new TTree("Stage1Efficiency", "Stage1Efficiency");


    // Vertex variables (truth):
    _tree->Branch("_vertex_x_true", &_vertex_x_true);
    _tree->Branch("_vertex_y_true", &_vertex_y_true);
    _tree->Branch("_vertex_z_true", &_vertex_z_true);
    _tree->Branch("_is_fiducial", &_is_fiducial);


    // Neutrino Variables:
    _tree->Branch("_neutrino_energy", &_neutrino_energy);
    _tree->Branch("_neutrino_pdg", &_neutrino_pdg);



    // Number of reconstructed vertexes:
    _tree->Branch("_n_vertexes", &_n_vertexes);

    // Information about number of clusters:
    _tree->Branch("_n_clusters", &_n_clusters);

    // Information about tracks:
    _tree->Branch("_n_tracks", &_n_tracks);
    _tree->Branch("_n_long_tracks", &_n_long_tracks);
    _tree->Branch("_longest_track", &_longest_track);

    // How many clustered hits in the collection plane
    // are associated to the longest track?
    _tree->Branch("_percent_hits_of_longest_track",
                  &_percent_hits_of_longest_track);




    return true;
}

void Stage1Efficiency::clear() {
    _vertex_x_true = 0;
    _vertex_y_true = 0;
    _vertex_z_true = 0;
    _is_fiducial = false;
    _neutrino_energy = 0;
    _neutrino_pdg = 0;
    _n_vertexes = 0;
    _n_clusters = 0;
    _n_tracks = 0;
    _n_long_tracks = 0;
    _longest_track = 0.0;
    _percent_hits_of_longest_track = 0;
}

bool Stage1Efficiency::analyze(gallery::Event * ev) {

    // For each file, loop over all events.
    //
    // Determine criteria for rejecting muon like events
    // Save metadata for each event (neutrino pdg, energy, vertex)
    // as well as filter results.

    clear();


//     // MC Information:
//     // First, try to get the mctruth
//     // information from the neutrino interaction
//     larlite::event_mctruth * event_truth
//         = storage -> get_data<larlite::event_mctruth>("generator");
//     if (event_truth != 0) {
//         if (event_truth -> size() != 0) {
//             auto truth = event_truth -> at(0);
//             // Get the neutrino:
//             auto neutrino = truth.GetNeutrino();
//             auto vertex = neutrino.Lepton().Position();
//             _vertex_x_true = vertex.X();
//             _vertex_y_true = vertex.Y();
//             _vertex_z_true = vertex.Z();
//             _is_fiducial = isFiducial(vertex,3);
//             _neutrino_energy = neutrino.Nu().Momentum().E();
//             _neutrino_pdg = neutrino.Nu().PdgCode();
//         }
//     }
//     else {
//         // There is not mctruth by 'generator' so this
//         // must be a tgmuon file
//         larlite::event_mctruth * event_truth
//             = storage -> get_data<larlite::event_mctruth>("tgmugenerator");
//     }


    // Get all of the tracks from the event:
    art::InputTag tracks_tag(_track_producer);
    auto const & tracks
        = ev -> getValidHandle<std::vector <recob::Track> >(tracks_tag);

    if (tracks->size() == 0) {
        if (_verbose)
            std::cout << "No tracks found." << std::endl;
        return false;
    }

    if (_verbose) {
        std::cout << "Number of tracks by " << _track_producer
                  << ": " << tracks->size() << std::endl;
    }

    // Get the associations from tracks (pmtrack)
    // to hits (trajcluster)

    art::InputTag assn_tag(_track_producer);

    art::FindMany<recob::Hit, unsigned short>
    hits_for_track(tracks, *ev, assn_tag);


    std::vector<recob::Hit const*> hits;
    hits_for_track.get(0, hits);

    std::cout << "Number of hits associated with the first track: "
              << hits.size() << std::endl;


    // larlite::event_hit * ev_hit = nullptr;
    // auto track_to_hit_ass
    //     = storage -> find_one_ass(ev_track->id(),
    //                               ev_hit,
    //                               ev_track->name());

//     if (! ev_hit ) {
//         if (_verbose)
//             std::cout << "No hits found." << std::endl;
//         return false;
//     }
//     else if (ev_hit -> size() == 0) {
//         if (_verbose)
//             std::cout << "No hits found." << std::endl;
//         return false;
//     }

//     if (_verbose) {
//         std::cout << "Number of hits by " << ev_hit -> name()
//                   << ": " << ev_hit -> size() << std::endl;
//     }

//     // Now, get the associations from hits (trajcluster)
//     // to clusters (trajcluster)

//     larlite::event_cluster * ev_cluster = nullptr;
//     auto hit_to_cluster_ass
//         = storage -> find_one_ass(ev_hit->id(),
//                                   ev_cluster,
//                                   ev_hit->name());

//     if (! ev_cluster ) {
//         if (_verbose)
//             std::cout << "No clusters found." << std::endl;
//         return false;
//     }
//     else if (ev_cluster -> size() == 0) {
//         if (_verbose)
//             std::cout << "No clusters found." << std::endl;
//         return false;
//     }


//     if (_verbose) {
//         std::cout << "Number of clusters by " << ev_cluster -> name()
//                   << ": " << ev_cluster -> size() << std::endl;
//     }


//     // Getting the clusters associated with a track:
//     // for

//     // Last, get the vertexes associated with the tracks:
//     larlite::event_vertex * ev_vertex = nullptr;
//     auto track_to_vertex_ass
//         = storage -> find_one_ass(ev_track->id(),
//                                   ev_vertex,
//                                   ev_track->name());

//     if (! ev_vertex ) {
//         if (_verbose)
//             std::cout << "No vertexes found." << std::endl;
//         return false;
//     }
//     else if (ev_vertex -> size() == 0) {
//         if (_verbose)
//             std::cout << "No vertexes found." << std::endl;
//         return false;
//     }


//     if (_verbose) {
//         std::cout << "Number of vertexes by " << ev_vertex -> name()
//                   << ": " << ev_vertex -> size() << std::endl;
//     }


//     // Populate some of the variables for the first selection:

//     _n_vertexes = ev_vertex->size();
//     _n_tracks = ev_track->size();
//     _n_clusters = ev_cluster->size();

//     // Loop over the tracks and find the longest track:
//     int longest_track_index = -1;
//     _longest_track = 0;
//     if (_n_tracks > 0) {
//         for (size_t i = 0; i < _n_tracks; i++) {
//             float this_track_length = ev_track->at(i).Length();
//             if (this_track_length > _longest_track) {
//                 _longest_track = this_track_length;
//                 longest_track_index = i;
//             }
//             if (this_track_length > _min_track_length) {
//                 _n_long_tracks ++;
//             }
//         }
//     }
//     _longest_track = _longest_track;

//     // // Next, find the clusters associated with the longest track:
//     // // auto clust_indexes = cluster_to_track_ass[longest_track_index];

//     if (_verbose) {
//         std::cout << "Number of hits associated to longest track: "
//                   << track_to_hit_ass[longest_track_index].size() << std::endl;
//         std::cout << "Number of hits total: "
//                   << ev_hit->size() << std::endl;
//     }
//     _percent_hits_of_longest_track
//         = (1.0 * track_to_hit_ass[longest_track_index].size())
//           / ev_hit->size();


//     _tree->Fill();


    return true;
}

// Decide if this particular event passes the cuts or not
// Designed to be implementable in larsoft without much hassle
bool Stage1Efficiency::filterEvent() {}

bool Stage1Efficiency::finalize() {

    // This function is called at the end of event loop.
    // Do all variable finalization you wish to do here.
    // If you need, you can store your ROOT class instance in the output
    // file. You have an access to the output file through "_fout" pointer.
    //
    // Say you made a histogram pointer h1 to store. You can do this:
    //
    if (_fout) { _fout->cd(); _tree->Write(); }



    return true;
}

bool Stage1Efficiency::isFiducial(const TLorentzVector & vertex, double cut) {
    // Argoneut specific
    if (vertex.X() > 23.5 - cut || vertex.X() < -23.5 + cut) return false;
    if (vertex.Y() > 20 - cut || vertex.Y() < -20 + cut) return false;
    if (vertex.Z() > 90 - cut || vertex.Z() < 0 + cut) return false;

    return true;
}


}
#endif
