#ifndef LARLITE_EXAMPLE_ANA_CXX
#define LARLITE_EXAMPLE_ANA_CXX

#include "example_ana.h"


namespace galleryfmwk {

bool example_ana::initialize() {

    //
    // This function is called in the beggining of event loop
    // Do all variable initialization you wish to do here.
    // If you have a histogram to fill in the event loop, for example_ana,
    // here is a good place to create one on the heap (i.e. "new TH1D").
    //



    return true;
}


bool example_ana::analyze(gallery::Event * ev) {

    // For each file, loop over all events.
    //
    // Determine criteria for rejecting muon like events
    // Save metadata for each event (neutrino pdg, energy, vertex)
    // as well as filter results.


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

    // Get associations for tracks to hits:
    art::InputTag assn_tag(_track_producer);

    art::FindMany<recob::Hit> hits_for_tracks(tracks, *ev, assn_tag);



    // Loop over the tracks, and find the number of hits per track:

    int average_hits = 0;

    size_t index = 0;
    for (auto & track : * tracks) {
        std::vector<recob::Hit const*> hits;

        hits_for_tracks.get(index, hits);

        average_hits += hits.size();

        // Loop over individual hits, if needed:
        // for (auto const& hit : hits) {
        //     std::cout << "Hit start time: " << hit->StartTick << std::endl;
        // }

        index++;
    }

    if (_verbose)
        std::cout << "Average number of associated hits per track for this event: "
                  << average_hits / index << std::endl;


    return true;
}


bool example_ana::finalize() {

    // This function is called at the end of event loop.
    // Do all variable finalization you wish to do here.
    // If you need, you can store your ROOT class instance in the output
    // file. You have an access to the output file through "_fout" pointer.
    //
    // Say you made a histogram pointer h1 to store. You can do this:
    //
    // if (_fout) { _fout->cd(); _tree->Write(); }

    return true;
}


}
#endif
