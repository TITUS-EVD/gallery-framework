#ifndef LARLITE_EXAMPLE_ANA_CXX
#define LARLITE_EXAMPLE_ANA_CXX

#include "example_ana.h"

namespace galleryfmwk {

bool example_ana::inFV(double x_vtx, double y_vtx, double z_vtx, double fromWall)
{
	//is vertex in given FV?
	const double x_boundary1 = 0;
	const double x_boundary2 = 256.35;
	const double y_boundary1 = -116.5;
	const double y_boundary2 = 116.5;
	const double z_boundary1 = 0;
	const double z_boundary2 = 1036.8;

	if(x_vtx > x_boundary1 + fromWall &&
	   x_vtx < x_boundary2 - fromWall &&
	   y_vtx > y_boundary1 + fromWall &&
	   y_vtx < y_boundary2 - fromWall &&
	   z_vtx > z_boundary1 + fromWall &&
	   z_vtx < z_boundary2 - fromWall)
	{
		return true;
	}
	return false;

}



bool example_ana::initialize() {

	//
	// This function is called in the beggining of event loop
	// Do all variable initialization you wish to do here.
	// If you have a histogram to fill in the event loop, for example_ana,
	// here is a good place to create one on the heap (i.e. "new TH1D").
	//

	x_boundary1 = 0;
	x_boundary2 = 256.35;
	y_boundary1 = -116.5;
	y_boundary2 = 116.5;
	z_boundary1 = 0;
	z_boundary2 = 1036.8;

	//fiducial cut beyond TPC
	fromWall = 0;

	num_cosmic = 0;
	num_primary_pfp = 0;
	num_nue = 0;
	num_numu = 0;

	//max fv cut for plotting (in cm)
	fv_cut_max = 50;
	h_nue_fv_cuts = new TH1D("h_nue_fv_cuts", "h_nue_fv_cuts", 50, 0, 50);
	h_numu_fv_cuts = new TH1D("h_numu_fv_cuts", "h_numu_fv_cuts", 50, 0, 50);

	h_nue_like_daughters = new TH2D("h_nue_like_daughters", "h_nue-like_daughters", 6, 0, 6, 6, 0, 6);
	h_numu_like_daughters = new TH2D("h_numu_like_daughters", "h_numu-like_daughters", 6, 0, 6, 6, 0, 6);

	h_nue_like_shwr_daughters_xy = new TH2D("h_nue_like_shwr_daughters_xy", "h_nue_like_shwr_daughters_xy", 52, 0, 260, 60, -120, 120);
	h_nue_like_shwr_daughters_yz = new TH2D("h_nue_like_shwr_daughters_yz", "h_nue_like_shwr_daughters_yz", 50, 0, 1050, 60, -120, 120);
	h_nue_like_trk_daughters_xy = new TH2D("h_nue_like_trk_daughters_xy", "h_nue_like_trk_daughters_xy", 52, 0, 260, 60, -120, 120);
	h_nue_like_trk_daughters_yz = new TH2D("h_nue_like_trk_daughters_yz", "h_nue_like_trk_daughters_yz", 50, 0, 1050, 60, -120, 120);
	h_nue_like_vtx_xy = new TH2D("h_nue_like_vtx_xy", "h_nue_like_vtx_xy", 52, 0, 260, 60, -120, 120);
	h_nue_like_vtx_yz = new TH2D("h_nue_like_vtx_yz", "h_nue_like_vtx_yz", 50, 0, 1050, 60, -120, 120);

	h_numu_like_shwr_daughters_xy = new TH2D("h_numu_like_shwr_daughters_xy", "h_numu_like_shwr_daughters_xy", 50, 0, 260, 60, -120, 120);
	h_numu_like_shwr_daughters_yz = new TH2D("h_numu_like_shwr_daughters_yz", "h_numu_like_shwr_daughters_yz", 50, 0, 1050, 60, -120, 120);
	h_numu_like_trk_daughters_xy = new TH2D("h_numu_like_trk_daughters_xy", "h_numu_like_trk_daughters_xy", 50, 0, 260, 60, -120, 120);
	h_numu_like_trk_daughters_yz = new TH2D("h_numu_like_trk_daughters_yz", "h_numu_like_trk_daughters_yz", 50, 0, 1050, 60, -120, 120);
	h_numu_like_vtx_xy = new TH2D("h_numu_like_vtx_xy", "h_numu_like_vtx_xy", 50, 0, 260, 60, -120, 120);
	h_numu_like_vtx_yz = new TH2D("h_numu_like_vtx_yz", "h_numu_like_vtx_yz", 50, 0, 1050, 60, -120, 120);

	h_nue_cosmic_closest = new TH1D("h_nue_cosmic_closest", "h_nue_cosmic_closest", 60, 0, 60);

	return true;
}


bool example_ana::analyze(gallery::Event * ev) {

	num_cosmic++;

	// For each file, loop over all events.
	//
	// Determine criteria for rejecting muon like events
	// Save metadata for each event (neutrino pdg, energy, vertex)
	// as well as filter results.


	// Get all of the tracks from the event:
	art::InputTag tracks_tag(_track_producer);
	art::InputTag showers_tag(_shower_producer);
	auto const & tracks
	        = ev->getValidHandle<std::vector <recob::Track> >(tracks_tag);
	auto const & showers
	        = ev->getValidHandle<std::vector <recob::Shower> >(showers_tag);
	auto const & pfp
	        = ev->getValidHandle<std::vector <recob::PFParticle> > ("pandoraNu");
	auto const & pfparticles(*pfp);
	auto const & cosmic_pfp
	        = ev->getValidHandle<std::vector < recob::PFParticle> > ("pandoraCosmic");
	auto const & cosmicpfps(*cosmic_pfp);

	art::FindMany<recob::Vertex> vertex_for_pfp(pfp, *ev, "pandoraNu");
	art::FindMany<recob::Track> track_for_pfp(pfp, *ev, "pandoraNu");
	art::FindMany<recob::Shower> shower_for_pfp(pfp, *ev, "pandoraNu");

	art::FindMany<recob::Track> cosmic_track_for_pfp(cosmic_pfp, *ev, "pandoraCosmic");

	std::vector < geoalgo::Trajectory_t > track_trajectory_list;
	std::vector < geoalgo::Trajectory_t > cosmic_track_trajectory_list;
	std::vector < geoalgo::Point_t> nue_vertex_list;

	const int num_pfps = pfparticles.size();
	const int num_cosmics = cosmicpfps.size();
	//const int num_tracks = tracks->size();
	//const int num_showers = showers->size();

	//pfp loop
	for(std::size_t this_pfp = 0; this_pfp < num_pfps; this_pfp++)
	{
		//******************************
		//check for reconstructed vertex
		//******************************
		std::vector<recob::Vertex const*> vertex;
		vertex_for_pfp.get(this_pfp,vertex);
		if(vertex.size() ==0 )
		{
			std::cout << "No vertex association found!" << std::endl;
			return false;
		}
		//get vertex vector
		double xyz [3];
		vertex.at(0)->XYZ(xyz);
		if( inFV(xyz[0], xyz[1], xyz[2], fromWall) == false)
		{
			//std::cout << "Reco vertex outside fiducial volume!" << std::endl;
			//std::cout << "Skipping event..." << std::endl;
			continue;
		}

		//************************************
		//check if pfp is neutrino-like object
		//************************************
		auto const pfparts = pfparticles.at(this_pfp);
		int shwr_daughters = 0;
		int trk_daughters = 0;
		if(pfparts.IsPrimary() == true)
		{
			num_primary_pfp++;
			//nues!
			if(pfparts.PdgCode() == 12)
			{
				num_nue++;
				geoalgo::Point_t const nue_vtx (xyz[0], xyz[1], xyz[2]);
				nue_vertex_list.push_back(nue_vtx);

				h_nue_like_vtx_xy->Fill(xyz[0], xyz[1]);
				h_nue_like_vtx_yz->Fill(xyz[2], xyz[1]);

				auto const daughters = pfparts.Daughters();
				for(std::size_t const i : daughters)
				{
					auto const daughter = pfparticles.at(i);
					std::vector<recob::Vertex const*> d_vertex;
					vertex_for_pfp.get(i, d_vertex);
					if(d_vertex.size() == 0 )
					{
						std::cout << "No vertex association found for daughter!" << std::endl;
						return false;
					}
					//get vertex vector
					double d_xyz [3];
					d_vertex.at(0)->XYZ(d_xyz);

					//shwr daughters
					if(daughter.PdgCode() == 11)
					{
						shwr_daughters++;
						h_nue_like_shwr_daughters_xy->Fill(d_xyz[0], d_xyz[1]);
						h_nue_like_shwr_daughters_yz->Fill(d_xyz[2], d_xyz[1]);

						std::vector<recob::Shower const*> shower;
						shower_for_pfp.get(i, shower);
						//shower.at(0)->Vertex().X();

					}
					//trk daughters
					if(daughter.PdgCode() == 13)
					{
						trk_daughters++;
						h_nue_like_trk_daughters_xy->Fill(d_xyz[0], d_xyz[1]);
						h_nue_like_trk_daughters_yz->Fill(d_xyz[2], d_xyz[1]);

						std::vector<recob::Track const*> track;
						track_for_pfp.get(i, track);
						std::vector<geoalgo::Point_t> track_path;
						for(int pts = 0; pts < track.at(0)->NPoints(); pts++)
						{
							geoalgo::Point_t const track_point (
							        track.at(0)->LocationAtPoint(pts).X(),
							        track.at(0)->LocationAtPoint(pts).Y(),
							        track.at(0)->LocationAtPoint(pts).Z());
							track_path.push_back(track_point);
						}
						const geoalgo::Trajectory_t trj = track_path;
						if(!track_path.empty()) {track_path.clear(); }
						track_trajectory_list.push_back(trj);
					}

				}
				h_nue_like_daughters->Fill(shwr_daughters, trk_daughters);
				for(int fv_cut = 0; fv_cut < fv_cut_max; fv_cut++)
				{
					if(inFV(xyz[0], xyz[1], xyz[2], fv_cut) == true)
					{
						h_nue_fv_cuts->Fill(fv_cut);
					}
				}
			}
			//numus!
			if(pfparts.PdgCode() == 14)
			{
				num_numu++;
				h_numu_like_vtx_xy->Fill(xyz[0], xyz[1]);
				h_numu_like_vtx_yz->Fill(xyz[2], xyz[1]);

				auto const daughters = pfparts.Daughters();
				for(std::size_t const i : daughters)
				{
					auto const daughter = pfparticles.at(i);
					std::vector<recob::Vertex const*> d_vertex;
					vertex_for_pfp.get(i, d_vertex);
					if(d_vertex.size() ==0 )
					{
						std::cout << "No vertex association found for daughter!" << std::endl;
						return false;
					}
					//get vertex vector
					double d_xyz [3];
					d_vertex.at(0)->XYZ(d_xyz);

					//shwr daughters
					if(daughter.PdgCode() == 11)
					{
						shwr_daughters++;
						h_numu_like_shwr_daughters_xy->Fill(d_xyz[0], d_xyz[1]);
						h_numu_like_shwr_daughters_yz->Fill(d_xyz[2], d_xyz[1]);
					}
					//trk daughters
					if(daughter.PdgCode() == 13)
					{
						trk_daughters++;
						h_numu_like_trk_daughters_xy->Fill(d_xyz[0], d_xyz[1]);
						h_numu_like_trk_daughters_yz->Fill(d_xyz[2], d_xyz[1]);
					}

				}
				h_numu_like_daughters->Fill(shwr_daughters, trk_daughters);
				for(int fv_cut = 0; fv_cut < fv_cut_max; fv_cut++)
				{
					if(inFV(xyz[0], xyz[1], xyz[2], fv_cut) == true)
					{
						h_numu_fv_cuts->Fill(fv_cut);
					}
				}
			}
		}//end if nu-like
	}//end loop pfps

	//loop over pandora cosmics
	for(std::size_t this_cosmic = 0; this_cosmic < num_cosmics; this_cosmic++)
	{
		auto const cosmic = cosmicpfps.at(this_cosmic);
		if(cosmic.PdgCode() == 13)
		{
			std::vector<recob::Track const*> cosmic_track;
			cosmic_track_for_pfp.get(this_cosmic, cosmic_track);
			if(cosmic_track.size() == 0)
			{
				std::cout << "No track for pfp!" << std::endl;
				continue;
			}
			std::vector<geoalgo::Point_t> cosmic_track_path;
			for(int pts = 0; pts < cosmic_track.at(0)->NPoints(); pts++)
			{
				geoalgo::Point_t const cosmic_track_point (
				        cosmic_track.at(0)->LocationAtPoint(pts).X(),
				        cosmic_track.at(0)->LocationAtPoint(pts).Y(),
				        cosmic_track.at(0)->LocationAtPoint(pts).Z());
				cosmic_track_path.push_back(cosmic_track_point);
			}
			const geoalgo::Trajectory_t trj = cosmic_track_path;
			if(!cosmic_track_path.empty()) {cosmic_track_path.clear(); }
			cosmic_track_trajectory_list.push_back(trj);
		}

	}

	//Geometry studies!
	//std::vector < double > closest_points;
	for(int nNue = 0; nNue < nue_vertex_list.size(); nNue++)
	{
		if(!cosmic_track_trajectory_list.empty())
		{
			geoalgo::Point_t nue_vertex = nue_vertex_list.at(nNue);
			double closest_point = _geo_algo_instance.SqDist(nue_vertex, cosmic_track_trajectory_list);
			//std::cout << "Closest Point Between Nue-like vertex and Cosmic-like track: " << closest_point << std::endl;
			//closest_points.push_back(closest_point);
			h_nue_cosmic_closest->Fill(closest_point);
		}
	}


	// Get associations for tracks to hits:
	art::InputTag assn_tag(_track_producer);

	art::FindMany<recob::Hit> hits_for_tracks(tracks, *ev, assn_tag);

	// Loop over the tracks, and find the number of hits per track:

	int average_hits = 0;

	std::size_t index = 0;
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

	// if (_verbose && index > 0)
	//      // std::cout << "Average number of associated hits per track for this event: "
	//      //           << average_hits / index << std::endl;


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
	std::cout << "Number of events: " << num_cosmic << std::endl;
	std::cout << "Number of primary pfps: " << num_primary_pfp << std::endl;
	std::cout << "Number of nue-like: " << num_nue << std::endl;
	std:: cout << "Number of numu-like: " << num_numu << std::endl;

	TCanvas * c1 = new TCanvas();
	c1->cd();
	h_nue_like_daughters->Draw("colz");
	h_nue_like_daughters->GetXaxis()->SetTitle("showers");
	h_nue_like_daughters->GetYaxis()->SetTitle("tracks");
	c1->Print("nue-like_daughters.pdf");
	TCanvas * c2 = new TCanvas();
	c2->cd();
	h_numu_like_daughters->Draw("colz");
	h_numu_like_daughters->GetXaxis()->SetTitle("showers");
	h_numu_like_daughters->GetYaxis()->SetTitle("tracks");
	c2->Print("numu-like_daughters.pdf");

	TCanvas * c3 = new TCanvas();
	c3->cd();
	h_nue_fv_cuts->Draw();
	h_nue_fv_cuts->GetXaxis()->SetTitle("Fiducial Volume Cut [cm]");
	h_nue_fv_cuts->GetYaxis()->SetTitle("Events in Volume");
	c3->Print("nue-like_fiducial_volume.pdf");
	TCanvas * c4 = new TCanvas();
	c4->cd();
	h_numu_fv_cuts->Draw();
	h_numu_fv_cuts->GetXaxis()->SetTitle("Fiducial Volume Cut [cm]");
	h_numu_fv_cuts->GetYaxis()->SetTitle("Events in Volume");
	c4->Print("numu-like_fiducial_volume.pdf");

	TCanvas * c5 = new TCanvas();
	c5->cd();
	h_nue_like_shwr_daughters_xy->Draw("colz");
	h_nue_like_shwr_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_nue_like_shwr_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c5->Print("nue-like_shwr_daughters_xy.pdf");
	TCanvas * c6 = new TCanvas();
	c6->cd();
	h_nue_like_shwr_daughters_yz->Draw("colz");
	h_nue_like_shwr_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_nue_like_shwr_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c6->Print("nue-like_shwr_daughters_zy.pdf");
	TCanvas * c7 = new TCanvas();
	c7->cd();
	h_nue_like_trk_daughters_xy->Draw("colz");
	h_nue_like_trk_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_nue_like_trk_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c7->Print("nue-like_trk_daughters_xy.pdf");
	TCanvas * c8 = new TCanvas();
	c8->cd();
	h_nue_like_trk_daughters_yz->Draw("colz");
	h_nue_like_trk_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_nue_like_trk_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c8->Print("nue-like_trk_daughters_zy.pdf");
	TCanvas * c9 = new TCanvas();
	c9->cd();
	h_nue_like_vtx_xy->Draw("colz");
	h_nue_like_vtx_xy->GetXaxis()->SetTitle("x [cm]");
	h_nue_like_vtx_xy->GetYaxis()->SetTitle("y [cm]");
	c9->Print("nue-like_vtx_xy.pdf");
	TCanvas * c10 = new TCanvas();
	c10->cd();
	h_nue_like_vtx_yz->Draw("colz");
	h_nue_like_vtx_yz->GetXaxis()->SetTitle("z [cm]");
	h_nue_like_vtx_yz->GetYaxis()->SetTitle("y [cm]");
	c10->Print("nue-like_vtx_zy.pdf");

	TCanvas * c11 = new TCanvas();
	c11->cd();
	h_numu_like_shwr_daughters_xy->Draw("colz");
	h_numu_like_shwr_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_numu_like_shwr_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c11->Print("numu-like_shwr_daughters_xy.pdf");
	TCanvas * c12 = new TCanvas();
	c12->cd();
	h_numu_like_shwr_daughters_yz->Draw("colz");
	h_numu_like_shwr_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_numu_like_shwr_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c12->Print("numu-like_shwr_dauhters_zy.pdf");
	TCanvas * c13 = new TCanvas();
	c13->cd();
	h_numu_like_trk_daughters_xy->Draw("colz");
	h_numu_like_trk_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_numu_like_trk_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c13->Print("numu-like_trk_daughters_xy.pdf");
	TCanvas * c14 = new TCanvas();
	c14->cd();
	h_numu_like_trk_daughters_yz->Draw("colz");
	h_numu_like_trk_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_numu_like_trk_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c14->Print("numu-like_trk_daughters_zy.pdf");
	TCanvas * c15 = new TCanvas();
	c15->cd();
	h_numu_like_vtx_xy->Draw("colz");
	h_numu_like_vtx_xy->GetXaxis()->SetTitle("x [cm]");
	h_numu_like_vtx_xy->GetYaxis()->SetTitle("y [cm]");
	c15->Print("numu-like_vtx_xy.pdf");
	TCanvas * c16 = new TCanvas();
	c16->cd();
	h_numu_like_vtx_yz->Draw("colz");
	h_numu_like_vtx_yz->GetXaxis()->SetTitle("z [cm]");
	h_numu_like_vtx_yz->GetYaxis()->SetTitle("y [cm]");
	c16->Print("numu-like_vtx_zy.pdf");

	TCanvas * c17 = new TCanvas();
	c17->cd();
	h_nue_cosmic_closest->Draw();
	h_nue_cosmic_closest->GetXaxis()->SetTitle("Distance [cm]");
	h_nue_cosmic_closest->GetYaxis()->SetTitle("Events");
	c17->Print("nue-like_cosmic_closest.pdf");

	return true;
}


}
#endif
