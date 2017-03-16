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

	//gROOT->SetBatch();

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
	h_nue_like_trk_daughters = new TH1D ("h_nue_like_trk_daughters", "h_nue-like_trk_daughters", 6, 0, 6);
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
	h_nue_shwr_cosmic_closest = new TH1D("h_nue_shwr_cosmic_closest", "h_nue_shwr_cosmic_closest", 60, 0, 60);
	h_nue_shwr_vtx_dist = new TH1D("h_nue_shwr_vtx_dist", "h_nue_shwr_vtx_dist", 60, 0, 120);

	h_nue_shwr_E = new TH1D("h_nue_shwr_E", "h_nue_shwr_E", 100, 0, 2);
	h_nue_shwr_cosmic_closest_vs_E = new TH2D("h_nue_shwr_cosmic_closest_vs_E", "h_nue_shwr_cosmic_closest_vs_E", 100, 0, 2, 60, 0, 120);

	h_cosmic_trk_length = new TH1D ("h_cosmic_trk_length", "h_cosmic_trk_length", 50, 0, 100);
	h_nue_trk_length = new TH1D("h_nue_trk_length", "h_nue_trk_length", 50, 0, 100);

	h_nue_trk_closest = new TH1D("h_nue_trk_closest", "h_nue_trk_closest", 60, 0, 60);
	h_nue_shwr_trk_closest = new Th1D("h_nue_shwr_trk_closest", "h_nue_shwr_trk_closest", 60, 0, 60);

	c1 = new TCanvas();
	c1b = new TCanvas();
	c2 = new TCanvas();
	c3 = new TCanvas();
	c4 = new TCanvas();
	c5 = new TCanvas();
	c6 = new TCanvas();
	c7 = new TCanvas();
	c8 = new TCanvas();
	c9 = new TCanvas();
	c10 = new TCanvas();
	c11 = new TCanvas();
	c12 = new TCanvas();
	c13 = new TCanvas();
	c14 = new TCanvas();
	c15 = new TCanvas();
	c16 = new TCanvas();
	c17 = new TCanvas();
	c17b = new TCanvas();
	c18 = new TCanvas();
	c19 = new TCanvas();
	c19b = new TCanvas();
	c20a = new TCanvas();
	c20b = new TCanvas();
	c21a = new TCanvas();
	c21b = new TCanvas();

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
	art::FindMany<recob::Shower> cosmic_shower_for_pfp(cosmic_pfp, *ev, "pandoraCosmic");

	std::vector < geoalgo::Trajectory_t > track_trajectory_list;
	std::vector < geoalgo::Trajectory_t > cosmic_track_trajectory_list;
	std::vector < geoalgo::Point_t> nue_vertex_list;
	std::vector < geoalgo::Point_t> shwr_vertex_list;
	std::vector < double > shwr_energy_list;

	const int num_pfps = pfparticles.size();
	const int num_cosmics = cosmicpfps.size();

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
					//let's get the vertex associations for the daughters
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

						//let's get the shower associations
						std::vector<recob::Shower const*> shower;
						shower_for_pfp.get(i, shower);
						if(shower.size() == 0)
						{
							std::cout << "No shower for this pfp shower!" << std::endl;
							continue;
						}

						//let's check the distance between the shwr vtx and the nue vtx
						const double dist_x = d_xyz[0] - xyz[0];
						const double dist_y = d_xyz[1] - xyz[1];
						const double dist_z = d_xyz[2] - xyz[2];
						const double dist =
						        sqrt((dist_x * dist_x)+
						             (dist_y * dist_y)+
						             (dist_z * dist_z));
						h_nue_shwr_vtx_dist->Fill(dist);

						//let's get the energy! Energy() - GeV?
						double total_energy = 0;
						const std::vector < double > plane_energy = shower.at(0)->Energy();
						const int best_plane = shower.at(0)->best_plane();
						total_energy = plane_energy.at(best_plane);
						h_nue_shwr_E->Fill(total_energy);
						shwr_energy_list.push_back(total_energy);

						geoalgo::Point_t const shwr_vtx (d_xyz[0], d_xyz[1], d_xyz[2]);
						shwr_vertex_list.push_back(shwr_vtx);

						//let's look at the shower directions
						const double dir_x = shower.at(0)->Direction().X();
						const double dir_y = shower.at(0)->Direction().Y();
						const double dir_z = shower.at(0)->Direction().Z();
						//std::cout << dir_x << ", " << dir_y << ", " << dir_z << std::endl;

					}
					//trk daughters
					if(daughter.PdgCode() == 13)
					{
						trk_daughters++;
						h_nue_like_trk_daughters_xy->Fill(d_xyz[0], d_xyz[1]);
						h_nue_like_trk_daughters_yz->Fill(d_xyz[2], d_xyz[1]);

						std::vector<recob::Track const*> track;
						track_for_pfp.get(i, track);
						if(track.size() == 0)
						{
							std::cout << "No track for this pfp track!" << std::endl;
							continue;
						}
						//let's construct the path of the tracks
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

						//let's get the track length!
						const double track_length = track.at(0)->Length();
						h_nue_trk_length->Fill(track_length);

						//let's look at the track directions
						const double dir_x = track.at(0)->VertexDirection().X();
						const double dir_y = track.at(0)->VertexDirection().Y();
						const double dir_z = track.at(0)->VertexDirection().Z();
						//std::cout << dir_x << ", " << dir_y << ", " << dir_z << std::endl;

					}//end nue track daughters
				}//end nue daughters
				h_nue_like_trk_daughters->Fill(trk_daughters);
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

	//**************************
	//loop over pandora cosmics
	//*************************
	for(std::size_t this_cosmic = 0; this_cosmic < num_cosmics; this_cosmic++)
	{
		//std::cout << "cosmic!" << std::endl;
		auto const cosmic = cosmicpfps.at(this_cosmic);
		if(cosmic.PdgCode() == 13)
		{
			//let's get the cosmic to track associations
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

			//let's get the track length
			const double cosmic_length = cosmic_track.at(0)->Length();
			h_cosmic_trk_length->Fill(cosmic_length);

			//let's get the track energy
			const double cosmic_trk_energy = cosmic_track.at(0)->StartMomentum();

		} //end loop tracks
		if(cosmic.PdgCode() == 11)
		{
			//let's get the cosmic to shower associations
			std::vector<recob::Shower const*> cosmic_shower;
			cosmic_shower_for_pfp.get(this_cosmic, cosmic_shower);
			if(cosmic_shower.size() == 0)
			{
				std::cout << "No shower for pfp!" << std::endl;
				continue;
			}

			//let's get the cosmic shower energy
			double cosmic_shwr_energy;
			const std::vector < double > plane_energy = cosmic_shower.at(0)->Energy();
			const int bestplane = cosmic_shower.at(0)->best_plane();
			const double total_energy = plane_energy.at(bestplane);
		}        //end loop showers
	}        //end loop cosmics

	//Geometry studies!
	std::vector < geoalgo::Point_t > cut_nue_vertex;
	//closest point between nue vertex and cosmic track
	for(int nNue = 0; nNue < nue_vertex_list.size(); nNue++)
	{
		if(!cosmic_track_trajectory_list.empty())
		{
			geoalgo::Point_t nue_vertex = nue_vertex_list.at(nNue);
			double closest_point = _geo_algo_instance.SqDist(nue_vertex, cosmic_track_trajectory_list);
			h_nue_cosmic_closest->Fill(closest_point);
			//*****************************************************************
			//let's see what happens if we remove nue vtx close to tagged cosmic
			//*****************************************************************
			if(closest_point >= 5)
			{
				cut_nue_vertex.push_back(nue_vertex);
			}
		}
		if(!track_trajectory_list.empty())
		{
			geoalgo::Point_t nue_vtx = nue_vertex_list.at(nNue);
			double closest_point = _geo_algo_instance.SqDist(nue_vertex, track_trajectory_list);
			h_nue_trk_closest->Fill(closest_point);
		}
	}
	//closest point between nue shwr vertex and cosmic track
	for (int nE = 0; nE < shwr_vertex_list.size(); nE++)
	{
		if(!cosmic_track_trajectory_list.empty())
		{
			geoalgo::Point_t shwr_vertex = shwr_vertex_list.at(nE);
			double closest_point = _geo_algo_instance.SqDist(shwr_vertex, cosmic_track_trajectory_list);
			h_nue_shwr_cosmic_closest->Fill(closest_point);
			h_nue_shwr_cosmic_closest_vs_E->Fill(shwr_energy_list.at(nE), closest_point);
		}
		if(!track_trajectory_list.empty())
		{
			geoalgo::Point_t shwr_vtx = shwr_vertex_list.at(nE);
			double closest_point = _geo_algo_instance.SqDist(shwr_vertex, track_trajectory_list);
			h_nue_shwr_trk_closest->Fill(closest_point);
		}
	}
	//cut on sqdist to remove small distances - does this change vertex ave. position

	return true;
}

/*

   after cut -> tpc position
   more stats
   shower start point activity - any neutrino tracks nearby?

 */

bool example_ana::finalize() {

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

	/*********************************
	** Histogram Saving and Editing **
	*///******************************
	c1->cd();
	h_nue_like_daughters->Draw("colz");
	h_nue_like_daughters->GetXaxis()->SetTitle("showers");
	h_nue_like_daughters->GetYaxis()->SetTitle("tracks");
	c1->Print("nue-like_daughters.pdf");
	c1b->cd();
	h_nue_like_trk_daughters->Draw();
	h_nue_like_trk_daughters->GetXaxis()->SetTitle("Tracks");
	h_nue_like_trk_daughters->GetYaxis()->SetTitle("Events");
	c1b->Print("nue-like_trk_daughters.pdf");
	c2->cd();
	h_numu_like_daughters->Draw("colz");
	h_numu_like_daughters->GetXaxis()->SetTitle("showers");
	h_numu_like_daughters->GetYaxis()->SetTitle("tracks");
	c2->Print("numu-like_daughters.pdf");

	c3->cd();
	h_nue_fv_cuts->Draw();
	h_nue_fv_cuts->GetXaxis()->SetTitle("Fiducial Volume Cut [cm]");
	h_nue_fv_cuts->GetYaxis()->SetTitle("Events in Volume");
	c3->Print("nue-like_fiducial_volume.pdf");
	c4->cd();
	h_numu_fv_cuts->Draw();
	h_numu_fv_cuts->GetXaxis()->SetTitle("Fiducial Volume Cut [cm]");
	h_numu_fv_cuts->GetYaxis()->SetTitle("Events in Volume");
	c4->Print("numu-like_fiducial_volume.pdf");

	c5->cd();
	h_nue_like_shwr_daughters_xy->Draw("colz");
	h_nue_like_shwr_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_nue_like_shwr_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c5->Print("nue-like_shwr_daughters_xy.pdf");
	c6->cd();
	h_nue_like_shwr_daughters_yz->Draw("colz");
	h_nue_like_shwr_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_nue_like_shwr_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c6->Print("nue-like_shwr_daughters_zy.pdf");
	c7->cd();
	h_nue_like_trk_daughters_xy->Draw("colz");
	h_nue_like_trk_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_nue_like_trk_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c7->Print("nue-like_trk_daughters_xy.pdf");
	c8->cd();
	h_nue_like_trk_daughters_yz->Draw("colz");
	h_nue_like_trk_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_nue_like_trk_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c8->Print("nue-like_trk_daughters_zy.pdf");
	c9->cd();
	h_nue_like_vtx_xy->Draw("colz");
	h_nue_like_vtx_xy->GetXaxis()->SetTitle("x [cm]");
	h_nue_like_vtx_xy->GetYaxis()->SetTitle("y [cm]");
	c9->Print("nue-like_vtx_xy.pdf");
	c10->cd();
	h_nue_like_vtx_yz->Draw("colz");
	h_nue_like_vtx_yz->GetXaxis()->SetTitle("z [cm]");
	h_nue_like_vtx_yz->GetYaxis()->SetTitle("y [cm]");
	c10->Print("nue-like_vtx_zy.pdf");

	c11->cd();
	h_numu_like_shwr_daughters_xy->Draw("colz");
	h_numu_like_shwr_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_numu_like_shwr_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c11->Print("numu-like_shwr_daughters_xy.pdf");
	c12->cd();
	h_numu_like_shwr_daughters_yz->Draw("colz");
	h_numu_like_shwr_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_numu_like_shwr_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c12->Print("numu-like_shwr_dauhters_zy.pdf");
	c13->cd();
	h_numu_like_trk_daughters_xy->Draw("colz");
	h_numu_like_trk_daughters_xy->GetXaxis()->SetTitle("x [cm]");
	h_numu_like_trk_daughters_xy->GetYaxis()->SetTitle("y [cm]");
	c13->Print("numu-like_trk_daughters_xy.pdf");
	c14->cd();
	h_numu_like_trk_daughters_yz->Draw("colz");
	h_numu_like_trk_daughters_yz->GetXaxis()->SetTitle("z [cm]");
	h_numu_like_trk_daughters_yz->GetYaxis()->SetTitle("y [cm]");
	c14->Print("numu-like_trk_daughters_zy.pdf");
	c15->cd();
	h_numu_like_vtx_xy->Draw("colz");
	h_numu_like_vtx_xy->GetXaxis()->SetTitle("x [cm]");
	h_numu_like_vtx_xy->GetYaxis()->SetTitle("y [cm]");
	c15->Print("numu-like_vtx_xy.pdf");
	c16->cd();
	h_numu_like_vtx_yz->Draw("colz");
	h_numu_like_vtx_yz->GetXaxis()->SetTitle("z [cm]");
	h_numu_like_vtx_yz->GetYaxis()->SetTitle("y [cm]");
	c16->Print("numu-like_vtx_zy.pdf");

	c17->cd();
	h_nue_cosmic_closest->Draw();
	h_nue_cosmic_closest->GetXaxis()->SetTitle("Distance [cm]");
	h_nue_cosmic_closest->GetYaxis()->SetTitle("Events");
	c17->Print("nue-like_cosmic_closest.pdf");
	c17b->cd();
	h_nue_shwr_cosmic_closest->Draw();
	h_nue_shwr_cosmic_closest->GetXaxis()->SetTitle("Distance [cm]");
	h_nue_shwr_cosmic_closest->GetYaxis()->SetTitle("Events");
	c17b->Print("nue-like_shwr_cosmic_closest.pdf");

	c18->cd();
	h_nue_shwr_vtx_dist->Draw();
	h_nue_shwr_vtx_dist->GetXaxis()->SetTitle("Distance [cm]");
	h_nue_shwr_vtx_dist->GetYaxis()->SetTitle("Events");
	c18->Print("nue-like_shwr_vtx_distance.pdf");

	c19->cd();
	h_nue_shwr_E->Draw();
	h_nue_shwr_E->GetXaxis()->SetTitle("Total Shower Energy [GeV]");
	h_nue_shwr_E->GetYaxis()->SetTitle("Events");
	c19->Print("nue-like_shwr_E.pdf");
	c19b->cd();
	h_nue_shwr_cosmic_closest_vs_E->Draw("colz");
	h_nue_shwr_cosmic_closest_vs_E->GetXaxis()->SetTitle("Total Shower Energy [GeV]");
	h_nue_shwr_cosmic_closest_vs_E->GetYaxis()->SetTitle("Distance [cm]");
	c19b->Print("nue-like_shwr_vtx_distance_vs_E.pdf");

	c20a->cd();
	h_cosmic_trk_length->Draw();
	h_cosmic_trk_length->GetXaxis()->SetTitle("Length [cm]");
	h_cosmic_trk_length->GetYaxis()->SetTitle("Events");
	c20a->Print("cosmic_trk_length.pdf");
	c20b->cd();
	h_nue_trk_length->Draw();
	h_nue_trk_length->GetXaxis()->SetTitle("Length [cm]");
	h_nue_trk_length->GetYaxis()->SetTitle("Events");
	c20b->Print("nue-like_trk_length.pdf");

	c21a->cd();
	h_nue_trk_closest->Draw();
	h_nue_trk_closest->GetXaxis()->SetTitle("Distance to nearest track [cm]");
	h_nue_trk_closest->GetYaxis()->SetTitle("Events");
	c21a->Print("nue-like_trk_closest");
	c21b->cd();
	h_nue_shwr_trk_closest->Draw();
	h_nue_shwr_trk_closest->GetXaxis()->SetTitle("Distance to nearest track [cm]");
	h_nue_shwr_trk_closest->GetYaxis()->SetTitle("Events");
	c21b->Print("nue-like_shwr_trk_closest");

	return true;
}


}
#endif
