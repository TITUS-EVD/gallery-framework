import ROOT
from ROOT import gallery, galleryfmwk, larutil, argoana

import sys

samples = dict()
samples.update({'nue' : "/data/argoneut/nue_xsec_files/nue/genie_nuecc_genie_nuecc_mc_9985687_56_c9d7a7ab-78f1-4aa2-9e18-d080d0bfd2e8_reco0_16701457_200.root"})
# samples.update({'numu' : "/data/argoneut/nue_xsec_files/numu_cc_sim/numu_stage0_larlite.root"})
# samples.update({'xing' : "/data/argoneut/nue_xsec_files/xing_muons/xing_muons_stage0_larlite.root"})
# samples.update({'anu_sim' : "/data/argoneut/nue_xsec_files/anu_sim/anu_sim_stage0_larlite.root"})


def process_sample(_type, _file):


    # Create ana_processor instance
    my_proc = galleryfmwk.ana_processor()

    # Set input root file
    # for _f in f:
        # my_proc.add_input_file(_f)

    my_proc.add_input_file(_file)


    # Specify output root file name
    my_proc.set_ana_output_file(_file.replace('.root', '') + "_ana.root")
    # my_proc.set_output_file("")

    filterModule = argoana.Stage1Efficiency()
    filterModule.setTrackProducer("pmtrack")
    filterModule.setClusterProducer("trajcluster")
    filterModule.setMinTrackLength(15)
    filterModule.setVerbose(True)

    # Attach an analysis unit ... here we use a base class which do
    my_proc.add_process(filterModule)

    my_proc.run(10)

def main():

    larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kArgoNeuT)

    for _type in samples:
        process_sample(_type, samples[_type])


if __name__ == '__main__':
  main()