import ROOT
from ROOT import gallery, galleryfmwk, larutil
from ROOT import larcv
from ROOT import supera

import sys
import os
import glob

def process_file(_file):


    # Create ana_processor instance
    my_proc = galleryfmwk.ana_processor()

    # Set input root file
    # for _f in f:
        # my_proc.add_input_file(_f)

    my_proc.add_input_file(_file)

    out_dir = '/data/sbnd/dl_larcv/'
    out_name = os.path.basename(_file).rstrip('.root') + '_larcv.root'

    io = larcv.IOManager(larcv.IOManager.kWRITE)
    io.set_out_file(out_dir + out_name)

    # Specify output root file name
    # my_proc.set_ana_output_file(_file.replace('.root', '') + "_larcv.root")
    # my_proc.set_output_file("")

    supera_light = supera.supera_light()
    supera_light.set_io_manager(io)

    supera_light.add_supera_module(supera.SBNDWire())
    supera_light.add_supera_module(supera.SBNDCluster())
    supera_light.add_supera_module(supera.SBNDNeutrino())

    # supera_light.initialize()

    # Attach an analysis unit ... here we use a base class which do
    my_proc.add_process(supera_light)

    my_proc.run()

def main():

    larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kSBND)

#    if len(sys.argv) < 2:
#        print "Error: must include an input file."
#        exit()

#    _file = sys.argv[-1]
#    process_file(_file)

    for _file in glob.glob('/data/sbnd/dl_larsoft/*.root'):
        print _file
        process_file(_file)


if __name__ == '__main__':
  main()
