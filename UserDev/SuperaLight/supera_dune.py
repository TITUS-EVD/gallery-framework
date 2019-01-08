import sys
import os
import glob
import uuid
import argparse

from ROOT import gallery, galleryfmwk, larutil
from ROOT import larcv
from ROOT import supera
from ROOT import TFile

def process_files(file_list):


    # Create ana_processor instance
    my_proc = galleryfmwk.ana_processor()

    # Set input root file
    for _f in file_list:
        my_proc.add_input_file(_f)

    out_dir = './'
    # out_name = 'dune_supera_larcv_{0}.root'.format(uuid.uuid4())
    out_name = 'dune_supera_larcv_dev.root'

    io = larcv.IOManager(larcv.IOManager.kWRITE)
    io.set_out_file(out_dir + out_name)


    supera_light = supera.supera_light()
    supera_light.set_io_manager(io)

    # supera_light.add_supera_module(supera.DUNEWire())
    supera_light.add_supera_module(supera.DUNECluster())
    supera_light.add_supera_module(supera.DUNENeutrino())

    # supera_light.initialize()

    # Attach an analysis unit ... here we use a base class which do
    my_proc.add_process(supera_light)

    my_proc.run()

    # Open the output file and find out how many events were processed:
    f = TFile.Open(out_name)
    tree = f.GetListOfKeys()[0].GetName()
    print "Number of entries processed: {0}".format(f.Get(tree).GetEntries())
    print "Output file name: {0}".format(out_name)


def main():



    parser = argparse.ArgumentParser(description='Gallery based larsoft to larcv converter.')
    # geom = parser.add_mutually_exclusive_group()
    # geom.add_argument('--DUNE',
    #                   action='store_true',
    #                   help="Run with the DUNE Geometry")
    parser.add_argument('--files', nargs='+', help="Optional input file to use")

    args = parser.parse_args()

    larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kSBND)

    file_list = args.files
    process_files(file_list)
    # for _file in glob.glob('/data/DUNE/dl_larsoft/*.root'):
    #     print _file
    #     process_file(_file)


if __name__ == '__main__':
  main()
