import sys
import os
import glob
import uuid
import argparse

from ROOT import gallery, galleryfmwk, larutil
from ROOT import larcv
from ROOT import supera


def process_files(file_list):


    # Create ana_processor instance
    my_proc = galleryfmwk.ana_processor()

    # Set input root file
    for _f in file_list:
        my_proc.add_input_file(_f)

    out_dir = './'
    out_name = 'sbnd_supera_larcv_{0}.root'.format(uuid.uuid4())

    io = larcv.IOManager(larcv.IOManager.kWRITE)
    io.set_out_file(out_dir + out_name)


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



    parser = argparse.ArgumentParser(description='Gallery based larsoft to larcv converter.')
    # geom = parser.add_mutually_exclusive_group()
    # geom.add_argument('--sbnd',
    #                   action='store_true',
    #                   help="Run with the SBND Geometry")
    parser.add_argument('--files', nargs='+', help="Optional input file to use")

    args = parser.parse_args()

    larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kSBND)


   _file = sys.argv[-1]
#    process_file(_file)

    for _file in glob.glob('/data/sbnd/dl_larsoft/*.root'):
        print _file
        process_file(_file)


if __name__ == '__main__':
  main()
