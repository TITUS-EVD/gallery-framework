import sys
import os
import glob
import uuid
import argparse

from ROOT import gallery, galleryfmwk, larutil
from ROOT import supera

def process_files(f):
    # a = supera.SuperaModuleBase()


    # Create ana_processor instance
    my_proc = galleryfmwk.ana_processor()

    # Set input root file
    my_proc.add_input_file(f)

    base_name = os.path.basename(f)

    out_dir = './'
    out_name = out_dir + base_name.replace(".root", "_larcv.h5")

    supera_light = supera.supera_light()
    supera_light.set_output_file(out_name)

    # Attach an analysis unit ... here we use a base class which do
    my_proc.add_process(supera_light)

    my_proc.run()



def main():



    parser = argparse.ArgumentParser(description='Gallery based larsoft to larcv converter.')

    parser.add_argument('--file','-f', help="Input file to use")

    args = parser.parse_args()

    larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kSBND)

    process_files(args.file)
    # for _file in glob.glob('/data/sbnd/dl_larsoft/*.root'):
    #     print _file
    #     process_file(_file)


if __name__ == '__main__':
  main()
