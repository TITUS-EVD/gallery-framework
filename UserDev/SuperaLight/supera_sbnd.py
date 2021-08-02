import sys
import os
import glob
import uuid
import argparse

from ROOT import gallery, galleryfmwk, larutil
from ROOT import supera
from ROOT import TFile

def process_files(file_list):
    # a = supera.SuperaModuleBase()


    # Create ana_processor instance
    my_proc = galleryfmwk.ana_processor()

    # Set input root file
    for _f in file_list:
        my_proc.add_input_file(_f)

    out_dir = './'
    out_name = 'sbnd_supera_larcv.h5'.format(uuid.uuid4())

    supera_light = supera.supera_light()
    supera_light.set_output_file(out_name)

    # module_list = [supera.SBNDRawDigit()]
    # print("Printing name")
    # print(module_list[0].name())
    # print("done")
    # supera_light.add_supera_module(module_list[0])
    # supera_light.add_supera_module(supera.SBNDCluster())
    # supera_light.add_supera_module(supera.SBNDNeutrino())

    # supera_light.initialize()

    # Attach an analysis unit ... here we use a base class which do
    my_proc.add_process(supera_light)

    my_proc.run(10)

    # # Open the output file and find out how many events were processed:
    # f = larcv.IOManager()
    # tree = f.GetListOfKeys()[0].GetName()
    # print("Number of entries processed: {0}".format(f.Get(tree).GetEntries()))
    # print("Output file name: {0}".format(out_name))


def main():



    parser = argparse.ArgumentParser(description='Gallery based larsoft to larcv converter.')
    # geom = parser.add_mutually_exclusive_group()
    # geom.add_argument('--sbnd',
    #                   action='store_true',
    #                   help="Run with the SBND Geometry")
    parser.add_argument('--files', nargs='+', help="Optional input file to use")

    args = parser.parse_args()

    larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kSBND)

    file_list = args.files
    process_files(file_list)
    # for _file in glob.glob('/data/sbnd/dl_larsoft/*.root'):
    #     print _file
    #     process_file(_file)


if __name__ == '__main__':
  main()
