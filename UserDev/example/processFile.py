import ROOT
from ROOT import gallery, galleryfmwk, larutil

import sys


def process_file(_file):

    # Create ana_processor instance
    my_proc = galleryfmwk.ana_processor()

    myfile = open(_file, 'r')

    # Set input root file
    for _f in myfile:
        # print _f
        my_proc.add_input_file(_f[:-1])

    # print my_proc._input_files
    # exit()

    # my_proc.add_input_file(_file)

    # make root run in batch mode
    ROOT.gROOT.SetBatch(1)

    # Specify output root file name
    my_proc.set_ana_output_file(_file.replace('.root', '') + "_ana.root")
    # my_proc.set_output_file("")

    opFilterModule = galleryfmwk.opFilter()
    opFilterModule.setTrackProducer("pandoraNuKHit")
    opFilterModule.setShowerProducer("showerrecopandora")
    opFilterModule.setFlashProducer("simpleFlashBeam")

    exampleModule = galleryfmwk.example_ana()
    exampleModule.setTrackProducer("pandoraNuKHit")
    exampleModule.setShowerProducer("showerrecopandora")
    exampleModule.setNearestCutDist(5)
    exampleModule.fiducial_volume_x_right(0)
    exampleModule.fiducial_volume_x_left(0)
    exampleModule.fiducial_volume_y_up(0)
    exampleModule.fiducial_volume_y_down(0)
    exampleModule.fiducial_volume_z_back(0)
    exampleModule.fiducial_volume_z_front(0)
    exampleModule.setVerbose(False)

    # Attach an analysis unit ... here we use a base class which do
    # my_proc.add_process(opFilterModule)
    my_proc.add_process(exampleModule)

    my_proc.run()


def main():

    if len(sys.argv) < 2:
        print "Error: must include an input file."
        exit()

    _file = sys.argv[-1]
    process_file(_file)


if __name__ == '__main__':
    main()
