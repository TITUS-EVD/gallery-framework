import ROOT
from ROOT import gallery
from ROOT import evd

a = evd.DrawWire()
a.initialize()
a.setYDimension(9600,0)
a.setYDimension(9600,1)
a.setYDimension(9600,2)


files = ROOT.vector(ROOT.string)()
files.push_back("/data/uboone/pulser_data/10V-pulses/TestRun-2017_2_21_14_10_54-0010153-00000_20170222T014119_mucs_20170222T021127_merged.root")
b = gallery.Event(files)

a.analyze(b)

c = a.getArrayByPlane(0)

print type(c)
print c.shape

print b