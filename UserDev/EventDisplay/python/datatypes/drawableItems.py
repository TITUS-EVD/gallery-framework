import hit
# import match
import shower
import track
import wire
import cluster
import endpoint2d
import vertex
import mctrack
import mcshower
import spacepoint
import simch
import opflash
import seed
import pfpart
import neutrino

# This is the class that maintains the list of drawable items.
# If your class isn't here, it can't be drawn
import collections


class drawableItems(object):

    """This class exists to enumerate the drawableItems"""
    # If you make a new drawing class, add it here

    def __init__(self):
        super(drawableItems, self).__init__()
        # items are stored as pointers to the classes (not instances)
        self._drawableClasses = collections.OrderedDict()
        self._drawableClasses.update({'Hit': [hit.hit,"recob::Hit"]})
        self._drawableClasses.update({'Cluster': [cluster.cluster,"recob::Cluster"]})
        # self._drawableClasses.update({'Match': [match.match,"pfpart"]})
        self._drawableClasses.update({'Shower': [shower.shower,"recob::Shower"]})
        self._drawableClasses.update({'Track': [track.track,"recob::Track"]})
        # # self._drawableClasses.update({'Neutrino': [neutrino.neutrino,"ass"]})
        self._drawableClasses.update({'Endpoint 2D': [endpoint2d.endpoint2d,"recob::EndPoint2D"]})
        self._drawableClasses.update({'Vertex': [vertex.vertex,"recob::Vertex"]})
        self._drawableClasses.update({'SPS': [spacepoint.spacepoint,"recob::SpacePoint"]})
        self._drawableClasses.update({'MCTrack': [mctrack.mctrack,"sim::MCTrack"]})

    def getListOfTitles(self):
        return self._drawableClasses.keys()

    def getListOfItems(self):
        return zip(*self._drawableClasses.values())[1]

    def getDict(self):
        return self._drawableClasses


try:
    import pyqtgraph.opengl as gl
    class drawableItems3D(object):

        """This class exists to enumerate the drawableItems in 3D"""
        # If you make a new drawing class, add it here

        def __init__(self):
            super(drawableItems3D, self).__init__()
            # items are stored as pointers to the classes (not instances)
            self._drawableClasses = collections.OrderedDict()
            self._drawableClasses.update({'Spacepoints': [spacepoint.spacepoint3D,"recob::SpacePoint"]})
            self._drawableClasses.update({'PFParticle': [pfpart.pfpart3D,"recob::PFParticle"]})
            self._drawableClasses.update({'Seed': [seed.seed3D,"recob::Seed"]})
            self._drawableClasses.update({'Vertex': [vertex.vertex3D,"recob::Vertex"]})
            self._drawableClasses.update({'Shower': [shower.shower3D,"recob::Shower"]})
            self._drawableClasses.update({'Track': [track.track3D,"recob::Track"]})
            self._drawableClasses.update({'Opflash': [opflash.opflash3D,"recob::OpFlash"]})
            self._drawableClasses.update({'MCTrack': [mctrack.mctrack3D,"sim::MCTrack"]})
            self._drawableClasses.update({'MCShower': [mcshower.mcshower3D,"sim::MCShower"]})
            self._drawableClasses.update({'Simch': [simch.simch3D,"sim::SimChannel"]})

        def getListOfTitles(self):
            return self._drawableClasses.keys()

        def getListOfItems(self):
            return zip(*self._drawableClasses.values())[1]

        def getDict(self):
            return self._drawableClasses



except:
    pass

