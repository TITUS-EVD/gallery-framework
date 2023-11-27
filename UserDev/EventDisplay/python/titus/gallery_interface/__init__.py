# from .event import event, manager
from titus.gallery_interface.geometry import argoneut, microboone, lariat
from titus.gallery_interface.gallery_interface import GalleryInterface

# from .liveevdmanager import live_evd_manager_2D
try:
    import pyqtgraph.opengl as gl
    from .evdmanager import evd_manager_3D
except:
    pass
