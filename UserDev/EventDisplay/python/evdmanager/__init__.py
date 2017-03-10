from .event import event, manager
from .geometry import argoneut, microboone, lariat
from .evdmanager import evd_manager_2D
try:
    import pyqtgraph.opengl as gl
    from .evdmanager import evd_manager_3D
except:
    pass
