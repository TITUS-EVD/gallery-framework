#
# This is an example GNUmakefile for my packages
#

# specific names for this package
DICT  = EventDisplay_RawViewerCint
SHLIB = libEventDisplay_RawViewer.so
SOURCES = $(filter-out $(DICT).cxx, $(wildcard *.cxx))
FMWK_HEADERS = LinkDef.h $(DICT).h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
OBJECTS = $(SOURCES:.cxx=.o)

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += -I$(GALLERY_FMWK_COREDIR)
INCFLAGS += -I$(GALLERY_FMWK_USERDEVDIR)/EventDisplay
INCFLAGS += $(shell python-config --includes)
INCFLAGS += -I$(shell python -c "import numpy; print(numpy.get_include())")
INCFLAGS += $(shell gallery-config --includes)
INCFLAGS += $(shell gallery-fmwk-config --includes)

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)



include $(GALLERY_FMWK_BASEDIR)/Makefile/Makefile.${OSNAME}

# set compiler options for ROOT
CXXFLAGS +=  `root-config --cflags`

# call the common GNUmakefile
LDFLAGS += -L$(shell python-config --prefix)/lib/
LDFLAGS += $(shell python-config --ldflags)
LDFLAGS += $(shell gallery-config --libs)
LDFLAGS += $(shell gallery-fmwk-config --libs)
include $(GALLERY_FMWK_BASEDIR)/Makefile/GNUmakefile.CORE

