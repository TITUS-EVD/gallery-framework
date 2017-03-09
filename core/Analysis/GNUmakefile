#
# This is an example GNUmakefile for my packages
#

# specific names for this package
DICT  = gallery_framework_AnalysisDict
SHLIB = libgallery_framework_Analysis.so
SOURCES = $(filter-out $(DICT).cxx, $(wildcard *.cxx))
FMWK_HEADERS = LinkDef.h $(DICT).h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
OBJECTS = $(SOURCES:.cxx=.o)

# include options for this package
INCFLAGS  = -I. 
INCFLAGS += $(shell gallery-config --includes)
INCFLAGS += -I$(GALLERY_FMWK_COREDIR)

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(GALLERY_FMWK_BASEDIR)/Makefile/Makefile.${OSNAME}

# set compiler options for ROOT
CXXFLAGS +=  `root-config --cflags`

# set linker flags
LDFLAGS += -L$(GALLERY_FMWK_LIBDIR) -lgallery_framework_Base
LDFLAGS += -L$(GALLERY_LIB) -lgallery

# call the common GNUmakefile
include $(GALLERY_FMWK_BASEDIR)/Makefile/GNUmakefile.CORE


