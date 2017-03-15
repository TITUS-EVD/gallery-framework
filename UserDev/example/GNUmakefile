#
# This is an example GNUmakefile for my packages
#

# specific names for this package
DICT  = gallery_framework_exampleCint
SHLIB = libgallery_framework_example.so
SOURCES = $(filter-out $(DICT).cxx, $(wildcard *.cxx))
FMWK_HEADERS = LinkDef.h $(DICT).h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
OBJECTS = $(SOURCES:.cxx=.o)

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell gallery-config --includes)
INCFLAGS += $(shell gallery-fmwk-config --includes)
INCFLAGS += -I$(GALLERY_FMWK_USERDEVDIR)/BasicTool

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

# call kernel specific compiler setup
include $(GALLERY_FMWK_BASEDIR)/Makefile/Makefile.${OSNAME}

# call the common GNUmakefile
LDFLAGS += $(shell gallery-config --libs)
LDFLAGS += $(shell gallery-fmwk-config --libs)
LDFLAGS += -L$(GALLERY_FMWK_LIBDIR) -lBasicTool_GeoAlgo


include $(GALLERY_FMWK_BASEDIR)/Makefile/GNUmakefile.CORE


