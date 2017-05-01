
# specific names for this package
DICT  = uboone_NoiseFilterCint
SHLIB = libub_noise_filter_NoiseFilter.so
SOURCES = $(filter-out $(DICT).cxx, $(wildcard *.cxx))
FMWK_HEADERS = LinkDef.h 
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
OBJECTS = $(SOURCES:.cxx=.o)

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell gallery-config --includes)
INCFLAGS += $(shell gallery-fmwk-config --includes)


# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)


###################################################

# call kernel specific compiler setup
include $(GALLERY_FMWK_BASEDIR)/Makefile/Makefile.${OSNAME}


# call the common GNUmakefile

# LDFLAGS += $(shell larlite-config --libs)
# LDFLAGS += -L$(shell python-config --prefix)/lib/
# LDFLAGS += $(shell python-config --ldflags)

include $(GALLERY_FMWK_BASEDIR)/Makefile/GNUmakefile.CORE

