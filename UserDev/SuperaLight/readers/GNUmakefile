#
# This is a GNUmakefile for my packages
#

# For this package, which compiles and links against hdf5, override
# the compilers to use h5cc and h5c++

# specific names for this package
DICT  = gallery_framework_SuperaCint
SHLIB = libgallery_framework_Supera.so
SOURCES = $(filter-out $(DICT).cxx, $(wildcard *.cxx))
FMWK_HEADERS = LinkDef.h $(DICT).h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
OBJECTS = $(SOURCES:.cxx=.o)

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell gallery-config --includes)
INCFLAGS += $(shell gallery-fmwk-config --includes)

INCFLAGS += -I$(shell python -c "import larcv; print(larcv.get_includes())" )
INCFLAGS += -I//home/cadams/Polaris/sbnd_parsl/singularity_software/larcv3/src/json/include/
INCFLAGS += -I//home/cadams/Polaris/sbnd_parsl/singularity_software/larcv3/src/pybind11_json/include

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

# call kernel specific compiler setup
include $(GALLERY_FMWK_BASEDIR)/Makefile/Makefile.${OSNAME}

# call the common GNUmakefile
LDFLAGS += $(shell gallery-config --libs)
LDFLAGS += $(shell gallery-fmwk-config --libs)
LDFLAGS += -L$(LARCV_LIBDR) -llarcv
LDFLAGS += -L$(shell python -c "import larcv; print(larcv.get_lib_dir())" ) -llarcv3

include $(GALLERY_FMWK_BASEDIR)/Makefile/GNUmakefile.CORE
