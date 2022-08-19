#!/usr/bin/env bash
#
#
# CXX= $(GALLERY_FMWK_CXX)
# LD= g++
# LDFLAGS+="-Wl,-rpath,$(GALLERY_FMWK_BASEDIR)/lib"
# FFLAGS          += -Wall
# FLDFLAGS        += -lstdc++ -shared
# CXXFLAGS        += -g -O3 -std=c++0x -W -Wall -Wno-deprecated -fPIC -D_CORE_$(shell uname -s)_
# SOMAKER         = g++
# SOFLAGS         = -g -fPIC -shared


# What's the cpp file?
TARGET="supera_light_exec.cpp"

${GALLERY_FMWK_CXX} \
$(gallery-fmwk-config  --includes) \
$(root-config --cflags) \
-I${CANVAS_ROOT_IO_INC} \
-I"/lus/grand/projects/neutrino_osc_ADSP/software/larsoft/products/python/v3_9_2/Linux64bit+3.10-2.17/lib/python3.9/site-packages/larcv-3.4.2-py3.9-linux-x86_64.egg/larcv/include" \
-I"/home/cadams/Polaris/sbnd_parsl/singularity_software/larcv3/src/json/include/" \
-I"/home/cadams/Polaris/sbnd_parsl/singularity_software/larcv3/src/pybind11_json/include" \
$(gallery-config --includes) \
$(gallery-config --libs) \
$(gallery-fmwk-config  --libs) \
$(root-config --libs) \
-L"/lus/grand/projects/neutrino_osc_ADSP/software/larsoft/products/python/v3_9_2/Linux64bit+3.10-2.17/lib/python3.9/site-packages/larcv-3.4.2-py3.9-linux-x86_64.egg/larcv/lib/" -llarcv3 \
supera_light.o \
-o supera \
${TARGET}
