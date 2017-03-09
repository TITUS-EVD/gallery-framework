#####################################################################################
#
# A top Makefile for building my project.
# One needs to define $GALLERY_FMWK_BASEDIR to build the sub directories.
#
#####################################################################################
#
# IMPOSE CONDITION BETWEEN GALLERY_FMWK_BASEDIR & PWD =>
#   do not compile if PWD !=$GALLERY_FMWK_BASEDIR is set elsewhere
#
ifndef GALLERY_FMWK_BASEDIR
 GALLERY_FMWK_BASEDIR := $(shell cd . && pwd -P)
endif
NORMALIZED_GALLERY_FMWK_BASEDIR := $(shell cd ${GALLERY_FMWK_BASEDIR} && pwd -P)
ifneq ($(NORMALIZED_GALLERY_FMWK_BASEDIR), $(shell cd . && pwd -P))
 ifneq ($(wildcard ./PMTDecoder/*),)
ERROR_MESSAGE := $(error Your source code detected in current dir, but GALLERY_FMWK_BASEDIR is not current dir.  \
   To avoid recompiling the wrong installation,\
   GALLERY_FMWK_BASEDIR must be set to the current directory when making.  \
   Currently it is set to ${GALLERY_FMWK_BASEDIR} [$(NORMALIZED_GALLERY_FMWK_BASEDIR)].  \
   Current directory is $(shell pwd).)
 endif
endif
export GALLERY_FMWK_BASEDIR

all:
	@mkdir -p $(GALLERY_FMWK_BASEDIR)/lib
	@echo "Start building core"
	@+make --directory=$(GALLERY_FMWK_COREDIR)
	@echo
	@echo "Start building UserDev"
	@+make --directory=$(GALLERY_FMWK_USERDEVDIR)
	@echo 
	@echo "Exiting"

clean:
	@echo "Cleaning core"
	@+make clean --directory=$(GALLERY_FMWK_COREDIR)
	@echo
	@echo "Cleaning UserDev"
	@+make clean --directory=$(GALLERY_FMWK_USERDEVDIR)
	@echo
	@echo "Exiting"

#####################################################################################
#
# DOCUMENTATION...
#
doxygen:
	@echo 'dOxygenising your code...'
	@doxygen $(GALLERY_FMWK_BASEDIR)/doc/doxygenMyProject.script

doxygen+:
	@echo 'dOxygenising MyProject + local-ROOT...'
	@doxygen $(GALLERY_FMWK_BASEDIR)/doc/doxygenMyProject+.script
#
#####################################################################################
