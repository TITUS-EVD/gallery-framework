#####################################################################################
#
# A top Makefile for building my project.
# One needs to define $GALLERY_FMWK_USERDEVDIR environment variable and set it to where this
# makefile exists. 
# One can type "make" and this builds packages that are added in $SUBDIR defined below.
# 
# The original is taken from Glenn A. Smith's example for Double Chooz experiment.
#
#####################################################################################
#
# IMPOSE CONDITION BETWEEN GALLERY_FMWK_USERDEVDIR & PWD =>
#   do not compile if PWD !=$GALLERY_FMWK_USERDEVDIR is set elsewhere
#
ifndef GALLERY_FMWK_USERDEVDIR
 GALLERY_FMWK_USERDEVDIR := $(shell cd . && pwd -P)
endif
NORMALIZED_GALLERY_FMWK_USERDEVDIR := $(shell cd ${GALLERY_FMWK_USERDEVDIR} && pwd -P)
ifneq ($(NORMALIZED_GALLERY_FMWK_USERDEVDIR), $(shell cd . && pwd -P))
 ifneq ($(wildcard ./Base/*),)
ERROR_MESSAGE := $(error Your source code detected in current dir, but GALLERY_FMWK_USERDEVDIR is not current dir.  \
   To avoid recompiling the wrong installation,\
   GALLERY_FMWK_USERDEVDIR must be set to the current directory when making.  \
   Currently it is set to ${GALLERY_FMWK_USERDEVDIR} [$(NORMALIZED_GALLERY_FMWK_USERDEVDIR)].  \
   Current directory is $(shell pwd).)
 endif
endif
export GALLERY_FMWK_USERDEVDIR
#
#####################################################################################
#
# Define directories to be compile upon a global "make"...
#
SUBDIRS :=  example SuperaLight

#####################################################################################
#
# COMPILATION...
#
#.phony: all configure default-config clean
.phony: all clean

all:
	@for i in $(SUBDIRS); do ( echo "" && echo "Compiling $$i..." && cd $$i && $(MAKE) ) || exit $$?; done
#####################################################################################
#
# CLEANs...
#
clean:
	@for i in $(SUBDIRS); do ( echo "" && echo "Cleaning $$i..." && cd $$i && $(MAKE) clean && rm -f $(GALLERY_FMWK_LIBDIR)/$$i.* ) || exit $$?; done

#####################################################################################
#
# DOCUMENTATION...
#
doxygen:
	@echo 'dOxygenising your code...'
	@mkdir -p $(GALLERY_FMWK_USERDEVDIR)/doc/dOxygenMyProject
	@doxygen $(GALLERY_FMWK_USERDEVDIR)/doc/doxygenMyProject.script

doxygen+:
	@echo 'dOxygenising MyProject + local-ROOT...'
	@mkdir -p $(GALLERY_FMWK_USERDEVDIR)/doc/dOxygenMyProject+
	@doxygen $(GALLERY_FMWK_USERDEVDIR)/doc/doxygenMyProject+.script
#
#####################################################################################
#EOF
