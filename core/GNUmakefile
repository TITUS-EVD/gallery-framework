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

#
#####################################################################################
#
# Define directories to be compile upon a global "make"...
#
SUBDIRS :=  Base Analysis LArUtil

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
