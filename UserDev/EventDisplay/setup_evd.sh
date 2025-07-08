#!/usr/bin/bash

# This section of code determines where the evd is stored.

if [ -z ${GALLERY_FMWK_BASEDIR+x} ]; then 
  echo "Must set up gallery framework to use this!";
  return 
fi

# This section extends the path and python path to run from anywhere
export DIR=$GALLERY_FMWK_USERDEVDIR/EventDisplay/


# This section verifies that python dependences are setup 

PYTHONPATH_backup=$PYTHONPATH
PATH_backup=$PATH

if [[ ! ":$PATH:" == *":$DIR/python:"* ]]; then
  export PATH=$DIR/python:$PATH
fi

if [[ ! ":$PYTHONPATH:" == *":$DIR/python:"* ]]; then
  export PYTHONPATH=$DIR/python/:$PYTHONPATH
fi

# Test argparse
if ! $(python -c "import argparse" &> /dev/null); then 
  echo "Warning: can not use evd due to missing package argparse"
  export PATH=$PATH_backup
  export PYTHONPATH=$PYTHONPATH_backup
  return
fi

# Test numpy
if ! $(python -c "import numpy" &> /dev/null); then 
  echo "Warning: can not use evd due to missing package numpy"
  export PATH=$PATH_backup
  export PYTHONPATH=$PYTHONPATH_backup 
  return
fi

# Test pyqt4
if ! $(python -c "import pyqtgraph.Qt" &> /dev/null); then 
  echo "Warning: can not use evd due to missing package PyQt"
  export PATH=$PATH_backup
  export PYTHONPATH=$PYTHONPATH_backup
  return
fi

export BUILD_GALLERY_FMWK_EVD=true

echo "TITUS event display has been set up."

