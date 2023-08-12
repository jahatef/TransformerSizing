#!/bin/bash

$PYTHON setup.py install --single-version-externally-managed --record record.txt

cd $PREFIX/bin
rm -f pip2* pip3*
rm -f $SP_DIR/__pycache__/pkg_res*
# Remove all bundled .exe files courtesy of distlib.
rm -f $SP_DIR/pip/_vendor/distlib/*.exe