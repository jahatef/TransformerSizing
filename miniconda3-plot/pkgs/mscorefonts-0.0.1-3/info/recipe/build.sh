#!/bin/bash
mkdir -p ${PREFIX}/fonts
for ar in $(ls *.exe); do cabextract $ar; done 
for fo in $(ls *.ttf *.TTF); do cp ${fo} ${PREFIX}/fonts/$(echo "${fo}" | awk '{print tolower($0)}'); done
