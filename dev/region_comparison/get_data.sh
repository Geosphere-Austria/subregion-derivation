#! /usr/bin/zsh

# HISTALP CRSM
echo "Working on HISTALP CRSM"
wget -P dat/raw \
    https://www.zamg.ac.at/histalp/download/crsm/Shape_CRSM.ZIP
unzip dat/rawShape_CRSM.ZIP \
    -d dat/rawcrsm/
rm dat/rawShape_CRSM.ZIP
echo "CRSM Shapefiles done"

# BFW growing regions
echo "Working on BFW Growing Regions"
wget --cipher 'DEFAULT:!DH' -P dat/raw \
    https://bfw.ac.at/300/pdf/1027_shape_Wuchs_Lambert.zip
unzip dat/raw1027_shape_Wuchs_Lambert.zip \
    -d dat/rawfgr/
rm dat/raw1027_shape_Wuchs_Lambert.zip 
echo "FGR Shapefiles done"

