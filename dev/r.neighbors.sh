#!/usr/bin/zsh

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# GRASS setup: 
#   - creation of GRASS LOCATION and a PERMANENT mapset
#   - compute r.neighbors for radius $1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
set -e

radius=$1
basedir="dat/interim"
geofile="${basedir}/06_postprocessing/main_clusters.tif"
grassdir="${basedir}/grassdata"
mapset="${grassdir}/PERMANENT"

# clean up existing dir
if [ -d "$grassdir" ]
then
	echo "\n`date "+%Y-%m-%d %H:%M:%S"`: Cleaning up $grassdir."
	rm -rf $grassdir
fi

# create new location from GeoTIFF
echo "\n`date "+%Y-%m-%d %H:%M:%S"`: Create GRASS LOCATION from $geofile"
grass -c -e ${geofile} ${grassdir}

# check 
if [ -d "$mapset" ]
then
	echo "\n`date "+%Y-%m-%d %H:%M:%S"`: Mapset PERMANENT successfully created at $grassdir."
else
	echo "\n`date "+%Y-%m-%d %H:%M:%S"`: ERROR: $grassdir not found. Please check GRASS location."
	exit
fi

# compute r.neighbors
echo "\n`date "+%Y-%m-%d %H:%M:%S"`: Link gdal raster as pseudo GRASS raster"
grass ${mapdir} --exec r.external input=${geofile} output=subregions

echo "\n`date "+%Y-%m-%d %H:%M:%S"`: Filter using r.neighbors"
grass ${mapdir} --exec r.neighbors input=subregions output=subregions_mode_{$radius} size=$radius method=mode

echo "\n`date "+%Y-%m-%d %H:%M:%S"`: Export GeoTIFF"
grass ${mapdir} --exec r.out.gdal input=subregions_mode_{$radius} output=subregions_mode_${radius}.tif

# cleaning up
echo "\n`date "+%Y-%m-%d %H:%M:%S"`: Cleaning up $grassdir."
rm -rf $grassdir
mv subregions_mode_${radius}.tif dat/interim/06_postprocessing/
