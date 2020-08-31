# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:58:19 2020

@author: linds
"""

##############################################################################
# IMPORT
##############################################################################

from pyspatialml import Raster
from pyspatialml.datasets import nc

import geopandas as gpd
from shapely.geometry import mapping

import rasterio
from rasterio.mask import mask


##############################################################################
# FUNCTIONS
##############################################################################


def nre_fun(x, y):
    nre = (x - y) / (x + y)
    return nre

##############################################################################
# RASTER
##############################################################################

raw_fp = "C:/Users/linds/NOAA/shapefile_training/data_raw/" # file path to raw data folder
raster_fp = raw_fp + 'white_small_sub.tif' # combine with tif file
stack = Raster(raster_fp)
stack.names
stack.rename({'white_small_sub_1': 'coastal'})
stack.rename({'white_small_sub_2': 'blue'})
stack.rename({'white_small_sub_3': 'green'})
stack.rename({'white_small_sub_4': 'yellow'})
stack.rename({'white_small_sub_5': 'red'})
stack.rename({'white_small_sub_6': 'rededge'})
stack.rename({'white_small_sub_7': 'NIR1'})
stack.rename({'white_small_sub_8': 'NIR2'})

# Check the dimensions and bands 
print(stack.shape) # dimensions
print(stack.count) # bands

##############################################################################
# SHAPEFILE
##############################################################################

shapefile_fp = raw_fp + 'Lindsay_white_river_land_cover/Lindsay_white_river_land_cover.shp'
shapefile = gpd.read_file(shapefile_fp)
shapefile.crs
shapefile.bounds

geoms = shapefile.geometry.values 

geometry = geoms[0] 
print(type(geometry))
print(geometry)

feature = [mapping(geometry)] # can also do this using polygon.__geo_interface__
print(type(feature))
print(feature)

out_image, out_transform = mask(stack, feature, crop=True, nodata = 0)
out_image.shape


##############################################################################
# RANDOM FOREST MODEL
##############################################################################


