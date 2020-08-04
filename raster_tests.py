# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:35:55 2020

@author: Lindsay Turner
"""
##############################################################################
# IMPORT
##############################################################################

import rasterio
from rasterio.mask import mask
from rasterio.plot import show

import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
import matplotlib.pyplot as plt

import os

##############################################################################
# RASTER
##############################################################################

# Read in data:
raw_fp = "C:/Users/linds/NOAA/shapefile_training/data_raw/" # file path to raw data folder
raster_fp = raw_fp + 'white_small_sub.tif' # combine with tif file
raster = rasterio.open(raster_fp)
type(raster)
raster.meta # This gives information on the raster file

# Read in single bands
red = raster.read(5)
green = raster.read(3)
blue = raster.read(2)
NIR2 = raster.read(8)

# individual plots of bands
show((raster, 5), cmap='Reds')
show((raster, 3), cmap='Greens')
show((raster, 2), cmap='Blues')

# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

# Normalize the bands
redn = normalize(red)
greenn = normalize(green)
bluen = normalize(blue)
NIR2n = normalize(NIR2)

# Create RGB natural color composite
rgb = np.dstack((redn, greenn, bluen))
ngb = np.dstack((NIR2n, greenn, bluen))


# Plot RGB
plt.imshow(rgb)
plt.imshow(ngb)

# Check the dimensions and bands 
print(raster.shape) # dimensions
print(raster.count) # bands

#clipped_img = raster.read([5,3,2])[:, 150:600, 250:1400]
#print(clipped_img.shape)
#fig, ax = plt.subplots(figsize=(10,7))
#show(clipped_img[:, :, :], ax=ax, transform=raster.transform) # add the transform arg to get it in lat long coords

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

from shapely.geometry import mapping
feature = [mapping(geometry)] # can also do this using polygon.__geo_interface__
print(type(feature))
print(feature)

out_image, out_transform = mask(raster, feature, crop=True)
out_image.shape

##############################################################################
# RANDOM FOREST MODEL
##############################################################################

# create training pixels matrix with corresponding classname labels for rf
X = np.array([], dtype=np.int8).reshape(0,8) # pixels for training
y = np.array([], dtype=np.string_) # labels for training

# extract the raster values within the polygon 
with rasterio.open(raster_fp) as src:
    band_count = src.count
    for index, geom in enumerate(geoms):
        feature = [mapping(geom)]

        # the mask function returns an array of the raster pixels within this feature
        out_image, out_transform = mask(src, feature, crop=True) 
        # eliminate all the pixels with 0 values for all 8 bands - AKA not actually part of the shapefile
        out_image_trimmed = out_image[:,~np.all(out_image == 0, axis=0)]
        # eliminate all the pixels with 255 values for all 8 bands - AKA not actually part of the shapefile
        out_image_trimmed = out_image_trimmed[:,~np.all(out_image_trimmed == 255, axis=0)]
        # reshape the array to [pixel count, bands]
        out_image_reshaped = out_image_trimmed.reshape(-1, band_count)
        # append the labels to the y array
        y = np.append(y,[shapefile["Classname"][index]] * out_image_reshaped.shape[0]) 
        # stack the pizels onto the pixel array
        X = np.vstack((X,out_image_reshaped))   
        
        
        


