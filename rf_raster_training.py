# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:46:52 2020

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

import pandas as pd
import numpy as np
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

##############################################################################
# FUNCTIONS
##############################################################################

def nre_fun(x, y):
    nre = (x - y) / (x + y)
    return nre


# Takes a portion of samples from each class
# Note: this code is slow and needs revision
def undersample_ds(x, classCol, nsamples_class, seed):
    for i in np.unique(x[classCol]):
        if (sum(x[classCol] == i) - nsamples_class != 0):            
            xMatch = x[(x[classCol]).str.match(i)]
            x = x.drop(xMatch.sample(n = len(xMatch) - nsamples_class,
                                     random_state = seed).index)
    return x
   
   
# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

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

feature = [mapping(geometry)] # can also do this using polygon.__geo_interface__
print(type(feature))
print(feature)

out_image, out_transform = mask(raster, feature, crop=True)
out_image.shape

##############################################################################
# TRAINING & TEST DATA
##############################################################################

# create training pixels matrix with corresponding classname labels for rf
# This segment of code is from the tutorial:
# http://patrickgray.me/open-geo-tutorial/chapter_5_classification.html
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
        
        
# What are our classification labels?
labels = np.unique(shapefile["Classname"])
print('The training data include {n} classes: {classes}\n'.format(n=labels.size, 
                                                                classes=labels))

#Convert X and y to a pd dataframe
df_raw = pd.DataFrame(data = X, index = y)

# Fix column names
df_raw['Classname'] = df_raw.index # change classname to a column

# Take the band names from the csv and rename df_raw
col_names = pd.read_csv('data_raw/training_data_1M_sub.csv',nrows=1).columns[0:8]
df_raw.rename(columns=dict(zip(df_raw.columns[0:8], col_names)),inplace=True)
nsamples_class = 10000 # Number of samples to take from each class
sample_seed = 12 # seed for random sample
training_bc = df_raw.groupby('Classname').apply(lambda s: s.sample(nsamples_class,
                                                                  random_state = sample_seed))
# Run NRE function on the combination of  indices that preformed best
green_red = nre_fun(training_bc['green'], training_bc['red'])
blue_coastal = nre_fun(training_bc['blue'], training_bc['coastal'])
NIR2_yellow = nre_fun(training_bc['NIR2'], training_bc['yellow'])
NIR1_red = nre_fun(training_bc['NIR1'], training_bc['red'])
rededge_yellow = nre_fun(training_bc['rededge'], training_bc['yellow'])
red_NIR2 = nre_fun(training_bc['red'], training_bc['NIR2'])
rededge_NIR2 = nre_fun(training_bc['rededge'], training_bc['NIR2'])
rededge_NIR1 = nre_fun(training_bc['rededge'], training_bc['NIR1'])
green_NIR1 = nre_fun(training_bc['green'], training_bc['NIR1'])
green_NIR2 = nre_fun(training_bc['green'], training_bc['NIR2'])
rededge_green = nre_fun(training_bc['rededge'], training_bc['green'])
rededge_red = nre_fun(training_bc['rededge'], training_bc['red'])
yellow_NIR1 = nre_fun(training_bc['yellow'], training_bc['NIR1'])
NIR2_blue = nre_fun(training_bc['NIR2'], training_bc['blue'])
blue_red = nre_fun(training_bc['blue'], training_bc['red'])

# Combine indices into a dataframe
indices_df = pd.concat([green_red, blue_coastal, NIR2_yellow, NIR1_red,
                        rededge_yellow, red_NIR2, rededge_NIR2,
                        rededge_NIR1, green_NIR1, green_NIR2, rededge_green,
                        rededge_red, yellow_NIR1, NIR2_blue, blue_red],
                       axis = 1)

feature_names = ['green red', 'blue coastal', 'NIR2 yellow', 'NIR1 red',
              'rededge yellow', 'red NIR2', 'rededge NIR2', 'rededge NIR1',
              'green NIR1', 'green NIR2', 'rededge green', 'rededge red',
              'yellow NIR1', 'NIR2 blue', 'blue red']
indices_df.columns = feature_names
indices_df = indices_df * 10000
indices_df['Classname'] = pd.Series(training_bc['Classname'],
                                    index = indices_df.index)
        
        

##############################################################################
# RANDOM FOREST MODEL
##############################################################################

# X data for rf
features = indices_df
#features.drop('Classname')

# y data for rf. The y data needs to be as integers for sklearn. 
#labels, labels_name = string_to_int(indices_df['Classname'])
#labels = pd.get_dummies(indices_df['Classname'])
labels = pd.factorize(indices_df['Classname'])[0]

# Partition data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(features[feature_names],
                                                    labels, train_size = 0.9,
                                                    random_state = 8,
                                                    stratify = labels)

t0 = time.time()
# random classifier 
rf = RandomForestClassifier(n_estimators = 200,
                            max_features = 5,
                            random_state = 8)

rf.fit(X_train, y_train)
t1 = time.time()
total_time = t1-t0

result = permutation_importance(rf, X_train, y_train, random_state = 8)

predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)
print(classification_report(y_test, predictions))
confmat = confusion_matrix(y_test, predictions)
df_confmat = pd.DataFrame(confmat)
plot_confusion_matrix(rf, X_test, y_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))





