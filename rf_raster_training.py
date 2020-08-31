# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:46:52 2020

@author: Lindsay Turner

http://patrickgray.me/open-geo-tutorial/chapter_5_classification.html
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
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from pyspatialml import Raster

from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.windows import Window
from rasterio.plot import reshape_as_raster, reshape_as_image

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

def str_class_to_int(class_array):
    class_array[class_array == 'Barren'] = 0
    class_array[class_array == 'Deciduous Forest'] = 1
    class_array[class_array == 'Dry Sandbar'] = 2
    #class_array[class_array == 'Dry sandbar'] = 2
    class_array[class_array == 'Evergreen Forest'] = 3
    class_array[class_array == 'Herbaceous'] = 4
    class_array[class_array == 'Low streamside vegetation'] = 5
    class_array[class_array == 'Shrubland'] = 6
    class_array[class_array == 'Water'] = 7
    class_array[class_array == 'Wood instream'] = 8
    return(class_array.astype(int))




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
nsamples_class = 10000

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

# Plots of spectral signatures of class (thank you Colin!)

# We will need a "X" matrix containing our features, and a "y" array containing our labels
print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))



# Explore the spectral signatures of each class now to make sure they're actually separable
fig, ax = plt.subplots(1,2, figsize=[20,8])

# numbers 1-8
band_count = np.arange(1,9)

classes = np.unique(y)
for class_type in classes:
    band_intensity = np.mean(X[y==class_type, :], axis=0)
    ax[0].plot(band_count, band_intensity, label=class_type)
    ax[1].plot(band_count, band_intensity, label=class_type)
# plot them as lines

# Add some axis labels
ax[0].set_xlabel('Band #')
ax[0].set_ylabel('Reflectance Value')
ax[1].set_ylabel('Reflectance Value')
ax[1].set_xlabel('Band #')
#ax[0].set_ylim(32,38)
ax[1].set_ylim(1475,1650)
#ax.set
ax[1].legend(loc="upper right")
# Add a title
ax[0].set_title('Band Intensities Full Overview')
ax[1].set_title('Band Intensities Mid Ref Subset')

##############

# Fix lowercase Dry sandbar classname so that it isn't oversampled
y = np.char.replace(y, 'Dry sandbar', 'Dry Sandbar')
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

indices_df['Classname'] = str_class_to_int(indices_df['Classname'])        
        

##############################################################################
# RANDOM FOREST MODEL
##############################################################################

# X data for rf
features = indices_df

# y data for rf. The y data needs to be as integers for sklearn. 
#labels = pd.factorize(indices_df['Classname'])[0]
labels = (indices_df['Classname'])

# Partition data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(features[feature_names],
                                                    labels, train_size = 0.9,
                                                    random_state = 8,
                                                    stratify = labels)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
confmat = confusion_matrix(y_test, predictions)
df_confmat = pd.DataFrame(confmat)
plot_confusion_matrix(rf, X_test, y_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(accuracy)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))


##############################################################################
# PLOT RF PREDICTIONS
##############################################################################

img = rasterio.open(raster_fp).read()[:, 150:600, 250:1400]

print(img.shape)
reshaped_img = reshape_as_image(img)
print(reshaped_img.shape)

# Reshape our classification map back into a 2D matrix so we can visualize it
class_prediction = class_prediction.reshape(reshaped_img[:, :, 0].shape)

class_prediction = str_class_to_int(class_prediction)

######### Messy, sorry
# predict original raster using trained model classifications
df_pred = pd.DataFrame(data = X)
col_names = pd.read_csv('data_raw/training_data_1M_sub.csv',nrows=1).columns[0:8]
df_pred.rename(columns=dict(zip(df_pred.columns[0:8], col_names)),inplace=True)

# Run NRE function on the combination of  indices that preformed best
green_red = nre_fun(df_pred['green'], df_pred['red'])
blue_coastal = nre_fun(df_pred['blue'], df_pred['coastal'])
NIR2_yellow = nre_fun(df_pred['NIR2'], df_pred['yellow'])
NIR1_red = nre_fun(df_pred['NIR1'], df_pred['red'])
rededge_yellow = nre_fun(df_pred['rededge'], df_pred['yellow'])
red_NIR2 = nre_fun(df_pred['red'], df_pred['NIR2'])
rededge_NIR2 = nre_fun(df_pred['rededge'], df_pred['NIR2'])
rededge_NIR1 = nre_fun(df_pred['rededge'], df_pred['NIR1'])
green_NIR1 = nre_fun(df_pred['green'], df_pred['NIR1'])
green_NIR2 = nre_fun(df_pred['green'], df_pred['NIR2'])
rededge_green = nre_fun(df_pred['rededge'], df_pred['green'])
rededge_red = nre_fun(df_pred['rededge'], df_pred['red'])
yellow_NIR1 = nre_fun(df_pred['yellow'], df_pred['NIR1'])
NIR2_blue = nre_fun(df_pred['NIR2'], df_pred['blue'])
blue_red = nre_fun(df_pred['blue'], df_pred['red'])

# Combine indices into a dataframe
all_index = pd.concat([green_red, blue_coastal, NIR2_yellow, NIR1_red,
                        rededge_yellow, red_NIR2, rededge_NIR2,
                        rededge_NIR1, green_NIR1, green_NIR2, rededge_green,
                        rededge_red, yellow_NIR1, NIR2_blue, blue_red],
                       axis = 1)

feature_names = ['green red', 'blue coastal', 'NIR2 yellow', 'NIR1 red',
              'rededge yellow', 'red NIR2', 'rededge NIR2', 'rededge NIR1',
              'green NIR1', 'green NIR2', 'rededge green', 'rededge red',
              'yellow NIR1', 'NIR2 blue', 'blue red']

all_index.columns = feature_names
all_index = all_index * 10000

class_prediction = rf.predict(all_index)
