#!/usr/bin/python

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

### training data includes: features_train, labels_train (both include fast and slow points)

grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# make sure sklearn is imported into ClassifyNB.py
clf = classify(features_train, labels_train)

# Decision Boundary with Text Points Drawing
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())