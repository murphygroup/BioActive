#initializes features

import os

import numpy as np

# import an appropriate classifier
from sklearn import svm
from sklearn.cluster import KMeans
# sklearn.metrics provides a means for which we can easily calculate the accuracy in which we have effectively a
# prediction and some form of ground truth
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from scipy import stats



import matplotlib.pyplot as plt

import provideFeatures

def main(campaignObject):

        database = campaignObject.data

        provideFeatures.main(campaignObject)

        setattr(campaignObject, 'featsDistances', [])
        #allFeats = np.array(database.retrieve(campaignObject, location="Derived Data 1", flag='all'))[:, 0]
        allFeats = np.stack(
                np.asarray(database.retrieve(campaignObject, 
                    location="Associated Variable 1", flag='all'))[:, 0])

        #print('allFeats')
        #print(allFeats.shape)
        #print(allFeats)

        #kmeans = KMeans(n_clusters=campaignObject.ESS.dVars[0][2], random_state=0).fit(allFeats)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(allFeats)
        campaignObject.model = kmeans.cluster_centers_
        campaignObject.predictions = kmeans.labels_
        #print('Cluster centers')
        #print(kmeans.cluster_centers_)
        #print('Labels')
        #print(kmeans.labels_)
        #print('Transform')
        #print(kmeans.transform(allFeats))

        dist2centers = kmeans.transform(allFeats)
        distmax = np.max(dist2centers, axis=0)
        #print('Distmax')
        #print(distmax)
        dist2centers = dist2centers/distmax
        #print('Dist2centers')
        #print(dist2centers)
        dmin = np.min(dist2centers, axis=1)
        #print('dmin')
        #print(dmin)
        campaignObject.yAccuracy = 1-dmin

        # get feature pairwise distances for use in diversity sampling in active learning stage
        campaignObject.featsDistances = euclidean_distances(allFeats)
