# get label for requested set of features

import os

import numpy as np
import csv


def main(dataRequested, campaignObject):

    print('Labels requested for samples:')
    print(dataRequested)
    labels = campaignObject.groundTruth
    print(labels[dataRequested])
    
    # engage database for storing values pertaining to data requested
    database = campaignObject.data

    for ind in dataRequested:
        #datum = np.array(labels[ind+1]).astype(int)
        print(np.array(labels[ind]).astype(int))
        datum = np.array(labels[ind]).astype(int)
        database.store([(datum,) + database.transform_num(ind)], destination="Dependent Variable 1")
        campaignObject.yAccuracy[ind] = 1  # should consider whether 1 or True is more appropriate for indicating that
        # data for this particular tuple has been collected

    return campaignObject
