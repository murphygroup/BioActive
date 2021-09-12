# createCampaign provides definitions for: 
# 1) data collection functions, used to obtain all currently existing data,
#    a cost for collecting some specific set of experimental data,
#    a specific set of new experimental data 
# 2) the types of dependent (measured) variables that can be returned, 
# 3) the types and ranges of independent (experimental) variables to be explored, 
# 4) the set of definitions for model construction functions,
# 5) the active learning function to be used, and 
# 6) the objective function to be optimized along with the termination criteria

# this version defines a pool-based active learning campaign in which there is a single 
# categorical variable which holds the index into a pool of unlabeled examples for which
# features are available to use for building a predictive model

import os

import csv

# Useful library for handling iterators
import itertools

# Important for storing results and history of campaign runs on disk or on a remote server
from database import *
import datetime

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
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

import util


def main(*arg):

    NumberOfPoints = arg[0]
    NumberOfAVars = arg[1]
    NumberOfClasses = arg[2]
    NumberOfClustersPerClass = arg[3]
    UseActiveLearning = arg[4]
        
    class campaign:

        def initAvars(self):
            from initModelFeat import main
            return main(self)
        
        def goalTether(self, arg):
            from tetherCat import main
            return main(arg)

        def modelData(self, *arg):
            from poolModeler import main
            return main(*arg)

        def activeLearner(self, arg):
            from poolactLearn import main
            return main(arg)

        def fetchData(self, arg1, arg2):
            from getpoolquery_Data import main
            return main(arg1, arg2)

        def provideNames():
            print("iVarNames:")
            print(campaign.ESS.iVarNames)
            print("iVarCategories:")
            print(campaign.ESS.iVarCategories)
            
        class ESS:
            # define data specifications

            # this is the index into the pool
            iVars = [('int', 0, NumberOfPoints-1)]
            iVarNames = ["iVar0"]
            iVarCategories = [()]
            # this is the class assigned to each item in the pool
            dVars = [('int', 0, NumberOfClasses-1)]

            listTuples = util.createTuples(iVars)
            dimarr = []
            for i in range(len(iVars)):
                dimarr.append(iVars[i][2] - iVars[i][1] + 1)

        data = Database(ESS.dimarr, "poolbased", dvars=1, avars=NumberOfAVars, reset=True)

        # use pairwise accuracy assessments when labels are 
        # assigned arbitrarily round-to-round
        pairwise = False

        # active learning (as opposed to random)
        flag = UseActiveLearning

        # number of experiments to perform round to round as 
        # modeling and subsequent active learning guided experimentation proceed
        confCount = 2

        # Simulation?
        simsFlag = True

        hasClassifier = False

        batchNaNs = []
        batchCorrect = []
        batchIncorrect = []

        Xfeats,Ylabels = make_classification(n_samples=NumberOfPoints,
                        n_features=NumberOfAVars, n_informative=NumberOfAVars,
                        n_redundant=0, n_repeated=0,
                        n_classes=NumberOfClasses,
                        n_clusters_per_class=NumberOfClustersPerClass,
                        flip_y=0.0, class_sep=1.0, hypercube=True,
                        shift=0.0, scale=1.0, shuffle=True, random_state=None)
        groundTruthAVars = Xfeats
        groundTruth = Ylabels

        class plotting:
            title = "Accuracy as Fraction of Experimental Space Coverage Increases"
            xlabel = "Experimental Space Coverage"
            ylabel = "Accuracy"
            filename = 'BioActive_PoolBasedCampaignTest_Sims.tif'
            intDir = 'simsDirectory_' + str(datetime.datetime.now()) + '/'
            filename = intDir + filename

    C = campaign()
    return C
