# contModel is a continuous modeler designed to used regression for sklearn's 
# linear model regression package and is potentially useful for multiple independent 
# variables

import os

import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

from accFunctions import * 
from plotFunctions import *


def main(*args):

    if len(args) == 1:

        campaignObject = args[0]
        # engages database for accessing values pertaining to all data acquired
        database = campaignObject.data
        haveData = [database.transform_ind(i) for i in database.storeind]
        noData = campaignObject.ESS.listTuples
        for ind in haveData:
            index = np.argwhere(noData == ind)
            np.delete(noData, index)

        X = campaignObject.ESS.listTuples[haveData]
        Y = [database.retrieve(campaignObject.ESS.listTuples[i]) for i in haveData]

        if not campaignObject.initModelComplete:

            # when there are no observations, i.e. first round, 
            # we initialize all the variables that will be needed and
            # attribute them to the campaignObject
            # normally, this is completed by initModel, however, 
            # for some campaigns, it might be imperfect/insufficient
            # if initModel is sufficient, leave this clause blank
            iVarDimensions = []
            for i in range(0,len(campaignObject.ESS.iVars)):
                iVarDimensions.append(len(np.unique(campaignObject.ESS.listTuples[:,i])))
            setattr(campaignObject, 'initPredictions',
                np.empty(iVarDimensions))
            campaignObject.initPredictions[:] = np.nan

            # Done adding stuff/overwriting for model initialization, 
            # so we set initModelComplete to True
            campaignObject.initModelComplete = True

        else:

            allIndices = range(len(campaignObject.ESS.listTuples))

            data = np.empty([len(campaignObject.ESS.listTuples),
                len(campaignObject.ESS.dVars)])
            data[:] = np.nan
            for i in allIndices:
                if i in haveData:
                    data[i] = database.retrieve(campaignObject.ESS.listTuples[i])

            fullSpace_last = np.reshape(np.ndarray.copy\
                    (campaignObject.initPredictions),
                    campaignObject.yAccuracy.shape)

            campaignObject.yAccuracy[haveData] = 1
            reg = linear_model.LinearRegression().fit(X, Y)
            campaignObject.model = reg
            campaignObject.modelR_value = reg.score(X, Y)
            campaignObject.initPredictions = \
                    reg.predict(campaignObject.ESS.listTuples)

            #print(campaignObject.yAccuracy.shape)
            groundTruth = np.reshape(campaignObject.groundTruth,
                    campaignObject.yAccuracy.shape)
            fullSpace = np.reshape(np.ndarray.copy\
                    (campaignObject.initPredictions),
                    campaignObject.yAccuracy.shape)

            if haveData:

                # reporting forward accuracy
                forwardAcc(campaignObject, fullSpace_last, data,
                        acc_f=r2_score)

                # reporting full space accuracy (predictions + 
                # observations or acquired data)
                for i in range(len(allIndices)):
                        if i in haveData:
                                fullSpace[i] = data[i]
                fullSpaceAcc(campaignObject, fullSpace, groundTruth,
                        allIndices, acc_f=r2_score)

                # reporting accuracy of predictions made only
                predsOnlyAcc(campaignObject, fullSpace, groundTruth,
                        allIndices, haveData, acc_f=r2_score)

            else:
                noDataAcc(campaignObject)

            # the yAccuracy is an estimated confidence in each prediction,
            # estimate for unsampled points as the shortest distance in
            # independent variable space to a sampled point
            for i in range(len(campaignObject.yAccuracy)):
                dist = np.empty([len(haveData), 1])
                iCoords = campaignObject.ESS.listTuples[i]
                j = 0
                for k in haveData:
                    kCoords = campaignObject.ESS.listTuples[k]
                    dist[j] = np.sqrt(np.sum((iCoords-kCoords)**2))
                    j+=1
                campaignObject.yAccuracy[i] = \
                        campaignObject.modelR_value/np.amin(dist[np.nonzero(dist)])
                if i in haveData:
                    campaignObject.yAccuracy[i] = 1

        return campaignObject

    elif len(args) > 0:

        campaignObject = args[0]
        ESC = args[1]

        plotAcc(campaignObject, ESC)
