# pool based modeler - just a classifier trained on labeled instances
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

from accFunctions import *
from plotFunctions import *

import matplotlib.pyplot as plt


def main(*args):
        if len(args) == 1:

                campaignObject = args[0]

                # engage database for accessing values pertaining to all data acquired
                database = campaignObject.data
                haveData = [database.transform_ind(i) for i in database.storeind]

                if not campaignObject.initModelComplete:
                        # when there are no observations, i.e. first round, we initialize all the variables that will be needed and
                        # attribute them to the campaignObject
                        # normally, this is completed by initModel, however, 
                        # for some campaigns, it might be imperfect/insufficient
                        # if initModel is sufficient, leave this clause blank
                        
                        setattr(campaignObject, 'ESC', [])
                        campaignObject.ESC.append(0)
                        # Done adding stuff/overwriting for model initialization, so we set initModelComplete to True
                        campaignObject.initModelComplete = True

                else:

                        # at this point, we should have labels and thus something to train our classifier on

                        allIndices = range(campaignObject.ESS.iVars[0][2])
                        allFeats = np.stack(
                                np.asarray(database.retrieve(campaignObject,
                                    location="Associated Variable 1", flag='all'))[:, 0])
                        fullSpace_last = np.ndarray.copy(campaignObject.predictions)

                        # noData = np.argwhere(campaignObject.yAccuracy != 1)[:, 0]
                        haveData = np.argwhere(campaignObject.yAccuracy == 1)[:, 0]
                        campaignObject.ESC.append(len(haveData)/(campaignObject.ESS.iVars[0][2]+1))
                        selectFeats = allFeats[haveData]
                        #print('selectFeats')
                        #print(selectFeats)
                        aLabels = database.retrieve(database, location="Dependent Variable 1", flag='all')
                        #print('aLabels')
                        #print(aLabels)
                        acqLabels = np.empty(len(aLabels), dtype=int)
                        for i in range(len(aLabels)):
                                t = aLabels[i]
                                t2 = t[0]
                                acqLabels[i] = t2
                        #print('acqLabels')
                        #print(acqLabels)

                        if len(np.unique(acqLabels))<=1:
                                campaignObject.predictions[:] = acqLabels[0]
                                #print('Single Class Predictions')
                                #print(np.transpose(campaignObject.predictions))
                        elif len(acqLabels) >= 3:
                                # Query by Committee
                                # cmmt member 1
                                clsfier1 = svm.SVC(kernel='linear', C=1).fit(selectFeats, acqLabels)
                                preds1 = clsfier1.predict(allFeats)
                                #print('Preds1')
                                #print(preds1)
                                decisionfxn1 = abs(clsfier1.decision_function(allFeats))

                                # cmmt member 2
                                clsfier2 = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(selectFeats, acqLabels)
                                preds2 = clsfier2.predict(allFeats)
                                #print('Preds2')
                                #print(preds2)
                                decisionfxn2 = abs(clsfier2.decision_function(allFeats))  # need to set gamma explicitly at some point

                                # cmmt member 3
                                clsfier3 = svm.SVC(kernel='sigmoid', C=1, gamma='scale').fit(selectFeats, acqLabels)
                                preds3 = clsfier3.predict(allFeats)
                                #print('Preds3')
                                #print(preds3)
                                decisionfxn3 = abs(clsfier3.decision_function(allFeats))  # need to set gamma explicitly at some point

                                campaignObject.model = [clsfier1, clsfier2, clsfier3]

                                # QoC average distance to hyperplane
                                campaignObject.yAccuracy = np.maximum(campaignObject.yAccuracy, 1/(1 + np.square(np.mean(
                                                                                                                  np.stack((decisionfxn1, decisionfxn2, decisionfxn3), axis=-1),
                                                                                                                  axis=1))))
                                # campaignObject.modelR_value = clsfier.score(X, Y)  # which one would be best, what to do with it, etc?

                                # QoC predictions are simply the mode of all 3 members' predictions
                                campaignObject.predictions = stats.mode(np.stack((preds1, preds2, preds3), axis=-1), axis=1).mode
                                #print('Predictions')
                                #print(np.transpose(campaignObject.predictions))

                        fullSpace = np.reshape(np.ndarray.copy(campaignObject.predictions), campaignObject.yAccuracy.shape)

                        if len(haveData) > 0:

                                # reporting forward accuracy
                                data = {}
                                for i in campaignObject.lastDataRequest:
                                    data[i] = database.retrieve(database.transform_num(i),
                                            location="Dependent Variable 1")

                                forwardAcc(campaignObject, fullSpace_last, data, acc_f=accuracy_score,
                                        nan_opt=False, nan_batchVal=True)
                                #print('Forward Modeling accuracy')
                                #print(campaignObject.accuracy_forwardModeling)
                                # reporting full space accuracy (predictions + observations or acquired data)
                                for i in range(len(allIndices)):
                                        if i in haveData:
                                                fullSpace[i] = database.retrieve(database.transform_num(i), location="Dependent Variable 1")

                                # this block may not be necessary anymore, especially if NaNs aren't possible with the structure of
                                # createCampaign
                                '''
                                fullSpace_noNaNs = []
                                groundTruth_adjusted = []
                                for i in allIndices:
                                        if np.isnan(fullSpace[i]):
                                                fullSpace_noNaNs.append(False)
                                        else:
                                                fullSpace_noNaNs.append(fullSpace[i])

                                if len(groundTruth_adjusted) < 1:
                                        campaignObject.accuracy_full.append(0)
                                else:
                                        campaignObject.accuracy_full.append(accuracy_score(campaignObject.groundTruth, fullSpace_noNaNs))
                                '''
                                campaignObject.accuracy_full.append(accuracy_score(campaignObject.groundTruth, fullSpace))
                                #print('Full space accuracy')
                                #print(campaignObject.accuracy_full)

                                # reporting accuracy of predictions made only
                                # placeholderFP = np.reshape(campaignObject.initPredictions, campaignObject.yAccuracy.shape)
                                onlyPredsGT = []
                                onlyPredsFP = []
                                for i in allIndices:
                                        if i in haveData:
                                                pass
                                        else:
                                                # this block (again) should not be necessary, i.e. NaNs, given the structure of Pool Based model
                                                '''
                                                if np.isnan(fullSpace[i]):
                                                        onlyPredsGT.append(True)
                                                        onlyPredsFP.append(False)
                                                else:
                                                        onlyPredsGT.append(campaignObject.groundTruth[i])
                                                        onlyPredsFP.append(fullSpace[i])
                                                '''
                                                onlyPredsGT.append(campaignObject.groundTruth[i])
                                                onlyPredsFP.append(fullSpace[i])

                                if len(onlyPredsGT) < 1:
                                        campaignObject.accuracy_onlyPredictions.append(0)
                                else:
                                        campaignObject.accuracy_onlyPredictions.append(accuracy_score(onlyPredsGT, onlyPredsFP))
                                #print('Predictions only accuracy')
                                #print(campaignObject.accuracy_onlyPredictions)
                        else:
                            noDataAcc(campaignObject)

                        '''
                        for i in range(len(campaignObject.yAccuracy)):
                                dist = np.empty([len(haveData), 1])
                                iCoords = campaignObject.ESS.listTuples[i]
                                j = 0
                                for k in haveData:
                                        kCoords = campaignObject.ESS.listTuples[k]
                                        dist[j] = np.sqrt(np.sum((iCoords - kCoords) ** 2))
                                        j += 1
                                campaignObject.yAccuracy[i] = campaignObject.modelR_value / np.amin(dist[np.nonzero(dist)])
                                if i in haveData:
                                        campaignObject.yAccuracy[i] = 1
                        '''


                return campaignObject


        elif len(args) > 0:

                campaignObject = args[0]
                ESC = campaignObject.ESC
                plotAcc(campaignObject, ESC)

