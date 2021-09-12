# catModel11 is the 11th revision of a categorical modeler, hard imputation, 
# designed to identify clusters and predict the values of an unknown experimental 
# space by clustering together rows and columns that do not conflict separately
# and then using the clusters as a model to predict the space they span while 
# simultaneously attributing a confidence
# based off 1 of 2 approaches: 
# 1) what the confidence should be if the prediction is based off the clustering model
#    alone or 
# 2) what the confidence should be if the prediction is based off the clustering model 
#    or off the assignment of the mode for a given block

# import libraries
import os
import numpy as np

# directories
import sys
dirpath = os.getcwd()
sys.path.insert(0, dirpath + '/assessment_resources')

# scipy is useful for mode determinations
from scipy import stats

# sklearn.metrics provides a means for which we can easily calculate 
# the accuracy in which we have effectively a prediction and some form 
# of ground truth
from sklearn.metrics import accuracy_score

# just using matplotlib for plotting
import matplotlib.pyplot as plt

# the heat-map functions written in house allow tracking of confidence 
# assessment over the course of the campaign
from heatmap_blocktracker import main as heatMapblocks

# the following heat-map and pairwise plotting provide for accuracy 
# scoring when in the context of say k-clustering applications in data acquisition,
# the assigned test labels are arbitrary and thus dynamic from round to round of
# experimentation
from heatMap import main as heatMap_pairwise
from pairwiseAccuracy import main as pairwiseAccuracy
from pairwiseForwardAccuracy import main as pairwiseForwardAccuracy
from pairwisePlot import main as pairwisePlot

from accFunctions import *
from plotFunctions import *

def main(*args):

        # during most of the campaign, the predictive modeler is modeling; 
        # however, once the campaign is complete, the predictive modeler is 
        # used to plot accuracy
        if len(args) == 1:

            # initialize the campaignObject for variable accessibility
            campaignObject = args[0]

            # get experimental space coverage
            ESCcurrent = float(len(campaignObject.data.storeind))/\
                    np.prod(campaignObject.ESS.dimarr)

            # dataShape provides the dimensions needed for reshaping arrays 
            # into the most accommodating variables
            dataShape = [len(set(campaignObject.ESS.listTuples[:, 0])),
                    len(set(campaignObject.ESS.listTuples[:, 1]))]

            # bring in the database to fill a framed data array
            database = campaignObject.data

            # identify subset of experiments for which there are observations
            haveData = [database.transform_ind(i) for i in database.storeind]

            # are there any data points yet, i.e. is this the first round or 
            # a subsequent round?
            if campaignObject.initModelComplete:
                # this block is the precursor to producing a framed data array 
                # in which the data is organized by the
                # dimensions provided in the campaignObject.
                allIndices = range(len(campaignObject.ESS.listTuples))

                data = np.empty([len(campaignObject.ESS.listTuples),
                    len(campaignObject.ESS.dVars)])
                data[:] = np.nan
                for i in allIndices:
                    if i in haveData:
                        try:
                            # for when there are multiple values for a given tuple
                            if len(database.retrieve(campaignObject.ESS.listTuples[i])) > 1:
                                if hasattr(campaignObject, 'multipleVals'):
                                    if campaignObject.multipleVals == 'First':
                                        data[i] = database.retrieve(\
                                                campaignObject.ESS.listTuples[i])[0]
                                    elif campaignObject.multipleVals == 'Recent':
                                        data[i] = database.retrieve(\
                                                campaignObject.ESS.listTuples[i])[1]
                                    elif campaignObject.multipleVals == 'Mean':
                                        data[i] = [np.mean(database.retrieve(\
                                                campaignObject.ESS.listTuples[i]))]
                            else:
                                data[i] = [np.mean(database.retrieve(\
                                        campaignObject.ESS.listTuples[i]))]
                        except:
                            data[i] = database.retrieve(\
                                    campaignObject.ESS.listTuples[i])

                framedData = np.reshape(data, dataShape)

                #********************#

                # fullSpace_last is all the predictions from the previous round 
                # of experimentation and modeling this must be called before 
                # findCLusters, because findClusters updates initPredictions
                # this block also identifies what predictions will be compared 
                # either to the batch of data returned in an subsequent 
                # round and then
                fullSpace_last = np.reshape(np.ndarray.copy(\
                        campaignObject.finalPredictions),
                        campaignObject.yAccuracy.shape)
                fullSpace_last_rows = np.reshape(np.ndarray.copy(\
                        campaignObject.initPredictions[:, :, 0]),
                        campaignObject.yAccuracy.shape)
                fullSpace_last_cols = np.reshape(np.ndarray.copy(\
                        campaignObject.initPredictions[:, :, 1]),
                        campaignObject.yAccuracy.shape)

                groundTruth = np.reshape(np.ndarray.copy(campaignObject.groundTruth),
                        campaignObject.yAccuracy.shape)

                #********************#

                # find clusters in both the rows and columns
                [res0, models0] = findClusters(framedData, 0, campaignObject)
                [res1, models1] = findClusters(framedData, 1, campaignObject)

                #********************#

                # this block is responsible for tracking what predictions are 
                # presumed to belong to clusters identified in "findClusters"
                for i in range(len(models0)):
                    for j in range(len(models0[i])):
                        campaignObject.clusterMatrix[int(models0[i][j]), :, 0] = int(i)

                for i in range(len(models1)):
                    for j in range(len(models1[i])):
                        campaignObject.clusterMatrix[:, int(models1[i][j]), 1] = int(i)

                # possibly useful code to integrate or replace other lines
                '''
                # figure out total possible combinations
                totalCombinations = len(campaignObject.model[0]) * \
                        len(campaignObject.model[1])
                rowSet = np.unique(campaignObject.clusterMatrix[:, :, 0])
                colSet = np.unique(campaignObject.clusterMatrix[:, :, 1])

                allCombinations = list(product(rowSet, colSet))
                for i in range(dataShape[0]):
                        for j in range(dataShape[1]):
                                framedEquivalencies[i, j] = allCombinations.index(
                                        (campaignObject.clusterMatrix[i, j, 0], 
                                        campaignObject.clusterMatrix[i, j, 1]))
                '''

                #********************#

                # store useful variables for quality assessment and assurance
                campaignObject.rowsCounted.append(len(models0))
                campaignObject.colsCounted.append(len(models1))
                campaignObject.model = [models0, models1]

                if ESCcurrent <= 0.25:
                    if hasattr(campaignObject, 'model25rows'):
                        campaignObject.model25rows = campaignObject.model[0]
                    if hasattr(campaignObject, 'model25cols'):
                        campaignObject.model25cols = campaignObject.model[1]
                elif ESCcurrent <= 0.50:
                    if hasattr(campaignObject, 'model50rows'):
                        campaignObject.model50rows = campaignObject.model[0]
                    if hasattr(campaignObject, 'model50cols'):
                        campaignObject.model50cols = campaignObject.model[1]

                #********************#

                # this block combines and interpolates predictions made from 
                # clustering rows and columns
                for i in range(dataShape[0]):
                    for j in range(dataShape[1]):
                        # are they both finite, i.e. not NaNs?
                        if (np.isfinite(campaignObject.initPredictions[i, j, 0]) &
                                        np.isfinite(\
                                                campaignObject.initPredictions[i,
                                                    j, 1])).all():
                            # are they equal to one another?
                            if campaignObject.initPredictions[i, j, 0] == \
                                    campaignObject.initPredictions[i, j, 1]:
                                # pick either, doesn't matter then
                                campaignObject.finalPredictions[i, j] = \
                                        campaignObject.initPredictions[i, j, 0]
                            else:
                                if res0[np.ravel_multi_index([i, j], dataShape)] > \
                                                res1[np.ravel_multi_index([i, j], dataShape)]:
                                    campaignObject.finalPredictions[i, j] = \
                                            campaignObject.initPredictions[i, j, 0]
                                elif res0[np.ravel_multi_index([i, j], dataShape)] < \
                                                res1[np.ravel_multi_index([i, j], dataShape)]:
                                    campaignObject.finalPredictions[i, j] = \
                                            campaignObject.initPredictions[i, j, 1]
                                else:
                                    # otherwise, pick randomly between the 2 possible options
                                    campaignObject.finalPredictions[i, j] = np.random.choice(
                                            [campaignObject.initPredictions[i, j, 0],
                                                campaignObject.initPredictions[i, j, 1]], 1)

                        # are one or the other or both of these NaNs?
                        else:
                            # if the 0th one is a NaN, but not the other,
                            # then pick the other, i.e. 1th
                            if np.isnan(campaignObject.initPredictions[i, j, 0]):
                                campaignObject.finalPredictions[i, j] = \
                                        campaignObject.initPredictions[i, j, 1]
                            # if the 1th one is a NaN, but not the other,
                            # then pick the other, i.e. 0th
                            elif np.isnan(campaignObject.initPredictions[i, j, 1]):
                                campaignObject.finalPredictions[i, j] = \
                                        campaignObject.initPredictions[i, j, 0]
                            # if both are NaNs, then there's nothing we can do,
                            # so set final/merged prediction as NaN
                            elif (np.isnan(campaignObject.initPredictions[i, j, 0]) &
                                            np.isnan(campaignObject.initPredictions[i, j, 1])).all():
                                campaignObject.finalPredictions[i, j] = np.nan

                #********************#

                # this block merges the confidences provided from the clustering processes in both the rows and columns
                '''
                if (np.sum(np.isnan(res0)) == len(campaignObject.yAccuracy) and \
                        np.sum(np.isnan(res1)) == len(
                                campaignObject.yAccuracy)).all():
                        campaignObject.yAccuracy[:] = 0
                else:
                        for i in range(len(campaignObject.yAccuracy)):
                                if np.isfinite(res0[i]) or np.isfinite(res1[i]):
                                        campaignObject.yAccuracy[i] = \
                                                np.nanmean([res0[i], res1[i]])
                                elif (np.isnan(res0[i]) and \
                                        np.isnan(res1[i])).all():
                                        campaignObject.yAccuracy[i] = 0
                '''

                campaignObject.yAccuracy = np.amax([res0, res1], axis=0)

                # following sub-block is to distinguish between 0 confidence 
                # predictions for which in the first case, there is no prediction
                # at all, i.e. a NaN or in the second case,
                # where there's a 0 confidence prediction of a finite value
                for ind in allIndices:
                    if np.isnan(np.reshape(campaignObject.finalPredictions,
                        campaignObject.yAccuracy.shape)[ind]):
                        campaignObject.yAccuracy[ind] = -1

                # lastly, the confidence of an observed or collected data is 1
                for ind in allIndices:
                    if ind in haveData:
                        campaignObject.yAccuracy[ind] = 1

                #********************#

                # pairwise accuracy is used for categorical campaigns in which 
                # labels are assigned arbitrarily round to round
                if campaignObject.pairwise == True:
                    campaignObject.filenames1.append(heatMap_pairwise(campaignObject, dimMaj=0))
                    campaignObject.filenames1.append(heatMap_pairwise(campaignObject, dimMaj=1))

                    # pairwiseAccuracy returns a tuple where the first element is fullspace and second is preds only
                    rowsAcc = pairwiseAccuracy(campaignObject,
                            campaignObject.initPredictions[:, :, 0])
                    colsAcc = pairwiseAccuracy(campaignObject,
                            campaignObject.initPredictions[:, :, 1])
                    finalAcc = pairwiseAccuracy(campaignObject,
                            campaignObject.finalPredictions)

                    campaignObject.fullSpaceAccuracyRows.append(rowsAcc[0])
                    campaignObject.fullSpaceAccuracyCols.append(colsAcc[0])
                    campaignObject.fullSpaceAccuracy.append(finalAcc[0])

                    # rowsAcc[1] == 1 happens when ESC= 1 so no preds
                    # GW made a change to discuss *should be finalAcc instead of rowsAcc 
                    # because while predictions may not be made from clustering in rows 
                    # with approach 1 (predict using only clustering models as opposed to
                    # the mode), predictions might still be made from column clustering model
                    # additionally, whenever the predictions are NaNs before the end of the 
                    # campaign, those are predictions that should be measured with respect 
                    # to accuracy

                    if finalAcc[1] != None:
                        campaignObject.predictionAccuracyRows.append(rowsAcc[1])
                        campaignObject.predictionAccuracyCols.append(colsAcc[1])
                        campaignObject.predictionAccuracy.append(finalAcc[1])

                    # forward acc
                    pairwiseForwardAccuracy(campaignObject, fullSpace_last_rows,
                            dimMaj=0, merged=False)
                    pairwiseForwardAccuracy(campaignObject, fullSpace_last_cols,
                            dimMaj=1, merged=False)
                    pairwiseForwardAccuracy(campaignObject, fullSpace_last,
                            dimMaj=0, merged=True)

                if campaignObject.pairwise == False:

                    # this block reports forward accuracy
                    batchPred = []
                    batchGT = []

                    forwardAcc(campaignObject, fullSpace_last, data, 
                            nan_opt=False, dataInd=0)

                    # this block reports full space accuracy 
                    # (predictions + observations or acquired data)
                    fullSpace = np.reshape(np.ndarray.copy\
                            (campaignObject.finalPredictions),
                            campaignObject.yAccuracy.shape)

                    for i in allIndices:
                        if i in haveData:
                            fullSpace[i] = data[i]

                    # this block accounts for NaNs
                    fullSpace_noNaNs, groundTruth_adjusted = \
                            fullSpaceAcc(campaignObject, fullSpace,
                                    groundTruth, allIndices,
                                    nan_opt=False, dataInd=0, GTInd=0)

                    if (accuracy_score(groundTruth_adjusted, fullSpace_noNaNs) >= 0.95)\
                            & campaignObject.percentileStop:
                        campaignObject.percentile95 = ESCcurrent
                        campaignObject.percentileStop = False

                    # this block reporting accuracy of only predictions that have been made, sans acquired data
                    predsOnlyAcc(campaignObject, fullSpace, groundTruth, allIndices,
                            haveData, nan_opt=False, dataInd=0, GTInd=0)

            # when there are no observations, i.e. first round, we initialize all 
            # the variables that will be needed and
            # attribute them to the campaignObject
            # normally, this is completed by initModel, however, 
            # for some campaigns, it might be imperfect/insufficient
            # if initModel is sufficient, leave this else clause blank 
            else:

                # active learner will be passed an additional argument to 
                # address mutual information
                if hasattr(campaignObject, 'activeLearning'):
                    pass
                else:
                    setattr(campaignObject, 'activeLearning', 'clustering')

                # track what predictions belong to which clusters
                setattr(campaignObject, 'clusterMatrix', 
                        np.empty([len(campaignObject.ESS.listTuples[:, 0]), len(campaignObject.ESS.listTuples[:, 1]), 2]))

                # quality assurance/assessment
                setattr(campaignObject, 'rowsCounted', [])
                setattr(campaignObject, 'colsCounted', [])
                campaignObject.rowsCounted.append(0)
                campaignObject.colsCounted.append(0)

                # for pairwise and/or pre-trained classifier--> more 
                # information needed to smooth over
                setattr(campaignObject, 'numberObtained', 0)

                campaignObject.filenames3.append(heatMapblocks(
                        np.reshape(campaignObject.yAccuracy,
                           [len(set(campaignObject.ESS.listTuples[:, 0])),
                            len(set(campaignObject.ESS.listTuples[:, 1]))]), 0,
                           campaignObject))

                campaignObject.filenames3.append(heatMapblocks(
                        np.reshape(campaignObject.yAccuracy,
                           [len(set(campaignObject.ESS.listTuples[:, 0])),
                            len(set(campaignObject.ESS.listTuples[:, 1]))]), 1,
                           campaignObject))

                # track the predictions row and column-wise
                setattr(campaignObject, 'model25rows', {})
                setattr(campaignObject, 'model25cols', {})
                setattr(campaignObject, 'model50rows', {})
                setattr(campaignObject, 'model50cols', {})

                if campaignObject.pairwise == True:

                    setattr(campaignObject, 'predictionAccuracyRows', [0])
                    setattr(campaignObject, 'predictionAccuracyCols', [0])

                    setattr(campaignObject, 'fullSpaceAccuracyRows', [0])
                    setattr(campaignObject, 'fullSpaceAccuracyCols', [0])

                    setattr(campaignObject, 'forwardAccuracyRows', [])
                    setattr(campaignObject, 'forwardAccuracyCols', [])

                    setattr(campaignObject, 'myESCTest3_0', [])
                    setattr(campaignObject, 'myESCTest3_1', [])
                    setattr(campaignObject, 'ESCForClusters', [0])

                    setattr(campaignObject, 'predictionAccuracy', [0])
                    setattr(campaignObject, 'predictionAccuracyExtra', [0])
                    setattr(campaignObject, 'fullSpaceAccuracy', [0])
                    setattr(campaignObject, 'fullSpaceAccuracyExtra', [0])
                    setattr(campaignObject, 'forwardAccuracy', [])
                    setattr(campaignObject, 'forwardAccuracyExtra', [])

                    setattr(campaignObject, 'resOfClustering', [])

                    setattr(campaignObject, 'lastIndexesRequested', [])

                # Done adding stuff/overwriting for model initialization, 
                # so we set initModelComplete to True
                campaignObject.initModelComplete = True

            return campaignObject

        elif len(args) > 1:

                campaignObject = args[0]
                # depending on where the routine terminates, i.e. either as 
                # a result of satisfying criteria in the active learner or 
                # as a result of satisfying goal criteria, at which point ESC 
                # and accuracy measure lengths are =, it is necessary to pull
                # the size back on the accuracy
                ESC = args[1]

                if campaignObject.pairwise==True:
                    pairwisePlot(campaignObject, ESC)

                elif campaignObject.pairwise==False:
                    [f, axarr] = plt.subplots(2, sharex=True)

                    if ESC[-1] != 1:
                        plotAcc(campaignObject, ESC, ax=axarr[0],
                                ylim=[0, 1.2], multiPlot=True)
                    elif ESC[-1] == 1:
                        # since there are not predictions to be made 
                        # once the space has been exhausted
                        plotAcc(campaignObject, ESC, ax=axarr[0],
                                hasPredsAcc=False, 
                                ylim=[0, 1.2], multiPlot=True)
                    axarr[0].set_title(campaignObject.plotting.xlabel)
                    axarr[0].set(ylabel=campaignObject.plotting.ylabel)

                    plotAcc(campaignObject, ESC, ax=axarr[1],
                            hasFPAcc=False, hasFwdAcc=False, 
                            hasPredsAcc=False, hasClusters=True,
                            multiPlot=True)
                    axarr[1].set(ylabel='Clusters Predicted')
                    plt.savefig(campaignObject.plotting.filename)
                    plt.close('all')

                    print('Full Space Accuracy is , ',
                            campaignObject.accuracy_full)
                    print('Forward Modeling Accuracy is , ',
                            campaignObject.accuracy_forwardModeling)
                    print('Predictions-Only Accuracy is , ',
                            campaignObject.accuracy_onlyPredictions)


def findClusters(framedData, dimMaj, campaignObject):
        # the shape of the framedData will be useful in reshaping 
        # arrays to be most amenable to a variety of purposes
        placeholder = framedData.shape

        # 'intermVagree' is a square array such that each block is 
        # compared to every other block, i.e. compare what rows
        # have in common with other rows and what columns have in 
        # common with other columns
        intermVagree = np.empty([placeholder[dimMaj], placeholder[dimMaj]])
        intermVagree[:] = 0

        # clustering section
        # we change the shape of the array so that we don't have to 
        # change how we loop through it
        for i in range(placeholder[dimMaj]):
            for j in range(placeholder[dimMaj]):
                if dimMaj == 0:
                    countsame = 0
                    for k in range(len(framedData[i, :])):
                        if np.allclose(framedData[i, k], framedData[j, k], equal_nan=True):
                            if (np.isfinite(framedData[i, k]) & np.isfinite(framedData[j, k])).all():
                                countsame = countsame + 1
                        else:
                            if (np.isfinite(framedData[i, k]) & np.isfinite(framedData[j, k])).all():
                                countsame = -1
                                break
                    intermVagree[i, j] = countsame

                if dimMaj == 1:
                    countsame = 0
                    for k in range(len(framedData[:, i])):
                        if np.allclose(framedData[k, i], framedData[k, j], equal_nan=True):
                            if (np.isfinite(framedData[k, i]) & np.isfinite(framedData[k, j])).all():
                                countsame = countsame + 1
                        else:
                            if (np.isfinite(framedData[k, i]) & np.isfinite(framedData[k, j])).all():
                                countsame = -1
                                break
                    intermVagree[i, j] = countsame

        #********************#

        # 'vCluster' is used to identify what blocks do not conflict 
        # with other blocks, hence the >=0 constraint
        # this block will be used to extract clusters
        vCluster = np.empty([placeholder[dimMaj], placeholder[dimMaj]])
        vCluster[:] = 0
        for i in range(placeholder[dimMaj]):
            for j in range(placeholder[dimMaj]):
                if intermVagree[i, j] >= 0:
                    vCluster[i, j] = 1

        # this block extracts clusters from vCluster while 
        # removing conflicts iteratively
        clusterIndices = []
        for i in range(placeholder[dimMaj]):
            array = vCluster[i, :]
            clusterIndices.append([a for a, e in enumerate(array) if e == 1])
            conflicts = []
            for j in clusterIndices[i]:
                if len(np.where(vCluster[i + 1:, j] == 0)[0]) > 0:
                    if j in conflicts:
                        pass
                    else:
                        conflicts.extend(np.where(vCluster[i + 1:, j] == 0)[0] + i + 1)
            clusterIndices[i] = np.setdiff1d(clusterIndices[i], conflicts)
            vCluster[i, :] = 0
            vCluster[:, i] = 0

        for i in range(len(clusterIndices) - 1):
            for j in range((i + 1), len(clusterIndices), 1):
                if len(np.intersect1d(clusterIndices[i], clusterIndices[j])) == \
                        np.amin([len(clusterIndices[i]), len(clusterIndices[j])]):
                    clusterIndices[i] = np.union1d(clusterIndices[i],
                            clusterIndices[j])
                    clusterIndices[j] = []
                elif len(np.intersect1d(clusterIndices[i], clusterIndices[j])) > 0:
                    clusterIndices[j] = np.setdiff1d(clusterIndices[j],
                            clusterIndices[i])
                else:
                    pass

        #********************#

        # this block is designed such that any clusters that are not
        # distinct from one another are compared against one
        # another so as to produce clusters that are distinct
        refinedClusters = []
        for i in range(len(clusterIndices)):
            if len(clusterIndices[i]) > 0:
                refinedClusters.append(clusterIndices[i])

        #********************#

        # initializing yAccuracy (confidence)
        vAccData = np.ndarray.copy(framedData)
        vAccData[:] = 0

        # predictions Section
        # 'newframedData' exists so that the framedData isn't modified and further both can be used to calculate confidence
        # I gave up on fixing this style :(
        newframedData = np.ndarray.copy(framedData)
        if dimMaj == 0:
                for i in range(len(refinedClusters)):
                        array = refinedClusters[i].astype(int)
                        length = len(array)
                        if length > 1:
                                # the purpose of this block is to build a matrix of values from the blocks belonging to the same cluster
                                # and collapse it to an array in such a way that the resulting array has as many values as could be
                                # present in any of the blocks found to not disagree
                                predictingMergerMat = newframedData[array[0], :]
                                for r in range(len(array) - 1):
                                        predictingMergerMat = np.vstack([predictingMergerMat, framedData[array[r + 1], :]])

                                predictingMerger = np.empty(len(framedData[array[0], :]))
                                predictingMerger[:] = np.nan
                                for r in range(len(predictingMerger)):
                                        if np.sum(np.isnan(predictingMergerMat[:, r])) != len(predictingMergerMat[:, r]):
                                                valIndex = np.where(np.isfinite(predictingMergerMat[:, r]))
                                                predictingMerger[r] = predictingMergerMat[valIndex[0][0], r]
                                #*******************************************************************************************************

                                # this block is for approach 1 in which the only values that are predicted are those from the predicting
                                # clustering
                                if campaignObject.confAssignmentChoice == 'Conf1':
                                        for k in range(len(array)):
                                                newframedData[array[k], :] = predictingMerger
                                                for j in range(len(framedData[array[k], :])):
                                                        if np.isfinite(framedData[array[k], j]):
                                                                vAccData[array[k], j] = 1
                                                        elif np.isfinite(newframedData[array[k], j]):
                                                                vAccData[array[k], j] = \
                                                                        np.sum(predictingMerger == framedData[array[k], :])/len(framedData[array[k], :])
                                #*******************************************************************************************************

                                # this block is for approach 2 in which the values predicted are those from the predicting clustering
                                # as well as from the assignment of the mode to values that cannot be predicted yet from the clustering
                                if campaignObject.confAssignmentChoice == 'Conf2':
                                        pmCommon = stats.mode(predictingMerger, axis=None)
                                        for k in range(len(array)):
                                                newframedData[array[k], :] = predictingMerger
                                                for j in range(len(framedData[array[k], :])):
                                                        if np.isfinite(framedData[array[k], j]):
                                                                vAccData[array[k], j] = 1
                                                        elif np.isfinite(newframedData[array[k], j]):
                                                                vAccData[array[k], j] = \
                                                                        np.sum(predictingMerger == framedData[array[k], :])/len(framedData[array[k], :])
                                                        elif np.isnan(newframedData[array[k], j]):
                                                                vAccData[array[k], j] = np.prod([
                                                                        pmCommon.count/np.sum(np.isfinite(predictingMerger)),
                                                                        np.sum(np.isfinite(predictingMerger))/len(predictingMerger),
                                                                        ])
                                        # we only assign the mode after we've calculated the confidence as we use the overlap of the
                                        # predictive merger with the block in question and the absence of a prediction to know that we
                                        # should be calculating the accuracy of the given mode assignment
                                        predictingMerger[np.isnan(predictingMerger)] = pmCommon.mode
                                        for k in range(len(array)):
                                                newframedData[array[k], :] = predictingMerger
                        if length == 1:
                                if campaignObject.confAssignmentChoice == 'Conf2':
                                        pmCommon = stats.mode(newframedData[array[0], :])
                                        for j in range(len(framedData[array[0], :])):
                                                if np.isfinite(newframedData[array[0], j]):
                                                        vAccData[array[0], j] = 1
                                                elif np.isnan(newframedData[array[0], j]):
                                                    vAccData[array[0], j] = np.prod([
                                                            pmCommon.count/np.sum(np.isfinite(newframedData[array[0], :])),
                                                            np.sum(np.isfinite(newframedData[array[0], :]))/len(newframedData[array[0], :]),
                                                    ])
                                        array0 = newframedData[array[0], :]
                                        array0[np.isnan(array0)] = pmCommon.mode
                                        newframedData[array[0], :] = array0
                                #*******************************************************************************************************

        if dimMaj == 1:
                for i in range(len(refinedClusters)):
                        array = refinedClusters[i].astype(int)
                        length = len(array)
                        if length > 1:
                                predictingMergerMat = newframedData[:, array[0]]
                                for r in range(len(array) - 1):
                                        predictingMergerMat = np.vstack([predictingMergerMat, framedData[:, array[r + 1]]])

                                predictingMerger = np.empty(len(framedData[:, array[0]]))
                                predictingMerger[:] = np.nan
                                for r in range(len(predictingMerger)):
                                        if np.sum(np.isnan(predictingMergerMat[:, r])) != len(predictingMergerMat[:, r]):
                                                valIndex = np.where(np.isfinite(predictingMergerMat[:, r]))
                                                predictingMerger[r] = predictingMergerMat[valIndex[0][0], r]

                                if campaignObject.confAssignmentChoice == 'Conf1':
                                        for k in range(len(array)):
                                                newframedData[:, array[k]] = predictingMerger
                                                for j in range(len(framedData[:, array[k]])):
                                                        if np.isfinite(framedData[j, array[k]]):
                                                                vAccData[j, array[k]] = 1
                                                        elif np.isfinite(newframedData[j, array[k]]):
                                                                vAccData[j, array[k]] = \
                                                                        np.sum(predictingMerger == framedData[:, array[k]]) / len(framedData[:, array[k]])

                                if campaignObject.confAssignmentChoice == 'Conf2':
                                        pmCommon = stats.mode(predictingMerger, axis=None)
                                        for k in range(len(array)):
                                                newframedData[:, array[k]] = predictingMerger
                                                for j in range(len(framedData[:, array[k]])):
                                                        if np.isfinite(framedData[j, array[k]]):
                                                                vAccData[j, array[k]] = 1
                                                        elif np.isfinite(newframedData[j, array[k]]):
                                                                vAccData[j, array[k]] = \
                                                                        np.sum(predictingMerger == framedData[:, array[k]]) / len(framedData[:, array[k]])
                                                        elif np.isnan(newframedData[j, array[k]]):
                                                                vAccData[j, array[k]] = np.prod([
                                                                        pmCommon.count/np.sum(np.isfinite(predictingMerger)),
                                                                        np.sum(np.isfinite(predictingMerger))/len(predictingMerger),
                                                                        ])

                                        predictingMerger[np.isnan(predictingMerger)] = pmCommon.mode
                                        for k in range(len(array)):
                                                newframedData[:, array[k]] = predictingMerger
                        if length == 1:
                                if campaignObject.confAssignmentChoice == 'Conf2':
                                        pmCommon = stats.mode(newframedData[:, array[0]])
                                        for j in range(len(framedData[:, array[0]])):
                                                if np.isfinite(newframedData[j, array[0]]):
                                                        vAccData[j, array[0]] = 1
                                                elif np.isnan(newframedData[j, array[0]]):
                                                        vAccData[j, array[0]] = np.prod([
                                                                pmCommon.count / np.sum(np.isfinite(newframedData[:, array[0]])),
                                                                np.sum(np.isfinite(newframedData[:, array[0]]))/len(newframedData[:, array[0]]),
                                                        ])
                                        array0 = newframedData[:, array[0]]
                                        array0[np.isnan(array0)] = pmCommon.mode
                                        newframedData[:, array[0]] = array0

        # this block pertains to Approach 2, but must be completed last as it assigns the mode when necessary when even
        # under the circumstances that the mode has been used, it's only been used in the clusters and thus this block
        # assigns the mode to the rest of the space, not in a cluster
        if campaignObject.confAssignmentChoice == 'Conf2':
                mostCommon = stats.mode(framedData, axis=None)
                vAccData[np.isnan(vAccData)] = np.prod([
                        mostCommon.count/np.sum(np.isfinite(np.reshape(vAccData, campaignObject.yAccuracy.shape))),
                        np.sum(np.isnan(np.reshape(vAccData, campaignObject.yAccuracy.shape))) /
                        len(np.reshape(vAccData, campaignObject.yAccuracy.shape)),
                ])
                newframedData[np.isnan(newframedData)] = mostCommon.mode
        #*******************************************************************************************************************

        # assign the predictions from the given row or column clustering to the campaignObject "initial predictions"
        campaignObject.initPredictions[:, :, dimMaj] = newframedData

        # reshape the confidences calculated by individual clustering so that it can be merged with the other
        res = np.reshape(vAccData, campaignObject.yAccuracy.shape)

        # heat-map for tracking confidence over the course of the campaign
        campaignObject.filenames3.append(heatMapblocks(vAccData, dimMaj, campaignObject))

        return res, refinedClusters
