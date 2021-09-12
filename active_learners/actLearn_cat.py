# actLearn_cat is an active learner for categorical variable modeling campaigns
# Active learners select experiments corresponding to the least confident
# predictions as determined during modeling
# This active learner samples randomly from the least confident predictions and
# when the modeler provides mutual information for the predictions, e.g.
# "clusterMatrix," it discriminates against selections that don't provide
# additional information to the modeler

import numpy as np

# The heatmap_activeLearner provides heatmaps so as to assess and evaluate
# how the active learner is choosing experiments to conduct
from heatmap_activeLearner import main as heatmap

def main(campaignObject):

    # set the random seed for the active learner
    if hasattr(campaignObject, 'randoseed'):
        np.random.seed(campaignObject.randoseed)
    else:
        np.random.seed(7)

    # retrieve confidences pertaining to the predictions made in the modeler
    database = campaignObject.data
    haveData = [database.transform_ind(i) for i in database.storeind]
    allIndices = list(range(len(campaignObject.yAccuracy)))

    dataShape = [len(set(campaignObject.ESS.listTuples[:, 0])),
        len(set(campaignObject.ESS.listTuples[:, 1]))]

    confidence = np.ndarray.copy(campaignObject.yAccuracy)
    keepConfidence = np.ndarray.copy(confidence)

    # keep track of the number of indices for which there is no data and only
    # predictions
    noData = []
    for ind in allIndices:
        if ind in haveData:
            pass
        else:
            noData.append(ind)

    # what is the first least confidence
    minVal = np.amin(confidence)

    # the active learner will randomly sample from the experiments to perform
    # for which the campaign has the least confident predictions; in the case
    # that there are too few experiments of the minimum confidence to meet the
    # batch size, then using the second criterion, other experiments for which
    # the prediction has an incrementally higher confidence are allowed into
    # the pool.

    # active learning heatmap assessment and evaluative tool
    if hasattr(campaignObject, "filenames2"):
        campaignObject.filenames2.append(heatmap(np.reshape(confidence,
                    [len(set(campaignObject.ESS.listTuples[:, 0])),
                    len(set(campaignObject.ESS.listTuples[:, 1]))]),
                    campaignObject))

    # active learning can no longer commence once the number of experiments
    # to perform is less than that of the batch size
    if len(noData) < campaignObject.confCount:
        # quality control step--should not proceed if there are fewer
        # predictions than the predetermined number of experiments to do
        # iteration to iteration
        dataRequested = []
        print('data exhausted')
    else:
        # active learning component
        if campaignObject.flag:
            # random selection when nothing is none
            if len(noData) == len(allIndices):
                dataRequested = np.random.choice(np.asarray(noData),
                campaignObject.confCount, replace=False)
            # active learning when there some confidence about predictions made
            else:
                # if modeler does not provide mutual information pertaining to
                # predictions made in modeler
                if campaignObject.activeLearning == 'non-clustering':
                    # block responsible for enlarging pool of experiments
                    # corresponding to least confident predictions when pool
                    # size drops beneath desired batch size
                    leastConfidenceIndices = []
                    for i in range(len(allIndices)):
                        if confidence[i] == minVal:
                            leastConfidenceIndices.append(allIndices[i])
                            confidence[i] = 1
                    indicesLength = len(leastConfidenceIndices)

                    nextSmallest = np.amin(confidence)
                    while indicesLength < campaignObject.confCount:
                        for i in range(len(allIndices)):
                            if confidence[i] == nextSmallest:
                                leastConfidenceIndices.append(allIndices[i])
                                confidence[i] = 1
                        indicesLength = len(leastConfidenceIndices)
                        nextSmallest = np.amin(confidence)

                    dataRequested = np.random.choice(leastConfidenceIndices,
                        campaignObject.confCount, replace=False)

                elif campaignObject.activeLearning == 'clustering':
                    # block responsible for enlarging pool of experiments
                    # corresponding to least confident predictions when pool
                    # size drops beneath desired batch size while simultaneously
                    # discriminating in the selection of experiments belonging
                    # to the same cluster
                    leastConfidenceIndices = []
                    accountedMI = []
                    for i in range(len(allIndices)):
                        pointMI = np.ndarray.tolist(
                            campaignObject.clusterMatrix[
                                np.unravel_index(allIndices[i], dataShape)
                                ])
                        if confidence[i] == minVal:
                            if pointMI not in accountedMI:
                                leastConfidenceIndices.append(allIndices[i])
                                confidence[i] = 1
                                accountedMI.append(pointMI)
                    indicesLength = len(leastConfidenceIndices)

                    nextSmallest = np.amin(confidence)
                    while indicesLength < campaignObject.confCount:
                        for i in range(len(allIndices)):
                            pointMI = np.ndarray.tolist(
                                campaignObject.clusterMatrix[
                                    np.unravel_index(allIndices[i], dataShape)
                                    ])
                            if confidence[i] == nextSmallest:
                                if pointMI not in accountedMI:
                                    leastConfidenceIndices.append(
                                        allIndices[i])
                                    confidence[i] = 1
                                    accountedMI.append(pointMI)
                        indicesLength = len(leastConfidenceIndices)

                        if indicesLength >= campaignObject.confCount:
                            break
                        elif indicesLength < campaignObject.confCount:
                            for i in range(len(allIndices)):
                                if confidence[i] == nextSmallest:
                                    leastConfidenceIndices.append(
                                        allIndices[i])
                                    confidence[i] = 1
                            indicesLength = len(leastConfidenceIndices)
                        nextSmallest = np.amin(confidence)

                    dataRequested = np.random.choice(
                        leastConfidenceIndices, campaignObject.confCount,
                            replace=False)

        # random sampling component
        else:
            dataRequested = np.random.choice(np.asarray(noData),
                campaignObject.confCount, replace=False)
        print('Data requested: indices')
        print(dataRequested)
        print('Data requested: uncertainties')
        print(keepConfidence[dataRequested].T)
            
    # print statement for wanting to know when predictions are NaNs or finite
    # values
    '''
    print(np.reshape(campaignObject.finalPredictions,
        campaignObject.yAccuracy.shape)[dataRequested])
        '''
        
    return dataRequested
