# actLearn_contModularized is an active learner for continuous variable modeling campaigns
# Active learners select experiments corresponding to the least confident
# predictions as determined during modeling
# This active learner samples randomly from the least confident predictions

import numpy as np

def main(campaignObject):

    # set the random seed for the active learner
    if hasattr(campaignObject, 'randoseed'):
        np.random.seed(campaignObject.randoseed)
    else:
        np.random.seed(32)

    # retrieve confidences pertaining to the predictions made in the modeler
    confidence = campaignObject.yAccuracy
    allNaNs = np.isnan(confidence)
    confidence[allNaNs] = 0
    database = campaignObject.data
    haveData = [database.transform_ind(i) for i in database.storeind]

    # keep track of the number of indices for which there is no data and only
    # predictions
    noData = list(range(len(campaignObject.ESS.listTuples)))
    for ind in haveData:
        noData.remove(ind)

    # what is the lowest confidence prediction
    minConf = min(confidence)
    if minConf >= 0.90:
        # quality control step
        dataRequested = []
        print('Stopping: All predicted confidences above 0.9')
        return dataRequested
    else:
        # identify all combinations for which confidence is within 2 standard
        # deviations of the mean of the total array (~a radius like metric,
        # but with the min as the center)
        confCount = campaignObject.confCount
        bar = 0.9
        barIndices =[]
        for j in range(len(noData)):
            if confidence[noData[j]] <= bar:
                barIndices.append(noData[j])

        # active learning can no longer commence once the number of experiments
        # to perform is fewer than that of the batch size
        if confCount <= len(barIndices):
            dataRequested = np.random.choice(
                np.asarray(barIndices), confCount, replace=False)
        else:
            dataRequested = []
            print('Stopping: Not enough points with confidence below 0.9 ('
                  + str(len(barIndices))
                  + ') to fill a batch ('+str(confCount)+')')

        return dataRequested
