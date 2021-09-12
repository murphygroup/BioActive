# poolactLearn is an active learning module designed for pool based active learning

import numpy as np


def main(campaignObject):
    """
    Based on the current accuracies, this function returns a variable DataRequested which holds
    the indices for which Data should be extracted from datafile
    """

    # set the random seed for the active learner
    if hasattr(campaignObject, 'randoseed'):
        np.random.seed(campaignObject.randoseed)
    else:
        np.random.seed(7)

    # retrieve confidences pertaining to the predictions made in the modeller
    allIndices = list(range(len(campaignObject.yAccuracy)))
    # print(allIndices)

    # The following variable confidence is a local copy of campaignObject.yaccuracy
    confidence = np.ndarray.copy(campaignObject.yAccuracy)
    # print(confidence)

    # Indices for which Data is yet to be extracted from datafile
    noData = [i for i, val in enumerate(confidence) if val < 1]
    # print(noData)

    # the active learner will randomly sample from the experiments to perform for which the campaign has the least
    # confident predictions; in the case that there are too few experiments of the minimum confidence to meet the batch
    # size, then using the second criterion, other experiments for which the prediction has an incrementally higher
    # confidence are allowed into the pool.

    if len(noData) < campaignObject.confCount:
        # quality control step--should not proceed if there are fewer
        # predictions than the predetermined number of experiments to do
        # iteration to iteration
        dataRequested = []
        print('data exhausted')
    else:
        # active learning component
        if campaignObject.flag:

            # active learning when there is some confidence about predictions made
            if len(noData) != len(allIndices):

                # imagine we have a matrix in which every feature set's distance is measured from every other
                # feature set, with the diagonal being all 0s. let's take the ith index, whose corresponding
                # confidence satisfies minVal and in the subsequent round, select from those i+1 indice(s)
                # who also satisfy minVal and are the furthest away

                complete_set=[]
                while len(complete_set) < campaignObject.confCount:
                    nextSmallest = np.amin(confidence)
                    leastConfidenceIndices = []

                    #Build a set of leastConfidenceIndices based on the smallest value currently available
                    for i in range(len(campaignObject.yAccuracy)):
                        if confidence[i] == nextSmallest:
                            leastConfidenceIndices.append(i)

                            # Set local accuracy to be one if the index is to be picked
                            confidence[i] = 1

                    indicesLength = len(leastConfidenceIndices)

                    # check if this set is larger than number required
                    if indicesLength >= campaignObject.confCount - len(complete_set):
                        distMat = np.ndarray.copy(campaignObject.featsDistances)

                        # For indices picked earlier, set those distances to be zero
                        distMat[:,complete_set] = 0

                        # For indices which already have been picked in any earlier iteration, set those distance to zero
                        ind = campaignObject.yAccuracy == 1
                        distMat[:,ind] = 0
 
                        # Take only the Upper Triangular Matrix
                        distMat = np.triu(distMat)
 
                        # Pick confCount number of indices which have the greatest distance in the matrix
                        # and add into the current list
                        x = distMat.argsort()[0, -campaignObject.confCount:]
                        complete_set += list(x)

                    # When complete set falls short of confCount, add whole of leastConfidenceIndices
                    else:
                        complete_set+=leastConfidenceIndices
                    print('complete_set')
                    print(complete_set)

                dataRequested = np.random.choice(complete_set, campaignObject.confCount, replace=False)
                print('First case')
                print(dataRequested)


            # random selection when nothing is known but active learning is desired
            else:
                dataRequested = np.random.choice(np.asarray(noData), campaignObject.confCount, replace=False)
                print('Second case')
                print(dataRequested)

        # random sampling component
        else:
            dataRequested = np.random.choice(np.asarray(noData), campaignObject.confCount, replace=False)
            print('third case')
            print(dataRequested)

    return dataRequested
