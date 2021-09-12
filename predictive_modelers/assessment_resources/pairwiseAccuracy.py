# calculates pairwise accuracy and designed for a campaign with two independent categorical variables
# addresses need to assess accuracy in campaigns when data is comprised of unlabeled data and labels can change
# arbitrarily from round to round

import numpy as np
import matplotlib.pylab as plt


def main(campaignObject, givenArray):

    # judges the accuracy of both predictions AND observations
    points = 0
    pointsPossible = 0
    predictionsMade = 0
    allIndices = range(len(campaignObject.ESS.listTuples))
    for i in allIndices:
        pointsPossible = pointsPossible + i  # because each index is compared to all the indices after it
     
    dataShape = [len(set(campaignObject.ESS.listTuples[:, 0])), len(set(campaignObject.ESS.listTuples[:, 1]))]
    # ESS_size = the # of indexes in the experimental space
    ESS_size = len(set(campaignObject.ESS.listTuples[:, 0])) * len(set(campaignObject.ESS.listTuples[:, 1]))
    
    predictions = np.reshape(givenArray, dataShape)
    groundTruth = np.reshape(np.ndarray.copy(campaignObject.groundTruth), dataShape)
                      
    database = campaignObject.data
    haveData = [database.transform_ind(i) for i in database.storeind]
    if haveData:
        allIndices = range(len(campaignObject.ESS.listTuples))
        data = np.empty([len(campaignObject.ESS.listTuples), len(campaignObject.ESS.dVars)])
        data[:] = np.nan

        for i in allIndices:
            if i in haveData:
                data[i] = database.retrieve(campaignObject.ESS.listTuples[i])
    # data = acquired data
    data = np.reshape(data, dataShape)
    
    # compares current to all things after it (if also compared to those before it, would be double counting)
    for i in allIndices:
        iTuple = campaignObject.data.transform_num(i)  # this fxn turns index into tuple
        current = predictions[iTuple[0]][iTuple[1]]
        if np.isnan(current) == False:
            predictionsMade = predictionsMade + 1
        else:
            current = data[iTuple[0]][iTuple[1]]
        for j in range(i+1, len(campaignObject.ESS.listTuples)):
            jTuple = campaignObject.data.transform_num(j)
            after = predictions[jTuple[0]][jTuple[1]]
            if np.isnan(after) == True:
                after = data[jTuple[0]][jTuple[1]]
            # is the relationship as expected?
            shouldBeSame = (groundTruth[iTuple[0]][iTuple[1]] == groundTruth[jTuple[0]][jTuple[1]])
            isSame = (current == after)
            if (shouldBeSame == isSame):
                points = points + 1

    total = len(campaignObject.ESS.listTuples)
    acc = (points / pointsPossible) * 100
    '''
    print("ACC FXN2 ", predictionsMade, points, pointsPossible, total, acc)
    '''
    
    plt.clf()
    plt.figure(1)
    
    fullSpaceAcc = acc

    # just for predictions
    points = 0
    pointsPossible = 0
    predictionsMade = 0
    allIndices = range(len(campaignObject.ESS.listTuples))

    # modified this to remove acquired from predictions
    
    predictions = np.reshape(givenArray, (ESS_size))
    database = campaignObject.data
    haveData = [database.transform_ind(i) for i in database.storeind]
    if haveData:
        allIndices = range(len(campaignObject.ESS.listTuples))

        for i in allIndices:
            if i in haveData:
                predictions[i] = np.nan
    predictions = np.reshape(predictions, dataShape)

    # compare current to all things after it (if also compared to those before it, would be double counting)
    for i in allIndices:
        iTuple = campaignObject.data.transform_num(i)  # this fxn turns index into tuple
        current = predictions[iTuple[0]][iTuple[1]]
        if np.isnan(current) == False:
            predictionsMade = predictionsMade + 1
            for j in range(i+1, len(campaignObject.ESS.listTuples)):
                jTuple = campaignObject.data.transform_num(j)
                after = predictions[jTuple[0]][jTuple[1]]
                if np.isnan(after) == False:
                    # is the relationship as expected?
                    shouldBeSame = (groundTruth[iTuple[0]][iTuple[1]] == groundTruth[jTuple[0]][jTuple[1]])
                    isSame = (current == after)
                    if (shouldBeSame == isSame):
                        points = points + 1
    # not penalized for nans but also doesnt benefit from them
    '''
    pointsPossible == 0 when predictionsMade is 0 or 1
    '''

    # also needs to account for when the model can't make predictions based off clusters, in which case predictions are
    # NaNs and are wrong
    campaignProgress = float(len(campaignObject.data.storeind))/np.prod(campaignObject.ESS.dimarr)
    if (predictionsMade == 0) & (campaignProgress < 1):  # this line should make the following elif statement obsolete
        predictionsAcc = 0
        return (fullSpaceAcc, predictionsAcc)
    elif predictionsMade == 0:
        predictionsAcc = None
        return (fullSpaceAcc, predictionsAcc)
    if predictionsMade == 1:
        predictionsAcc = 0
        return (fullSpaceAcc, predictionsAcc)
    for i in range(predictionsMade):
        pointsPossible = pointsPossible + i # because each index is compared to all the indices after it
    total = len(campaignObject.ESS.listTuples)
    # pointsPossible == 0 when predictionsMade is 0 or 1
    acc = (points / pointsPossible) * 100
    '''
    print("ACC FXN ", predictionsMade, points, pointsPossible, total, acc)
    '''
    
    plt.clf()
    plt.figure(1)
       
    predictionsAcc = acc

    return fullSpaceAcc, predictionsAcc
