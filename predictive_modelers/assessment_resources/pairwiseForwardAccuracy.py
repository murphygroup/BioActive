# calculates point-wise forward accuracy and designed for a campaign with two independent categorical variables
# addresses need to assess accuracy in campaigns when data is comprised of unlabeled data and labels can change
# arbitrarily from round to round

import numpy as np


def main(campaignObject, givenArray, dimMaj=0, merged=False):

    # below code is for forward accuracy.
    # for each batch obtained, checks if the new info obtained matches what was in the predictions.
    # for batch size of 3, should be checking 3 items each time.
      
    points = 0
    # pointsPossible = n + n-1 + n-2 + ... 1 where n is batchSize - 1
    pointsPossible = 0
    predictionsMade = 0
    allIndices = range(len(campaignObject.ESS.listTuples))
    # ESS_size = the number of indexes in the experimental space
    ESS_size = len(set(campaignObject.ESS.listTuples[:, 0])) * len(set(campaignObject.ESS.listTuples[:, 1]))
    dataShape = [len(set(campaignObject.ESS.listTuples[:, 0])), len(set(campaignObject.ESS.listTuples[:, 1]))]
    
    # predictions represents the predictions
    predictions = np.reshape(givenArray, dataShape)
    groundTruth = np.reshape(np.ndarray.copy(campaignObject.groundTruth), dataShape)
                    
    # find acquired data- this is the same code as in catmodel and contmodel
    database = campaignObject.data
    haveData = [database.transform_ind(i) for i in database.storeind]
    if haveData:
        allIndices = range(len(campaignObject.ESS.listTuples))
        data = np.empty([len(campaignObject.ESS.listTuples), len(campaignObject.ESS.dVars)])
        data[:] = np.nan

        for i in allIndices:
            if i in haveData:
                data[i] = database.retrieve(campaignObject.ESS.listTuples[i])

    data = np.reshape(data, dataShape)
    dataUnshaped = np.reshape(data, ESS_size)
    batchValues = []
    batchIndexes = campaignObject.lastDataRequest
    for index in batchIndexes:
        batchTuple = campaignObject.data.transform_num(index)
        batchValues.append(data[batchTuple[0]][batchTuple[1]])
    for i in range(len(batchIndexes)):
        pointsPossible = pointsPossible + i  # because each index is compared to all the indices after it
    
    # ONLY COMPARE IT TO OTHER THINGS IN ITS BATCH
    for index in range(len(batchIndexes)):
        # notice the following line is added in the case of fwd accuracy
        i = batchIndexes[index]
        iTuple = campaignObject.data.transform_num(i)  # this fxn turns index into tuple
        current = predictions[iTuple[0]][iTuple[1]]
        
        for index2 in range(index+1, len(batchIndexes)):
            j = batchIndexes[index2]
            jTuple = campaignObject.data.transform_num(j)
            after = predictions[jTuple[0]][jTuple[1]]
            
            # is the relationship as expected?
            shouldBeSame = (groundTruth[iTuple[0]][iTuple[1]] == groundTruth[jTuple[0]][jTuple[1]])
            isSame = (current == after)
            if (shouldBeSame == isSame):
                points = points + 1
                
    for i in allIndices:
        iTuple = campaignObject.data.transform_num(i)  # this fxn turns index into tuple
        current = predictions[iTuple[0]][iTuple[1]]
        if np.isnan(current) == False:
            predictionsMade = predictionsMade + 1
            
    total = len(campaignObject.ESS.listTuples)
    # predictionsMade == 0 when ESC = 0 and ESC = 1
    dataCollected = 0
    for i in dataUnshaped:
        if np.isnan(i) == False:
            dataCollected = dataCollected + 1
    ESC = dataCollected/ total
    if pointsPossible != 0:
        acc = (points / pointsPossible) * 100
    else:  # if no ESC has been covered yet
        acc = 0   
    # in the case where 100% of ESC is covered, there would be no predictions so this graph cannot extend to 1
    
    '''
    print("ACC FXN3 ", predictionsMade, points, pointsPossible, total, acc, ESC)
    '''
    
    if merged == False:
        if dimMaj == 0:
            campaignObject.forwardAccuracyRows.append(acc)
            campaignObject.myESCTest3_0.append(ESC)
            print(campaignObject.myESCTest3_0, campaignObject.forwardAccuracyRows)
            
        if dimMaj == 1:
            campaignObject.forwardAccuracyCols.append(acc)
            campaignObject.myESCTest3_1.append(ESC)
            
    if merged == True:
        if dimMaj == 0:
            campaignObject.forwardAccuracy.append(acc)
            '''
            campaignObject.myESCTest3_0.append(ESC)
            '''
    
            print(campaignObject.myESCTest3_0, campaignObject.forwardAccuracy)
        if dimMaj == 1:
            campaignObject.forwardAccuracyExtra.append(acc)
            '''
            campaignObject.myESCTest3_1.append(ESC)
            '''

    return
