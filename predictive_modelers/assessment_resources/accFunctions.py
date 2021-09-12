# This file contains functions that are used to report different types of 'accuracy'
# every iteration. The functions are currently split into functions for models with
# discrete data, and models with continuous data.

# Created by Piratach Yoovidhya (piratacy)
# Date of last change: 05/09/2020

import numpy as np
from sklearn.metrics import accuracy_score

# common optional argument explanations:
# - acc_f is the function used to evaluate accuracy using 
# ground truth and predictions as arguments
# - defAcc is the default value appended when nothing is calculated
# - nan_opt is used to specify what to do when the value is NaN
# - dataInd is used when the prediction data we want is at a specific
# index of fullSpace_last[i]
# - GTInd is the same as above but for "ground truth" values

################ DISCRETE MODELS ################ 

# reports forward accuracy
# for this case, "data" is acts similarly to a ground truth
def forwardAcc(campaignObject, fullSpace_last, data, acc_f=accuracy_score, defAcc=0, 
        nan_opt="break", nan_batchVal=None, dataInd=None, GTInd=None):
    batchGT = []
    batchPred = []
    for i in campaignObject.lastDataRequest:
        if np.isnan(fullSpace_last[i]):
            if nan_opt == "break":
                break
            elif nan_opt == "pass": # ignore NaN values
                pass
            else: # specify value to append
                batchPred.append(nan_opt)
                if nan_batchVal is not None:
                    batchGT.append(nan_batchVal)
                else:
                    batchGT.append(data[i])
        else:
            if dataInd is None:
                batchPred.append(fullSpace_last[i])
            else:
                batchPred.append(fullSpace_last[i][dataInd])
            if GTInd is None:
                batchGT.append(data[i])
            else:
                batchGT.append(data[i][GTInd])

    if len(batchGT) < 1:
        campaignObject.accuracy_forwardModeling.append(defAcc)
    else:
        campaignObject.accuracy_forwardModeling.append(acc_f(batchGT, batchPred))
    #print('Forward: '+str(campaignObject.accuracy_forwardModeling))

    return (batchPred, batchGT)

# reports full space accuracy
# requires groundTruth
def fullSpaceAcc(campaignObject, fullSpace, groundTruth, allIndices, acc_f=accuracy_score, 
        defAcc=0, nan_opt="break", dataInd=None, GTInd=None):
    fullSpace_noNaNs = []
    groundTruth_adjusted = []
    for i in allIndices:
        if np.isnan(fullSpace[i]):
            if nan_opt == "break":
                break
            elif nan_opt == "pass": # ignore NaN values
                pass
            else: # specify value to append
                fullSpace_noNaNs.append(nan_opt)
                if GTInd is None:
                    groundTruth_adjusted.append(groundTruth[i])
                else:
                    groundTruth_adjusted.append(groundTruth[i][GTInd])
        else:
            if dataInd is None:
                fullSpace_noNaNs.append(fullSpace[i])
            else:
                fullSpace_noNaNs.append(fullSpace[i][dataInd])
            if GTInd is None:
                groundTruth_adjusted.append(groundTruth[i])
            else:
                groundTruth_adjusted.append(groundTruth[i][GTInd])

    if len(groundTruth_adjusted) < 1:
        campaignObject.accuracy_full.append(defAcc)
    else:
        campaignObject.accuracy_full.append(acc_f(groundTruth, fullSpace_noNaNs))
    #print('Full space: '+str(campaignObject.accuracy_full))

    return (fullSpace_noNaNs, groundTruth)

# reports predictions only accuracy
# requires groundTruth
def predsOnlyAcc(campaignObject, fullSpace, groundTruth, allIndices, haveData,
        acc_f=accuracy_score, defAcc=0, nan_opt="break", dataInd=None, GTInd=None):
    onlyPredsGT = []
    onlyPredsFP = []
    for i in allIndices:
        if i in haveData:
            continue
        if np.isnan(fullSpace[i]):
            if nan_opt == "break":
                break
            elif nan_opt == "pass": # ignore NaN values
                pass
            else: # specify value to append
                onlyPredsFP.append(nan_opt)
                if GTInd is None:
                    onlyPredsGT.append(groundTruth[i])
                else:
                    onlyPredsGT.append(groundTruth[i][GTInd])
        else:
            if dataInd is None:
                onlyPredsFP.append(fullSpace[i])
            else:
                onlyPredsFP.append(fullSpace[i][dataInd])
            if GTInd is None:
                onlyPredsGT.append(groundTruth[i])
            else:
                onlyPredsGT.append(groundTruth[i][GTInd])

    if len(onlyPredsGT) < 1:
        campaignObject.accuracy_onlyPredictions.append(defAcc)
    else:
        campaignObject.accuracy_onlyPredictions.append(acc_f(onlyPredsGT, onlyPredsFP))
    #print('Preds: '+str(campaignObject.accuracy_onlyPredictions))

    return (onlyPredsGT, onlyPredsFP)

# reporting accuracy when not haveData is true
def noDataAcc(campaignObject, defaultAcc=0, defaultTVD=1):
    campaignObject.accuracy_forwardModeling.append(defaultAcc)
    campaignObject.accuracy_full.append(defaultAcc)
    campaignObject.accuracy_onlyPredictions.append(defaultAcc)
    campaignObject.total_var_distance.append(defaultTVD)

################ CONTINUOUS MODELS ################ 

# used to determine "accuracy" for continuous variables
def contAccuracy(groundTruth, preds):
    acc = 0
    for i in range(len(groundTruth)):
        GTVal = groundTruth[i]
        pred = preds[i]
        error = abs(GTVal - pred)/GTVal 
        if (error >= 1):
            # must be between 0 and 1
            error = 1
        acc += 1 - error
    return acc/len(preds)

# reporting forward accuracy using prediction functions
def forwardAccCont(campaignObject, previousPF, X, Y, acc_f=contAccuracy):
    batchGT = [] # "ground truth" = prev round's observed data
    batchPreds = [] # previous batch predictions
    index = 0

    for elem in campaignObject.lastDataRequest: # elem = two indep var.
        elem2 = np.atleast_2d(elem) # 2D array
        pred = previousPF.predict(elem2)
        if (np.isnan(pred)):
            break

        for i in range(len(X)):
            if (X[i][0] == elem[0] and X[i][1] == elem[1]):
                index = i
                break
        observedData = Y[index]

        batchPreds.append(pred)
        batchGT.append(observedData)
    
    if (len(batchGT) < 0):
        campaignObject.accuracy_forwardModeling.append(0)
    else:
        campaignObject.accuracy_forwardModeling.append(acc_f(batchGT, batchPreds))

###### CALCULATING THE TOTAL VARIATION DISTANCE ######

# function has to be of the form z = ax + by + c
# finds exact volume of plane given certain bounds
def findVolume(a, b, c, low, high):
    range1 = high - low
    range2 = (high**2) - (low**2)
    return range1 * (0.5 * a * range2 + 0.5 * b * range2 + c * range1)

# very rough approximation of the volume
# might not be needed
def iterateVolume(a, b, c, low, high, step=0.1):
    vol = 0
    f = lambda x, y: (a*x + b*y + c)
    (lo, hi) = (int(low/step), int(high/step))
    for i in range(lo, hi):
        for j in range(lo, hi):
            (x, y) = (i * step, j * step)
            vol += abs(x * y * f(x, y))
    return vol

# calculates the total variation distance of two linear functions
# the functions consist of two independent variables of the form
# z = ax + by + c. 
def reportTVD(campaignObject, predFunction, groundTruth, step=0.1):

    from scipy.integrate import dblquad

    low = campaignObject.ESS.iVars[0][1]
    high = campaignObject.ESS.iVars[0][2]
    bounds = (low, high)

    # defining the normalized prediction function
    predCoef = predFunction.coef_
    a1, b1, c1 = predCoef[0], predCoef[1], predFunction.intercept_
    predVol = findVolume(a1, b1, c1, low, high)
    norm_Pred = lambda x, y: (x*a1 + y*b1 + c1) / predVol

    # defining the normalized groundTruth function
    # perhaps this can be stored without having to be re-calculated
    gtCoef = groundTruth.coef_
    a2, b2, c2 = gtCoef[0], gtCoef[1], groundTruth.intercept_
    gtVol = findVolume(a2, b2, c2, low, high)
    norm_GT = lambda x, y: (x*a2 + y*b2 + c2) / gtVol

    # function to be integrated
    diff_f = lambda x, y: abs(norm_Pred(x, y) - norm_GT(x, y))

    (TVD, err) = dblquad(diff_f, low, high, lambda x: low, lambda x: high)

    campaignObject.total_var_distance.append(TVD)

    return TVD

