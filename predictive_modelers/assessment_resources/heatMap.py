# Author: Bria Persaud and Eunice Chen
# Date: June 30, 2019
# Purpose: make a heat-map to visualize the progress of the categorical modeler

import os
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import copy


def main(campaignObject, dimMaj):
    # One figure, but 3 subplots- first is of the current predictions, second
    # acquired data and third, the ground truth
    plt.clf()
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Batch " + str(campaignObject.iterate))
    axs[0].set_title("Model Predictions")
    axs[1].set_title("Acquired Data")
    axs[2].set_title("Ground Truth")

    # set ground truth graph
    ideal = campaignObject.groundTruth
    ideal = np.reshape(ideal, (3, 9))
    axs[2] = sns.heatmap(ideal, linewidth=0.5, ax=axs[2], vmin=0, vmax=5)

    # right now, the code is only plotting the combined predictions (combination
    # of row and column space)
    preds = copy.deepcopy(campaignObject.finalPredictions)

    # uncomment to plot the predictions from the row and column space instead
    '''
    preds = copy.deepcopy(campaignObject.initPredictions[:, :, dimMaj])
    '''

    # add predictions to the heat-map, if there are any
    hasPrediction = False
    for row in preds:
        for col in row:
            if np.isfinite(col):
                hasPrediction = True

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

    acquiredData = np.reshape(data, (3, 9))

    # set the graph for the acquired data
    axs[1] = sns.heatmap(acquiredData, linewidth=0.5, ax=axs[1], vmin=0, vmax=5)

    tempdirectory1 = campaignObject.plotting.intDir + 'heatmaptempdir1_/'
    if not os.path.exists(tempdirectory1):
        os.makedirs(tempdirectory1)

    if not hasPrediction:
        # if there are no predictions, show a blank graph for the predictions
        axs[0].axis('off')
        plotName = tempdirectory1 + "HeatMap_batch_" + \
                   str(campaignObject.iterate) + "_with_" + str(dimMaj) + "_dimMaj.jpg"
        # save the filename of the heat-map generated from this batch
        plt.savefig(plotName)
        plt.close('all')
    else:
        # take out the acquired data from the predictions
        # this step should be unnecessary
        for i in haveData:
            r, c = database.transform_num(i)
            preds[r][c] = np.nan

        # set the heat-map for the predictions
        sns.heatmap(preds, linewidth=0.5, mask=preds == np.nan, ax=axs[0], vmin=0, vmax=5)
        # if you show the plot then try to save it, a blank image will be saved
        # plt.show()
        # save the filename of the heat-map generated from this batch
        plotName = tempdirectory1 + "HeatMap_batch_" + \
                   str(campaignObject.iterate) + "_with_" + str(dimMaj) + "_dimMaj.jpg"
        plt.savefig(plotName)
        plt.close('all')

    return plotName
