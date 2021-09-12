# This file consists of the function(s) needed for plotting the graphs
# for each campaign.

# Created by Piratach Yoovidhya (piratacy)
# Date of Last Change: 05/09/2020

import matplotlib.pyplot as plt
import os

# Campaign plotting function
# How to use MultiPlot: specify the axis-object obtained from 
# plt.subplots in the ax argument (e.g. ax=axObject[0]), and 
# set multiPlot to true.
def plotAcc(campaignObject, ESC, ax=None, hasFPAcc=True, hasFwdAcc=True, 
        hasPredsAcc=True, hasTVD=False, hasClusters=False,
        ylim=None, multiPlot=False):

    legendLabels = []

    # requires ground truth
    if hasFPAcc:
        if ax is not None:
            ax.plot(ESC, campaignObject.accuracy_full)
        else:
            plt.plot(ESC, campaignObject.accuracy_full)
        legendLabels += ['Accuracy (Predictions + Obs)']

    # should have forward modeling for all cases
    if hasFwdAcc:
        if ax is not None:
            ax.plot(ESC, campaignObject.accuracy_forwardModeling)
        else:
            plt.plot(ESC, campaignObject.accuracy_forwardModeling)
        legendLabels += ['Accuracy (Forward Modeling)']
   
    # requires ground truth
    if hasPredsAcc:
        if ax is not None:
            ax.plot(ESC, campaignObject.accuracy_onlyPredictions)
        else:
            plt.plot(ESC, campaignObject.accuracy_onlyPredictions)
        legendLabels += ['Accuracy (Predictions Only)']

    # only available for continuous case for now
    if hasTVD:
        if ax is not None:
            ax.plot(ESC, campaignObject.total_var_distance)
        else:
            plt.plot(ESC, campaignObject.total_var_distance)
        legendLabels += ['Total Variation Distance']

    if hasClusters:
        if ax is not None:
            ax.plot(ESC, campaignObject.rowsCounted, 
                    ESC, campaignObject.colsCounted)
        else:
            plt.plot(ESC, campaignObject.rowsCounted, 
                    ESC, campaignObject.colsCounted)
        legendLabels += ['Rows', 'Columns']
        
    if ylim is not None:
        if ax is not None:
            ax.set_ylim(ylim)
        else:
            plt.set_ylim(ylim)

    if ax is not None:
        ax.legend(legendLabels)
    else:
        plt.legend(legendLabels)

    # no labels ot titles for multiPlot
    # do it outside the function
    if not multiPlot:

        plt.title(campaignObject.plotting.title)
        plt.ylabel(campaignObject.plotting.ylabel)
        plt.xlabel(campaignObject.plotting.xlabel)

        try:
            plt.savefig(campaignObject.plotting.filename)
        except:
            if not os.path.isdir(campaignObject.plotting.intDir):
                os.makedirs(campaignObject.plotting.intDir)
            plt.savefig(campaignObject.plotting.filename)

        plt.close('all')

    return legendLabels

