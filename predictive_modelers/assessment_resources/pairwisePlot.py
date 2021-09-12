# serves to plot pairwise accuracy assessment data from categorical campaigns in which the data is unlabeled

import os
import numpy as np
import datetime
import matplotlib.pyplot as plt


def main(campaignObject, ESC):
            dirStore = campaignObject.plotting.intDir
            if not os.path.exists(dirStore):
                os.makedirs(dirStore)

            plt.clf()
            plt.figure(1)

            # note: we also need to be mindful of how this resource could be used when the space is not exhausted

            # -1 because cannot have an ESC of 1 for Just Predictions, since at ESC=1, there are no predictions
            # the accuracy of no predictions cannot be reported
            minimum1 = np.amax([len(campaignObject.predictionAccuracyRows), len(ESC)]) - \
                       np.amin([len(campaignObject.predictionAccuracyRows), len(ESC)])
            '''
            minimum2 = np.amax([len(campaignObject.fullSpaceAccuracyRows), len(ESC)]) - \
                       np.amin([len(campaignObject.fullSpaceAccuracyRows), len(ESC)])
            '''

            if ESC[-1] != 1:
                plt.plot(ESC, campaignObject.predictionAccuracyRows)
            elif ESC[-1] == 1:
                plt.plot(ESC[:-1], campaignObject.predictionAccuracyRows)
            plt.plot(ESC, campaignObject.fullSpaceAccuracyRows)
            plt.plot(campaignObject.myESCTest3_0, campaignObject.forwardAccuracyRows)
            plt.title('Accuracy over Course of Campaign (Rows)')
            plt.ylabel('Pointwise Accuracy')
            plt.xlabel('Experimental space coverage')
            plt.legend(["Predictions Only", "Full Space: Predictions+Observations", "Forward Accuracy"],
                       loc = 'upper left')
            plt.savefig(dirStore + 'pointbypointAcc' + campaignObject.myName +
                        "DimMaj0" + str(datetime.datetime.now()) + '.tif')
            
            plt.clf()
            plt.figure(1)
            # -1 because cannot have an ESC of 1 for Just Predictions, since at ESC=1, there are no predictions
            # the accuracy of no predictions cannot be reported
            minimum1 = np.amax([len(campaignObject.predictionAccuracyCols), len(ESC)]) - \
                       np.amin([len(campaignObject.predictionAccuracyCols), len(ESC)])
            minimum2 = np.amax([len(campaignObject.fullSpaceAccuracyCols), len(ESC)]) - \
                       np.amin([len(campaignObject.fullSpaceAccuracyCols), len(ESC)])
            if ESC[-1] != 1:
                plt.plot(ESC, campaignObject.predictionAccuracyCols)
            elif ESC[-1] == 1:
                plt.plot(ESC[:-1], campaignObject.predictionAccuracyCols)

            plt.plot(ESC, campaignObject.fullSpaceAccuracyCols)
            plt.plot(campaignObject.myESCTest3_1, campaignObject.forwardAccuracyCols)
            plt.title('Accuracy over Course of Campaign (Columns)')
            plt.ylabel('Pointwise Accuracy')
            plt.xlabel('Experimental space coverage')
            plt.legend(["Predictions Only", "Full Space: Predictions+Observations", "Forward Accuracy"],
                       loc = 'upper left')
            plt.savefig(dirStore + 'pointbypointAcc' + campaignObject.myName +
                        "DimMaj1" + str(datetime.datetime.now()) + '.tif')
            
            plt.clf()
            plt.figure(1)
            # -1 because cannot have an ESC of 1 for Just Predictions, since at ESC=1, thereare no predictions
            # the accuracy of no predictions cannot be reported
            minimum1 = np.amax([len(campaignObject.predictionAccuracy), len(ESC)]) - \
                       np.amin([len(campaignObject.predictionAccuracy), len(ESC)])
            minimum2 = np.amax([len(campaignObject.fullSpaceAccuracy), len(ESC)]) - \
                        np.amin([len(campaignObject.fullSpaceAccuracy), len(ESC)])
            if ESC[-1] != 1:
                plt.plot(ESC, campaignObject.predictionAccuracy)
            elif ESC[-1] == 1:
                plt.plot(ESC[:-1], campaignObject.predictionAccuracy)

            plt.plot(ESC, campaignObject.fullSpaceAccuracy)
            plt.plot(campaignObject.myESCTest3_0, campaignObject.forwardAccuracyRows)
            plt.title('Merged Predictions Accuracy over Course of Campaign')
            plt.ylabel('Pointwise Accuracy')
            plt.xlabel('Experimental space coverage')
            plt.legend(["Predictions Only", "Full Space: Predictions+Observations", "Forward Accuracy"],
                       loc = 'upper left')
            plt.savefig(dirStore + 'FpointbypointAcc' + campaignObject.myName + str(datetime.datetime.now()) + '.tif')
            
            plt.clf()
            plt.figure(1)
            plt.plot(ESC, campaignObject.rowsCounted, ESC, campaignObject.colsCounted)
            plt.legend(['Clusters in Rows', 'Clusters in Columns'], loc='upper left')
            plt.title('Number of Clusters Detected over Course of Campaign')
            plt.ylabel('Number of Clusters')
            plt.xlabel('Experimental Space Coverage')
            plt.savefig(dirStore + 'numClusters_' + campaignObject.myName + str(datetime.datetime.now()) + '.tif')