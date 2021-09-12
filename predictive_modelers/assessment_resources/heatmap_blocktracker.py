# designed to track, store, and provide for the assessment of how confidence is being assigned during the modeling
# process for rows and columns individually during their own respective distinct modeling processes

import os
import shutil
import datetime

# useful package for heat-map construction
import seaborn as sns
import matplotlib.pylab as plt


def main(accuracyFramed, dimMaj, campaignObject):

    axs = sns.heatmap(accuracyFramed, vmin=0, vmax=1)
    if dimMaj == 0:
        plt.title("Iteration " + str(campaignObject.iterate) + "_with Rows")
        plotName = "Confidence_Assignment_Rows_" + str(campaignObject.iterate) + '.tif'
    elif dimMaj == 1:
        plt.title("Iteration " + str(campaignObject.iterate) + "_with Columns")
        plotName = "Confidence_Assignment_Columns_" + str(campaignObject.iterate) + '.tif'

    plt.xlabel("Independent Variable 1")
    plt.ylabel("Independent Variable 2")

    tempdirectory3 = campaignObject.plotting.intDir + 'heatmaptempdir3_/'
    if not os.path.exists(tempdirectory3):
        os.makedirs(tempdirectory3)

    plotName = tempdirectory3 + plotName
    plt.savefig(plotName)
    plt.close('all')

    return plotName


def animate(campaignObject):

    filenames = campaignObject.filenames3
    import imageio

    # the following block produces a movie from all the heat-maps created from the active learning heat-map "progress
    # reports"
    videoName = campaignObject.plotting.intDir + 'SD_ConfAssignment_'

    videoName_rows = videoName + "Rows_" + str(datetime.datetime.now())
    videoName_cols = videoName + "Columns_" + str(datetime.datetime.now())
    endCap = '.mp4'
    with imageio.get_writer(videoName_rows+endCap, fps=1) as writer:
        for filename in filenames:
            if "_Rows" in filename:
                image = imageio.imread(filename)
                writer.append_data(image)
    with imageio.get_writer(videoName_cols+endCap, fps=1) as writer:
        for filename in filenames:
            if "_Columns" in filename:
                image = imageio.imread(filename)
                writer.append_data(image)
    writer.close()
    #***************************************************************************************************************

    # uncomment to clean-up intermediate files used to construct the videos
    '''
    shutil.rmtree('heatmaptempdir3_')
    '''