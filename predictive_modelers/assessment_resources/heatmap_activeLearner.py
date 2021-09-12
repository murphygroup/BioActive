# designed to track, store, and provide for the assessment of the active learner's performance

import os
import datetime

# useful package for heat-map construction
import seaborn as sns
import matplotlib.pylab as plt


def main(accuracyFramed, campaignObject):

    # accuracyFramed is the confidence array
    axs = sns.heatmap(accuracyFramed, vmin=0, vmax=1)
    plt.title("Iteration " + str(campaignObject.iterate))
    plt.xlabel("Independent Variable 1")
    plt.ylabel("Independent Variable 2")

    plotName = "Active_Learning_HeatMap_" + str(campaignObject.iterate) + '.tif'
    tempdirectory2 = campaignObject.plotting.intDir + 'heatmaptempdir2_/'
    plotName = tempdirectory2 + plotName

    if not os.path.exists(tempdirectory2):
        os.makedirs(tempdirectory2)

    plt.savefig(plotName)
    plt.close('all')

    return plotName


def animate(campaignObject):

    filenames = campaignObject.filenames2
    import imageio

    # the following block produces a movie from all the heat-maps created from the active learning heat-map "progress
    # reports"
    videoName = campaignObject.plotting.intDir + 'SD_Clustering_AL' + str(datetime.datetime.now())
    endCap = '.mp4'
    with imageio.get_writer(videoName+endCap, fps=1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    writer.close()
    #***************************************************************************************************************

    # uncomment to clean-up intermediate files used to construct the videos
    '''
    shutil.rmtree('heatmaptempdir2_')
    '''