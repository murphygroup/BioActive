# runCampaign2 integrates the functionality of a potentially highly variable ESS (experimentalspace specification) into
# a single, common active learning framework delivering a model

# import libraries
import numpy as np
try:
    import initModel
except ImportError:
    from predictive_modelers import initModel

def main(campaignObject):

    ESC = []

    initModel.main(campaignObject)
    campaignObject.initAvars() # currently only used for pool-based campaigns

    while campaignObject.goalTether(campaignObject):

        # models the data
        campaignObject.modelData(campaignObject)

        # the number of experiments completed should = the number of points for which
        # yaccs(:) = 1 ************ definitely cleaner ways to keep track
        ESC.append(float(len(campaignObject.data.storeind))/np.prod(campaignObject.ESS.dimarr))

        # active learning component--selects from highest uncertainties or least confidences a random
        # subset to get data for
        dataRequested = campaignObject.activeLearner(campaignObject)

        # bookkeeping for forward modeling and experimental space coverage
        campaignObject.lastDataRequest = dataRequested
        if len(dataRequested) == 0:
            # push everything into HDF5 when it's over
            campaignObject.data.flush()
            break

        print('Fetching data for round '+str(campaignObject.iterate))
        # fetches data requested
        campaignObject.fetchData(dataRequested, campaignObject)

        campaignObject.iterate = campaignObject.iterate + 1

    # returns to predictive modeler for any plotting or graphing
    campaignObject.modelData(campaignObject, ESC)
    campaignObject.data.flush()

    return campaignObject, ESC
