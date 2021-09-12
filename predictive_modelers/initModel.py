# Created by Piratach Yoovidhya (piratacy) and Raymond Hua (rhua2)
# Date: 03/20/2020

# This file initializes most/everything you need for your modelers
# It is very possible that initializes more variables to your campaignObject
# than you actually need.
# That is completely fine. It shouldn't affect anything.

import os
import numpy as np

def main(campaignObject):

	# initModelComplete is set to False if the initModel is insufficient
	# to initialize all variables (i.e. we need to initialize more or overwrite stuff)
	# we set initModelComplete to False by default
	setattr(campaignObject, 'initModelComplete', False)

	# 'yAccuracy" represents the confidence of a given experiment's value, be it acquired or predicted
	print(campaignObject.ESS.listTuples.shape)
	print(campaignObject.ESS.dVars)
	setattr(campaignObject, 'yAccuracy',
			np.empty([len(campaignObject.ESS.listTuples), len(campaignObject.ESS.dVars)]))
	campaignObject.yAccuracy[:] = 0

	print(campaignObject.ESS)
	# row clustering and column clustering also produce predictions for which reason there needs to be a space
	# and array in which to store them
	setattr(campaignObject, 'initPredictions', np.empty([*campaignObject.ESS.dimarr,2]))
	campaignObject.initPredictions[:] = np.nan

	print('dimarr')
	print(campaignObject.ESS.dimarr)
	# after merging the predictions made from the 2 initial predictions elicited by row and column clustering
	setattr(campaignObject, 'finalPredictions', np.empty([*campaignObject.ESS.dimarr]))
	campaignObject.finalPredictions[:] = np.nan
	print('Length of finalPredictions'+str(len(campaignObject.finalPredictions)))

	setattr(campaignObject, 'predictions',
			np.empty([len(campaignObject.ESS.listTuples), len(campaignObject.ESS.dVars)]))
	campaignObject.predictions[:] = np.nan    

	setattr(campaignObject, 'lastDataRequest', [])

	setattr(campaignObject, 'modelR_value', 0)
	setattr(campaignObject, 'model', {})

	setattr(campaignObject, 'crossValScores', [])

	setattr(campaignObject, 'iterate', 0)

	# track at what point full-space accuracy exceeds 95%
	setattr(campaignObject, 'percentile95', np.nan)
	setattr(campaignObject, 'percentileStop', True)

	# provide space and tracking for heat-maps generated for the purpose of tracking how the campaign assigns
	# confidence and selects experiments
	setattr(campaignObject, 'filenames1', []) #for pairwise only
	setattr(campaignObject, 'filenames2', [])
	setattr(campaignObject, 'filenames3', [])

	setattr(campaignObject, 'accuracy_full', [])
	setattr(campaignObject, 'accuracy_forwardModeling', [])
	setattr(campaignObject, 'accuracy_onlyPredictions', [])
	campaignObject.accuracy_full.append(0)
	campaignObject.accuracy_forwardModeling.append(0)
	campaignObject.accuracy_onlyPredictions.append(0)

	setattr(campaignObject, 'total_var_distance', [])
	campaignObject.total_var_distance.append(1)

	
