# tetherCont is the objective goal acquisition function or goal-tether function designed to evaluate whether or not
# the campaigns that use the continuous modeler have met or exceeded the desired confidence or accuracy
import numpy as np

def main(campaignObject):

	# at the beginning, before the categorical modeler can set the attribute of the confidence array, "yAccuracy," the
	# campaign has to pass through the goal tether first
	if hasattr(campaignObject, 'yAccuracy'):
		confidence = campaignObject.yAccuracy
		allNaNs1 = np.isnan(confidence)
		confidence[allNaNs1] = 0
		if min(confidence) >= 0.90 and campaignObject.modelR_value >= 0.90:
			print('tether satisfied')
			return False
		else:
			return True
	else:
		return True