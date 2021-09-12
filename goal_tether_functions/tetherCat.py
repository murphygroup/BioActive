# tetherCat is the objective goal acquisition function or goal-tether function designed to evaluate whether or not
# the campaigns that use the categorical modeler have met or exceeded the desired confidence or accuracy

import numpy as np

def main(campaignObject):

	# at the beginning, before the categorical modeler can set the attribute of the confidence array, "yAccuracy," the
	# campaign has to pass through the goal tether first
	if hasattr(campaignObject, 'yAccuracy'):
		#print('yAccuracy')
		#print(campaignObject.yAccuracy)
		if np.mean(campaignObject.yAccuracy) >= 0.97:
			print('accuracy criteria satisfied')
			return False
		else:
			return True
	else:
		return True
