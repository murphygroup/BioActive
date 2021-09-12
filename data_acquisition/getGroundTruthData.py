# this function simulates acquiring dependent variable values for
# specific combinations of independent variables.  It uses values
# stored as "ground Truth" when the campaign was created.

def main(requested, campaignObject):

	# engage database for storing values pertaining to data requested
	database = campaignObject.data
	neededData = []
	for ind in requested:
		data = campaignObject.groundTruth[ind]
		neededData.append((data,) + database.transform_num(ind))
	database.store(neededData)
