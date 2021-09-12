import os
import sys
import shutil
import pickle
from pathlib import Path
import outputManager
# cleans-up cache directories
from cleanCachefiles import main as cleanCachedirs
import importlib
from runCampaign2 import main as runCampaign

def CreateAndRunCampaign(campaign_name,
                         create_campaign_python_file,
                         create_function_args,
                         randomseed,
                         outdir,
                         visualizationfiles):

	dirpath = os.getcwd()
	#print(dirpath)
	sys.path.insert(0, dirpath + '/goal_tether_functions')
	sys.path.insert(0, dirpath + '/predictive_modelers')
	sys.path.insert(0, dirpath + '/predictive_modelers/assessment_resources')
	sys.path.insert(0, dirpath + '/active_learners')
	sys.path.insert(0, dirpath + '/data_acquisition')

	dtbsExt = '.hdf5'
	
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	os.chdir(outdir)

	# creates campaign
	create_campaign_module = importlib.import_module(create_campaign_python_file)
	campaignObject = create_campaign_module.main(*create_function_args)

	setattr(campaignObject, 'randoseed', randomseed)

	# run campaign(s)
	runCampaign(campaignObject)

	# clean-up
	# save the full campaign object to a pickle file
	pickle.dump(campaignObject.__dict__, open(campaign_name + '_Seed=' + str(randomseed) + ".p", "wb"))

	os.chdir(dirpath)

	#print(campaignObject.plotting.intDir)
	filesep = os.path.sep
	outd = Path(outdir + filesep + campaignObject.plotting.intDir)
	visualizations = [list(outd.glob(glob)) for glob in visualizationfiles]

	#print(visualizations)
	visualizations = sum(visualizations, []) # flatten nested list

	cleanCachedirs()

	return visualizations
