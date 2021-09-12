# sets up directories appropriate for storing output from BioActive

# arguments are: integer seed, noise modulation, and desired campaign directory name

# directories
import os
import sys

dirpath = os.getcwd()
sys.path.insert(0, dirpath + '/goal_tether_functions')
sys.path.insert(0, dirpath + '/predictive_modelers')
sys.path.insert(0, dirpath + '/predictive_modelers/assessment_resources')
sys.path.insert(0, dirpath + '/active_learners')
sys.path.insert(0, dirpath + '/data_acquisition')
sys.path.insert(0, dirpath + '/diagnostics')


# directory storage business
def dirSetup(*arg):

    try:
        campaignsdir = arg[0][1] + '/'
    except:
        campaignsdir = 'campaignDirectory/'
    if not os.path.exists(campaignsdir):
        os.makedirs(campaignsdir)

    return campaignsdir