# createCampaignDiscreteRegressioCategoricalModeler provides definitions for:
# 1) data collection functions, used to obtain all currently existing data,
#    a cost for collecting some specific set of experimental data, a specific 
#    set of new experimental data, 
# 2) the types of dependent (measured) variables that can be returned, 
# 3) the types and ranges of independent (experimental) variables to be explored, 
# 4) the set of definitions for model construction functions, 
# 5) the active learning function to be used, and,  
# 6) the objective function to be optimized along with the termination criteria. 
# The argument specifies how many independent variables to create.

# Useful library for handling iterators
import itertools

# Important for storing results and history of campaign runs on disk or on a remote server
from database import *
import datetime

# directories
import os

import util

#arguments:
# Number of Independent Variables
# Number of Values per Independent Variable
# Number of Groups per Independent Variable
# Amount of Added Noise
# Number of Experiments per Round

def main(*arg):

    NumberOfIvars = arg[0]
    NumberPerIvar = arg[1]
    NumberGroups = arg[2]
    AddedNoise = arg[3]
    ExperimentsPerRound = arg[4]
    UseActiveLearning = arg[5]
    
    class campaign:

        def initAvars(self):
            return

        # this goal function stops when min confidence is >0.9 and
        # R2 is greater than 0.9
        def goalTether(self, arg):
            from tetherCat import main
            return main(arg)

        # this modeler fits a row/column cluster model and estimates
        # confidence in predictions from distance to nearest sampled point
        def modelData(self, *arg):
            from catModel11 import main
            return main(*arg)

        # this active learner chooses randomly among the points
        # with estimated accuracy below 0.9 and stops when there are
        # not enough points to fill a batch
        def activeLearner(self, arg):
            from actLearn_cat import main
            return main(arg)

        def fetchData(self, arg1, arg2):
            from getGroundTruthData import main
            return main(arg1, arg2)

        def provideNames():
            print("iVarNames:")
            print(campaign.ESS.iVarNames)
            print("iVarCategories:")
            print(campaign.ESS.iVarCategories)

        class ESS:
            # define data specifications

            if NumberOfIvars!=2:
                raise ValueError("Number of Independent Variables must be 2:" + str(NumberOfIvars))
            
            #trailing comma creates tuple of tuples
            iVars = (('int',0,NumberPerIvar-1),)*NumberOfIvars 
            #print(iVars)
 
            iVarNames = ["iVar"+str(i) for i in range(NumberOfIvars)]
            #print(iVarNames)

            iVarCategories = [()]*NumberOfIvars
            #print(iVarCategories)

            dVars = [('int', 1, )]
            #print(dVars)

            listTuples = util.createTuples(iVars)

            dimarr = []
            for i in range(len(iVars)):
                dimarr.append(iVars[i][2] - iVars[i][1] + 1)

            # constraints on data collection function(s), e.g. QC
            '''    
            class acquisitionOptions:
                    replicates = [] # 4. Data acquisition function
                    tolerance = [] # 4. Data acquisition function
                    cost = [0] # 4. Data acquisition function
                    getcurrent = [] # 5. Data access object
            '''

        # all data is initialized as np.nan inside database.py
        data = Database(ESS.dimarr,
                'catmodel_test_' + str(NumberOfIvars) + '_' +
                str(datetime.datetime.now()), 0, reset=True)

        confCount = ExperimentsPerRound

        NGroups = [NumberGroups]*NumberOfIvars 
        groundTruth = generate_ground_truth_categorical(ESS.listTuples,NGroups,AddedNoise)

        # do not use pairwise accuracy, because we will have ground truth;
        # otherwise, accuracy assessment used when
        # data is unlabeled and classes are assigned arbitrarily round to round
        pairwise = False

        # active learning (as opposed to random)
        flag = UseActiveLearning

        hasClassifier = False

        # Clustering model only (Conf1) or Clustering model plus Mode (Conf2)
        confAssignmentChoice = 'Conf1'

        # simulation?
        simsFlag = True
        
        class plotting:
            title = "Accuracy as Fraction of Experimental Space Coverage Increased"
            xlabel = "Experimental Space Coverage"
            ylabel = "Accuracy"
            filename = "BioActive_LinReCatModelg("+str(NumberOfIvars) + "iVar)Campaign_batchsize" + str(ExperimentsPerRound) + ".tif"
            intDir = 'simsDirectory_' + str(datetime.datetime.now()) + '/'
            filename = intDir + filename

    C = campaign()
    return C

def generate_ground_truth_categorical(listTuples,nGroups,AddedNoise):

    # create categorically structured data and store it both as framed data
    # as well as a 1D array for assessment and use of ground truth
    # respectively
    xs = listTuples[:, 0]
    ys = listTuples[:, 1]
    allData = np.empty(len(xs))
    
    for i in range(len(allData)):
        allData[i] = 1 + xs[i] % nGroups[0] + ys[i] % nGroups[1]
    
    if AddedNoise!=0:
        rng = np.random.default_rng()
        randuni = rng.random(len(allData))
        changeit = np.where(randuni < AddedNoise)
        newvals = rng.integers(low=1, high=max(allData), size=len(changeit))
        for i in range(len(changeit)):
            allData[changeit[i]] = newvals[i]

    return allData
