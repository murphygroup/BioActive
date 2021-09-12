# createCampaignDiscreteRegression provides definitions for:
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
# Amount of Added Noise
# Number of Experiments per Round

def main(*arg):

    NumberOfIvars = arg[0]
    NumberPerIvar = arg[1]
    AddedNoise = arg[2]
    ExperimentsPerRound = arg[3]
    
    class campaign:

        def initAvars(self):
            return

        # this goal function stops when min confidence is >0.9 and
        # R2 is greater than 0.9
        def goalTether(self, arg):
            from tetherCont import main
            return main(arg)

        # this modeler fits a linear model and estimates confidence
        # in predictions from distance to nearest sampled point
        def modelData(self, *arg):
            from contModel import main
            return main(*arg)

        # this active learner chooses randomly among the points
        # with estimated accuracy below 0.9 and stops when there are
        # not enough points to fill a batch
        def activeLearner(self, arg):
            from actLearn_contModularized import main
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

            if NumberOfIvars<1:
                raise ValueError("Number of Independent Variables must be positive:" + str(NumberOfIvars))
            
            #trailing comma creates tuple of tuples
            iVars = (('int',0,NumberPerIvar),)*NumberOfIvars 
            #print(iVars)
 
            iVarNames = ["iVar"+str(i) for i in range(NumberOfIvars)]
            #print(iVarNames)

            iVarCategories = [()]*NumberOfIvars
            #print(iVarCategories)

            dVars = [('int', -50, 50)]
            #print(dVars)

            listTuples = util.createTuples(iVars)
            #listName = 'listTuples' + str(NumberOfIvars) + '_' + str(datetime.datetime.now()) + '.txt'
            #print(listName)
            #np.savetxt(listName, listTuples, fmt="%s")

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
                'linreg_test_' + str(NumberOfIvars) + '_' +
                str(datetime.datetime.now()), 0, reset=True)

        confCount = ExperimentsPerRound

        # define the coefficients for generating data
        betas = [(i+2)**2 for i in range(NumberOfIvars)]
        #print(betas)
        groundTruth = generate_ground_truth_linear(ESS.listTuples,betas,AddedNoise)

        class plotting:
            title = "Accuracy as Fraction of Experimental Space Coverage Increased"
            xlabel = "Experimental Space Coverage"
            ylabel = "Accuracy"
            filename = "BioActive_LinReg("+str(NumberOfIvars) + "iVar)Campaign_batchsize" + str(ExperimentsPerRound) + ".tif"
            intDir = 'simsDirectory_' + str(datetime.datetime.now()) + '/'
            filename = intDir + filename

    C = campaign()
    return C

def generate_ground_truth_linear(listTuples,betas,AddedNoise):

    y = np.empty([listTuples.shape[0]])
    for i in range(listTuples.shape[0]):
        y[i] = sum(betas[j]*listTuples[i][j] for j in range(listTuples.shape[1]))
    if AddedNoise!=0:
        rng = np.random.default_rng()
        stnoise = rng.standard_normal(listTuples.shape[0])
        y = y + stnoise*AddedNoise
    return y
