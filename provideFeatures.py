# works in concert with pool based active learning campaign
# assumes all features are in hand and can be accessed within a pickle or more conveniently, a csv file

import numpy as np
import csv
#import pickle


def main(C):

    #print('in provideFeatures')
    feats = C.groundTruthAVars
    #print(type(feats))
    #print(feats.shape)
    #print(feats)
    
    database = C.data

    print('iVars=')
    #print(C.ESS.iVars)
    for index in range(C.ESS.iVars[0][2]+1):
        ind = database.transform_num(index)
        database.store([(index,) + ind])
        database.store([(np.array(feats[index]).astype(float),) + ind],
                destination="Associated Variable 1")

    database.flush()

    return C
