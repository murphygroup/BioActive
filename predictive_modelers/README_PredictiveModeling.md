#Predictive Modeling Functions

The purpose of all the predictive modeling functions provided in the BioActive project is to model the experimental 
space defined explicitly by tuples comprised of all the independent variables in the campaign Object. Some of the 
predictive modelers make use of resources such as sklearn and others have been written in-house by the team behind 
BioActive.

Each predictive modeler initializes all necessary variables needed to model the experimental space, track, assess, and 
evaluate accuracy over the course of the campaign.

##Continuous Modeling

The continuous variable modeler "contModel.py" uses linear regression and is not limited to 2 independent variables. It 
has been deployed for analysis for up to 10 independent variables.

##Categorical Modeling

The categorical variable modeler "catModel11" uses a clustering algorithm to identify what clusters in "rows" or
"columns" are not dissimilar and thus produces a model that can then predict values mapping the experimental space.

The categorical modeler is equipped to model a variety of different experimental spaces as well as to assess its own 
accuracy. In particular, it is designed to be flexible and to accommodate campaigns in which the test labels do not
remain static, i.e. pairwise analysis for cases where k-means clustering is used to determine what labels an 
experimental dataset should take given new data.

The categorical modeler: identifies clusters in rows and columns separately, provides initial predictions based off 
those models separately, and then merges the predictions into final predictions based off the confidence calculated 
for each of the predictions. It then averages the confidence between the modeling processes for clustering in rows and 
columns for a final confidence that can then be used to guide the active learner.

Two approaches are currently available for modeling in rows and columns. In both approaches, "conf1" and "conf2," a 
predictive model is built based off clustering in the rows and columns, respectively, however, in "conf2," the approach
makes use of the mode to predict values that can't be otherwise determined due to incompleteness of a predictive cluster.

##Accuracy Assessment

When ground truth is available, defined within the campaign object, e.g. createCampaign_linReg.py or
createCampaign_clusterinteractions.py, 3 types of accuracy assessment are deployed: forward modeling, full space 
accuracy, and predictions-only accuracy. In the absence of ground truth, only forward modeling may be used.

Heatmaps are deployed throughout so that confidence assignment and accuracy may be tracked over the course of the
campaign either in a .gif file as with the pairwise plots or in a video as in the confidence assignment tracking.

###Inputs
Currently available predictive modelers take as their input the campaign object and at the end of the campaign, the
campaign object and the experimental space coverage calculated within "runCampaign2.py" on every round to perform the 
final accuracy assessments and corresponding plotting.

The categorical modeler should only take campaigns with categorical variables just as the continuous variable modeler 
should only take campaigns with continuous variables.

###Outputs
The predictive modelers produce a model for the experimental space defined within the campaign object as well as several
accuracy assessments, e.g. plots of accuracy over the course of the campaign along with heatmaps revealing how the
campaign identified experiments to perform.

###Assessment Resources
The predictive modelers available and particularly the categorical modeler make great use of standalone resources such 
as ground truth for the various pertinent campaigns using them as well as the heat map functions and pairwise accuracy
computation.