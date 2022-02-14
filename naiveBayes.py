# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import collections
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k


  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid): 
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"

    # Dictionary for prior. Key = legal labels, Value = count of each label
    prior = dict()
    for label in range(len(self.legalLabels)):
        prior[label] = 0
    for i in range(len(trainingLabels)):
            prior[trainingLabels[i]] += 1
    # Normalize the prior dictionary, Key = legal labels, Value = probability of each label
    for y in prior:
        prior[y] = prior[y] / float(len(trainingLabels))

    # Dictionary for conditional P(x1..xn/y),
    conditional = dict() 
    # key = y label, value = dictionary of features x1...xn 
    for y in prior: 
        conditional[y] = collections.defaultdict(list) 

        # store a list containing the index to each training data,for each y-th label
        data_indices_list = list()
        # enumerate for indices to each correct label in trainingLabels data
        for datum_index, correct_label in enumerate(trainingLabels):        
            # if y-th label matches correct label, add data index to list
            if y == correct_label:                               
                data_indices_list.insert(0,trainingData[datum_index])

        # Fill in conditional dictionary table with its feature values
        for i in range(len(data_indices_list)): 
            for feat, value in data_indices_list[i].items():
                conditional[y][feat].append(value)

    # Normalize the conditional dictionary, Key = feat/label, Value = probability of each feat/label
    for y in prior:    
        for feat, value in data_indices_list[y].items(): 
            prob = dict(collections.Counter(conditional[y][feat]))
            for key in prob:
                prob[key] = prob[key] / float(len(conditional[y][feat]))
            conditional[y][feat] = prob

    self.prior = prior   
    self.conditional = conditional  


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"   
    for y in self.prior:    
        logPosterior_y = math.log(self.prior[y])
        for feat, value in datum.items():
            if value >= 0:
                value = 0.01
            logPosterior_y = logPosterior_y + math.log(float((self.conditional[y][feat]).get(datum[feat], value))) 
        logJoint[y] = logPosterior_y  

    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
