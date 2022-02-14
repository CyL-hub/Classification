import util
from math import sqrt
PRINT = True

class KNNClassifier:

  def __init__(self, data=None, labels=None, k=3):
    self.type = "knn"
    self.data = data
    self.labels = labels
    self.n = None
    self.k = k
    print("init")

  def classify(self, inputs):
    predictions = []
    for i in range(len(inputs)): # For each input to predict
      distance = []
      for j in range(self.n): # For each self.data - input
        d = self.data[j].__sub__(inputs[i]) # (1, D)
        d = d.__mul__(d) # (1,1)
        d = sqrt(d)
        # print("d", type(d), d)
        d = zip([d], [self.labels[j]])
        distance.append(d)

      # print("distance", distance)
      nearestNeighbors = sorted(distance)[:self.k]
      # print("nn", nearestNeighbors)
      freq = util.most_frequent([label[0][1] for label in nearestNeighbors])
      # print("freq", freq)

      predictions.append(freq)

    return predictions


  def train(self, trainingData, trainingLabels, validationData, validationLabels ):
    self.data = trainingData
    self.labels = trainingLabels
    self.n = len(self.data)



