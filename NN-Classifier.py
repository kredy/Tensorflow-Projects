'''

Neural Network to classify Iris Dataset


'''

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


def LoadData(filename):
  df = pd.read_csv(filename, delimiter=",")
  return df

  
def CleanData(dataframe):
  dataframe = dataframe.drop("Id", axis=1)
  dataframe["Species"] = dataframe["Species"].replace(to_replace="Iris-setosa", value=0)
  dataframe["Species"] = dataframe["Species"].replace(to_replace="Iris-versicolor", value=1)
  dataframe["Species"] = dataframe["Species"].replace(to_replace="Iris-virginica", value=2)
  return dataframe

  
def TrainTestSplit(dataframe):
  inputdata = dataframe.drop("Species", axis=1).values
  response = dataframe.Species.values
  X_train, X_test, Y_train, Y_test = train_test_split(inputdata, response, test_size=0.20)
  return X_train, X_test, Y_train, Y_test


def MLP(X, W1, W2, b1, b2):
  layer1 = tf.add((tf.matmul(X, W1)), b1)
  layer1 = tf.nn.relu(layer1)
  layer2 = tf.add((tf.matmul(layer1, W2)), b2)
  return layer2 

  
dataframe = LoadData("Iris.csv")
dataframe = CleanData(dataframe)
X_train, X_test, Y_train, Y_test = TrainTestSplit(dataframe)
Y_test = Y_test.reshape(-1,1)
features = X_train.shape[1]
classes = 3
epochs = 2001
alpha = 0.001

with tf.name_scope("inputdata") as inpdata:
  X = tf.placeholder(tf.float32, [None, features])
  Y = tf.placeholder(tf.int32, [None, ])
  
with tf.name_scope("onehot") as one_hot:
  Y_oh = tf.one_hot(Y, depth = 3)
  
with tf.name_scope("Parameters") as parameters:
  W1 = tf.get_variable("W1", shape=[features, 7], initializer=tf.contrib.layers.xavier_initializer()) 
  b1 = tf.Variable(tf.zeros([1, 7]))
  W2 = tf.get_variable("W2", shape=[7, classes], initializer=tf.contrib.layers.xavier_initializer()) 
  b2 = tf.Variable(tf.zeros([1, classes]))
  
Yhat = MLP(X, W1, W2, b1, b2 )
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_oh, logits=Yhat))
train_step = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
  
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(epochs):
    sess.run(train_step, feed_dict={X:X_train, Y:Y_train})
    if (i%100 == 0):
      print ("loss at", i, "=", sess.run(loss, feed_dict={X:X_train, Y:Y_train}))
  pred = tf.argmax(Yhat, 1)
  pred_train = (sess.run(pred, feed_dict={X:X_train}))
  pred_test = (sess.run(pred, feed_dict={X:X_test}))    

print ("Train Score:")
print (accuracy_score(Y_train, pred_train))
print ("Test Score:")
print (accuracy_score(Y_test, pred_test))
