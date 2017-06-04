from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris = load_iris()

# UNCOMMENT THE BELOW section to visualize the data

# check the data in the dataset:
# print(iris.feature_names)
# print(iris.target_names)
#
# print(iris.data[0]) # contains the data of the features
# print(iris.target[0]) # contains the code for the target flower
#                       # setosa - 0  versicolor - 1  virginica - 2
#
# # Note the data is found here
# # https://en.wikipedia.org/wiki/Iris_flower_data_set?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt
# # It has 150 entries.
# #To print all the data from the dataset
# for index in range(len(iris.target)):
#     print("Example %d : label - %s , features -  %s" %(index, iris.target[index] , iris.data[index]))

###################################################################################################

# Train and test data
#--------------------------------
test_idx = [0,50,100] # takes 3 values in position 0, 50 , 100 as the test data.

# train data set
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx, axis = 0)

#testing  dataset
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Training process
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

print(" Actual data ")
print(test_target)
print("predicted data")
print(clf.predict(test_data))

# Visualization : not covered!



