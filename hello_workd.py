from sklearn import tree

# 0- bumpy | 1 - smooth
features = [[140,1],
            [130, 1],
            [160, 0],
            [170, 0],
            [180, 0]]

# 0 - apple  | 1 - orange
labels = [0,
          0,
          1,
          1,
          1 ]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[110, 1]]))

