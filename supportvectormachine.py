from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
#SVC()
clf.predict([[2.,2.]])

#Multi-class classification
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
#SVC(decision_function_shape='ovo')
dec = clf.decision_function([[1]])
print(dec.shape[1])  # 4 classes: 4*3/2 = 6
#6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]]) 
print(dec.shape[1])  # 4 classes
