LDA:

Step 1: Load Necessary Libraries:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Step 2: Load the Data

iris = datasets.load_iris()

#convert dataset to pandas DataFrame
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],columns = iris['feature_names'] + ['target'])
print(df)

df['species'] =pd.Categorical.from_codes(iris.target, iris.target_names)

df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']

print(df.head())

Step 3: Fit the LDA Model

X = df[['s_length', 's_width', 'p_length', 'p_width']]
y = df['species']

#Fit the LDA model
model = LinearDiscriminantAnalysis()
model.fit(X, y)

Step 4: Use the Model to Make Predictions

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores))   

0.9777777777777779


