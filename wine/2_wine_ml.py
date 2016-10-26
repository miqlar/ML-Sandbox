import numpy as np
from sklearn import preprocessing, neighbors
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('wine.csv')

X=np.array(df.drop(['Class'], 1))
y=np.array(df['Class'])

iter_list=[]
score_list=[]

for k in range(1, 101):
	score_temporal=[]
	for _ in range(1000):
		clf = RandomForestClassifier(n_estimators=k)
		scores = cross_val_score(clf, X, y, cv=5)
		score = scores.mean()
		score_temporal.append(score)

	score = sum(score_temporal) / float(len(score_temporal))

	print(k, score)

	iter_list.append(k)
	score_list.append(score)

plt.plot(iter_list, score_list)
#plt.show()
plt.savefig("fig_01.png")
