import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('wine.csv')

print(df.shape)

X=np.array(df.drop(['Class'], 1))
y=np.array(df['Class'])

iter_list=[]
accuracy_list=[]

n_iterations=1000

for k in range(20):

	total_accuracy = 0

	k=k+1

	for i in range(n_iterations):

		#print(i)

		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

		#clf= neighbors.KNeighborsClassifier(n_neighbors=k)
		clf = RandomForestClassifier(n_estimators=k)
		clf.fit(X_train, y_train)

		accuracy = clf.score(X_test, y_test)

		total_accuracy = total_accuracy + accuracy

	total_accuracy=total_accuracy/n_iterations
	print(k, total_accuracy)

	iter_list.append(k)
	accuracy_list.append(total_accuracy)

plt.plot(iter_list, accuracy_list)
plt.show()




