# Example of getting neighbors for an instance
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors



# Test distance function
dataset = [[2.7810836,2.550537003,0],
		  [1.465489372,2.362125076,0],
		  [3.396561688,4.400293529,0],
		  [1.38807019,1.850220317,0],
		  [3.06407232,3.005305973,0],
		  [7.627531214,2.759262235,1],
		  [5.332441248,2.088626775,1],
		  [6.922596716,1.77106367,1],
		  [8.675418651,-0.242068655,1],
		  [7.673756466,3.508563011,1]]

df = pd.read_csv('output/Clean_text_Sentiment.csv')
X = df['text']
y = df['Sentiment']

enc = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf = TfidfVectorizer()
X_train = tf.fit_transform(X_train)
X_test = tf.transform(X_test)
y_train = enc.fit_transform(y_train)

#neighbors = get_neighbors(dataset, dataset[0], 3)
#for neighbor in neighbors:
	#print(neighbor)

clf = NearestNeighbors(n_neighbors=3)
clf.fit(X_train)
result = clf.kneighbors(X_test, return_distance=False)
print(result)