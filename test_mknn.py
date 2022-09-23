from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from models.MKNN import ModifiedKNN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from more_itertools import split_at
from heapq import nlargest as nMax
from heapq import nsmallest as nMin
#X, y = load_iris(return_X_y=True)
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('output/Clean_text_Sentiment.csv')
X = df['text']
y = df['Sentiment']

#tf = TfidfVectorizer(decode_error='replace', lowercase=False)
enc = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf = TfidfVectorizer()
X_train = tf.fit_transform(X_train)
X_test = tf.transform(X_test)
y_train = enc.fit_transform(y_train)
nilai_k = 3
clf = ModifiedKNN(k=nilai_k)
clf.fit(X_train, y_train)
predict, jarak =  clf.predict(X_test)
print(X_train)

y_test = y_test.apply(lambda x: 1 if 'Positive' else 0)
acc = accuracy_score(y_test, predict)*100
print("ACC :",acc)

#print(predict)
print("XTrain :",X_train.shape);print("XTest",X_test.shape);print("yTrain",y_train.shape);print("ytest:",y_test.shape)
#jarak = [x for l in jarak for x in l]
jarak = [nMin(nilai_k,map(float,i)) for i in jarak]
    
with open('jarak tetangga.txt', 'w') as f, open('label pred.txt', 'w') as g:
    f.write('\n'.join(str(ls) for ls in jarak))
    g.write(str(predict))

print('\n'.join(str(el) for el in jarak))