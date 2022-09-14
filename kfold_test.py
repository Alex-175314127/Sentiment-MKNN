from random import shuffle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from models.MKNN import ModifiedKNN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#X, y = load_iris(return_X_y=True)

df = pd.read_csv('output\Clean_text_Sentiment.csv')
X = df['text'].values
y = df['Sentiment'].values

n = 5
k=3
kfold = KFold(n, shuffle=True, random_state=33)
i=1
sum_akurasi = 0

#tf = TfidfVectorizer(decode_error='replace', lowercase=False)
enc = LabelEncoder()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for train_index, test_index in kfold.split(X):
    print("fold", i)
    print("Train", train_index, "Test :",test_index)
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    tf = TfidfVectorizer(decode_error="replace")
    X_train = tf.fit_transform(X_train)
    X_test = tf.transform(X_test)

    #enc = LabelEncoder()
    #y_train = enc.fit_transform(y_train)
    #y_test = enc.transform(y_test)

    clf = ModifiedKNN(k)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    conf_mat = confusion_matrix(y_test, predict)
   #a,b,c,d = conf_mat.ravel()
    #acc = (a+b) / (a+b+c+d) * 100
    acc = accuracy_score(y_test, predict)*100
    sum_akurasi += acc
    print("Accuracy :", str("%.4f" % acc)+'%')
    i += 1

sum_acc = sum_akurasi/n
print("Avearge accuracy : ", str("%.4f" % sum_acc)+'%')