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

df = pd.read_csv('Twitter_fresh\sentiment_result_200.csv')
X = df['text'].values
y = df['Sentiment'].values

n = 5
k=12
kfold = KFold(n, shuffle=True, random_state=33)
i=1
sum_akurasi = 0
acc, rc ,pr, f1 = [],[],[],[]

#tf = TfidfVectorizer(decode_error='replace', lowercase=False)
enc = LabelEncoder()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def calculate_conf_metrics(y_test, pred):
    TP, FP, TN, FN = 0,0,0,0
    for i in range(len(pred)):
        if (pred[i] == 0) & (y_test[i] == 0):
            TP += 1
        elif (pred[i] == 0) & (y_test[i] == 1):
            FP += 1
        elif (pred[i] == 1) & (y_test[i] == 1):
            TN += 1
        else:
            FN += 1
    print(f'TP : {TP} \t FP : {FP} \t TN : {TN} \t FN : {FN}')
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1_score = (2 * precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score

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

    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    clf = ModifiedKNN(12)
    clf.fit(X_train, y_train)
    predict, jarak = clf.predict(X_test)

    accuracy, precision, recall, f1_score = calculate_conf_metrics(y_test, predict)
    sum_akurasi += accuracy
    acc.append(accuracy)
    pr.append(precision)
    rc.append(recall)
    f1.append(f1_score)
    print("Accuracy :", str("%.4f" % accuracy)+'%')
    i += 1

sum_acc = sum_akurasi/n
print("Avearge accuracy : ", str("%.4f" % sum_acc)+'%')

print('Accuracy \t Precision\t Recall \t  F1 Score ')
print(f'{acc} \t\t {pr} \t\t {rc} \t\t {f1}')