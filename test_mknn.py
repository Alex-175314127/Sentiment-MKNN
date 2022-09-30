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

df = pd.read_csv('Twitter_fresh\sentiment_result_200.csv')
X = df['text']
y = df['Sentiment']

#tf = TfidfVectorizer(decode_error='replace', lowercase=False)
enc = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf = TfidfVectorizer()
X_train = tf.fit_transform(X_train)
X_test = tf.transform(X_test)
y_train = enc.fit_transform(y_train)
nilai_k = 12
clf = ModifiedKNN(k=nilai_k)
clf.fit(X_train, y_train)
predict, jarak =  clf.predict(X_test)
#print(X_train)

y_test = enc.transform(y_test)
#acc = accuracy_score(y_test, predict)*100
#print("ACC :",acc)

#print(predict)
#print("XTrain :",X_train.shape);print("XTest",X_test.shape);print("yTrain",y_train.shape);print("ytest:",y_test.shape)
#jarak = [x for l in jarak for x in l]
jarak = [nMin(nilai_k,map(float,i)) for i in jarak]
    
with open('jarak tetangga.txt', 'w') as f, open('label pred.txt', 'w') as g:
    f.write('\n'.join(str(ls) for ls in jarak))
    g.write(str(predict))

#print('\n'.join(str(el) for el in jarak))

neigbor_index = clf.get_neigbors(X_test)
#print("index Tetangga = ", neigbor_index)

svN = open('output/test_index.txt', 'w')
svN.write('\n'.join(str(a) for a in neigbor_index))

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


accuracy, precision, recall, f1_score = calculate_conf_metrics(y_test, predict)
print('Accuracy \t Precision\t Recall \t  F1 Score ')
print(f'{accuracy} \t\t {precision} \t\t {recall} \t\t {f1_score}')