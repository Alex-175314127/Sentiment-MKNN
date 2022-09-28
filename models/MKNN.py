import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

# method untuk mencari tetangga terdekat
def tetangga_terdekat(k):
    neigbor = {}
    near = ('', 0)
    # near = (value terjadi, tetangganya)
    for i in k:
        if i in neigbor:
            neigbor[i] += 1
        else:
            neigbor[i] = 1
        # menyimpan tetangga sebanyak k
        if neigbor[i] > near[1]:
            near = (i, neigbor[i])
    return near

# menghitung jarak euclidean
def jarak_euclidean(x, y):
        return euclidean_distances(x,y)

# fungsi S untuk Validity
def fungsi_S(a, b):
    if a == b:
        return True
    return False

# menghitung validity
def validity(distance, y, k):
    v_arr = []
    current_index = 0
    n_val = open('output/neigbor.txt', 'w')
    for i in distance:
        sorted_index = sorted(range(len(i)), key=lambda k: i[k])
        fungsi_k = []
        for j in range(k):
            fungsi_k.append(y[sorted_index[j + 1]])

        label = 0
        print('{} = neigbor: {}'.format(y[current_index], fungsi_k))
        n_val.write('{} = neigbor: {}'.format(y[current_index], fungsi_k) + '\n')
    
        #st.write('{}, neigbor: {}'.format(y[current_index], fungsi_k))
        for w in fungsi_k:
            if fungsi_S(y[current_index], w):
                label += 1
        validity_result = 1 / k * label
        v_arr.append(validity_result)
        current_index += 1
    n_val.close()
    return v_arr

class ModifiedKNN(object):
    def __init__(self, k=3):
        self.k = k

    # fit data x train dan data y train
    def fit(self, X, y):
        self.X_train = X

        if isinstance(y, pd.Series):
            self.y = y.values.ravel()
        else:
            self.y = y

        self.distance = jarak_euclidean(X, X)
        self.validity = validity(self.distance, self.y, self.k)

    def predict(self, X_test):
        if isinstance(X_test, pd.Series):
            test = X_test.values
        else:
            test = X_test

        pred_label = []
        distances = jarak_euclidean(X_test, self.X_train)
        #print(distances)

        #Hitung Weighted Voting
        for i in distances:
            weight_voting = []
            for j in range(len(self.validity)):
                weight = self.validity[j] * (1 / (i[j] + 0.5))
                weight_voting.append(weight)
            sorted_index = sorted(range(len(weight_voting)), key=lambda k: weight_voting[k], reverse = True)
            mknn_label = []
            y = self.y
            for w in range(self.k):
                mknn_label.append(y[sorted_index[w]])

            neigbor,count = tetangga_terdekat(mknn_label)
            #print(mknn_label)

            pred_label.append(neigbor)

        return pred_label, distances

    def get_neigbors(self, test_X):
        k_value = self.k
        train_X = self.X_train
        clf = NearestNeighbors(n_neighbors=k_value).fit(train_X)
        neigbor_index = clf.kneighbors(test_X, return_distance=False)
        return neigbor_index