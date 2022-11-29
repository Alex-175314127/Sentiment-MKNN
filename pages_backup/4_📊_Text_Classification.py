from matplotlib import collections
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

def run():
    st.set_page_config(page_title="Text Classification", page_icon="📊", layout='wide')
    st.title("Text Classification Movie Review")
    st.write("Using K-NN Algorithm with dataset from Twitter")
    #df_upload = st.sidebar.file_uploader("Input your data ")
    
    def load_data():
        #file_ex = exists("output/Clean_text_Sentiment.csv")
        data = None
        if 'sen_result' in st.session_state:
            #data = pd.read_csv("output/Clean_text_Sentiment.csv", encoding="utf-8")
            data = st.session_state['sen_result']
            st.dataframe(data)
        else:
            st.warning("The Dataset is Empty, Please do Preprocessing to Continue")
        return data
        

    def k_value():
        params = dict()
        K = st.sidebar.slider("Set value of K",1,25)
        params['K'] = K
        return params

    def unpack_word_weight(vect, word_weight):
        feature_names = np.array(vect.get_feature_names_out())
        data = word_weight.data
        indptr = word_weight.indptr
        indices = word_weight.indices
        n_docs = word_weight.shape[0]
    
        word_weight_list = []
        for i in range(n_docs):
            doc = slice(indptr[i], indptr[i + 1])
            count, idx = data[doc], indices[doc]
            feature = feature_names[idx]
            word_weight_dict = dict({k: v for k, v in zip(feature, count)})
            word_weight_list.append(word_weight_dict)
        return word_weight_list

    st.subheader("Dataset")
    df_file = load_data()

    if df_file is not None:
        text_df = st.sidebar.selectbox("Select Text Column", df_file.columns)
        labels_df = st.sidebar.selectbox("Label to Classify ", df_file.columns)

        df_file = df_file[[text_df, labels_df]]
        Xfeatures = df_file.iloc[:, 0]
        ylabels = df_file.iloc[:, 1]
       
        #Data Splitting
        split_size = st.sidebar.slider("Data Split Ration (% for training dataset",10,90,70,5)
        X_train, X_test, y_train, y_test = train_test_split(Xfeatures,ylabels, test_size=(100-split_size)/100, random_state=42)

        st.markdown("**Data Splits**")
        st.write("Training set")
        st.info(X_train.shape)
        st.write("Test set")
        st.info(X_test.shape)

        try:
            #Vectorize
            #cv = CountVectorizer(binary=True)
            #TF_IDF
            tf = TfidfVectorizer()
            textTF = tf.fit_transform(Xfeatures)
            X_train_tfidf = tf.transform(X_train)
            X_test_tfidf = tf.transform(X_test)
            #df_file['TF_IDF'] = unpack_word_weight(tf, textTF)
            #df_file.to_csv('output/TF_IDF.csv', index=False)
            df_TFIDF = unpack_word_weight(tf, textTF)
            df_TFIDF = pd.DataFrame(df_TFIDF)
            df_TFIDF.fillna(0, inplace=True)
            df_TFIDF.to_csv('data/TFIDF_result.csv', index=False)

            # Encoding label to be a value between 0 and classn-1
            Encoder = LabelEncoder()
            y_train = Encoder.fit_transform(y_train)
            y_test = Encoder.fit_transform(y_test)

            #Transfrom X_train dan X_test to vector term presence
            #X_train_TP = cv.transform(X_train)
            #X_test_TP = cv.transform(X_test)

            st.markdown("**Term Frequency - Inverse Fequency (TF-IDF)**")
            with st.expander("TF_IDF Result "):
                #tfidf_result= pd.read_csv('output/TF_IDF.csv', encoding='utf-8')
                st.dataframe(df_TFIDF)
                
            with st.expander("KNN Classification Result "):
                params = k_value()
                algo = KNeighborsClassifier(n_neighbors=params['K']).fit(X_train_tfidf,y_train)
                y_pred = algo.predict(X_test_tfidf)
                acc = accuracy_score(y_test, y_pred)*100
                st.write('Accuracy Score :',acc)
                st.write('f1_score :', f1_score(y_test,y_pred,average='weighted').round(2))
                st.write('Precision :', precision_score(y_test,y_pred,average='weighted').round(2))
                st.write('Recall :', recall_score(y_test,y_pred,average='weighted').round(2))
                st.write('error rate :', 1-acc.round(2))

                #Confusion Matrix
                st.set_option('deprecation.showPyplotGlobalUse', False)
                #st.write(confusion_matrix(y_test,y_pred))
                plot_confusion_matrix(algo,X_test_tfidf,y_test, display_labels=ylabels.unique().tolist())
                st.pyplot()
        except:
            st.error("Please Select Text and Label Column")

    
            
        


if __name__ == "__main__":
    main()