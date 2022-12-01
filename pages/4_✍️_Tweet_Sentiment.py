import pandas as pd
import numpy as np
import streamlit as st
import json
from datetime import datetime
from models.MKNN import ModifiedKNN
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re, unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import plotly.express as px
import altair as alt


###########################################################################################################################################

def casefolding(Text):
        Text = Text.lower()
        return Text

# Punctuation Removal
def punc_clean(Text):
    #remove url
    Text = re.sub(r'http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+', ' ', Text)
    #remove mention
    Text = re.sub(r"@[A-Za-z0-9]+","", Text)
    #remove hastag
    Text = re.sub(r"#[A-Za-z0-9_]+","", Text)
    #tanda baca
    Text = re.sub(r'[^\w]|_', ' ', Text)
    Text = re.sub(r'[!$%^&*@#()_+|~=`{}\[\]%\-:";\'<>?,.\/]', ' ', Text)
    #remove number in string
    Text = re.sub(r"\S*\d\S*", "", Text).strip()
    #remove number(int/float)
    Text =  re.sub(r"[0-9]", " ", Text)
    Text = re.sub(r"\b\d+\b", " ", Text)
    #remove double Space
    Text = re.sub(r'[\s]+', ' ', Text)
    #remove non-ASCII
    Text = unicodedata.normalize('NFKD', Text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    Text = ' '.join( [w for w in Text.split() if len(w)>1] )
    return Text

def word_norm(tweets):
    word_dict = pd.read_csv('data/indonesia_slangWords.csv')
    norm_word_dict = {}
    for index, row in word_dict.iterrows():
        if row[0] not in norm_word_dict:
            norm_word_dict[row[0]] = row[1]
    return [norm_word_dict[term] if term in norm_word_dict else term for term in tweets]

# Tokenize
def word_tokenize_wrapper(Text):
    return word_tokenize(Text)

# Stopwords
def remove_stopword(Text):
    stopW = stopwords.words('indonesian', 'english')
    sw = pd.read_csv('data/stopwordbahasa.csv')
    stopW.extend(sw)
    remove_sw = ' '.join(Text)
    clean_sw = [word for word in remove_sw.split() if word.lower() not in stopW]
    return clean_sw

## Indonesia Stemming
def indo_stem(Text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    result = []
    for w in Text:
        result.append(stemmer.stem(w))
        result.append(" ")
    return "".join(result)


def main():
    from track_util import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotSen_table
    st.set_page_config(page_title="Predict Text Sentiment", layout='wide', page_icon="✍️")
    menu = ["Home", "History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotSen_table()
    if choice == "Home":
        st.subheader("Sentiment Text")
        add_page_visited_details("Tweet Sentiment",datetime.now())

        with st.form(key='emotion_clf_form'):
            df= pd.read_csv('data/data_Train.csv', encoding='utf-8')
            Xfeature = df['text'].values
            ylabels = df['Sentiment'].apply(lambda x: 'Positive' if x == 1 else 'Negative').values
            
            raw_text = st.text_area("Input Text to Predict the Sentiment")
            
            submit_text = st.form_submit_button('Analyze')
            X_test = []
            if submit_text:
                with st.spinner('Loading....!'):
                    raw_text = casefolding(raw_text)
                    raw_text = punc_clean(raw_text)
                    raw_text = word_tokenize_wrapper(raw_text)
                    raw_text = remove_stopword(raw_text)
                    raw_text = indo_stem(raw_text)
                    
                    Tf = TfidfVectorizer(decode_error='replace')
                    X_train = Tf.fit_transform(Xfeature)
                    X_test.append(raw_text)
                    X_test = Tf.transform(X_test)
                    
                    enc = LabelEncoder()
                    #y_train = enc.fit_transform(ylabels)
                    
                    clf = ModifiedKNN(k=5)
                    clf.fit(X_train, ylabels)
                    pred, jarak = clf.predict(X_test)
                    
                    #pred = enc.inverse_transform(pred)
                    pred = ' '.join([str(w) for w in pred])
                    st.markdown(f'<h1 style="color:white; text-align:center;">Sentiment : {pred}</h1>',unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(' ')
                    with col2:
                        if pred == "Positive":
                            st.image('data/images/pos_image.png')
                        else:
                            st.image('data/images/neg_image.png')
                    with col3:
                        st.write(' ')
                    
                        
                

    elif choice == "History":
        st.subheader("Manage & History Results")
        add_page_visited_details("History", datetime.now())

        with st.expander("Page Metrics"):
            add_page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename', 'Time_of_Visit'])
            st.dataframe(add_page_visited_details)

            count_page = add_page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            cnt = alt.Chart(count_page).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
            st.altair_chart(cnt,use_container_width=True)
        
        with st.expander("Sentiment Analysis"):
            df_hs =pd.DataFrame(view_all_prediction_details(), columns=['raw_text','Sentiment','sentiment_score','Time_of_visit'])
            st.dataframe(df_hs)

            #Sentiment Plot
            sen_count = df_hs['Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Counts')
            sen_plt = alt.Chart(sen_count).mark_bar().encode(x='Sentiment', y='Counts', color='Sentiment')
            st.altair_chart(sen_plt,use_container_width=True)

    else:
        st.subheader()






if __name__ =='__main__':
    main()