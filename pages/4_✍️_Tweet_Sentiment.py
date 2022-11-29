import pandas as pd
import numpy as np
import streamlit as st
import json
from datetime import datetime
from models.MKNN import ModifiedKNN
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

import plotly.express as px
import altair as alt

def Sentiment_analysis(Text):
    senti = SentimentIntensityAnalyzer()
    #Delete Default lexicon VADER
    senti.lexicon.clear()

    #Read costom Lexicon
    sentidata = open('data\sentiwords_id.txt', 'r').read()
    #convert lexicon to dictonary
    senti_word = json.loads(sentidata)

    #update lexicon vader with custom lexicon (b.indo)
    senti.lexicon.update(senti_word)

    #True if text positive
    def is_label(tweet: str) -> bool:
        return senti.polarity_scores(tweet)['compound'] >= 0.05

    df = [Text]
    for tweet in df:
        if is_label(tweet) == True:
            label = "Positive"
            st.image('data/images/pos_image.png')
        else:
            label = "Negative"
            st.image('data/images/neg_image.png')

    sen_score = senti.polarity_scores(tweet)['compound']

    return label, sen_score
###########################################################################################################################################

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
            raw_text = raw_text.lower()
            submit_text = st.form_submit_button('Analyze')
            X_test = []
            if submit_text:
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
                st.write("Label = ",pred)
                
            #raw_text = st.text_area("Masukan Teks")
            
            #submit_text = st.form_submit_button("Analyze")

        #if submit_text:
            #col1,col2 = st.columns(2)
            
            #Sentiment Analysis
            #pred_sentiment, sen_score = Sentiment_analysis(raw_text)

            #col1.metric("Sentiment Score", sen_score)
            #col2.metric("Sentiment",pred_sentiment)

            #add_prediction_details(raw_text,pred_sentiment,sen_score,datetime.now())

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