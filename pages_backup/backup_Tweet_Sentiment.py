from secrets import choice
from unittest import result
from matplotlib.collections import Collection
import pandas as pd
import numpy as np
import streamlit as st
import os
import json
from datetime import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

from transformers import pipeline


import plotly.express as px
import altair as alt

def Emotion_Analysis(Text):
    pretrained_name = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
    emotion = pipeline("sentiment-analysis", 
                        model=pretrained_name,
                        tokenizer=pretrained_name)
    return emotion(Text)
################################################################################################################

def Sentiment_analysis(Text):
    sia1A, sia1B = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
    #Delete Default lexicon VADER
    sia1A.lexicon.clear()
    sia1B.lexicon.clear()

    #Read costom Lexicon
    with open('data/lexicon_sentimen_negatif.txt') as f:
        data1A = f.read()
    with open('data/lexicon_sentimen_positif.txt') as f:
        data1B = f.read()

    #convert lexicon to dictonary
    insetNeg = json.loads(data1A)
    insetPos = json.loads(data1B)

    #update lexicon vader with custom lexicon (b.indo)
    sia1A.lexicon.update(insetNeg)
    sia1B.lexicon.update(insetPos)

    #True if text positive
    def is_positive_inset(tweet: str) -> bool:
        return sia1A.polarity_scores(tweet)["compound"] + sia1B.polarity_scores(tweet)["compound"] >= 0.05
    #True if text negative
    def is_negative_inset(tweet: str) -> bool:
        return sia1A.polarity_scores(tweet)["compound"] + sia1B.polarity_scores(tweet)["compound"] <= -0.05

    if is_positive_inset(Text) == True:
        label = "Positive"
        st.image('data/images/pos_image.png')
    elif is_negative_inset(Text) == True:
        label = "Negative"
        st.image('data/images/neg_image.png')
    else:
        label = "Neutral"
        st.image('data/images/neg_image.png')

    sen_score = sia1A.polarity_scores(Text)["compound"] + sia1B.polarity_scores(Text)["compound"]

    return label, sen_score
###########################################################################################################################################

emotion_label_dict = {"sadness":"😭 Sedih", "anger":"😡🤬 Marah", "love":"😍💕 Cinta", "fear":"😨 Takut", "happy":"😁🤩 Bahagia"}

def main():
    from track_util import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotSen_table
    st.set_page_config(page_title="Predict Text Sentiment", layout='wide', page_icon="✍️")
    menu = ["Home", "History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotSen_table()
    if choice == "Home":
        st.subheader("Sentiment Text")
        add_page_visited_details("Home",datetime.now())

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Masukan Teks")
            
            submit_text = st.form_submit_button("Analyze")

        if submit_text:
            col1,col2 = st.columns(2)

            #Emotion Analysis
            pred_emotion = Emotion_Analysis(raw_text)
            #Sentiment Analysis
            pred_sentiment, sen_score = Sentiment_analysis(raw_text)

            for i in pred_emotion:
                prediction = i['label']
                pred_proba = i['score']

            col1.metric("Sentiment Score", sen_score)
            col2.metric("Sentiment",pred_sentiment)

            emotion_label = emotion_label_dict[prediction]
            col1.metric("Emotion Score", pred_proba)
            col2.metric("Emotion Result", emotion_label)
            add_prediction_details(raw_text,pred_sentiment,sen_score,prediction,pred_proba,datetime.now())

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
            df_hs =pd.DataFrame(view_all_prediction_details(), columns=['raw_text','Sentiment','sentiment_score','Emotion','emotion_score','Time_of_visit'])
            st.dataframe(df_hs)

            #Sentiment Plot
            sen_count = df_hs['Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Counts')
            sen_plt = alt.Chart(sen_count).mark_bar().encode(x='Sentiment', y='Counts', color='Sentiment')
            st.altair_chart(sen_plt,use_container_width=True)

            #Emotion Plot
            emo_count = df_hs['Emotion'].value_counts().rename_axis('Emotion').reset_index(name='Counts')
            emo_plt =alt.Chart(emo_count).mark_bar().encode(x='Emotion', y='Counts', color='Emotion')
            st.altair_chart(emo_plt,use_container_width=True)

    else:
        st.subheader()






if __name__ =='__main__':
    main()