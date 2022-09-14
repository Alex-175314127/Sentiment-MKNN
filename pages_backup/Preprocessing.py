from distutils.command.upload import upload
import pandas as pd
import numpy as np
import time
import nltk
import json
import reprlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import neattext.functions as nfx
import streamlit as st
import os
from collections import Counter
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import OrderedDict
from multiapp import MultiApp
from pages import test_view

def CleanData():
    st.title('Sentiment Analysis Twitter')
    st.markdown('This application is all about sentiment analysis of movie Review. All data is Crawling from Twitter')
    st.sidebar.title('Sentiment Analysis Movie Review')
    st.sidebar.markdown('We can analyse all movie Review from twitter')

    file_df = st.file_uploader("Pick a File ...")


    def load_data():
        data = pd.read_csv(file_df, encoding="utf-8")
        data = data[['user_name','date','text']]
        data['date'] = pd.to_datetime(data['date'])
        return data

    def casefolding(Text):
        Text = Text.lower()
        return Text

    #Punctuation Removal
    def punc_clean(Text):
        Text = nfx.remove_punctuations(Text)
        Text = nfx.remove_emojis(Text)
        Text = nfx.remove_special_characters(Text)
        Text = nfx.remove_numbers(Text)
        return Text

    #Normalization
    

    def word_norm(tweets):
        word_dict = pd.read_csv('data/indonesia_slangWords.csv')
        tweets = tweets.lower()
        res = ''
        for i in tweets.split():
            if i in word_dict.slang.values:
                res += word_dict[word_dict['slang'] == i]['formal'].iloc[0]
            else:
                res += i
            res += ' '
        return res

    # Tokenize
    def word_tokenize_wrapper(Text):
        return word_tokenize(Text)

    # Stopwords
    def remove_stopword(Text):
        stopW = stopwords.words('indonesian','english')
        sw = pd.read_csv('data/stopwordbahasa.csv')
        stopW.extend(sw)
        remove_sw = ' '.join(Text)
        clean_sw = [word for word in remove_sw.split() if word.lower() not in stopW]
        return clean_sw

    ## Indonesia Stemming
    def indo_stem(Text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        result =[]
        for w in Text:
            result.append(stemmer.stem(w))
            result.append(" ")
        return "".join(result)

    if file_df is not None:
        df = load_data()

    submit = st.button("Star Prepocessing")
    if file_df is not None and submit:
        df['text'] = df['text'].apply(casefolding)
        df['text'] = df['text'].apply(punc_clean)
        df['text'] = df['text'].apply(word_norm)
        df['text'] = df['text'].apply(word_tokenize_wrapper)
        df['text'] = df['text'].apply(remove_stopword)
        df['text'] = df['text'].apply(indo_stem)


        df.to_csv('output/Clean_text.txt', index=False)
        
    elif file_df is None and submit:
        st.error("Please pick a File")



if __name__ == "__main__":
    CleanData()