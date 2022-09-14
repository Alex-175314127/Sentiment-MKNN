import PIL.Image as Image
import numpy as np
import streamlit as st

import neattext.functions as nfx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud, ImageColorGenerator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px

import pandas as pd
import os
import json
from collections import Counter
from soupsieve import select

import base64
import time


### SENTIMENT ANALYSIS ###
def Sentiment_analysis():
    df = pd.read_csv('data/Clean_text.txt', encoding='utf-8')

    # LexiconVader dengan custom Lexicon(bahasa indonesia)
    sia1A, sia1B = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
    # Hapus Default lexicon VADER
    sia1A.lexicon.clear()
    sia1B.lexicon.clear()

    # Read custom Lexicon Bahasa Indonesia
    with open('data/lexicon_sentimen_negatif.txt') as f:
        data1A = f.read()
    with open('data/lexicon_sentimen_positif.txt') as f:
        data1B = f.read()

    # convert lexicon to dictonary
    insetNeg = json.loads(data1A)
    insetPos = json.loads(data1B)

    # update lexicon vader with custom lexicon (b.indo)
    sia1A.lexicon.update(insetNeg)
    sia1B.lexicon.update(insetPos)

    # method untuk cek apa sentimen pos,neg,neu
    def is_positive_inset(Text: str) -> bool:
        return sia1A.polarity_scores(Text)["compound"] + sia1B.polarity_scores(Text)["compound"] >= 0.05

    #def is_negative_inset(tweet: str) -> bool:
        #return sia1A.polarity_scores(tweet)["compound"] + sia1B.polarity_scores(tweet)["compound"] <= -0.05

    tweets = df['text'].to_list()

    with open('output/Sentiment-result.txt', 'w+') as f:
        for tweet in tweets:
            if is_positive_inset(tweet):
                label = "Positive"
            else:
                label = "Negative"
            f.write(str(label+"\n"))

    sen = pd.read_csv('output/Sentiment-result.txt', names=['Sentiment'])
    df = df.join(sen)

    ## Save clean Dataset
    df.to_csv('CleanText_Sentiment.csv', index=False)
    return df


############################################################################################
### EMOTION ANALYSIS ###
@st.cache
def Emotion_Analysis(Text):
    pretrained_name = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
    emotion = pipeline("sentiment-analysis",
                       model=pretrained_name,
                       tokenizer=pretrained_name)
    return emotion(Text)


# Extract the most common word in each emotion
def extract_keyword(Text, num=50):
    tokens = [i for i in Text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)


# Visualize Keuyword with WorldCloud
def visual_WordCould(Text):
    mask = np.array(Image.open('data/mask.jpg'))
    mywordcould = WordCloud(background_color="white", max_words=1000, mask=mask).generate(Text)
    img_color = ImageColorGenerator(mask)
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(mywordcould.recolor(color_func=img_color), interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

# Visualize Keyword with plot
def plot_most_common_word(mydict, emotion_name):
    df_emotion = pd.DataFrame(mydict.items(), columns=['token', 'count'])
    fig = px.bar(df_emotion, x='token', y='count', color='token', height=500, labels=emotion_name)
    st.plotly_chart(fig)


#############################################################################################################

timestr = time.strftime("%Y%m%d-%H%M%S")


def download_result(Text):
    st.markdown("### Download File ###")
    newFile = "Clean_dataset_{}_.csv".format(timestr)
    newFile2 = "Clean_dataset_{}_.txt".format(timestr)
    save_df = Text.to_csv(index=False)
    b64 = base64.b64encode(save_df.encode()).decode()
    href2 = f'<a download="{newFile2}" href="data:text/txt;base64,{b64}">🔰Download .txt</a>'
    href = f'<a download="{newFile}" href="data:text/csv;base64,{b64}">🔰Download .csv</a>'
    st.markdown(href2, unsafe_allow_html=True)
    st.markdown(href, unsafe_allow_html=True)


def download_Sentiment_result(Text):
    st.markdown("### Download File ###")
    newFile = "Sentiment_Result_{}_.csv".format(timestr)
    newFile2 = "Sentiment_Result_{}_.txt".format(timestr)
    save_df = Text.to_csv(index=False)
    b64 = base64.b64encode(save_df.encode()).decode()
    href2 = f'<a download="{newFile2}" href="data:text/txt;base64,{b64}">📥Download .txt</a>'
    href = f'<a download="{newFile}" href="data:text/csv;base64,{b64}">📥Download .csv</a>'
    st.markdown(href2, unsafe_allow_html=True)
    st.markdown(href, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="😶", layout='wide')
    st.title('Sentiment Analysis Twitter')
    st.markdown('This application is all about sentiment analysis of movie Review. All data is Crawling from Twitter')
    st.sidebar.title('Sentiment Analysis Movie Review')
    st.sidebar.markdown('We can analyse all movie Review from twitter')

    st.subheader("Preprocessing")
    df_file = st.file_uploader("Upload a dataset to clean", type=['csv'])

    # case_lower = st.sidebar.checkbox("Casefolding Text")
    # punc_del = st.sidebar.checkbox("Punctuation Removal")
    # normalize_text = st.sidebar.checkbox("Normalize Text")
    # token_text = st.sidebar.checkbox("Tokenize")
    # clean_stopword = st.sidebar.checkbox("Stopwords")
    # stem_text = st.sidebar.checkbox("Sastrawi Stemming IND")

    def load_data():
        data = pd.read_csv(df_file, encoding="utf-8")
        data = data[['user_name', 'date', 'text']]
        data['date'] = pd.to_datetime(data['date'])
        return data

    def casefolding(Text):
        Text = Text.lower()
        return Text

    # Punctuation Removal
    def punc_clean(Text):
        Text = nfx.remove_urls(Text)
        Text = nfx.remove_punctuations(Text)
        Text = nfx.remove_emojis(Text)
        Text = nfx.remove_special_characters(Text)
        Text = nfx.remove_numbers(Text)
        return Text

    @st.cache
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
    @st.cache
    def indo_stem(Text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        result = []
        for w in Text:
            result.append(stemmer.stem(w))
            # result.append(" ")
        return " ".join(result)

    if df_file is not None:
        raw_text = load_data()

        file_details = {"Filename": df_file.name, "Filesize": df_file.size, "Filetype": df_file.type}
        st.success(str(raw_text.shape[0]) + ' Dataset Loaded')
        st.write(file_details)

        # Decode Text
        # raw_text = df_file.read().decode('utf-8')
        # raw_ori = raw_text.copy()

        # col1,col2 = st.columns(2)

        with st.expander("Original Text"):
            st.write(raw_text)

        with st.spinner("Wait Preprocessing text in progress"):
            with st.expander("Pre-processing"):
                st.subheader('Case Folding (lower case)')
                raw_text['Clean_text'] = raw_text['text'].apply(casefolding)
                st.dataframe(raw_text[['text', 'Clean_text']].head(5))

                st.subheader("Remove Punctuations,url,numbers,emoji,and Special Character")
                raw_text['Clean_text'] = raw_text['Clean_text'].apply(punc_clean)
                st.dataframe(raw_text[['text', 'Clean_text']].head(5))

                st.subheader("Tokenization")
                raw_text['Clean_text'] = raw_text['Clean_text'].apply(word_tokenize_wrapper)
                st.table(raw_text[['text', 'Clean_text']].head(5))

                st.subheader("Normalisation")
                raw_text['Clean_text'] = raw_text['Clean_text'].apply(word_norm)
                st.table(raw_text[['text', 'Clean_text']].head(5))

                st.subheader("Stopword")
                raw_text['Clean_text'] = raw_text['Clean_text'].apply(remove_stopword)
                st.table(raw_text[['text', 'Clean_text']].head(5))

                st.subheader("Stemming Sastrawi")
                raw_text['Clean_text'] = raw_text['Clean_text'].apply(indo_stem)
                st.dataframe(raw_text[['text', 'Clean_text']].head(5))

                # st.write(raw_text)
                del raw_text['text']
                raw_text.rename(columns={'Clean_text': 'text'}, inplace=True)
                download_result(raw_text)
                raw_text.to_csv('data/Clean_text.txt')

        with st.spinner("Wait Sentiment Analysis in progress:"):
            with st.expander("Sentiment Result"):
                sen_result = Sentiment_analysis()
                sen_result['Emotion'] = Emotion_Analysis(sen_result['text'].to_list())
                sen_result['Emotion'] = sen_result.Emotion.apply(lambda x: x['label'])
                st.dataframe(sen_result)
                st.session_state['sen_result'] = sen_result
                sen_result.to_csv('output/Clean_text_Sentiment.csv')

            with st.expander("Tweets Sentiment and Emotion Visualize"):
                st.sidebar.markdown("Sentiment and Emotion Plot")

                emotion_list = sen_result['Emotion'].unique().tolist()
                sen_list = sen_result['Sentiment'].unique().tolist()
                # emotion_list.extend(sen_list)

                # Dalam Setiap sentiment dan emosi terdapat :
                # 1. List of Emotions
                # 2. Document of emotions
                # 3. Extract Keyword

                sentiment = sen_result['Sentiment'].value_counts()
                sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Tweets': sentiment.values})
                emotions = sen_result['Emotion'].value_counts()
                emotions = pd.DataFrame({'Emotion': emotions.index, 'Tweets': emotions.values})
                pl = st.sidebar.radio('Tweets visual of ...', ('Sentiment', 'Emotion'))
                if pl == "Sentiment":
                    select = st.sidebar.selectbox("Visual of Tweets Sentiment", ['Histogram', 'Wordcloud', 'Pie Chart'],
                                                  key=0)
                    if select == "Wordcloud":
                        ch = st.sidebar.selectbox("Sentiment", sen_list, key=0)
                        if ch == 'Positive':
                            pos_list = sen_result[sen_result['Sentiment'] == 'Positive']['text'].tolist()
                            pos_docx = ' '.join(pos_list)
                            keyword_pos = extract_keyword(pos_docx)
                            visual_WordCould(pos_docx)
                            plot_most_common_word(keyword_pos, "Positive")
                        elif ch == 'Negative':
                            neg_list = sen_result[sen_result['Sentiment'] == 'Negative']['text'].tolist()
                            neg_docx = ' '.join(neg_list)
                            keyword_neg = extract_keyword(neg_docx)
                            visual_WordCould(neg_docx)
                            plot_most_common_word(keyword_neg, "Negative")
                        else:
                            neu_list = sen_result[sen_result['Sentiment'] == 'Neutral']['text'].tolist()
                            neu_docx = ' '.join(neu_list)
                            keyword_neu = extract_keyword(neu_docx)
                            visual_WordCould(neu_docx)
                            plot_most_common_word(keyword_neu, "Neutral")
                    elif select == "Histogram":
                        st.subheader("Sentiment Plot")
                        fig = px.bar(sentiment, x='Sentiment', y='Tweets', color='Tweets', height=500)
                        st.plotly_chart(fig)
                    else:
                        fig = px.pie(sentiment, values='Tweets', names='Sentiment')
                        st.plotly_chart(fig)
                else:
                    select = st.sidebar.selectbox("Visual of Tweets Emotions", ['Histogram', 'Wordcloud', 'Pie Chart'])
                    if select == 'Wordcloud':
                        # ch = st.sidebar.selectbox("Emotions", (emotion_list), key=0)
                        if select == 'happy':
                            happy_list = sen_result[sen_result['Emotion'] == 'happy']['text'].tolist()
                            happy_docx = ' '.join(happy_list)
                            keyword_happy = extract_keyword(happy_docx)
                            visual_WordCould(happy_docx)
                            plot_most_common_word(keyword_happy, "Happy")
                        elif select == 'fear':
                            fear_list = sen_result[sen_result['Emotion'] == 'fear']['text'].tolist()
                            fear_docx = ' '.join(fear_list)
                            keyword_fear = extract_keyword(fear_docx)
                            visual_WordCould(fear_docx)
                            plot_most_common_word(keyword_fear, "Fear")
                        elif select == 'love':
                            love_list = sen_result[sen_result['Emotion'] == 'love']['text'].tolist()
                            love_docx = ' '.join(love_list)
                            keyword_love = extract_keyword(love_docx)
                            visual_WordCould(love_docx)
                            plot_most_common_word(keyword_love, "Love")
                        elif select == 'sadness':
                            sadness_list = sen_result[sen_result['Emotion'] == 'sadness']['text'].tolist()
                            sadness_docx = ' '.join(sadness_list)
                            keyword_sadness = extract_keyword(sadness_docx)
                            visual_WordCould(sadness_docx)
                            plot_most_common_word(keyword_sadness, "Sadness")
                        else:
                            angry_list = sen_result[sen_result['Emotion'] == 'anger']['text'].tolist()
                            angry_docx = ' '.join(angry_list)
                            visual_WordCould(angry_docx)
                            keyword_angry = extract_keyword(angry_docx)
                            plot_most_common_word(keyword_angry, "Angry")
                    elif select == "Histogram":
                        fig = px.bar(emotions, x='Emotion', y='Tweets', color='Tweets', height=500)
                        st.plotly_chart(fig)
                    else:
                        fig = px.pie(emotions, values='Tweets', names='Emotion')
                        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
