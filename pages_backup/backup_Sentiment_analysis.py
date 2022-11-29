from distutils import text_file
from locale import normalize
from tkinter.tix import Select
from numpy import empty
import streamlit as st

import neattext.functions as nfx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import tokenizers
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px

import pandas as pd
import os
import json
import reprlib
from collections import Counter
from collections import OrderedDict
from soupsieve import select

import base64
import time

            ### SENTIMENT ANALYSIS ###
def Sentiment_analysis():
    orig_dir = os.getcwd()
    try:
        os.chdir("output")
        base = "Clean_text.txt"

        input_stream = open(base, "r", encoding="utf-8")
        text = input_stream.readlines()
        input_stream.close()
        print("before:", len(text))

        #Remove duplicate word in text with convert to OrderedDict
        newText = list(OrderedDict.fromkeys(text))
        print("after:", len(newText))

        output = os.path.splitext(base)[0]+'-dup.txt'
        with open(output, 'w', encoding="utf-8") as f:
            for line in newText:
                f.write(str(line))

        df = pd.read_csv(output, encoding='latin-1', header=None, sep=",", names=['','user_name', 'date', 'text'], usecols=['user_name', 'date', 'text'], dtype=str)
    
        #Sentiment analysis using nltk Vader with custom Lexicon(bahasa indonesia)
        sia1A, sia1B = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
        #Delete Default lexicon VADER
        sia1A.lexicon.clear()
        sia1B.lexicon.clear()

        #Read costom Lexicon
        #there are 2 lexicon that is positive and negative polarity
        #value of Sentiment obtained from compound value
        with open('_json_inset-neg.txt') as f:
            data1A = f.read()
        with open('_json_inset-pos.txt') as f:
            data1B = f.read()

        #convert lexicon to dictonary
        insetNeg = json.loads(data1A)
        insetPos = json.loads(data1B)

        #update lexicon vader with custom lexicon (b.indo)
        sia1A.lexicon.update(insetNeg)
        sia1B.lexicon.update(insetPos)

        def is_positive_inset(tweet: str) -> bool:
            """True if tweet has positive compound sentiment, False otherwise."""
            return sia1A.polarity_scores(tweet)["compound"] + sia1B.polarity_scores(tweet)["compound"] >= 0.05

        def is_negative_inset(tweet: str) -> bool:
            """True if tweet has Negative compound sentiment, False otherwise."""
            return sia1A.polarity_scores(tweet)["compound"] + sia1B.polarity_scores(tweet)["compound"] <= -0.05


        tweets = df['text'].to_list()

        output = os.path.splitext(base)[1]+'-Sentiment-result.txt'
        with open(output, 'w') as f:
            for tweet in tweets:
                if is_positive_inset(tweet) == True:
                    label = "Positive"
                elif is_negative_inset(tweet) == True:
                    label = "Negative"
                else:
                    label = "Neutral"
                f.write(str(label+'\n'))

        sen = pd.read_csv('.txt-Sentiment-result.txt', names=['Sentiment'])
        df = df.join(sen)

        ## Save clean Dataset
        df.to_csv('CleanText_Sentiment.csv', index=False)
    finally:
        os.chdir(orig_dir)
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

#Extract the most common word in each emotion
def extract_keyword(Text, num=50):
    tokens = [i for i in Text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)

#Visualize Keuyword with WorldCloud
def visual_WordCould(Text):
    mywordcould = WordCloud().generate(Text)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(mywordcould,interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

#Visualize Keyword with plot
def plot_most_common_word(mydict,emotion_name):
    df_emotion = pd.DataFrame(mydict.items(),columns=['token','count'])
    fig = plt.figure(figsize=(20,10))
    plt.title("Plot of {}".format(emotion_name))
    sns.barplot(x='token',y='count',data=df_emotion)
    plt.xticks(rotation=45)
    st.pyplot(fig)


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
    
    st.title('Sentiment Analysis Twitter')
    st.markdown('This application is all about sentiment analysis of movie Review. All data is Crawling from Twitter')
    st.sidebar.title('Sentiment Analysis Movie Review')
    st.sidebar.markdown('We can analyse all movie Review from twitter')

    st.subheader("Preprocessing")
    df_file = st.file_uploader("Upload a dataset to clean", type=['csv'])
    #case_lower = st.sidebar.checkbox("Casefolding Text")
    #punc_del = st.sidebar.checkbox("Punctuation Removal")
    #normalize_text = st.sidebar.checkbox("Normalize Text")
    #token_text = st.sidebar.checkbox("Tokenize")
    #clean_stopword = st.sidebar.checkbox("Stopwords")
    #stem_text = st.sidebar.checkbox("Sastrawi Stemming IND")



    def load_data():
        data = pd.read_csv(df_file, encoding="utf-8")
        data = data[['user_name','date','text']]
        data['date'] = pd.to_datetime(data['date'])
        return data

    def casefolding(Text):
        Text = Text.lower()
        return Text

    #Punctuation Removal
    def punc_clean(Text):
        Text = nfx.remove_urls(Text)
        Text = nfx.remove_punctuations(Text)
        Text = nfx.remove_emojis(Text)
        Text = nfx.remove_special_characters(Text)
        Text = nfx.remove_numbers(Text)
        return Text

    def word_norm(tweets):
        word_dict = pd.read_csv('data/indonesia_slangWords.csv')
        #tweets = tweets.lower()
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
    @st.cache
    def indo_stem(Text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        result =[]
        for w in Text:
            result.append(stemmer.stem(w))
            #result.append(" ")
        return " ".join(result)

    
    if df_file is not None:
        file_details = {"Filename": df_file.name, "Filesize": df_file.size, "Filetype":df_file.type}
        st.write(file_details)

        #Decode Text
        #raw_text = df_file.read().decode('utf-8')
        raw_text = load_data()

        col1,col2 = st.columns(2)

        with col1:
            with st.expander("Original Text"):
                st.write(raw_text)


        with col2:
            with st.expander("Preprocessed Text"):
                raw_text['text'] = raw_text['text'].apply(casefolding)
                raw_text['text'] = raw_text['text'].apply(punc_clean)
                raw_text['text'] = raw_text['text'].apply(word_norm)
                raw_text['text'] = raw_text['text'].apply(word_tokenize_wrapper)
                raw_text['text'] = raw_text['text'].apply(remove_stopword)
                raw_text['text'] = raw_text['text'].apply(indo_stem)
                    
                st.write(raw_text)

                download_result(raw_text)
                raw_text.to_csv('output/Clean_text.txt')

      
        with st.expander("Sentiment Result"):
            sen_result = Sentiment_analysis()
            sen_result['Emotion'] = Emotion_Analysis(sen_result['text'].to_list())
            sen_result['Emotion'] = sen_result.Emotion.apply(lambda x: x['label'])
            st.dataframe(sen_result)
            st.session_state['sen_result'] = sen_result
            sen_result.to_csv('output/Clean_text_Sentiment.csv')

        with st.expander("Tweets Visualize"):
            st.sidebar.markdown("Sentiment and Emotion Plot")

            emotion_list = sen_result['Emotion'].unique().tolist()
            sen_list = sen_result['Sentiment'].unique().tolist()
            #emotion_list.extend(sen_list)

            #List of Emotions
            happy_list = sen_result[sen_result['Emotion'] == 'happy']['text'].tolist()
            fear_list = sen_result[sen_result['Emotion'] == 'fear']['text'].tolist()
            love_list = sen_result[sen_result['Emotion'] == 'love']['text'].tolist()
            sadness_list = sen_result[sen_result['Emotion'] == 'sadness']['text'].tolist()
            angry_list = sen_result[sen_result['Emotion'] == 'anger']['text'].tolist()
            pos_list = sen_result[sen_result['Sentiment'] == 'Positive']['text'].tolist()
            neg_list = sen_result[sen_result['Sentiment'] == 'Negative']['text'].tolist()
            neu_list = sen_result[sen_result['Sentiment'] == 'Neutral']['text'].tolist()

            #Document of emotions
            happy_docx = ' '.join(happy_list)
            fear_docx = ' '.join(fear_list)
            love_docx = ' '.join(love_list)
            sadness_docx = ' '.join(sadness_list)
            angry_docx = ' '.join(angry_list)
            pos_docx = ' '.join(pos_list)
            neg_docx = ' '.join(neg_list)
            neu_docx = ' '.join(neu_list)
            

            #Extract Keyword
            keyword_happy = extract_keyword(happy_docx)
            keyword_fear = extract_keyword(fear_docx)
            keyword_love = extract_keyword(love_docx)
            keyword_sadness = extract_keyword(sadness_docx)
            keyword_angry= extract_keyword(angry_docx)
            keyword_pos = extract_keyword(pos_docx)
            keyword_neg = extract_keyword(neg_docx)
            keyword_neu = extract_keyword(neu_docx)


            sentiment = sen_result['Sentiment'].value_counts()
            sentiment = pd.DataFrame({'Sentiment':sentiment.index, 'Tweets':sentiment.values})
            emotions = sen_result['Emotion'].value_counts()
            emotions = pd.DataFrame({'Emotion':emotions.index, 'Tweets':emotions.values})
            pl = st.sidebar.radio('Tweets visual of ...',('Sentiment','Emotion'))
            if pl == "Sentiment":
                select = st.sidebar.selectbox("Visual of Tweets Sentiment", ['Wordcloud','Histogram','Pie Chart'],key=0)
                if select == "Wordcloud":
                    ch = st.sidebar.selectbox("Sentiment", (sen_list), key=0)
                    if ch == 'Positive':
                        visual_WordCould(pos_docx) 
                    elif ch == 'Negative':
                        visual_WordCould(neg_docx)
                    else:
                        visual_WordCould(neu_docx)
                elif select == "Histogram":
                    fig = px.bar(sentiment, x='Sentiment', y='Tweets', color='Tweets', height=500)
                    st.plotly_chart(fig)
                else :
                    fig = px.pie(sentiment, values='Tweets', names='Sentiment')
                    st.plotly_chart(fig)
            else:
                select = st.sidebar.selectbox("Visual of Tweets Emotions", ['Wordcloud','Histogram','Pie Chart'])
                if select == 'Wordcloud':
                    ch = st.sidebar.selectbox("Emotions",(emotion_list), key=0)
                    if select == 'happy':
                        visual_WordCould(happy_docx)
                    elif select == 'fear':
                        visual_WordCould(fear_docx)
                    elif select == 'love':
                        visual_WordCould(love_docx)
                    elif select == 'sadness':
                        visual_WordCould(sadness_docx)
                    else:
                        visual_WordCould(angry_docx)
                elif select == "Histogram":
                    fig = px.bar(emotions, x='Emotion', y='Tweets', color='Tweets', height=500)
                    st.plotly_chart(fig)
                else :
                    fig = px.pie(emotions, values='Tweets', names='Emotion')
                    st.plotly_chart(fig)


            
        







if __name__ == "__main__":
    main()
