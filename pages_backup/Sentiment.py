import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import reprlib
import streamlit as st
from collections import Counter
from wordcloud import WordCloud


                ### Sentiment Analysis###
os.chdir("output")

base = "Clean_text.txt"

input_stream = open(base, "r", encoding="utf-8")
text = input_stream.readlines()
input_stream.close()
print("before:", len(text))

# menghapus duplikasi kalimat dengan mengonversinya ke 'ordered dictionary'
newText = list(OrderedDict.fromkeys(text))
print("after:", len(newText))

output = os.path.splitext(base)[0]+'-dup.txt'
with open(output, 'w', encoding="utf-8") as f:
    for line in newText:
        f.write(str(line))

df = pd.read_csv(output, encoding='latin-1', header=None, sep=",", names=['','user_name', 'date', 'text'], usecols=['user_name', 'date', 'text'], dtype=str)

# Memanfaatkan nltk VADER untuk menggunakan leksikon kustom
sia1A, sia1B = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
# membersihkan leksikon VADER default
sia1A.lexicon.clear()
sia1B.lexicon.clear()

# Membaca leksikon InSet
# Leksikon InSet lexicon dibagi menjadi dua, yakni polaritas negatif dan polaritas positif;
# kita akan menggunakan nilai compound saja untuk memberi label pada suatu kalimat
with open('_json_inset-neg.txt') as f:
    data1A = f.read()
with open('_json_inset-pos.txt') as f:
    data1B = f.read()

# Mengubah leksikon sebagai dictionary
insetNeg = json.loads(data1A)
insetPos = json.loads(data1B)

# Update leksikon VADER yang sudah 'dimodifikasi'
sia1A.lexicon.update(insetNeg)
sia1B.lexicon.update(insetPos)

print(reprlib.repr(sia1A.lexicon))
print(reprlib.repr(sia1B.lexicon))

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

plt.figure(figsize=(20,10))
fig = sns.countplot(x='Sentiment',data=df)
st.pyplot(fig)

## Save clean Dataset
df.to_csv('CleanText_Sentiment.csv', index=False)




                ### Emotion Analysis ###
from transformers import pipeline
pretrained_name = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
nlp = pipeline(
    "sentiment-analysis",
    model=pretrained_name,
    tokenizer=pretrained_name
)
df['Emotion'] = nlp(df['text'].to_list())
df['Emotion'] = df.Emotion.apply(lambda x: x['label'])

df.to_csv('CleanText_Sentiment_Emotion.csv')

df_clean = df.copy()
def extract_keyword(Text, num=50):
    tokens = [i for i in Text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)

emotion_list = df_clean['Emotion'].unique().tolist()

#List of Emotions
happy_list = df_clean[df_clean['Emotion'] == 'happy']['text'].tolist()
fear_list = df_clean[df_clean['Emotion'] == 'fear']['text'].tolist()
love_list = df_clean[df_clean['Emotion'] == 'love']['text'].tolist()
sadness_list = df_clean[df_clean['Emotion'] == 'sadness']['text'].tolist()
angry_list = df_clean[df_clean['Emotion'] == 'angry']['text'].tolist()

#Document of emotions
happy_docx = ' '.join(happy_list)
fear_docx = ' '.join(fear_list)
love_docx = ' '.join(love_list)
sadness_docx = ' '.join(sadness_list)
angry_docx = ' '.join(angry_list)

#Extract Keyword
keyword_happy = extract_keyword(happy_docx)
keyword_fear = extract_keyword(fear_docx)
keyword_love = extract_keyword(love_docx)
keyword_sadness = extract_keyword(sadness_docx)
keyword_angry= extract_keyword(angry_docx)

#Visualize Keyword with plot
def plot_most_common_word(mydict,emotion_name):
    df_emotion = pd.DataFrame(mydict.items(),columns=['token','count'])
    plt.figure(figsize=(20,10))
    plt.title("Plot of {}".format(emotion_name))
    sns.barplot(x='token',y='count',data=df_emotion)
    plt.xticks(rotation=45)
    return plt.show()

#Create if for option of wordcloud 
plot_most_common_word(keyword_sadness,"sadness")


#Visualize Keuyword with WorldCloud
def visual_WordCould(docx):
    mywordcould = WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcould,interpolation='bilinear')
    plt.axis('off')
    return plt.show()

#Create if statemen for other emotion
visual_WordCould(happy_docx)