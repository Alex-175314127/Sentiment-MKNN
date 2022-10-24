import streamlit as st
import tweepy as tw
import pandas as pd
import base64
import time
import configparser
from stqdm import stqdm
from datetime import datetime
from track_util import create_page_visited_table,add_page_visited_details

timestr = time.strftime("%Y%m%d-%H%M%S")
def download_result(Text):
    st.markdown("### Download File ###")
    File_csv = "Twitter_dataset_{}_.csv".format(timestr)
    File_txt = "Twitter_dataset_{}_.txt".format(timestr)
    save_df = Text.to_csv(index=False)
    b64 = base64.b64encode(save_df.encode()).decode()
    href_txt = f'<a download="{File_txt}" href="data:text/txt;base64,{b64}">🔰Download .txt</a>'
    href_csv = f'<a download="{File_csv}" href="data:text/csv;base64,{b64}">🔰Download .csv</a>'
    st.markdown(href_txt, unsafe_allow_html=True)
    st.markdown(href_csv, unsafe_allow_html=True)

def main():
    create_page_visited_table()
    st.set_page_config(page_title="Twitter Data Crawling",page_icon="🐤",layout='wide')
    st.image('data/twitter_banner.jpg')
    st.subheader("Tools for Crawling Data from Twiiter")
    add_page_visited_details("Twitter Crawling",datetime.now())

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    cfg = configparser.ConfigParser()
    cfg.read('models/Twitter_keys.ini')

    consumer_api_key = cfg['Twitter_keys']['api_key']
    consumer_api_secret = cfg['Twitter_keys']['api_key_secret']
    auth = tw.OAuthHandler(consumer_api_key, consumer_api_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    with st.form(key='Twitter Form'):
        search_w = st.text_input("Enter keyword to search")
        limit_tw = st.slider("How many tweets to get", 0, 500, step=10)
        search_word = f'{search_w} -filter:retweets'
        date_since = st.date_input("Tweets Since")
        date_until = st.date_input("Tweets until")
        submit_btn = st.form_submit_button(label="Search")

        if submit_btn:
            tweets = tw.Cursor(api.search_tweets, q=search_word, lang='id', since=date_since, until=date_until).items(limit_tw)

            tweets_cp = []
            for tweet in stqdm(tweets):
                tweets_cp.append(tweet)
            print(f"new tweets retrieved: {len(tweets_cp)}")

            tweets_df = pd.DataFrame()
            for tweet in stqdm(tweets_cp):
                hashtags = []
                try:
                    for hashtag in tweet.entities["hashtags"]:
                        hashtags.append(hashtag["text"])
                    text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
                except:
                    pass
                tweets_df = tweets_df.append(pd.DataFrame({'user_id': tweet.user.id,
                                                           'user_name': tweet.user.name,
                                                           'user_location': tweet.user.location,\
                                                           'user_description': tweet.user.description,
                                                           'user_created': tweet.user.created_at,
                                                           'user_followers': tweet.user.followers_count,
                                                           'user_friends': tweet.user.friends_count,
                                                           'user_favourites': tweet.user.favourites_count,
                                                           'user_verified': tweet.user.verified,
                                                           'date': tweet.created_at,
                                                           'text': text,
                                                           'hashtags': [hashtags if hashtags else None],
                                                           'source': tweet.source,
                                                           'is_retweet': tweet.retweeted}, index=[0]))
            st.dataframe(tweets_df)
            download_result(tweets_df)



if __name__ == "__main__":
    main()