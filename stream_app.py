from youtube_comment_downloader import *
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

downloader = YoutubeCommentDownloader()

sia = SentimentIntensityAnalyzer()


# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

run_scraper = st.button('Analyze video')
you_video = st.text_input('youtube video URL')

@st.cache
def scrape(video_url):

    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    df_scraped = pd.DataFrame(comments)    
    return df_scraped

def analyze_df(df_comments):
    
    df_comments['compound_scores'] = [sia.polarity_scores(el)['compound'] for el in df_comments['text']]

    return df_comments

if (run_scraper and you_video):         

    df_scraped = scrape(you_video)

    st.write("number of comments:",len(df_scraped))

    df_comments = analyze_df(df_scraped)

    # fig = plt.figure()    
    # plt.box(df_comments, y="score")

    # st.pyplot(fig)    

    # Creating histogram
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(df_comments['compound_scores'], bins = [-1, -0.75, -0.50, -0.25, 0, 0.25, 0.5, 0.75, 1])
    
    # show plot
    st.pyplot(fig) 

    df_coments = df_comments.sort_values(by=['compound_scores'],ascending = False)

    st.markdown('**Highest scoring comments**')
    st.dataframe(df_coments[['text','compound_scores']].head(10))

    st.markdown('**Lowest scoring comments**')
    st.dataframe(df_coments[['text','compound_scores']].tail(10))