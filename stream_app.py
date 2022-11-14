import itertools
from youtube_comment_downloader import *
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import numpy as np
from transformers import pipeline


downloader = YoutubeCommentDownloader()

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
    comments_list = list(df_scraped['text'])

    sentiment_pipeline = pipeline(model='distilbert-base-uncased-finetuned-sst-2-english')

    inference = sentiment_pipeline(comments_list)

    values_list = [list(inference[i].values()) for i in range(len(inference))]

    df_scraped[['label','score']] = pd.DataFrame(values_list, columns = ['label','score'])

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
    ax.hist(df_comments['score'], bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # show plot
    st.pyplot(fig) 

    df_coments = df_comments.sort_values(by=['score'],ascending = False)

    st.markdown('**Highest scoring comments**')
    st.dataframe(df_coments[['text','score']].head(10))

    st.markdown('**Lowest scoring comments**')
    st.dataframe(df_coments[['text','score']].tail(10))


    #auto bining
    # df_coments['binning'] = pd.cut(df_coments['score'],5, labels=["bad","neutral_low", "neutral", "neutral-high","good"])

    # hist_plot = alt.Chart(df_coments).mark_bar().encode(
    #     x='binning',y='score',)

    # st.write("Barplot Score")    

    # st.altair_chart(hist_plot, use_container_width=True)