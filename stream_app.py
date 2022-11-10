import itertools
from youtube_comment_downloader import *
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from transformers import pipeline


downloader = YoutubeCommentDownloader()

you_video = st.text_input('https://www.youtube.com/watch?v=YClmpnpszq8&ab_channel=CodingIsFun')

st.write('youtube video', you_video)

comments = downloader.get_comments_from_url('https://www.youtube.com/watch?v=YClmpnpszq8&ab_channel=CodingIsFun', sort_by=SORT_BY_POPULAR)

df_coments = pd.DataFrame(comments)

st.write("number of comments:",len(df_coments))

comments_list = list(df_coments['text'])

sentiment_pipeline = pipeline(model='distilbert-base-uncased-finetuned-sst-2-english')

inference = sentiment_pipeline(comments_list)

values_list = [list(inference[i].values()) for i in range(len(inference))]

df_coments[['label','score']] = pd.DataFrame(values_list, columns = ['label','score'])


#auto bining
df_coments['binning'] = pd.cut(df_coments['score'],5, labels=["bad","neutral_low", "neutral", "neutral-high","good"])

hist_plot = alt.Chart(df_coments).mark_bar().encode(
    x='binning',y='score')

st.write("Barplot Score")    

st.altair_chart(hist_plot, use_container_width=True)

df_coments = df_coments.sort_values(by=['score'])

st.dataframe(df_coments['text'].head(5))

st.dataframe(df_coments['text'].tail(5  ))