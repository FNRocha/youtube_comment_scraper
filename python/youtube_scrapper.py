# Databricks notebook source
#!/usr/bin/env python
# coding: utf-8



import itertools
from youtube_comment_downloader import *
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline
import altair as alt




downloader = YoutubeCommentDownloader()

comments = downloader.get_comments_from_url('https://www.youtube.com/watch?v=FJI23liDg4g&ab_channel=JovemNerd', sort_by=SORT_BY_POPULAR)




df_coments = pd.DataFrame(comments)




len(df_coments)




comments_list = list(df_coments['text'])




sentiment_pipeline = pipeline(model='distilbert-base-uncased-finetuned-sst-2-english')
    #"sentiment-analysis")




idx = 22
inference = sentiment_pipeline(comments_list)
#print("inference: ",inference[idx])




values_list = [list(inference[i].values()) for i in range(len(inference))]




df_coments[['label','score']] = pd.DataFrame(values_list, columns = ['label','score'])




fig, ax = plt.subplots()
ax.hist(df_coments[['score']], bins=20)

st.pyplot(fig)




#auto bining
df_coments['binning'] = pd.cut(df_coments['score'],5, labels=["bad","neutral_low", "neutral", "neutral-high","good"])




df_coments[['score','binning']].head()




alt.Chart(df_coments).mark_bar().encode(
    x='binning',y='score')


# In[ ]:




