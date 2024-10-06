import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils import resample
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os

def page_config(title):
    st.set_page_config(page_title=title,layout="wide")
