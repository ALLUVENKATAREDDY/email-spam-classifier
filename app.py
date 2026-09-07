from unittest import result

import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    res = []
    for i in text:
        if i.isalnum():
            res.append(i)

    text = res[:]
    res.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            res.append(i)

    text = res[:]
    res.clear()

    for i in text:
        res.append(ps.stem(i))

    return ' '.join(res)


transform_text('Im still looking for a car to buy. And have not gone 4the driving test yet.')

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load((open('model1.pkl','rb')))

st.title('Email/SMS Spam Classifier')

input_sms=st.text_area('Enter a message to classify')

if st.button("Predict"):
    # 1. Preprocessing
    transform_sms=transform_text((input_sms))

    # 2.Vectorize
    vector_input= tfidf.transform([transform_sms])

    # 3.Predict
    result=model.predict(vector_input)[0]

    # 4.Display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
