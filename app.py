import nltk
import string
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# text preprocessing
sw=stopwords.words('english')
puntuation=string.punctuation
ps=PorterStemmer()
def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            if i not in sw and i not in puntuation:
                y.append(ps.stem(i))
    text=' '.join(y)
    return text
tfidf=pickle.load(open('transformar.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.header("SMS Spam Prediction.")
input_data=st.text_area('enter SMS.')
if st.button('Predict'):
    input_data_transform=tfidf.transform([text_transform(input_data)])
    if model.predict(input_data_transform)[0]:
        st.write('SPAM')
    else:
        st.write('NOT SPAM')