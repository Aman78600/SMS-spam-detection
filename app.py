import nltk
import string
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')  # PorterStemmer may also require this

# Download NLTK resources
download_nltk_resources()
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

page_bg_img = '''
<style>
body {
background-image: url("SMS_img.jpeg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.header("SMS Spam Prediction.")
input_data=st.text_area('Enter SMS.')
if st.button('Predict'):
    input_data_transform=tfidf.transform([text_transform(input_data)])
    if model.predict(input_data_transform)[0]:
        st.markdown('<b style="color:red;">Spam Massage.</b>', unsafe_allow_html=True)
    else:
        st.markdown('<b style="color:green;">NOT Spam Massage.</b>', unsafe_allow_html=True)
     
