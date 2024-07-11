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

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('SMS_img.jpg')  

st.header("SMS Spam Prediction.")
input_data=st.text_area('Enter SMS.')
if st.button('Predict'):
    input_data_transform=tfidf.transform([text_transform(input_data)])
    if model.predict(input_data_transform)[0]:
        st.markdown('<b style="color:red;">Spam Massage.</b>', unsafe_allow_html=True)
    else:
        st.markdown('<b style="color:green;">NOT Spam Massage.</b>', unsafe_allow_html=True)
     
