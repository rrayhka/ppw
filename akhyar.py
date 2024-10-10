import streamlit as st
from prepos import *
import pickle


# Config Page
st.set_page_config(
    page_title="Prediksi Kategori Berita CNBC Indonesia",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': "https://rrayhka.github.io/pencarian-penambangan-web/",
        'About': "# Tugas PPW Klasifikasi Berita\nLink source:\nhttps://github.com/rrayhka/ppw"
    }
)

# Title Page
st.markdown("# Prediksi Kategori Berita CNBC Indonesia")

# Load model
lr_model = pickle.load(open('model/logistic_regression.pkl', 'rb'))
tfidf_model = pickle.load(open('model/tfidf_model.pkl', 'rb'))

# Label encode
labels_encode = {
    1: "Research",
    0: "News",
}

# Main Section
"""
Masukkan link berita dari CNN Indonesia untuk mengetahui kategori berita tersebut.
Kategori berita yang tersedia adalah **Internasional** dan **Nasional**.
"""
link_news = st.text_input("Masukkan link berita:", key="link_news")

if st.button("Prediksi"):
    # Check if link is empty
    if link_news == "":
        st.write("Link tidak boleh kosong")

    # Check if link is valid
    elif link_news != "" and ("cnbcindonesia.com" not in link_news or 
                          ("research" not in link_news and "news" not in link_news)):
        st.write("Link tidak valid, pastikan link berita dari CNBC Indonesia dan benar")

    # If link is valid
    else:
        news = scrapeNews(link_news)
        news['cleaned_text'] = news["isi"].apply(preprocessing)

        tfidf = model_tf_idf(news['cleaned_text'], tfidf_model, news['kategori'])
        tfidf = tfidf.drop(['Kategori Berita'], axis=1)

        prediction = lr_model.predict(tfidf)

        st.write(f"Kategori berita: {labels_encode[prediction[0]]}")