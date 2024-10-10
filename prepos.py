import streamlit as st
import requests
from bs4 import BeautifulSoup

# Library untuk data manipulation & visualisasi
import pandas as pd
import re
import string

# Library untuk text preprocessing
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

def scrapeNews(url):
    """
    https://www.cnbcindonesia.com/news/20241010135430-4-578546/bos-bos-batu-bara-mau-bikin-pusat-hiburan-di-ikn

    https://www.cnbcindonesia.com/research/20241010111125-128-578481/prabowo-yakin-ri-bisa-jadi-produsen-emas-terbesar-dunia-ini-buktinya
    """
    isi = []
    judul = []
    response = requests.get(url)
    if response.status_code == 200:
        articleFull = BeautifulSoup(response.content, "html.parser")
        judulArtikel = articleFull.find(
        "h1", class_="mb-4 text-32 font-extrabold"
        ).text.strip()  # Untuk mendapatkan judul artikel
        artikel_element = articleFull.find("div", class_="detail-text") # Isi artikel terdapat pada tag div dengan class detail-text
        artikelTeks = [p.get_text(strip=True) for p in artikel_element.find_all("p")] # Mengambil semua isi artikel yang terdapat di tag p
        artikel_content = "\n".join(artikelTeks)
        isi.append(artikel_content)
        judul.append(judulArtikel)
    else:
        print(f"Error: {article_response.status_code}")
    return pd.DataFrame({"judul":judul, "isi":isi})

# Cleansing function to clean text
def cleansing(text):
    text = re.sub(r'[\s]+', ' ', text)  # Menghapus tambahan spasi
    text = text.encode('ascii', 'ignore').decode('utf-8')  # Menghapus karakter non-ASCII
    text = re.sub(r'[^\x00-\x7f]', r'', text)  # Menghapus karakter non-printable
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(r'\b-\b', ' ', text)  # Menghapus tanda hubung yang tidak berada di antara dua huruf
    text = re.sub(r'[^\w\s]+', ' ', text)  # Menghapus tanda baca
    text = text.replace('\n', '')  # Menghapus baris baru
    return text

# Stopword removal function
def stopword(text):
    stop_words = set(stopwords.words('indonesian'))  # Mengambil daftar stopwords Bahasa Indonesia
    text = text.split()  # Memisahkan teks menjadi kata-kata
    text = [word for word in text if word not in stop_words]  # Menghapus stopwords dari teks
    text = ' '.join(text)  # Menggabungkan kata-kata kembali menjadi teks
    return text

# Stemming function using Sastrawi
def stemming_indo(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)  # Melakukan stemming menggunakan Sastrawi
    return text

# Main preprocessing function
def preprocessing(text):
    clean_text = cleansing(text)  # Cleansing
    stopword_text = stopword(clean_text)  # Menghapus stopwords
    stemmed_text = stemming_indo(stopword_text)  # Stemming
    return stemmed_text

@st.cache_data
def model_tf_idf(data, _model, kategori):
    """
    Membuat model TF-IDF dari data
    """
    tfidf_matrix = _model.transform(data)
    feature_names = _model.get_feature_names_out()
    
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    df_tfidf.insert(0, 'Kategori Berita', kategori.reset_index(drop=True))
    return df_tfidf