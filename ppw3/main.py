from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model klasifikasi
lr_model = pickle.load(open('model/lr_modelNormal.pkl', 'rb'))
tfidf_model = pickle.load(open('model/tfidf_modelLatest.pkl', 'rb'))

# Mengunduh data NLTK yang diperlukan
nltk.download('stopwords')
nltk.download('punkt')

# Enkode label untuk kategori
labels_encode = {1: "Research", 0: "News"}
# stop_words = set(stopwords.words('indonesian'))
stop_words = stopwords.words('indonesian')
# save stopwords
with open('stopwords.txt', 'w') as f:
  for item in stop_words:
    f.write("%s\n" % item)
# Fungsi untuk mengambil konten berita dari URL
def scrape_news(url):
    isi = []
    judul = []
    response = requests.get(url)
    if response.status_code == 200:
        article_full = BeautifulSoup(response.content, "html.parser")
        judul_artikel = article_full.find("h1", class_="mb-4 text-32 font-extrabold").text.strip()
        artikel_element = article_full.find("div", class_="detail-text")
        artikel_teks = [p.get_text(strip=True) for p in artikel_element.find_all("p")]
        artikel_content = "\n".join(artikel_teks)
        isi.append(artikel_content)
        judul.append(judul_artikel)
    return pd.DataFrame({"judul": judul, "isi": isi})

# Fungsi pembersihan teks
def cleansing(text):
    text = re.sub(r'[\s]+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\b-\b', ' ', text)
    text = re.sub(r'[^\w\s]+', ' ', text)
    text = text.replace('\n', '')
    return text

# Fungsi untuk menghapus stopword
def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Fungsi stemming
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# Fungsi utama untuk preprocessing teks
def preprocess_text(text):
    clean_text = cleansing(text)
    stopword_text = remove_stopwords(clean_text)
    return stemming(stopword_text)

# Fungsi untuk mengklasifikasikan teks
def classify_news(text):
    processed_text = preprocess_text(text)
    text_vectorized = tfidf_model.transform([processed_text])
    prediction = lr_model.predict(text_vectorized)
    prediction_proba = lr_model.predict_proba(text_vectorized)
    return prediction[0], prediction_proba[0]

# Fungsi untuk mengklasifikasikan teks dengan model yang berbeda
def classify_news_with_model(text, model):
    processed_text = preprocess_text(text)
    text_vectorized = tfidf_model.transform([processed_text])
    prediction = model.predict(text_vectorized)
    prediction_proba = model.predict_proba(text_vectorized)
    
    # Mengembalikan kategori, probabilitas berita, dan probabilitas penelitian
    return prediction[0], prediction_proba[0]  # prediction[0] untuk kategori, prediction_proba[0] untuk probabilitas


# Rute untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        link_news = request.form.get("link_news")
        selected_model = request.form.get("model")

        # Validasi input
        if not link_news:
            return render_template('index.html', error="Link tidak boleh kosong.")

        if "cnbcindonesia" not in link_news:
            return render_template('index.html', error="Link tidak valid. Pastikan link berita dari CNBC Indonesia.")

        # Mengambil konten berita dari URL yang diberikan
        news = scrape_news(link_news)
        news['cleaned_text'] = news["isi"].apply(preprocess_text)

        # Melakukan klasifikasi dengan model yang dipilih
        if selected_model == "logistic_regression":
            prediction, probabilities = classify_news(news['cleaned_text'][0])
            category_name = labels_encode[prediction]
            prob_news_percent = round(probabilities[0] * 100, 3)
            prob_research_percent = round(probabilities[1] * 100, 3)
        elif selected_model == "lr_modelNcompo5":
            # Memuat pipeline untuk 5 komponen
            with open('model_pipeline_5.pkl', 'rb') as f:
                pipeline_5 = pickle.load(f)
            # Transformasi menggunakan model TF-IDF yang dimuat
            X_new_tfidf = tfidf_model.transform([news['cleaned_text'][0]])
            prediction = pipeline_5.predict(X_new_tfidf)
            probabilities = pipeline_5.predict_proba(X_new_tfidf)
            category_name = labels_encode[prediction[0]]
            prob_news_percent = round(probabilities[0][0] * 100, 3)  # Akses probabilitas untuk kelas berita
            prob_research_percent = round(probabilities[0][1] * 100, 3)  # Akses probabilitas untuk kelas pe
        elif selected_model == "lr_modelNcompo10":
            # Memuat pipeline untuk 10 komponen
            with open('model_pipeline_10.pkl', 'rb') as f:
                pipeline_10 = pickle.load(f)
            # Transformasi menggunakan model TF-IDF yang dimuat
            X_new_tfidf = tfidf_model.transform([news['cleaned_text'][0]])
            prediction = pipeline_10.predict(X_new_tfidf)
            probabilities = pipeline_10.predict_proba(X_new_tfidf)
            category_name = labels_encode[prediction[0]]
            prob_news_percent = round(probabilities[0][0] * 100, 3)  # Akses probabilitas untuk kelas berita
            prob_research_percent = round(probabilities[0][1] * 100, 3)  # Akses probabilitas untuk kelas pe
        
        # Membulatkan probabilitas dan mengubah ke persen

        return render_template('index.html', result=category_name, prob_news=prob_news_percent, prob_research=prob_research_percent)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
