import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
import pickle
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# # NLTK Downloads
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')


# NewsClassifierApp class 
class NewsClassifierApp:
    def __init__(self):
        # Load models
        self.lr_model = pickle.load(open('model/logistic_regression.pkl', 'rb'))
        self.tfidf_model = pickle.load(open('model/tfidf_model.pkl', 'rb'))
        
        # Label encoding
        self.labels_encode = {
            1: "Research",
            0: "News",
        }
        
        # Stopwords for Indonesian
        self.stop_words = set(stopwords.words('indonesian'))

    # Scrape news content from the given URL
    def scrapeNews(self, url):
        isi = []
        judul = []
        response = requests.get(url)
        if response.status_code == 200:
            articleFull = BeautifulSoup(response.content, "html.parser")
            judulArtikel = articleFull.find(
                "h1", class_="mb-4 text-32 font-extrabold"
            ).text.strip()
            artikel_element = articleFull.find("div", class_="detail-text")
            artikelTeks = [p.get_text(strip=True) for p in artikel_element.find_all("p")]
            artikel_content = "\n".join(artikelTeks)
            isi.append(artikel_content)
            judul.append(judulArtikel)
        else:
            print(f"Error: {response.status_code}")
        return pd.DataFrame({"judul": judul, "isi": isi})

    # Cleansing function
    def cleansing(self, text):
        text = re.sub(r'[\s]+', ' ', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\x00-\x7f]', r'', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = re.sub(r'\b-\b', ' ', text)
        text = re.sub(r'[^\w\s]+', ' ', text)
        text = text.replace('\n', '')
        return text

    # Stopword removal
    def stopword(self, text):
        text = text.split()
        text = [word for word in text if word not in self.stop_words]
        text = ' '.join(text)
        return text

    # Stemming function
    def stemming_indo(self, text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = stemmer.stem(text)
        return text

    # Main preprocessing function
    def preprocessing(self, text):
        clean_text = self.cleansing(text)
        stopword_text = self.stopword(clean_text)
        stemmed_text = self.stemming_indo(stopword_text)
        return stemmed_text

    # TF-IDF model function
    @st.cache_data
    def model_tf_idf(self, data, _model, kategori):
        tfidf_matrix = _model.transform(data)
        feature_names = _model.get_feature_names_out()
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        df_tfidf.insert(0, 'Kategori Berita', kategori.reset_index(drop=True))
        return df_tfidf

    # Classify text (from backup.py)
    def classify_news(self, text):
        processed_text = self.preprocessing(text)
        text_vectorized = self.tfidf_model.transform([processed_text])
        prediction = self.lr_model.predict(text_vectorized)
        prediction_proba = self.lr_model.predict_proba(text_vectorized)
        return prediction[0], prediction_proba[0]

    # Function to handle website input
    def web_input(self):
        link_news = st.text_input("Masukkan link berita:")
        if st.button("Prediksi"):
            if link_news == "":
                st.write("Link tidak boleh kosong")
            elif "research" in link_news:
                kategori = "Research"
            elif "news" in link_news:
                kategori = "News"
            else:
                st.error("Link tidak valid, pastikan link berita dari CNBC Indonesia dan dari topik Research atau News.")
                return  
            
            
            news = self.scrapeNews(link_news)
            news['cleaned_text'] = news["isi"].apply(self.preprocessing)
            news["kategori"] = kategori
            tfidf = self.model_tf_idf(news['cleaned_text'], self.tfidf_model, news['kategori'])
            tfidf = tfidf.drop(['Kategori Berita'], axis=1)
            prediction = self.lr_model.predict(tfidf)
            st.write(f"Kategori berita: {self.labels_encode[prediction[0]]}")

    # Function to handle text input
    def text_input(self):
        user_input = st.text_area("Masukkan teks berita:")
        if st.button("Klasifikasikan"):
            if user_input.strip() == "":
                st.write("Silakan masukkan teks berita untuk diklasifikasikan.")
            else:
                category, probabilities = self.classify_news(user_input)
                category_name = "News" if category == 0 else "Research"
                st.subheader("Hasil Klasifikasi:")
                st.write(f"Kategori yang diprediksi: **{category_name}**")
                st.write(f"News: {probabilities[0] * 100:.2f}%")
                st.write(f"Research: {probabilities[1] * 100:.2f}%")

    # Main function to run the app
    def run(self):
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

        # Sidebar - Input Selection
        input_choice = st.sidebar.radio(
            "Pilih Metode Input:",
            ("Masukkan Link Berita", "Masukkan Teks Berita")
        )

        # Handle input based on choice
        if input_choice == "Masukkan Link Berita":
            self.web_input()
        else:
            self.text_input()

# Run the Streamlit app
if __name__ == "__main__":
    app = NewsClassifierApp()
    app.run()
