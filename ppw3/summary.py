from flask import Flask, request, render_template, jsonify
import pandas as pd
import requests
import os
import re
import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import nltk

# Inisialisasi NLTK
nltk.download("stopwords")
nltk.download("punkt")

# Inisialisasi Flask
app = Flask(__name__)

# Fungsi untuk scraping berita
def scrape_news(url):
    isi = []
    judul = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        article_full = BeautifulSoup(response.content, "html.parser")
        judul_artikel = article_full.find("h1", class_="mb-4 text-32 font-extrabold")
        if judul_artikel:
            judul_artikel = judul_artikel.text.strip()
        else:
            judul_artikel = "Judul tidak ditemukan"
        artikel_element = article_full.find("div", class_="detail-text")
        if artikel_element:
            artikel_teks = [p.get_text(strip=True) for p in artikel_element.find_all("p")]
            artikel_content = "\n".join(artikel_teks)
        else:
            artikel_content = "Konten artikel tidak ditemukan"
        isi.append(artikel_content)
        judul.append(judul_artikel)
    except requests.exceptions.RequestException as e:
        judul.append("Error")
        isi.append(f"Gagal mengambil data: {e}")

    return pd.DataFrame({"judul": judul, "isi": isi})

# Fungsi preprocessing
def preprocess_text(content):
    content = content.lower()
    content = re.sub(r"[0-9]|[/(){}\[\]\|@,;_]|[^a-z .]+", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    tokens = word_tokenize(content)
    stopword = set(stopwords.words("indonesian"))
    tokens = [word for word in tokens if word not in stopword]
    return " ".join(tokens)

# Fungsi untuk membuat ringkasan dan visualisasi graf
def summarize_and_visualize(content):
    kalimat = sent_tokenize(content)
    preprocessed_text = preprocess_text(content)
    kalimat_preprocessing = sent_tokenize(preprocessed_text)

    # TF-IDF dan cosine similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(kalimat_preprocessing)
    cossim_prep = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Analisis jaringan dengan NetworkX
    G = nx.DiGraph()
    for i in range(len(cossim_prep)):
        G.add_node(i)
        for j in range(len(cossim_prep)):
            if cossim_prep[i][j] > 0.1 and i != j:
                G.add_edge(i, j)

    # Hitung closeness centrality dan buat ringkasan
    closeness_scores = nx.closeness_centrality(G)
    sorted_closeness = sorted(closeness_scores.items(), key=lambda x: x[1], reverse=True)
    ringkasan = " ".join(kalimat[node] for node, _ in sorted_closeness[:3])

    # Visualisasi graf
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=2)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="b")
    nx.draw_networkx_edges(G, pos, edge_color="red", arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Graph Representation of Sentence Similarity")
    # Periksa apakah file graph.png sudah ada
    graph_path = "static/graph.png"
    if os.path.exists(graph_path):
        os.remove(graph_path)  # Hapus file jika sudah ada

    # Simpan graf sebagai file baru
    plt.savefig(graph_path)
    plt.close()

    return ringkasan

# Route utama untuk scraping dan analisis
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            # Scraping berita
            df = scrape_news(url)
            if not df.empty:
                content = df["isi"].iloc[0]
                title = df["judul"].iloc[0]

                # Preprocessing, summarizing, and visualizing
                ringkasan = summarize_and_visualize(content)
                return render_template("result.html", title=title, content=content, summary=ringkasan, graph_url="static/graph.png")
            else:
                return render_template("summary.html", error="Gagal mengambil data dari URL.")
        else:
            return render_template("summary.html", error="URL tidak boleh kosong.")
    return render_template("summary.html")

# Menjalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True)
