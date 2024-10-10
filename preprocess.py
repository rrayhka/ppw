import streamlit as st
import requests as req
from bs4 import BeautifulSoup as bs

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




# Cleaning text
def clean_text(text: str=None) -> str:
	"""
	Mmembersihkan text dari karakter-karakter yang tidak diperlukan
	"""
	text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text) # Menghapus https* and www*
	text = re.sub(r'@[^\s]+', ' ', text) # Menghapus username
	text = re.sub(r'[\s]+', ' ', text) # Menghapus tambahan spasi
	text = re.sub(r'#([^\s]+)', ' ', text) # Menghapus hashtags
	text = re.sub(r'rt', ' ', text) # Menghapus retweet
	text = text.translate(str.maketrans("","",string.punctuation)) # Menghapus tanda baca
	text = re.sub(r'\d', ' ', text) # Menghapus angka
	text = text.lower()
	text = text.encode('ascii','ignore').decode('utf-8') #Menghapus ASCII dan unicode
	text = re.sub(r'[^\x00-\x7f]',r'', text)
	text = text.replace('\n','') #Menghapus baris baru
	text = text.strip()
	return text

# Scraping berita
def scrape_news(soup: str) -> dict:
	"""
	Mengambil informasi berita dari url
	"""
	berita = {}
	texts = []
	# TODO:
	# ada struktur aneh https://www.cnnindonesia.com/olahraga/20240830134615-142-1139388/live-report-timnas-indonesia-vs-thailand-u-20
	
	berita["judul"] = soup.title.text

	if 'FOTO:' in berita["judul"]:
		div_content = soup.find("div", class_="detail-text text-cnn_black text-sm grow min-w-0")
		if div_content:
			full_text = div_content.get_text(strip=True)
			text = full_text.split('--', 1)[-1]
			text = text.split('var article')[0].strip()

			cleaned_text = clean_text(text)
			texts.append(cleaned_text)

		berita["tanggal"] = soup.find("div", class_="container !w-[1100px] overscroll-none").find_all("div")[1].find_all("div")[2].text

	else:
		text_list = soup.find("div", class_="detail-text text-cnn_black text-sm grow min-w-0")
		for text in text_list.find_all("p"):
			if 'para_caption' not in text.get('class', []):
				cleaned_text = clean_text(text.text)
				texts.append(cleaned_text)

		berita["tanggal"] = soup.find("div", class_="container !w-[1100px] overscroll-none").find_all("div")[1].find_all("div")[3].text

	berita["isi"] = "\n".join(texts)
	berita["kategori"] = soup.find("meta", attrs={'name': 'dtk:namakanal'})['content']
	return berita

# Mengambil html dari url
def get_html(url: str) -> str:
	"""
	Mengambil html dari url
	"""
	try:
		response = req.get(url).text
		return bs(response, "html5lib")
	
	except Exception as e:
		print(e)
		return ""

@st.cache_data
def get_news(news_url: str) -> pd.DataFrame:
	"""
	Mengambil informasi dari isi berita yang ada pada url
	"""
	news = []

	result = scrape_news(get_html(news_url))
	news.append(result)

	df = pd.DataFrame.from_dict(news)

	return df




def stemming_indo(text: str) -> str:
	"""
	Menstemming kata atau lemmisasi kata dalam bahasa Indonesia
	"""
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	text = ' '.join(stemmer.stem(word) for word in text)
	return text

def clean_stopword(tokens: list) -> list:
	"""
	Membersihkan kata yang merupakan stopword
	"""
	listStopword =  set(stopwords.words('indonesian'))
	removed = []
	for t in tokens:
		if t not in listStopword:
			removed.append(t)
	return removed

@st.cache_data
def preprocess_text(content):
	"""
	Memproses text berita, membersihkan text, memperbagus kata, dan menghilangkan stopword
	"""
	result = []
	for text in content:
		tokens = nltk.tokenize.word_tokenize(text)
		cleaned_stopword = clean_stopword(tokens)
		stemmed_text = stemming_indo(cleaned_stopword)
		result.append(stemmed_text)
	return result



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