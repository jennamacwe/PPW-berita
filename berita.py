# import streamlit as st
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle
# import re

# # stopwords
# import nltk
# from nltk.corpus import stopwords

# # stemming
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# # split data
# from sklearn.model_selection import train_test_split

# st.title("Pencarian dan Penambangan WEB")
# st.write("Nama  : Jennatul Macwe ")
# st.write("Nim   : 210411100151 ")
# st.write("Kelas : Pencarian dan Penambangan WEB B ")

# data_set_description, dataset, prepro, model, Input = st.tabs(["Deskripsi Data Set", "Dataset", "Pre-processing", "Model", "Klasifikasi"])

# with data_set_description:
#     st.write("### Deskripsi Dataset")
#     st.write("Dataset ini berisi berita yang dikelompokkan ke dalam dua kategori, yaitu 'Politik' dan 'Gaya Hidup'.")
#     st.write("***Jumlah Data :*** 100 baris data dengan 5 kolom")
#     st.write("***Dataset ini TIDAK MEMILIKI Missing Values*** yang berarti bahwa setiap bagian data pada dataset lengkap dengan informasi.")

#     st.write("### Penjelasan Fitur :")
#     st.write("Jumlah Fitur : 4")
#     st.write("* Judul Berita : Judul dari setiap berita")
#     st.write("* Isi Berita : Konten utama dari berita. Kolom ini memuat informasi lengkap tentang peristiwa, fakta, atau opini yang dibahas dalam berita. Fitur ini diproses lebih lanjut untuk analisis teks.")
#     st.write("* Tanggal Berita : Fitur ini menunjukkan kapan berita tersebut diterbitkan berupa tanggal dan waktu")
#     st.write("* Kategori Berita : Fitur ini merupakan label atau kategori dari setiap berita. Kategori ini menunjukkan apakah berita tersebut berkaitan dengan Politik atau Gaya Hidup. Fitur ini penting karena merupakan target yang akan diprediksi dalam model klasifikasi. Kolom kategori ini nantinya akan diubah menjadi nilai numerik (misalnya, 0 untuk Politik dan 1 untuk Gaya Hidup).")

#     st.write("### Sumber Dataset : https://timesindonesia.co.id/#google_vignette")
#     st.write("Dataset ini berisi kumpulan berita yang diambil dari situs TIMES Indonesia, sebuah portal berita yang memuat berbagai artikel terkait berbagai topik, termasuk politik dan gaya hidup. Data ini dikumpulkan melalui proses web crawling menggunakan Python, yang merupakan teknik pengumpulan data secara otomatis dari situs web.")
#     st.write("")

# with dataset:
#     st.write("### Dataset Berita Times Indonesia Kategori Politik dan Gaya Hidup")

#     # Membaca data dari URL
#     url = "https://raw.githubusercontent.com/jennamacwe/PPW-berita/refs/heads/main/Tugas-Crawling-Data-Berita-2-kategori.csv"
#     data = pd.read_csv(url, header=None)

#     # Menampilkan data
#     st.dataframe(data)

#     st.write("### Penjelasan Fitur :")
#     st.write("Jumlah Fitur : 4")
#     st.write("* Judul Berita : Judul dari setiap berita")
#     st.write("* Isi Berita : Konten utama dari berita. Kolom ini memuat informasi lengkap tentang peristiwa, fakta, atau opini yang dibahas dalam berita.")
#     st.write("* Tanggal Berita : Fitur ini menunjukkan kapan berita tersebut diterbitkan berupa tanggal dan waktu")
#     st.write("* Kategori Berita : Label yang menunjukkan apakah berita terkait Politik atau Gaya Hidup.")

# with prepro:
#     st.write("### Preprocessing Dataset")

#     data['Kategori'] = data[4].map({'Politik': 0, 'Gaya Hidup': 1})

#     st.write("***1. Cleansing***")

#     def remove_url(data_berita):
#         url = re.compile(r'https?://\S+|www\.S+')
#         return url.sub(r'', data_berita)

#     def remove_html(data_berita):
#         html = re.compile(r'<.#?>')
#         return html.sub(r'', data_berita)

#     def remove_emoji(data_berita):
#         emoji_pattern = re.compile("[" 
#                                    u"\U0001F600-\U0001F64F" 
#                                    u"\U0001F300-\U0001F5FF"  
#                                    u"\U0001F680-\U0001F6FF"  
#                                    u"\U0001F1E0-\U0001F1FF" 
#                                    "]+", flags=re.UNICODE)
#         return emoji_pattern.sub(r'', data_berita)

#     def remove_numbers(data_berita):
#         return re.sub(r'\d+', '', data_berita)

#     def remove_symbols(data_berita):
#         return re.sub(r'[^a-zA-Z0-9\s]', '', data_berita)

#     st.write("***2. Case folding***")

#     def case_folding(text):
#         return text.lower() if isinstance(text, str) else text

#     st.write("***3. Tokenization dan Stop Words***")

#     nltk.download('stopwords')
#     stop_words = stopwords.words('indonesian')

#     def remove_stopwords(text):
#         return [word for word in text.split() if word not in stop_words]

#     st.write("***4. Stemming***")

#     factory = StemmerFactory()
#     stemmer = factory.create_stemmer()

#     def stemming(text):
#         return ' '.join([stemmer.stem(word) for word in text])

#     st.write("### Split Data")

#     # x = data[1].apply(stemming).values  # Isi Berita yang sudah di-stemming
#     # y = data['Kategori'].values

#     # Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=2)

#     x = data['Isi Berita'].apply(stemming).values  # Isi Berita yang sudah di-stemming
#     y = data['Kategori'].values

#     Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=2)


#     ####################
#     #####################   TF-IDF
#     #####################

#     # # Inisialisasi TF-IDF Vectorizer untuk mengubah teks menjadi representasi numerik berdasarkan frekuensi kata.
#     # vect = TfidfVectorizer(min_df=1) 

#     # # Melakukan fit pada data Xtrain dan langsung mentransformasikan teks menjadi matriks TF-IDF.
#     # X = vect.fit_transform(Xtrain)  
    
#     # # Melakukan transformasi kembali pada Xtrain menjadi array (fit sudah dilakukan sebelumnya).
#     # X_array = vect.transform(Xtrain)  

#     # # Membuka file untuk menyimpan model vectorizer dalam format pkl.
#     # with open('tfidf_vectorizer.pkl', 'wb') as f:  
#     #     pickle.dump(vect, f) 
        
#     vect = TfidfVectorizer(min_df=1)
#     Xtrain_vect = vect.fit_transform(Xtrain)

#     with open('tfidf_vectorizer.pkl', 'wb') as f:
#         pickle.dump(vect, f)

#     st.write("Contoh hasil pre-processing Xtrain:", Xtrain[:5])


# with model:
#     st.write("### Logistic Regression")

#     model = LogisticRegression()
#     model.fit(Xtrain_vect, Ytrain)

#     with open('logistic_model.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     Xtest_vect = vect.transform(Xtest)
#     prediksi = model.predict(Xtest_vect)

#     accuracy = accuracy_score(Ytest, prediksi)
#     st.write(f'Akurasi Model Logistic Regression: {accuracy * 100:.2f}%')

#     results_lr = pd.DataFrame({'Real Values': Ytest, 'Prediksi': prediksi})
#     st.dataframe(results_lr)

# with Input:
#     def load_model_and_vectorizer():
#         with open('logistic_model.pkl', 'rb') as model_file:
#             model = pickle.load(model_file)

#         with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
#             vectorizer = pickle.load(vectorizer_file)
#         return model, vectorizer

#     def predict_category(berita_input, model, vectorizer):
#         def preprocessing(berita):
#             berita = remove_url(berita)
#             berita = remove_html(berita)
#             berita = remove_emoji(berita)
#             berita = remove_symbols(berita)
#             berita = remove_numbers(berita)
#             berita = berita.lower()
#             tokens = berita.split()
#             tokens = [word for word in tokens if word not in stop_words]
#             stemming_result = [stemmer.stem(word) for word in tokens]
#             return ' '.join(stemming_result)

#         preprocessed_berita = preprocessing(berita_input)
#         berita_vectorized = vectorizer.transform([preprocessed_berita])
#         prediction = model.predict(berita_vectorized)
#         return 'Politik' if prediction == 0 else 'Gaya Hidup'

#     st.title("Prediksi Kategori Berita")
#     user_input = st.text_area("Isi Berita", height=200)

#     if st.button("Prediksi"):
#         if user_input:
#             model, vectorizer = load_model_and_vectorizer()
#             predicted_category = predict_category(user_input, model, vectorizer)
#             st.write(f"**Kategori berita:** {predicted_category}")
#         else:
#             st.write("Harap masukkan isi berita.")


# ############################################################3

# # import streamlit as st
# # import pandas as pd
# # from sklearn.linear_model import LogisticRegression
# # import joblib
# # from sklearn.metrics import accuracy_score
# # from sklearn.decomposition import PCA
# # from sklearn.model_selection import train_test_split
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # import pickle
# # import re

# # #stopwords
# # import nltk
# # from nltk.corpus import stopwords

# # #stemming
# # from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# # #spit data
# # from sklearn.model_selection import train_test_split

# # st.title("Pencarian dan Penambagan WEB")
# # st.write("Nama  : Jennatul Macwe ")
# # st.write("Nim   : 210411100151 ")
# # st.write("Kelas : Pencarian dan Penambangan WEB B ")

# # data_set_description, dataset, prepro, model, Input = st.tabs(["Deskripsi Data Set", "Dataset", "Pre-processing", "Model", "Klasifikasi"])

# # with data_set_description:
# #     st.write("### Deskripsi Dataset")
# #     st.write("Dataset ini berisi berita yang dikelompokkan ke dalam dua kategori, yaitu 'Politik' dan 'Gaya Hidup'.")
# #     st.write("***Jumlah Data :*** 100 baris data dengan 5 kolom")
# #     st.write("***Dataset ini TIDAK MEMILIKI Missing Values*** yang berarti bahwa setiap bagian data pada dataset lengkap dengan informasi.")

# #     st.write("### Penjelasan Fitur :")
# #     st.write("Jumlah Fitur : 4")
# #     st.write("* Judul Berita : Judul dari setiap berita")
# #     st.write("* Isi Berita : Konten utama dari berita. Kolom ini memuat informasi lengkap tentang peristiwa, fakta, atau opini yang dibahas dalam berita. Fitur ini diproses lebih lanjut untuk analisis teks.")
# #     st.write("* Tanggal Berita : Fitur ini menunjukkan kapan berita tersebut diterbitkan berupa tanggal dan waktu")
# #     st.write("* Kategori Berita : Fitur ini merupakan label atau kategori dari setiap berita. Kategori ini menunjukkan apakah berita tersebut berkaitan dengan Politik atau Gaya Hidup. Fitur ini penting karena merupakan target yang akan diprediksi dalam model klasifikasi. Kolom kategori ini nantinya akan diubah menjadi nilai numerik (misalnya, 0 untuk Politik dan 1 untuk Gaya Hidup).")

# #     st.write("### Sumber Dataset : https://timesindonesia.co.id/#google_vignette")
# #     st.write("Dataset ini berisi kumpulan berita yang diambil dari situs TIMES Indonesia, sebuah portal berita yang memuat berbagai artikel terkait berbagai topik, termasuk politik dan gaya hidup. Data ini dikumpulkan melalui proses web crawling menggunakan Python, yang merupakan teknik pengumpulan data secara otomatis dari situs web.")
# #     st.write("")


# # with dataset:
# #     st.write("### Dataset Berita Times Indonesia Kategori Politik dan Gaya Hidup")

# #     # Membaca data dari URL
# #     url = "https://raw.githubusercontent.com/jennamacwe/klasifikasi-berita/refs/heads/main/Tugas-Crawling-Data-Berita-2-kategori%20(1).csv"
# #     data = pd.read_csv(url, header=None)

# #     # Menampilkan data
# #     st.dataframe(data)

# #     st.write("### Penjelasan Fitur :")
# #     st.write("Jumlah Fitur : 4")
# #     st.write("* Judul Berita : Judul dari setiap berita")
# #     st.write("* Isi Berita : Konten utama dari berita. Kolom ini memuat informasi lengkap tentang peristiwa, fakta, atau opini yang dibahas dalam berita. Fitur ini diproses lebih lanjut untuk analisis teks.")
# #     st.write("* Tanggal Berita : Fitur ini menunjukkan kapan berita tersebut diterbitkan berupa tanggal dan waktu")
# #     st.write("* Kategori Berita : Fitur ini merupakan label atau kategori dari setiap berita. Kategori ini menunjukkan apakah berita tersebut berkaitan dengan Politik atau Gaya Hidup. Fitur ini penting karena merupakan target yang akan diprediksi dalam model klasifikasi. Kolom kategori ini nantinya akan diubah menjadi nilai numerik (misalnya, 0 untuk Politik dan 1 untuk Gaya Hidup).")


# # with prepro:
# #     st.write("### Dataset Awal Berita")

# #     # Membaca data dari URL
# #     url = "https://raw.githubusercontent.com/jennamacwe/klasifikasi-berita/refs/heads/main/Tugas-Crawling-Data-Berita-2-kategori%20(1).csv"
# #     data = pd.read_csv(url, header=None)

# #     # Menampilkan data
# #     st.dataframe(data)

# #     st.write("***1. Encoding kolom kategori***")

# #     data['Kategori'] = data['Kategori Berita'].map({'Politik': 0, 'Gaya Hidup': 1})

# #     st.write("***2. Cleansing***")

# #     def remove_url(data_berita):
# #         url = re.compile(r'https?://\S+|www\.S+')
# #         return url.sub(r'', data_berita)

# #     def remove_html(data_berita):
# #         html = re.compile(r'<.#?>')
# #         return html.sub(r'', data_berita)

# #     def remove_emoji(data_berita):
# #         emoji_pattern = re.compile("["
# #             u"\U0001F600-\U0001F64F"
# #             u"\U0001F300-\U0001F5FF"
# #             u"\U0001F680-\U0001F6FF"
# #             u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
# #         return emoji_pattern.sub(r'', data_berita)

# #     def remove_numbers(data_berita):
# #         data_berita = re.sub(r'\d+', '', data_berita)
# #         return data_berita

# #     def remove_symbols(data_berita):
# #         data_berita = re.sub(r'[^a-zA-Z0-9\s]', '', data_berita)
# #         return data_berita

# #     st.write("***3. Case folding***")

# #     def case_folding(text):
# #         if isinstance(text, str):
# #             lowercase_text = text.lower()
# #             return lowercase_text
# #         else :
# #             return text

# #     st.write("***4. Tokenization***")

# #     def tokenize(text):
# #         tokens = text.split()
# #         return tokens

# #     st.write("***5. Stop Words***")

# #     nltk.download('stopwords')
# #     stop_words = stopwords.words('indonesian')

# #     def remove_stopwords(text):
# #         return [word for word in text if word not in stop_words]

# #     st.write("***6. Stemming***")

# #     # Inisialisasi stemmer
# #     factory = StemmerFactory()
# #     stemmer = factory.create_stemmer()

# #     # Fungsi stemming
# #     def stemming(text):
# #         return [stemmer.stem(word) for word in text]

# #     st.write("### Split Data")

# #     x = data['stemming'].values
# #     y = data['Kategori'].values

# #     Xtrain, Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.2,random_state=2)

# #     ######################
# #     ######################   TF-IDF
# #     ######################

# #     # Inisialisasi TF-IDF Vectorizer untuk mengubah teks menjadi representasi numerik berdasarkan frekuensi kata.
# #     vect = TfidfVectorizer() 

# #     # Melakukan fit pada data Xtrain dan langsung mentransformasikan teks menjadi matriks TF-IDF.
# #     X = vect.fit_transform(Xtrain)  

# #     # Melakukan transformasi kembali pada Xtrain menjadi array (fit sudah dilakukan sebelumnya).
# #     X_array = vect.transform(Xtrain)  

# #     # Membuka file untuk menyimpan model vectorizer dalam format pkl.
# #     with open('tfidf_vectorizer.pkl', 'wb') as f:  
# #         pickle.dump(vect, f)  


# # with model:

# #     st.write("### Logistic Regression")

# #     # Membuat model Logistic Regression
# #     model = LogisticRegression()

# #     # Melatih model dengan data pelatihan
# #     model.fit(X, Ytrain)

# #     # Meyimpan model
# #     with open('logistic_model', 'wb') as f:
# #         pickle.dump(model, f)

# #     # Transformasi data baru dengan TF-IDF yang sudah dilatih
# #     X_new = tfidf_vectorizer.transform(Xtest)

# #     # Lakukan prediksi dengan model logistic regression
# #     prediksi = model.predict(X_new)

# #     accuracy = accuracy_score(Ytest, prediksi)
# #     st.write(f'Akurasi Model Logistic Regression: {accuracy * 100:.2f}')

# #     # Menampilkan hasil perbandingan actual dan prediksi Logistic Regression
# #     results_lr = pd.DataFrame({'Real Values': Ytest, 'Prediksi': prediksi})
# #     st.write("Perbandingan Nilai yang Sesungguhnya dan Prediksi:")
# #     st.dataframe(results_lr)

# # with Input:

# #     # Fungsi untuk memuat model dan vectorizer
# #     def load_model_and_vectorizer():
# #         with open('logistic_model.pkl', 'rb') as model_file:
# #             model = pickle.load(model_file)

# #         with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
# #             vectorizer = pickle.load(vectorizer_file)
# #         return model, vectorizer

# #     # Fungsi untuk memprediksi kategori
# #     def predict_category(berita_input, model, vectorizer):
# #         # Preprocessing sesuai yang sudah kamu lakukan sebelumnya
# #         def preprocessing(berita):
# #             # Fungsi pembersihan (sesuaikan dengan preprocessing yang sudah ada)
# #             berita = remove_url(berita)
# #             berita = remove_html(berita)
# #             berita = remove_emoji(berita)
# #             berita = remove_symbols(berita)
# #             berita = remove_numbers(berita)
# #             berita = berita.lower()  # Case folding
# #             tokens = berita.split()  # Tokenisasi
# #             stop_words = stopwords.words('indonesian')
# #             tokens = [word for word in tokens if word not in stop_words]  # Stopword removal
# #             stemming_result = [stemmer.stem(word) for word in tokens]  # Stemming
# #             return ' '.join(stemming_result)

# #         # Preprocess input dari pengguna
# #         preprocessed_berita = preprocessing(berita_input)
        
# #         # Ubah input menjadi vektor TF-IDF
# #         berita_vectorized = vectorizer.transform([preprocessed_berita])
        
# #         # Prediksi kategori menggunakan model Logistic Regression
# #         prediction = model.predict(berita_vectorized)
        
# #         # Konversi hasil prediksi menjadi kategori (0 = Politik, 1 = Gaya Hidup)
# #         return 'Politik' if prediction == 0 else 'Gaya Hidup'

# #     # Streamlit code
# #     st.title("Prediksi Kategori Berita")
# #     st.write("Masukkan isi berita untuk memprediksi kategorinya (Politik atau Gaya Hidup):")

# #     # Input pengguna
# #     user_input = st.text_area("Isi Berita", height=200)

# #     if st.button("Prediksi"):
# #         if user_input:
# #             # Load model dan vectorizer
# #             model, vectorizer = load_model_and_vectorizer()
            
# #             # Prediksi kategori
# #             predicted_category = predict_category(user_input, model, vectorizer)
            
# #             # Tampilkan hasil prediksi
# #             st.write(f"**Kategori berita:** {predicted_category}")
# #         else:
# #             st.write("Harap masukkan isi berita.")

    
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

#stopwords
import nltk
from nltk.corpus import stopwords

#stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#spit data
from sklearn.model_selection import train_test_split

# Membaca data dari URL
url = "C:\\Users\\lenovo\\Documents\\PPW\\Tugas-Crawling-Data-Berita-2-kategori.csv"
data = pd.read_csv(url, header=None)

# Menampilkan data
st.dataframe(data)

data['Kategori'] = data[4].map({'Politik': 0, 'Gaya Hidup': 1})
st.dataframe(data)

def remove_url(data_berita):
  url = re.compile(r'https?://\S+|www\.S+')
  return url.sub(r'', data_berita)

def remove_html(data_berita):
  html = re.compile(r'<.#?>')
  return html.sub(r'', data_berita)

def remove_emoji(data_berita):
  emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F"
      u"\U0001F300-\U0001F5FF"
      u"\U0001F680-\U0001F6FF"
      u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', data_berita)

def remove_numbers(data_berita):
  data_berita = re.sub(r'\d+', '', data_berita)
  return data_berita

def remove_symbols(data_berita):
  data_berita = re.sub(r'[^a-zA-Z0-9\s]', '', data_berita)
  return data_berita

data['cleansing'] = data[2].apply(lambda x: remove_url(x))
data['cleansing'] = data['cleansing'].apply(lambda x: remove_html(x))
data['cleansing'] = data['cleansing'].apply(lambda x: remove_emoji(x))
data['cleansing'] = data['cleansing'].apply(lambda x: remove_symbols(x))
data['cleansing'] = data['cleansing'].apply(lambda x: remove_numbers(x))

st.write("CLEANSING")

st.dataframe(data)

def case_folding(text):
    if isinstance(text, str):
      lowercase_text = text.lower()
      return lowercase_text
    else :
      return text

data ['case_folding'] = data['cleansing'].apply(case_folding)

st.write("CASE FOLDING")

st.dataframe(data)

def tokenize(text):
    tokens = text.split()
    return tokens

data['tokenize'] = data['case_folding'].apply(tokenize)

st.write("TOKENISASI")

st.dataframe(data)

# Download stopwords (pastikan hanya perlu sekali saja)
nltk.download('stopwords')

# Mengambil daftar stopwords dalam bahasa Indonesia
stop_words = stopwords.words('indonesian')

# Fungsi untuk menghapus stopwords
def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

# Asumsi 'data' adalah dataframe, dengan kolom 'tokenize' yang berisi teks yang sudah di-tokenisasi
data['stopword_removal'] = data['tokenize'].apply(lambda x: ' '.join(remove_stopwords(x)))

# Menampilkan dataframe setelah penghapusan stopwords
st.dataframe(data)

st.write("STOPWORD")

st.dataframe(data)

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi stemming
def stemming(text):
    return [stemmer.stem(word) for word in text]

# Terapkan stemming setelah stopwords removal
data['stemming'] = data['stopword_removal'].apply(lambda x: ' '.join(stemming(x.split())))

st.write("STEMING")

st.dataframe(data)

x = data['stemming'].values
y = data['Kategori'].values

Xtrain, Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.2,random_state=2)

st.write("Xtrain")
st.dataframe(Xtrain)

st.write("Xtest")
st.dataframe(Xtest)

st.write("Ytrain")
st.dataframe(Ytrain)

st.write("Ytest")
st.dataframe(Ytest)
