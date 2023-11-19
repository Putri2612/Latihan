# Import modul
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import streamlit as st
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import pickle  # Untuk memuat model Anda
from sklearn.feature_extraction.text import TfidfVectorizer  # Untuk mengubah teks menjadi representasi numerik
import os  # Untuk manipulasi file

# Download stopwords and punkt if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Fungsi untuk memuat data normalisasi
@st.cache_data()
def load_normalized_word():
    normalized_word = pd.read_excel("normalisasi.xlsx")
    normalized_word_dict = {}

    for index, row in normalized_word.iterrows():
        if row[0] not in normalized_word_dict:
            normalized_word_dict[row[0]] = row[1]
    return normalized_word_dict

# Fungsi untuk normalisasi term
def normalized_term(document, normalized_word_dict):
    if isinstance(document, str):
        return ' '.join([normalized_word_dict[term] if term in normalized_word_dict else term for term in document.split()])

# Fungsi untuk tokenisasi
def tokenization(text):
    text = nltk.tokenize.word_tokenize(text)
    return text

# Fungsi untuk penghapusan stopwords
def stopwords_removal(words):
    list_stopwords = set(nltk.corpus.stopwords.words('indonesian'))
    with open('stopword.txt', 'r') as file:
        for line in file:
            line = line.strip()
            list_stopwords.add(line)

    hapus = {"tidak", "naik", "kenaikan"}
    for i in hapus:
        if i in list_stopwords:
            list_stopwords.remove(i)

    return [word for word in words if word not in list_stopwords]

# Inisialisasi objek Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk stemming
def stemming(text):
    text = [stemmer.stem(word) for word in text]
    return text

@st.cache_data()
def vectorize_data(data):
    tfidf = TfidfVectorizer()
    x_tfidf = tfidf.fit_transform(data['processed_text'])
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(x_tfidf.toarray(), columns=feature_names)
    return tfidf_df

# Melakukan oversampling menggunakan SMOTE
@st.cache_data()
def apply_smote(tfidf_df, y):
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(tfidf_df, y)
    return X_smote, y_smote

# Fungsi untuk memuat data
@st.cache_data()
def load_data():
    file_path = 'hasil_stemming.xlsx'
    data = pd.read_excel(file_path)
    return data

# Fungsi untuk menghitung Information Gain
def compute_impurity(feature, impurity_criterion):
    probs = feature.value_counts(normalize=True)
    if impurity_criterion == 'entropy':
        impurity = -(np.sum(np.log2(probs) * probs))
    else:
        raise ValueError('Unknown impurity criterion')
    return impurity

# Fungsi untuk menghitung Information Gain
def compute_information_gain(df, target, descriptive_feature, split_criterion):
    target_entropy = compute_impurity(df[target], split_criterion)
    entropy_list = []
    weight_list = []

    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_impurity(df_feature_level[target], split_criterion)
        entropy_list.append(entropy_level)
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(weight_level)

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    information_gain = target_entropy - feature_remaining_impurity
    return information_gain

# Fungsi untuk menghitung Information Gain untuk semua fitur
@st.cache_data()
def calculate_information_gains(X_smote_dftiga, y_smote_encoded):
    split_criterion = 'entropy'
    information_gains = {}

    for feature in X_smote_dftiga.columns:
        information_gain = compute_information_gain(
            pd.concat([X_smote_dftiga, pd.Series(y_smote_encoded, name='label')], axis=1),
            'label', feature, split_criterion)
        information_gains[feature] = information_gain

    return information_gains

# Inisialisasi model Random Forest
random_seed = 42
rf_model = RandomForestClassifier(random_state=random_seed)

# Definisikan daftar hyperparameter yang ingin Anda uji
param_grid = {
    'n_estimators': [111, 165, 255],
    'max_depth': [65, 77, 88],
    'min_samples_leaf': [1, 5, 10],
}

# Inisialisasi GridSearchCV dengan model dan parameter grid
grid_search_tiga = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

# Navigasi sidebar
with st.sidebar:
    selected = option_menu('Analisis Sentimen', 
        ['Dashboard', 'Analysis Data', 'Classification', 'Report', 'Testing'],
        default_index=0)

# Halaman dashboard
if selected == 'Dashboard':
    st.title('Analisis Sentimen')
    st.write("by Putri Lailatul Maghfiroh")
    st.header('Dashboard')
    st.write("Selamat datang di aplikasi analisis sentimen sederhana. Gunakan sidebar untuk navigasi ke halaman lain.")

# Halaman Analisis Data
elif selected == 'Analysis Data':
    st.title('Analisis Sentimen')
    st.header('Analysis Data')

    data = load_data()
    st.dataframe(data.head(10))

    st.subheader('Statistik Data')
    st.write("Jumlah Data:", len(data))
    st.write("Jumlah Data Positif:", len(data[data['label'] == 'Positif']))
    st.write("Jumlah Data Negatif:", len(data[data['label'] == 'Negatif']))

    # Menampilkan WordCloud
    st.subheader('Word Cloud')
    processed_text = data['processed_text'].str.cat(sep=' ')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Halaman Klasifikasi
elif selected == 'Classification':
    st.title('Analisis Sentimen')
    st.header('Classification')

    st.subheader('Pemilihan Model')
    st.write("Pilih model yang ingin Anda gunakan untuk analisis sentimen.")
    model_option = st.selectbox("Pilih Model:", ['Random Forest'])

    # Jika memilih Random Forest
    if model_option == 'Random Forest':
        st.write("Anda memilih model Random Forest.")
        st.write("Memuat data...")
        data = load_data()
        st.write("Melakukan preprocessing data...")
        normalized_word_dict = load_normalized_word()
        data['processed_text'] = data['Text'].apply(lambda x: normalized_term(x, normalized_word_dict))
        data['processed_text'] = data['processed_text'].apply(tokenization)
        data['processed_text'] = data['processed_text'].apply(stopwords_removal)
        data['processed_text'] = data['processed_text'].apply(stemming)

        # Label encoding target variable
        label_encoder = LabelEncoder()
        data['label_encoded'] = label_encoder.fit_transform(data['label'])

        # TF-IDF Vectorization
        st.write("Mengubah teks menjadi representasi numerik...")
        tfidf_df = vectorize_data(data)

        # Melakukan oversampling menggunakan SMOTE
        st.write("Melakukan oversampling menggunakan SMOTE...")
        X_smote_dftiga, y_smote_encoded = apply_smote(tfidf_df, data['label_encoded'])

        # Menghitung Information Gain
        st.write("Menghitung Information Gain...")
        information_gains = calculate_information_gains(X_smote_dftiga, y_smote_encoded)

        # Menampilkan feature importance
        st.subheader('Feature Importance')
        feature_importance = pd.Series(information_gains)
        feature_importance = feature_importance / feature_importance.sum()
        feature_importance = feature_importance.sort_values(ascending=False)
        st.bar_chart(feature_importance)

        # Split data menjadi training dan testing set
        X_train, X_test, y_train, y_test = train_test_split(X_smote_dftiga, y_smote_encoded, test_size=0.2, random_state=random_seed)

        # Melatih model Random Forest dengan hyperparameter terbaik
        st.write("Melatih model Random Forest dengan hyperparameter terbaik...")
        grid_search_tiga.fit(X_train, y_train)
        best_rf_model = grid_search_tiga.best_estimator_

        # Menyimpan model ke dalam file
        model_filename = 'best_rf_model.pkl'
        with open(model_filename, 'wb') as model_file:
            joblib.dump(best_rf_model, model_file)

        st.success("Model telah dilatih dan disimpan!")

# Halaman Testing
elif selected == 'Testing':
    st.title('Analisis Sentimen')
    st.header('Testing')

    # Muat model dari file yang telah disimpan sebelumnya
    model_filename = 'best_rf_model.pkl'
    loaded_model = joblib.load(model_filename)

    # Muat data training saat aplikasi dimulai
    @st.cache_data()
    def load_training_data():
        data = pd.read_excel('hasil_stemming.xlsx')
        return data

    data_training = load_training_data()

    # Contoh data uji yang diunggah oleh pengguna
    input_text = st.text_area("Masukkan Kalimat yang Akan Diuji:", "")

    # Melakukan preprocessing pada data uji
    if input_text:
        input_text = normalized_term(input_text, normalized_word_dict)
        input_text = tokenization(input_text)
        input_text = stopwords_removal(input_text)
        input_text = stemming(input_text)

        # Ubah teks menjadi representasi numerik menggunakan TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(data_training['processed_text'])  # Gunakan data training untuk fit TF-IDF Vectorizer
        input_text_tfidf = tfidf_vectorizer.transform([' '.join(input_text)])  # Ubah teks input menjadi representasi TF-IDF

        # Lakukan prediksi
        prediction = loaded_model.predict(input_text_tfidf)

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if prediction[0] == 0:
            st.write("Sentimen: Negatif")
        else:
            st.write("Sentimen: Positif")

# Halaman Report
elif selected == 'Report':
    st.title('Analisis Sentimen')
    st.header('Report')

    # Muat model dari file yang telah disimpan sebelumnya
    model_filename = 'best_rf_model.pkl'
    loaded_model = joblib.load(model_filename)

    # Muat data training saat aplikasi dimulai
    @st.cache_data()
    def load_training_data():
        data = pd.read_excel('hasil_stemming.xlsx')
        return data

    data_training = load_training_data()

    # Split data menjadi features (X) dan target (y)
    X = data_training['processed_text']
    y = data_training['label']

    # Label encoding target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Ubah teks menjadi representasi numerik menggunakan TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)  # Gunakan data training untuk fit TF-IDF Vectorizer
    X_tfidf = tfidf_vectorizer.transform(X)  # Ubah teks menjadi representasi TF-IDF

    # Split data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=random_seed)

    # Lakukan prediksi pada data uji
    y_pred = loaded_model.predict(X_test)

    # Evaluasi model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Menampilkan metrik evaluasi
    st.subheader("Metrik Evaluasi Model:")
    st.write("Akurasi:", accuracy)
    st.write("Presisi:", precision)
    st.write("Recall:", recall)
    st.write("F1-Score:", f1)

    # Menampilkan confusion matrix
    st.subheader("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[label_encoder.classes_], columns=[label_encoder.classes_])
    st.write(cm_df)
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))

    # Heatmap untuk confusion matrix
    st.subheader("Heatmap Confusion Matrix:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

