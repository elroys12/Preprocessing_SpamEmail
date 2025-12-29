import pandas as pd
import re
import string
import os
import joblib
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

# SETUP
nltk.download('stopwords', quiet=True)
STOP_WORDS = stopwords.words('english')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# DATA LOADING
def load_data(path):
    logging.info(f"Loading dataset from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di path: {path}")
    return pd.read_csv(path)

# TEXT CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

# PREPROCESSING PIPELINE
def preprocess_data(
    data_path,
    max_features=5000,
    test_size=0.2,
    random_state=42,
    split_data=True
):
    df = load_data(data_path)

    # Validasi kolom sesuai dataset Spam Email
    if 'message' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'message' and 'label'")

    logging.info("Handling missing values")
    df = df.dropna(subset=['message', 'label'])

    logging.info("Removing duplicate data")
    df = df.drop_duplicates(subset='message')

    logging.info("Cleaning text")
    df['clean_text'] = df['message'].apply(clean_text)

    logging.info("Encoding label")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    logging.info("Vectorizing text with TF-IDF")
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df['clean_text'])

    if split_data:
        logging.info("Splitting train and test data")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test, tfidf, label_encoder

    return X, y, tfidf, label_encoder

# SAVE PREPROCESSED DATA (REVISI: Mendukung CSV dan PKL)
def save_preprocessed_data(output_dir, X_train, X_test, y_train, y_test, tfidf, label_encoder):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simpan Objek Transformer sebagai .pkl (Untuk penggunaan di model nanti)
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    
    # 2. Simpan Matriks X sebagai .csv (Harus dikonversi ke Dense Array)
    logging.info("Converting sparse matrix to CSV...")
    pd.DataFrame(X_train.toarray()).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test.toarray()).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    
    # 3. Simpan Label y sebagai .csv
    pd.DataFrame(y_train, columns=['label']).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=['label']).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    logging.info(f"Semua file (CSV & PKL) berhasil disimpan di: {output_dir}")

# --- EKSEKUSI ---
if __name__ == "__main__":
    # Path disesuaikan dengan struktur Linux/GitHub Actions
    DATA_INPUT_PATH = "SpamEmail_raw/SpamEmail.csv" 
    OUTPUT_DIR = "preprocessing/SpamEmail_preprocessing"

    try:
        # Jalankan Preprocessing
        X_train, X_test, y_train, y_test, tfidf, le = preprocess_data(DATA_INPUT_PATH)
        
        # Simpan Hasil dengan fungsi revisi
        save_preprocessed_data(
            OUTPUT_DIR,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            tfidf=tfidf,
            label_encoder=le
        )
        logging.info("Pipeline Automasi Berhasil dijalankan sepenuhnya!")

    except Exception as e:
        logging.error(f"Gagal menjalankan automasi: {e}")