import mlflow
import logging
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import dagshub

nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/souvikpaul425/yt_mlops_capstone_project.mlflow/",
    "dagshub_repo_owner": "souvikpaul425",
    "dagshub_repo_name": "yt_mlops_capstone_project",
    "experiment_name": "word2vec_bert_comparison"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== TRAIN & EVALUATE MODELS ==========================

def train_and_evaluate(df):
    
    with mlflow.start_run(): 
        
        start_time = time.time()
        
        try:
            # Tokenize for Word2Vec
            X =df['review']
            y = df['sentiment']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"],stratify=y, random_state=42)
            train_tokens = [word_tokenize(doc.lower()) for doc in X_train]
            test_tokens = [word_tokenize(doc.lower()) for doc in X_test]

            # Word2Vec embeddings 
            #Word2Vec requires thousands or millions of sentences to learn good word vectors. With only 500 examples, the vectors will be poor and likely hurt your model’s performance.
            logging.info("Training Word2Vec model...")
            w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)
            def document_vector(doc_tokens):
                vectors = [w2v_model.wv[word] for word in doc_tokens if word in w2v_model.wv]
                return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)
            X_train_w2v = np.array([document_vector(doc) for doc in train_tokens])
            X_test_w2v = np.array([document_vector(doc) for doc in test_tokens])
            mlflow.log_param("embedding_word2vec", "Word2Vec")
            mlflow.log_param("embedding_dim", w2v_model.vector_size)

            # BERT embeddings
            logging.info("Generating BERT embeddings...")
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            X_train_bert = bert_model.encode(X_train.reset_index(drop=True), show_progress_bar=True)
            X_test_bert = bert_model.encode(X_test.reset_index(drop=True), show_progress_bar=True)
            # Choose which embeddings to use:
            #X_train, X_test = X_train_w2v, X_test_w2v
            X_train, X_test = X_train_bert, X_test_bert
            mlflow.log_param("embedding_method_bert", "BERT")

            logging.info("Training Logistic Regression model...")
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            logging.info("Predicting test data...")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")
            mlflow.log_param("model", "Logistic Regression")

            logging.info(f"Run completed in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)

# ========================== MAIN EXECUTION ==========================

if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)
