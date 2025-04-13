import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import sys
from flask import Flask, request, jsonify
from pyvi import ViTokenizer
import joblib
import re

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    data_dir_path = os.path.join(current_dir, "data")
    file_path = os.path.join(data_dir_path, filename)
    if os.path.exists(file_path):
        return file_path
    
    file_path = os.path.join(current_dir, filename)
    if os.path.exists(file_path):
        return file_path
    
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, filename)
    if os.path.exists(file_path):
        return file_path
    
    if os.path.exists(filename):
        return filename
    
    return filename

def load_json_data(json_file):
    try:
        json_path = get_data_path(json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if not all(col in df.columns for col in ['question', 'answer']):
            raise ValueError("JSON must contain 'question' and 'answer' columns")
        return df
    except FileNotFoundError:
        raise
    except json.JSONDecodeError:
        raise
    except Exception as e:
        raise

def load_stopwords_vi(stopwords_file="vietnamese-stopwords.txt"):
    try:
        stopwords_path = get_data_path(stopwords_file)
        if not os.path.exists(stopwords_path):
            raise FileNotFoundError(f"Tệp {stopwords_path} không tìm thấy.")
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        return stopwords
    except Exception as e:
        return []

json_file = "output.json"
tfidf_file = get_data_path("tfidf_matrix.pkl")
vectorizer_file = get_data_path("tfidf_vectorizer.pkl")

try:
    df = load_json_data(json_file)
except Exception as e:
    df = pd.DataFrame(columns=['question', 'answer'])

df['question'] = df['question'].astype(str).fillna('')
df['answer'] = df['answer'].astype(str).fillna('')

df = df.drop_duplicates(subset=['question'], keep='last').reset_index(drop=True)

def tokenize_vietnamese(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(ViTokenizer.tokenize(text).split())

df['question_tokenized'] = df['question'].apply(tokenize_vietnamese)
df['content'] = df['question_tokenized'] + ' ' + df['answer'].apply(tokenize_vietnamese)

try:
    vietnamese_stopwords = load_stopwords_vi()
except Exception as e:
    vietnamese_stopwords = []

try:
    if os.path.exists(tfidf_file) and os.path.exists(vectorizer_file):
        tfidf_matrix = joblib.load(tfidf_file)
        tfv = joblib.load(vectorizer_file)
    else:
        tfv = TfidfVectorizer(
            min_df=1,
            max_features=15000,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words=vietnamese_stopwords
        )
        tfidf_matrix = tfv.fit_transform(df['content'])
        tfidf_matrix = tfidf_matrix.astype(np.float32)
        
        try:
            joblib.dump(tfidf_matrix, tfidf_file)
            joblib.dump(tfv, vectorizer_file)
        except Exception as e:
            pass
except Exception as e:
    tfv = TfidfVectorizer()
    tfidf_matrix = tfv.fit_transform(["fallback content"])

try:
    indices = pd.Series(df.index, index=df['question'])
    indices = indices[~indices.index.duplicated(keep='last')]
except Exception as e:
    indices = pd.Series()

def recommend_similar_questions(query, top_n=5):
    try:
        query_tokenized = tokenize_vietnamese(query)
        query_tfidf = tfv.transform([query_tokenized])
        sim_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:top_n]
        question_indices = [i[0] for i in sim_scores]
        question_scores = [i[1] for i in sim_scores]
        return question_indices, question_scores
    except Exception as e:
        raise ValueError(f"Lỗi khi tạo gợi ý: {str(e)}")

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        query = request.args.get('text')
        if not query or not query.strip():
            return jsonify({
                'status': 'error',
                'message': 'Tham số truy vấn "text" là bắt buộc và không được rỗng'
            }), 400

        top_n = 5
        recommended_indices, similarity_scores = recommend_similar_questions(query, top_n)

        recommendations = []
        for idx, score in zip(recommended_indices, similarity_scores):
            if idx < len(df) and score > 0.2:
                recommendations.append({
                    'question': df.iloc[idx]['question'],
                    'answer': df.iloc[idx]['answer'],
                    'similarity_score': float(score)
                })

        if not recommendations:
            return jsonify({
                'status': 'success',
                'message': f'Không tìm thấy gợi ý phù hợp cho truy vấn "{query}"',
                'data': []
            })

        return jsonify({
            'status': 'success',
            'message': f'Đã gợi ý {len(recommendations)} mục cho truy vấn "{query}"',
            'data': recommendations
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 4000)), debug=False)