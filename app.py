import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from pathlib import Path
from flask import Flask, request, jsonify, current_app
from pyvi import ViTokenizer
import joblib
import re

app = Flask(__name__)

CURRENT_DIR = Path(__file__).parent.absolute()
DATA_DIR = CURRENT_DIR / "data"
JSON_FILE = "output.json"
TFIDF_MATRIX_FILE = "tfidf_matrix.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
STOPWORDS_FILE = "vietnamese-stopwords.txt"

@app.before_first_request
def initialize_app():
    df, vectorizer, tfidf_matrix = prepare_data()
    current_app.config['df'] = df
    current_app.config['vectorizer'] = vectorizer
    current_app.config['tfidf_matrix'] = tfidf_matrix

def prepare_data():
    df = load_json_data(JSON_FILE)
    df['question'] = df['question'].astype(str).fillna('')
    df['answer'] = df['answer'].astype(str).fillna('')
    df = df.drop_duplicates(subset=['question'], keep='last').reset_index(drop=True)
    df['question_tokenized'] = df['question'].apply(tokenize_vietnamese)
    df['answer_tokenized'] = df['answer'].apply(tokenize_vietnamese)
    df['content'] = df['question_tokenized'] + ' ' + df['answer_tokenized']
    
    vietnamese_stopwords = load_stopwords()
    
    tfidf_path = get_data_path(TFIDF_MATRIX_FILE)
    vectorizer_path = get_data_path(VECTORIZER_FILE)
    
    if tfidf_path.exists() and vectorizer_path.exists():
        try:
            tfidf_matrix = joblib.load(tfidf_path)
            vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            vectorizer, tfidf_matrix = create_tfidf_model(df, vietnamese_stopwords)
    else:
        vectorizer, tfidf_matrix = create_tfidf_model(df, vietnamese_stopwords)
    
    return df, vectorizer, tfidf_matrix

def load_json_data(json_file):
    try:
        json_path = get_data_path(json_file)
        if not json_path.exists():
            return pd.DataFrame(columns=['question', 'answer'])
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        if not all(col in df.columns for col in ['question', 'answer']):
            return pd.DataFrame(columns=['question', 'answer'])
            
        return df
    except Exception as e:
        return pd.DataFrame(columns=['question', 'answer'])
    
def get_data_path(filename):
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path
    current_path = CURRENT_DIR / filename
    if current_path.exists():
        return current_path
    return Path(filename)

def load_stopwords():
    try:
        stopwords_path = get_data_path(STOPWORDS_FILE)
        if not stopwords_path.exists():
            return []
            
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        return stopwords
    except Exception as e:
        return []

def tokenize_vietnamese(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(ViTokenizer.tokenize(text).split())

def create_tfidf_model(df, stopwords):
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=10000,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        stop_words=stopwords
    )
    
    if len(df) > 0:
        tfidf_matrix = vectorizer.fit_transform(df['content'])
    else:
        tfidf_matrix = vectorizer.fit_transform(["fallback content"])

    try:
        tfidf_path = get_data_path(TFIDF_MATRIX_FILE)
        vectorizer_path = get_data_path(VECTORIZER_FILE)
        joblib.dump(tfidf_matrix, DATA_DIR / tfidf_path)
        joblib.dump(vectorizer, DATA_DIR / vectorizer_path)
    except Exception as e:
        print(f"Lỗi khi lưu mô hình TF-IDF: {str(e)}")
    
    return vectorizer, tfidf_matrix

def recommend_similar_questions(query, top_n=5):
    try:
        vectorizer = current_app.config['vectorizer']
        tfidf_matrix = current_app.config['tfidf_matrix']
        
        query_tokenized = tokenize_vietnamese(query)
        query_tfidf = vectorizer.transform([query_tokenized])
        
        sim_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        
        sim_scores_with_indices = [];
        for idx, score in enumerate(sim_scores):
            if score > 0.1:
                sim_scores_with_indices.append((idx, score))
        
        sim_scores_with_indices = sorted(sim_scores_with_indices, 
                                         key=lambda x: x[1], 
                                         reverse=True)
        
        top_results = sim_scores_with_indices[:top_n]
        question_indices = [i[0] for i in top_results]
        question_scores = [i[1] for i in top_results]
        
        return question_indices, question_scores
    except Exception as e:
        print(f"Lỗi khi tạo gợi ý: {str(e)}")
        return [], []

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        query = request.args.get('text', '').strip()
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Tham số truy vấn "text" là bắt buộc và không được rỗng'
            }), 400
        
        recommended_indices, similarity_scores = recommend_similar_questions(query, 5)
        if not recommended_indices or not similarity_scores:
            return jsonify({
                'status': 'success',
                'message': f'Không tìm thấy gợi ý phù hợp cho truy vấn "{query}"',
                'data': []
            })

        df = current_app.config['df']
        
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
        print(f"Lỗi trong endpoint /recommend: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=False)