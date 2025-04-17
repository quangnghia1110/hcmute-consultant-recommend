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
import requests
import numpy as np
import google.generativeai as genai
from config import (
    GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE, TOP_P, TOP_K, MAX_OUTPUT_TOKENS,
    CURRENT_DIR, DATA_DIR, JSON_FILE, STOPWORDS_FILE, TFIDF_MATRIX_FILE, VECTORIZER_FILE
)

app = Flask(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

def initialize_app(app):
    df, vectorizer, tfidf_matrix = prepare_data()
    app.config['df'] = df
    app.config['vectorizer'] = vectorizer
    app.config['tfidf_matrix'] = tfidf_matrix

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
        vectorizer = app.config['vectorizer']
        tfidf_matrix = app.config['tfidf_matrix']
        query_tokenized = tokenize_vietnamese(query)
        query_tfidf = vectorizer.transform([query_tokenized])
        sim_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        sim_scores_with_indices = [];
        for idx, score in enumerate(sim_scores):
            if score > 0.3:
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
            if idx < len(df) and score > 0.3:
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

@app.route('/recommend-answers', methods=['GET'])
def get_recommend_answers():
    try:
        query = request.args.get('text', '').strip()
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Tham số truy vấn "text" là bắt buộc và không được rỗng'
            }), 400
        chat_url = f"https://hcmute-consultant-chatbot-production.up.railway.app/chat?text={query}"
        response = requests.get(chat_url)
        if response.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': f'Không thể lấy dữ liệu từ API chat: {response.status_code}'
            }), 500
        chat_data = response.json()
        question = chat_data['data']['question']
        answer = chat_data['data']['answer']
        alternative_answers = generate_alternative_answers(question, answer)
        if len(alternative_answers) > 5:
            alternative_answers = alternative_answers[:4]
        result_answers = []
        for a in alternative_answers:
            result_answers.append({
                "answer": a
            })
        return jsonify({
            'status': 'success',
            'message': 'Đã tạo 4 câu trả lời thay thế',
            'question': question,
            'answer': answer,
            'alternative_answers': result_answers
        })
    except Exception as e:
        print(f"Lỗi trong endpoint /recommend-answers: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

def generate_alternative_answers(question, answer):
    try:
        prompt = f"""
            Dựa vào câu hỏi và câu trả lời gốc dưới đây, hãy tạo chính xác 4 câu trả lời thay thế KHÁC BIỆT HOÀN TOÀN về cách trình bày.
            MỖI câu trả lời PHẢI có:
            - Độ dài khác nhau (ngắn, trung bình, dài)
            - Cách tiếp cận khác nhau (trực tiếp, chi tiết, ví dụ thực tế, dưới dạng hướng dẫn)
            - Giọng điệu khác nhau (trang trọng, thân thiện, chuyên nghiệp, đơn giản)
            
            CÂU HỎI: {question}
            CÂU TRẢ LỜI GỐC: {answer}
            
            CHỈ TRẢ VỀ 4 CÂU TRẢ LỜI THAY THẾ, MỖI CÂU TRÊN 1 ĐOẠN VĂN, KHÔNG ĐÁNH SỐ, KHÔNG THÊM BẤT KỲ GIẢI THÍCH NÀO KHÁC.
        """
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        )
        if hasattr(response, 'text'):
            raw_answers = []
            current_answer = ""
            for line in response.text.strip().split('\n'):
                if line.strip():
                    if not current_answer:
                        current_answer = line.strip()
                    else:
                        current_answer += " " + line.strip()
                else:
                    if current_answer:
                        raw_answers.append(current_answer)
                        current_answer = ""
            if current_answer:
                raw_answers.append(current_answer)
            
            return raw_answers
        else:
            print("Gemini API không trả về kết quả dạng text")
            return []
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API: {str(e)}")
        return []

if initialize_app(app):
    print("✅ Dữ liệu đã khởi tạo")
else:
    print("⚠️ Dữ liệu chưa được khởi tạo hoàn chỉnh")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=False)