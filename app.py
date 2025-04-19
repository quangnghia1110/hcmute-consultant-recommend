import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from pathlib import Path
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
from pyvi import ViTokenizer
import joblib
import re
import requests
import numpy as np
import google.generativeai as genai
import mysql.connector
from mysql.connector import Error
from config import (
    GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE, TOP_P, TOP_K, MAX_OUTPUT_TOKENS,
    CURRENT_DIR, DATA_DIR, JSON_FILE, STOPWORDS_FILE, TFIDF_MATRIX_FILE, VECTORIZER_FILE,
    MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_PORT, CHAT_URL
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://hcmute-consultant.vercel.app", "*"]}})

genai.configure(api_key=GOOGLE_API_KEY)

def create_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        if connection.is_connected():
            print("✅ Kết nối MySQL thành công")
            return connection
    except Error as e:
        print(f"Lỗi khi kết nối đến MySQL: {e}")
    return None

def fetch_data_from_mysql():
    connection = create_mysql_connection()
    if not connection:
        print("❌ Không thể kết nối đến MySQL")
        return pd.DataFrame()

    try:
        print("📊 Kiểm tra tổng số bản ghi")
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM question")
        total_questions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM answer")
        total_answers = cursor.fetchone()[0]
        print(f"✅ Tổng số: {total_questions} câu hỏi, {total_answers} câu trả lời trong database")
        
        query_questions = """
        SELECT q.id, q.content, q.created_at, q.title, q.status_approval, q.role_ask_id, q.user_id
        FROM question q
        WHERE q.status_delete = 0
        """
        
        query_answers = """
        SELECT a.id, a.content, a.created_at, a.question_id, a.status_answer, a.status_approval,
               a.title, a.role_consultant_id, a.user_id
        FROM answer a
        """
        
        print("📊 Truy vấn bảng question")
        questions_df = pd.read_sql(query_questions, connection)
        print(f"✅ Lấy được {len(questions_df)} câu hỏi")
        
        print("📊 Truy vấn bảng answer")
        answers_df = pd.read_sql(query_answers, connection)
        print(f"✅ Lấy được {len(answers_df)} câu trả lời")
        
        if questions_df.empty:
            print("❌ Không có dữ liệu trong bảng question thỏa mãn điều kiện")
            return pd.DataFrame()
            
        if answers_df.empty:
            print("❌ Không có dữ liệu trong bảng answer thỏa mãn điều kiện")
            return pd.DataFrame()
            
        print(f"📊 Thực hiện merge hai bảng, question.id -> answer.question_id")
        print(f"📊 ID trong bảng question: {questions_df['id'].tolist()[:5]}")
        print(f"📊 question_id trong bảng answer: {answers_df['question_id'].tolist()[:5]}")
        
        merged_df = pd.merge(
            questions_df,
            answers_df,
            left_on='id',
            right_on='question_id',
            how='inner',
            suffixes=('_question', '_answer')
        )
        
        print(f"✅ Sau khi merge còn {len(merged_df)} dòng")
        
        if merged_df.empty:
            print("❌ Không có dữ liệu sau khi merge")
            print("📊 Thử thực hiện left join để xem vấn đề")
            merged_df = pd.merge(
                questions_df,
                answers_df,
                left_on='id',
                right_on='question_id',
                how='left',
                suffixes=('_question', '_answer')
            )
            print(f"✅ Sau khi left join có {len(merged_df)} dòng")
            
            has_answer = merged_df['question_id'].notna().sum()
            print(f"📊 Số câu hỏi có câu trả lời: {has_answer}/{len(merged_df)}")
            
            if has_answer > 0:
                merged_df = merged_df[merged_df['question_id'].notna()]
                print(f"✅ Giữ lại {len(merged_df)} câu hỏi có câu trả lời")
        
        if merged_df.empty:
            print("❌ Vẫn không có dữ liệu sau khi thử các phương pháp join khác nhau")
            return pd.DataFrame()
            
        mysql_df = pd.DataFrame({
            'question': merged_df['content_question'],
            'answer': merged_df['content_answer'],
            'question_id': merged_df['id_question'],
            'answer_id': merged_df['id_answer'],
            'source': 'mysql'
        })
        
        print(f"✅ Trả về DataFrame với {len(mysql_df)} dòng từ MySQL")
        return mysql_df
    
    except Error as e:
        print(f"❌ Lỗi khi truy vấn MySQL: {e}")
        return pd.DataFrame()
    
    finally:
        if connection and connection.is_connected():
            connection.close()

def initialize_app(app):
    df, vectorizer, tfidf_matrix = prepare_data()
    app.config['df'] = df
    app.config['vectorizer'] = vectorizer
    app.config['tfidf_matrix'] = tfidf_matrix
    return True

def prepare_data():
    print("📊 Đọc dữ liệu từ file JSON")
    json_df = load_json_data(JSON_FILE)
    if not json_df.empty:
        json_df['source'] = 'json'
        print(f"✅ Đọc được {len(json_df)} dòng từ JSON")
    else:
        print("⚠️ Không đọc được dữ liệu từ JSON hoặc file JSON trống")
        
    print("📊 Đọc dữ liệu từ MySQL")
    mysql_df = fetch_data_from_mysql()
    if not mysql_df.empty:
        print(f"✅ Đọc được {len(mysql_df)} dòng từ MySQL")
    else:
        print("⚠️ Không đọc được dữ liệu từ MySQL")
    
    # Chỉ phối hợp dữ liệu nếu cả hai nguồn đều có dữ liệu
    if not mysql_df.empty and not json_df.empty:
        print("📊 Kết hợp dữ liệu từ JSON và MySQL")
        df = pd.concat([json_df, mysql_df], ignore_index=True)
        print(f"✅ Tổng số dòng sau khi kết hợp: {len(df)}")
    elif not mysql_df.empty:
        print("📊 Chỉ sử dụng dữ liệu từ MySQL")
        df = mysql_df
    else:
        print("📊 Chỉ sử dụng dữ liệu từ JSON")
        df = json_df
    
    # Nếu không có dữ liệu từ cả hai nguồn, tạo DataFrame rỗng để tránh lỗi
    if df.empty:
        print("⚠️ Không có dữ liệu từ cả MySQL và JSON, tạo DataFrame rỗng")
        df = pd.DataFrame(columns=['question', 'answer'])
    
    print("📊 Chuẩn hóa và tiền xử lý dữ liệu")
    df['question'] = df['question'].astype(str).fillna('')
    df['answer'] = df['answer'].astype(str).fillna('')
    df = df.drop_duplicates(subset=['question'], keep='last').reset_index(drop=True)
    print(f"✅ Sau khi loại bỏ trùng lặp còn {len(df)} dòng")
    
    print("📊 Tokenize dữ liệu tiếng Việt")
    df['question_tokenized'] = df['question'].apply(tokenize_vietnamese)
    df['answer_tokenized'] = df['answer'].apply(tokenize_vietnamese)
    df['content'] = df['question_tokenized'] + ' ' + df['answer_tokenized']
    
    print("📊 Tải stopwords")
    vietnamese_stopwords = load_stopwords()
    print(f"✅ Đã tải {len(vietnamese_stopwords)} stopwords")

    print("📊 Tạo mô hình TF-IDF")
    vectorizer, tfidf_matrix = create_tfidf_model(df, vietnamese_stopwords)
    print(f"✅ Đã tạo ma trận TF-IDF kích thước {tfidf_matrix.shape}")
    
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
        print(f"Lỗi khi đọc file JSON: {str(e)}")
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
        print(f"Lỗi khi đọc stopwords: {str(e)}")
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
        sim_scores_with_indices = []
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
            if idx < len(df) and score > 0.3:
                result = {
                    'question': df.iloc[idx]['question'],
                    'answer': df.iloc[idx]['answer'],
                    'similarity_score': float(score)
                }
                
                if 'source' in df.columns:
                    result['source'] = df.iloc[idx]['source']
                    
                if 'question_id' in df.columns and not pd.isna(df.iloc[idx].get('question_id')):
                    result['question_id'] = int(df.iloc[idx]['question_id'])
                    
                if 'answer_id' in df.columns and not pd.isna(df.iloc[idx].get('answer_id')):
                    result['answer_id'] = int(df.iloc[idx]['answer_id'])
                
                recommendations.append(result)
                
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
        chat_url = f"{CHAT_URL}/chat?text={query}"
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
            alternative_answers = alternative_answers[:5]
        result_answers = []
        for a in alternative_answers:
            result_answers.append({
                "answer": a
            })
        return jsonify({
            'status': 'success',
            'message': 'Đã tạo 5 câu trả lời thay thế',
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
            Dựa vào câu hỏi và câu trả lời gốc dưới đây, hãy tạo chính xác 5 câu trả lời thay thế KHÁC BIỆT HOÀN TOÀN về cách trình bày.
            MỖI câu trả lời PHẢI có:
            - Độ dài khác nhau (ngắn, trung bình, dài)
            - Cách tiếp cận khác nhau (trực tiếp, chi tiết, ví dụ thực tế, dưới dạng hướng dẫn)
            - Giọng điệu khác nhau (trang trọng, thân thiện, chuyên nghiệp, đơn giản)
            
            CÂU HỎI: {question}
            CÂU TRẢ LỜI GỐC: {answer}
            
            CHỈ TRẢ VỀ 5 CÂU TRẢ LỜI THAY THẾ, MỖI CÂU TRÊN 1 ĐOẠN VĂN, KHÔNG ĐÁNH SỐ, KHÔNG THÊM BẤT KỲ GIẢI THÍCH NÀO KHÁC.
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

@app.route('/refresh-data', methods=['POST'])
def refresh_data():
    try:
        if initialize_app(app):
            return jsonify({
                'status': 'success',
                'message': 'Dữ liệu và model đã được làm mới thành công'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Có lỗi xảy ra khi làm mới dữ liệu'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi máy chủ nội bộ: {str(e)}'
        }), 500

if __name__ == "__main__":
    initialize_app(app)
    print("✅ Dữ liệu đã khởi tạo")
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=False) 