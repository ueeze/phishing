import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import re
from urllib.parse import urlparse
import socket
import os

# NLP 라이브러리들 (AI 기반 이메일 분석용)
try:
    from textblob import TextBlob
    from spellchecker import SpellChecker
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.warning("AI 이메일 분석을 위해 다음을 설치하세요: pip install textblob pyspellchecker")

st.title("AI 기반 피싱 탐지 시스템")
st.write("URL과 이메일을 AI로 종합 분석하여 피싱 위험을 정확하게 탐지합니다!")

# ============ 기존 URL 분석 함수들 (그대로 유지) ============
def extract_url_features(url):
    """URL에서 피싱 탐지를 위한 특성들을 추출 - 기존 코드 그대로"""
    try:
        parsed_url = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path
        
        features = {}
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['has_https'] = 1 if url.startswith('https://') else 0
        features['num_subdomains'] = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
        features['has_suspicious_words'] = 1 if any(word in url.lower() for word in 
                                                  ['secure', 'account', 'update', 'confirm', 'verify', 'login']) else 0
        features['num_digits'] = len(re.findall(r'\d', url))
        features['num_params'] = len(parsed_url.query.split('&')) if parsed_url.query else 0
        features['path_length'] = len(path)
        features['has_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0
        
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        features['suspicious_tld'] = 1 if any(url.lower().endswith(tld) for tld in suspicious_tlds) else 0
        
        return features, None
        
    except Exception as e:
        return None, str(e)

def check_domain_reputation(domain):
    """도메인 평판 간단 체크 - 기존 코드 그대로"""
    try:
        socket.gethostbyname(domain)
        return "도메인이 존재합니다"
    except:
        return "도메인을 찾을 수 없습니다"

# ============ AI 기반 이메일 분석 시스템 (새로 추가) ============
class AIEmailAnalyzer:
    def __init__(self):
        self.ai_model = None
        self.ai_loaded = False
        self.model_info = None
        
        if AI_AVAILABLE:
            self.spell_checker = SpellChecker()
        
        # AI 모델 로드 시도
        self.load_ai_model()
    
    def load_ai_model(self):
        """통합 AI 모델 로드"""
        model_file = 'integrated_phishing_detector.pkl'
        
        try:
            if os.path.exists(model_file):
                print("AI 모델 로드 중...")
                model_data = joblib.load(model_file)
                
                self.ai_model = {
                    'model': model_data['model'],
                    'scaler': model_data.get('scaler'),
                    'tfidf_vectorizer': model_data.get('tfidf_vectorizer'),
                    'bigram_vectorizer': model_data.get('bigram_vectorizer'),
                    'label_encoder': model_data['label_encoder'],
                    'feature_columns': model_data.get('feature_columns', []),
                    'use_scaling': model_data.get('use_scaling', False)
                }
                
                self.model_info = {
                    'name': model_data.get('model_name', 'Unknown'),
                    'version': model_data.get('model_version', '1.0'),
                    'timestamp': model_data.get('training_timestamp', 'Unknown')
                }
                
                self.ai_loaded = True
                print(f"AI 모델 로드 완료: {self.model_info['name']}")
                return True
            else:
                print(f"AI 모델 파일을 찾을 수 없습니다: {model_file}")
                return False
                
        except Exception as e:
            print(f"AI 모델 로드 실패: {e}")
            return False
    
    def extract_ai_email_features(self, subject, body, sender):
        """AI 모델용 이메일 특성 추출"""
        if not AI_AVAILABLE:
            return {}
        
        # 안전한 텍스트 처리
        subject = str(subject) if pd.notna(subject) else ""
        body = str(body) if pd.notna(body) else ""
        sender = str(sender) if pd.notna(sender) else ""
        combined_text = f"{subject} {body}".strip()
        
        features = {}
        
        # 1. 기본 특성들
        features['subject_length'] = len(subject)
        features['body_length'] = len(body)
        features['combined_length'] = len(combined_text)
        features['sender_length'] = len(sender)
        
        # 2. 단어 개수
        features['subject_word_count'] = len(subject.split())
        features['body_word_count'] = len(body.split())
        features['combined_word_count'] = len(combined_text.split())
        
        # 3. 키워드 특성
        urgent_keywords = [
            'urgent', 'immediately', 'asap', 'expire', 'suspend', 'deadline',
            'action required', 'verify now', 'click now', 'act now', 'hurry'
        ]
        personal_info_keywords = [
            'verify', 'confirm', 'update', 'provide', 'enter your',
            'personal information', 'credit card', 'password', 'ssn'
        ]
        scam_keywords = [
            'congratulations', 'winner', 'prize', 'lottery', 'million',
            'inheritance', 'beneficiary', 'claim', 'reward'
        ]
        
        combined_lower = combined_text.lower()
        features['urgent_keyword_count'] = sum(1 for k in urgent_keywords if k in combined_lower)
        features['personal_info_keyword_count'] = sum(1 for k in personal_info_keywords if k in combined_lower)
        features['scam_keyword_count'] = sum(1 for k in scam_keywords if k in combined_lower)
        
        features['has_urgent_keywords'] = 1 if features['urgent_keyword_count'] > 0 else 0
        features['has_personal_info_keywords'] = 1 if features['personal_info_keyword_count'] > 0 else 0
        features['has_scam_keywords'] = 1 if features['scam_keyword_count'] > 0 else 0
        
        # 4. 특수문자 특성
        features['exclamation_count'] = combined_text.count('!')
        features['question_count'] = combined_text.count('?')
        features['dollar_count'] = combined_text.count('$')
        features['percent_count'] = combined_text.count('%')
        
        # 대문자 비율
        if len(combined_text) > 0:
            features['uppercase_ratio'] = sum(1 for c in combined_text if c.isupper()) / len(combined_text)
        else:
            features['uppercase_ratio'] = 0
        
        # 5. URL 특성
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, combined_text)
        features['url_count'] = len(urls)
        features['has_urls'] = 1 if len(urls) > 0 else 0
        
        short_domains = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly']
        features['short_url_count'] = sum(1 for url in urls for domain in short_domains if domain in url)
        
        # 6. 발신자 분석
        features['sender_suspicious'] = self.analyze_sender_domain(sender)
        
        # 7. 감정 분석 (TextBlob)
        try:
            blob = TextBlob(combined_text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
            
            if blob.sentiment.polarity > 0.1:
                features['sentiment_positive'] = 1
                features['sentiment_negative'] = 0
                features['sentiment_neutral'] = 0
            elif blob.sentiment.polarity < -0.1:
                features['sentiment_positive'] = 0 
                features['sentiment_negative'] = 1
                features['sentiment_neutral'] = 0
            else:
                features['sentiment_positive'] = 0
                features['sentiment_negative'] = 0
                features['sentiment_neutral'] = 1
        except:
            features['sentiment_polarity'] = 0
            features['sentiment_subjectivity'] = 0
            features['sentiment_positive'] = 0
            features['sentiment_negative'] = 0
            features['sentiment_neutral'] = 1
        
        # 8. 맞춤법 검사
        try:
            words = re.findall(r'\b[a-zA-Z]+\b', combined_text.lower())
            if len(words) > 0:
                misspelled = self.spell_checker.unknown(words)
                features['spelling_error_count'] = len(misspelled)
                features['spelling_error_ratio'] = len(misspelled) / len(words)
            else:
                features['spelling_error_count'] = 0
                features['spelling_error_ratio'] = 0
        except:
            features['spelling_error_count'] = 0
            features['spelling_error_ratio'] = 0
        
        return features, combined_text
    
    def analyze_sender_domain(self, sender):
        """발신자 도메인 분석"""
        if not sender or '@' not in sender:
            return 1
        
        try:
            domain = sender.split('@')[1].lower()
            
            trusted_domains = [
                'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
                'apple.com', 'amazon.com', 'paypal.com', 'microsoft.com'
            ]
            
            suspicious_patterns = ['.tk', '.ml', '.ga', '.cf', 'secure-', 'verify-', 'account-']
            
            if domain in trusted_domains:
                return 0
            
            for pattern in suspicious_patterns:
                if pattern in domain:
                    return 1
                    
            return 0.5
        except:
            return 1
    
    def predict_with_ai(self, subject, body, sender):
        """AI 모델로 피싱 여부 예측"""
        if not self.ai_loaded or not AI_AVAILABLE:
            return self.fallback_prediction(subject, body, sender)
        
        try:
            # 1. 기본 특성 추출
            basic_features, combined_text = self.extract_ai_email_features(subject, body, sender)
            
            # 2. TF-IDF 특성 추출
            tfidf_features = {}
            if self.ai_model['tfidf_vectorizer'] is not None:
                # 텍스트 전처리
                processed_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text.lower())
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
                
                # TF-IDF 변환
                tfidf_vector = self.ai_model['tfidf_vectorizer'].transform([processed_text])
                tfidf_feature_names = [f'tfidf_{word}' for word in self.ai_model['tfidf_vectorizer'].get_feature_names_out()]
                
                for i, name in enumerate(tfidf_feature_names):
                    tfidf_features[name] = tfidf_vector.toarray()[0][i]
            
            # 3. 2-gram 특성 추출
            bigram_features = {}
            if self.ai_model['bigram_vectorizer'] is not None:
                try:
                    processed_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text.lower())
                    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
                    
                    bigram_vector = self.ai_model['bigram_vectorizer'].transform([processed_text])
                    bigram_feature_names = [f'2gram_{gram}' for gram in self.ai_model['bigram_vectorizer'].get_feature_names_out()]
                    
                    for i, name in enumerate(bigram_feature_names):
                        bigram_features[name] = bigram_vector.toarray()[0][i]
                except:
                    pass  # 2-gram 실패 시 무시
            
            # 4. 모든 특성 결합
            all_features = {**basic_features, **tfidf_features, **bigram_features}
            
            # 5. 모델 특성 순서에 맞춰 정렬
            feature_vector = []
            for feature_name in self.ai_model['feature_columns']:
                feature_vector.append(all_features.get(feature_name, 0))
            
            # 6. 예측 실행
            input_data = np.array([feature_vector])
            
            # 스케일링 적용 (필요시)
            if self.ai_model['use_scaling'] and self.ai_model['scaler'] is not None:
                input_data = self.ai_model['scaler'].transform(input_data)
            
            # AI 모델 예측
            prediction = self.ai_model['model'].predict(input_data)[0]
            probabilities = self.ai_model['model'].predict_proba(input_data)[0]
            
            # 결과 해석
            predicted_label = self.ai_model['label_encoder'].inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            # 피싱 확률 계산
            phishing_prob = probabilities[1] if len(probabilities) > 1 else (
                probabilities[0] if self.ai_model['label_encoder'].classes_[0] == 'phishing' else 1 - probabilities[0]
            )
            
            return {
                'prediction': predicted_label,
                'phishing_probability': phishing_prob,
                'confidence': confidence,
                'risk_score': int(phishing_prob * 100),
                'method': 'AI',
                'features_used': len(feature_vector)
            }
            
        except Exception as e:
            st.warning(f"AI 예측 중 오류 발생: {e}")
            return self.fallback_prediction(subject, body, sender)
    
    def fallback_prediction(self, subject, body, sender):
        """AI 모델 실패 시 폴백 예측 (규칙 기반)"""
        features, _ = self.extract_ai_email_features(subject, body, sender)
        
        risk_score = 0
        
        # 기본 규칙들
        if features.get('has_urgent_keywords', 0):
            risk_score += 25
        if features.get('has_personal_info_keywords', 0):
            risk_score += 30
        if features.get('has_scam_keywords', 0):
            risk_score += 20
        
        sender_risk = features.get('sender_suspicious', 0)
        if sender_risk == 1:
            risk_score += 15
        elif sender_risk == 0.5:
            risk_score += 7
        
        if features.get('has_urls', 0):
            risk_score += 10
        
        if features.get('uppercase_ratio', 0) > 0.3:
            risk_score += 10
        
        risk_score = min(risk_score, 100)
        phishing_prob = risk_score / 100
        
        prediction = 'phishing' if risk_score >= 50 else 'legitimate'
        
        return {
            'prediction': prediction,
            'phishing_probability': phishing_prob,
            'confidence': 0.7,  # 규칙 기반은 낮은 신뢰도
            'risk_score': risk_score,
            'method': 'Rule-based',
            'features_used': len(features)
        }

# ============ 기존 URL 시스템 클래스 (그대로 유지) ============
class UnifiedPhishingDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['url_length', 'domain_length', 'num_dots', 'num_hyphens', 
                            'has_https', 'num_subdomains', 'has_suspicious_words', 
                            'num_digits', 'num_params', 'path_length', 'has_ip', 'suspicious_tld']
        self.model_loaded = False
    
    def load_external_model(self, model_path='phishing_detector.pkl'):
        """외부에서 생성된 고급 모델 로드"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    if model_data.get('feature_columns'):
                        self.feature_names = model_data['feature_columns']
                else:
                    self.model = model_data
                
                self.model_loaded = True
                return True, "고급 외부 모델"
        except Exception as e:
            st.warning(f"외부 모델 로드 실패: {e}")
        
        return False, None
    
    def create_enhanced_sample_data(self):
        """향상된 샘플 데이터 생성"""
        np.random.seed(42)
        data = []
        
        for i in range(1500):
            data.append({
                'url_length': np.random.randint(10, 50),
                'domain_length': np.random.randint(5, 20),
                'num_dots': np.random.randint(1, 3),
                'num_hyphens': np.random.randint(0, 2),
                'has_https': np.random.choice([0, 1], p=[0.1, 0.9]),
                'num_subdomains': np.random.randint(0, 2),
                'has_suspicious_words': np.random.choice([0, 1], p=[0.9, 0.1]),
                'num_digits': np.random.randint(0, 3),
                'num_params': np.random.randint(0, 2),
                'path_length': np.random.randint(0, 20),
                'has_ip': 0,
                'suspicious_tld': 0,
                'label': 0
            })
        
        for i in range(1500):
            data.append({
                'url_length': np.random.randint(50, 150),
                'domain_length': np.random.randint(20, 60),
                'num_dots': np.random.randint(3, 8),
                'num_hyphens': np.random.randint(2, 10),
                'has_https': np.random.choice([0, 1], p=[0.4, 0.6]),
                'num_subdomains': np.random.randint(2, 6),
                'has_suspicious_words': np.random.choice([0, 1], p=[0.2, 0.8]),
                'num_digits': np.random.randint(3, 15),
                'num_params': np.random.randint(2, 8),
                'path_length': np.random.randint(15, 80),
                'has_ip': np.random.choice([0, 1], p=[0.7, 0.3]),
                'suspicious_tld': np.random.choice([0, 1], p=[0.6, 0.4]),
                'label': 1
            })
        
        return pd.DataFrame(data)
    
    def train_fallback_model(self):
        """외부 모델이 없을 경우 사용할 폴백 모델 학습"""
        df = self.create_enhanced_sample_data()
        
        X = df[self.feature_names]
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def predict_from_features(self, features):
        """특성으로부터 예측"""
        if self.model is None:
            return None
        
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        input_data = np.array([feature_vector])
        
        if self.scaler is not None:
            input_data = self.scaler.transform(input_data)
        
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        
        return {
            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
            'phishing_probability': probabilities[1] if len(probabilities) > 1 else (1 - probabilities[0]),
            'confidence': max(probabilities)
        }
    
    def get_feature_importance(self):
        """특성 중요도 반환"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

# ============ 시스템 초기화 ============
@st.cache_resource
def initialize_systems():
    # URL 시스템 초기화 (기존)
    url_system = UnifiedPhishingDetectionSystem()
    success, model_type = url_system.load_external_model('phishing_detector.pkl')
    
    if success:
        st.success(f"URL 분석 모델 로드 완료: {model_type}")
        url_model_type = model_type
    else:
        with st.spinner("URL 분석 모델 학습중..."):
            try:
                accuracy = url_system.train_fallback_model()
                st.success(f"URL 분석 모델 학습 완료! 정확도: {accuracy:.2%}")
                url_model_type = "내장 URL 모델"
            except Exception as e:
                st.error(f"URL 모델 초기화 실패: {e}")
                return None, None, None, None
    
    # 이메일 AI 시스템 초기화 (신규)
    email_ai = AIEmailAnalyzer()
    
    if email_ai.ai_loaded:
        st.success(f"이메일 AI 모델 로드 완료: {email_ai.model_info['name']}")
        email_model_type = f"AI 모델 ({email_ai.model_info['name']})"
    else:
        st.warning("이메일 AI 모델을 찾을 수 없어 규칙 기반으로 동작합니다.")
        email_model_type = "규칙 기반"
    
    return url_system, url_model_type, email_ai, email_model_type

# 시스템 초기화
if 'systems' not in st.session_state:
    url_system, url_model_type, email_ai, email_model_type = initialize_systems()
    st.session_state.systems = {
        'url_system': url_system,
        'url_model_type': url_model_type,
        'email_ai': email_ai,
        'email_model_type': email_model_type
    }
else:
    systems = st.session_state.systems
    url_system = systems['url_system']
    url_model_type = systems['url_model_type']
    email_ai = systems['email_ai']
    email_model_type = systems['email_model_type']

# ============ 메인 인터페이스 ============
if url_system and email_ai:
    # 탭 생성
    tab1, tab2 = st.tabs(["URL 분석", "이메일 분석"])
    
    # ====== URL 분석 탭 (기존 코드 그대로) ======
    with tab1:
        st.header("실시간 URL 피싱 탐지")
        
        url_input = st.text_input(
            "URL을 입력하세요:",
            placeholder="예: https://www.google.com 또는 suspicious-site.com",
            help="http:// 또는 https://를 포함하여 입력하거나, 도메인만 입력해도 됩니다."
        )
        
        st.subheader("빠른 테스트")
        col1, col2, col3, col4 = st.columns(4)
        
        test_urls = {
            "정상 사이트": "https://www.google.com",
            "의심 사이트 1": "http://secure-bank-verify-account-12345.suspicious-domain.tk/login?confirm=true&urgent=yes",
            "의심 사이트 2": "https://192.168.1.100/paypal-security-update-urgent-verify-now.html?user=12345",
            "의심 사이트 3": "http://www-amazon-security-update-confirm-account.verify-login-details.ml/secure/update"
        }
        
        for i, (label, url) in enumerate(test_urls.items()):
            col = [col1, col2, col3, col4][i]
            with col:
                if st.button(label, key=f"url_test_{i}"):
                    st.session_state.selected_url = url
                    st.rerun()
        
        if 'selected_url' in st.session_state:
            url_input = st.session_state.selected_url
            st.info(f"선택된 URL: {url_input}")
        
        if url_input and st.button("URL 분석 실행", type="primary"):
            with st.spinner("AI가 URL을 분석중입니다..."):
                features, error = extract_url_features(url_input)
                
                if error:
                    st.error(f"URL 분석 중 오류 발생: {error}")
                else:
                    parsed_url = urlparse(url_input if url_input.startswith(('http://', 'https://')) else 'http://' + url_input)
                    domain_status = check_domain_reputation(parsed_url.netloc)
                    
                    result = url_system.predict_from_features(features)
                    
                    if result:
                        st.subheader(f"URL 분석 결과")
                        
                        if result['prediction'] == 'Phishing':
                            st.error("**피싱 사이트로 의심됩니다!**")
                            st.error(f"**피싱 확률: {result['phishing_probability']:.1%}**")
                            st.warning("이 사이트에 개인정보를 입력하지 마세요!")
                        else:
                            st.success("**비교적 안전한 사이트입니다.**")
                            st.success(f"**안전 확률: {1-result['phishing_probability']:.1%}**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("신뢰도", f"{result['confidence']:.1%}")
                        with col2:
                            st.info(f"{domain_status}")
                        
                        # 위험 요소 분석 (기존 코드)
                        risk_factors = []
                        if features['url_length'] > 75:
                            risk_factors.append(f"URL이 매우 깁니다 ({features['url_length']}자)")
                        if features['domain_length'] > 30:
                            risk_factors.append(f"도메인이 비정상적으로 깁니다 ({features['domain_length']}자)")
                        if features['num_dots'] > 5:
                            risk_factors.append(f"서브도메인이 많습니다 (점 {features['num_dots']}개)")
                        if features['num_hyphens'] > 4:
                            risk_factors.append(f"하이픈이 많습니다 ({features['num_hyphens']}개)")
                        if features['has_https'] == 0:
                            risk_factors.append("HTTPS를 사용하지 않습니다")
                        if features['has_suspicious_words'] == 1:
                            risk_factors.append("의심스러운 단어가 포함되어 있습니다")
                        if features['has_ip'] == 1:
                            risk_factors.append("IP 주소를 직접 사용합니다")
                        if features['suspicious_tld'] == 1:
                            risk_factors.append("의심스러운 최상위 도메인을 사용합니다")
                        
                        if risk_factors:
                            st.subheader("발견된 위험 요소")
                            for factor in risk_factors:
                                st.write(f"• {factor}")
                        else:
                            st.success("**특별한 위험 요소가 발견되지 않았습니다.**")

    # ====== 이메일 분석 탭 (AI 기반으로 업그레이드) ======
    with tab2:
        st.header("AI 피싱 이메일 탐지")
        
        # AI 모델 상태 표시
        if email_ai.ai_loaded:
            st.success(f"AI 모델 활성화: {email_ai.model_info['name']}")
        else:
            st.warning("AI 모델 미사용 - 규칙 기반으로 동작")
        
        # 이메일 입력 폼
        with st.form("email_analysis_form"):
            st.subheader("이메일 정보 입력")
            
            email_sender = st.text_input(
                "발신자 이메일:",
                placeholder="예: sender@domain.com",
                help="이메일을 보낸 사람의 이메일 주소를 입력하세요"
            )
            
            email_subject = st.text_input(
                "이메일 제목:",
                placeholder="예: URGENT: Verify Your Account Now!",
                help="이메일의 제목을 입력하세요"
            )
            
            email_body = st.text_area(
                "이메일 본문:",
                placeholder="이메일의 전체 내용을 입력하세요...",
                height=200,
                help="이메일의 본문 내용을 모두 입력하세요"
            )
            
            analyze_button = st.form_submit_button("AI 이메일 분석", type="primary")
        
        # 빠른 테스트 샘플들
        st.subheader("빠른 테스트 샘플")
        
        test_emails = {
            "피싱 이메일 1": {
                "sender": "security@fake-bank.com",
                "subject": "URGENT: Verify Your Account Now!",
                "body": "Dear Customer, Your account will be suspended in 24 hours unless you verify immediately. Click here: http://secure-verify-account.suspicious-domain.tk/login?confirm=true"
            },
            "피싱 이메일 2": {
                "sender": "winner@lottery-scam.net",
                "subject": "Congratulations! You Won $1,000,000!",
                "body": "You have been selected as the winner of our lottery! To claim your prize, please provide your personal information and credit card details immediately."
            },
            "정상 이메일": {
                "sender": "orders@amazon.com",
                "subject": "Your Order Confirmation #123456",
                "body": "Thank you for your order. Your items will be shipped within 2 business days. You can track your order in your account."
            }
        }
        
        col1, col2, col3 = st.columns(3)
        for i, (label, email_data) in enumerate(test_emails.items()):
            col = [col1, col2, col3][i]
            with col:
                if st.button(label, key=f"email_test_{i}"):
                    st.session_state.selected_email = email_data
                    st.rerun()
        
        # 선택된 샘플 이메일 처리
        if 'selected_email' in st.session_state:
            selected = st.session_state.selected_email
            email_sender = selected['sender']
            email_subject = selected['subject'] 
            email_body = selected['body']
            st.info(f"선택된 샘플: {email_subject[:50]}...")
        
        # AI 이메일 분석 실행
        if analyze_button and (email_subject or email_body):
            with st.spinner("AI가 이메일을 분석중입니다..."):
                
                # AI 모델로 예측 (규칙 기반 폴백 포함)
                result = email_ai.predict_with_ai(email_subject, email_body, email_sender)
                
                # 결과 표시
                st.subheader("AI 분석 결과")
                
                # 메인 결과
                risk_score = result['risk_score']
                method = result['method']
                
                if risk_score >= 70:
                    st.error(f"!!**높은 위험도!** 피싱 이메일로 의심됩니다!")
                    st.error(f"**위험도: {risk_score}%**")
                    st.warning("!이 이메일의 링크를 클릭하거나 개인정보를 제공하지 마세요!")
                elif risk_score >= 30:
                    st.warning(f"!**중간 위험도** - 주의가 필요합니다")
                    st.warning(f"**위험도: {risk_score}%**")
                    st.info("발신자와 내용을 신중하게 검토하세요")
                else:
                    st.success(f"**낮은 위험도** - 비교적 안전해 보입니다")
                    st.success(f"**위험도: {risk_score}%**")
                
                # 상세 정보
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("AI 신뢰도", f"{result['confidence']:.1%}")
                
                with col2:
                    st.metric("분석 방법", method)
                
                with col3:
                    st.metric("사용된 특성", f"{result['features_used']}개")
                
                # AI vs 규칙 기반 구분 표시
                if method == 'AI':
                    st.info("**AI 기반 분석**: 고도화된 머신러닝 모델을 사용한 정확한 분석")
                else:
                    st.warning("**규칙 기반 분석**: AI 모델을 사용할 수 없어 기본 규칙으로 분석")
                
                # 예측 정보 표시
                st.subheader("예측 상세 정보")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric("피싱 확률", f"{result['phishing_probability']:.1%}")
                with pred_col2:
                    st.metric("정상 확률", f"{1-result['phishing_probability']:.1%}")

    # ====== 사이드바 (업데이트) ======
    st.sidebar.title("시스템 정보")
    st.sidebar.success(f"**URL 분석**: {url_model_type}")
    st.sidebar.success(f"**이메일 분석**: {email_model_type}")
    
    if url_system:
        importance = url_system.get_feature_importance()
        if importance:
            st.sidebar.write("### URL 탐지 특성")
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, imp in sorted_importance:
                st.sidebar.write(f"**{feature}**: {imp:.3f}")
    
    if email_ai.ai_loaded:
        st.sidebar.write("### AI 이메일 모델")
        st.sidebar.write(f"**버전**: {email_ai.model_info['version']}")
        st.sidebar.write(f"**학습일**: {email_ai.model_info['timestamp'][:10]}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 피싱 탐지 가이드
    
    **위험 신호:**
    - 긴급함을 조성하는 언어
    - 개인정보 제공 요구
    - 의심스러운 링크 포함
    - 문법 오류가 많은 텍스트
    - 알 수 없는 발신자
    
    **안전 신호:**
    - 신뢰할 수 있는 발신자
    - 정상적인 언어 패턴
    - HTTPS 사용
    - 명확한 연락처 정보
    
    ### 주의사항
    AI 분석 결과는 참고용입니다. 의심스러운 경우 직접 확인하세요.
    """)

else:
    st.error("시스템 초기화에 실패했습니다. 페이지를 새로고침해주세요.")

# 푸터
st.markdown("---")
st.markdown("**AI 피싱 탐지 시스템** | URL + 이메일 AI 통합 분석으로 완벽한 피싱 탐지")