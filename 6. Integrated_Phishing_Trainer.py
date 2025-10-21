"""
통합 AI 피싱 탐지 모델 학습기
URL 분석 + 이메일 텍스트 분석 (레벨 2 NLP)

새로운 기능:
- 이메일 텍스트 NLP 분석
- TF-IFD 벡터화
- 감정 분석 (TextBlob)
- N-gram 패턴 분석
- 문법 오류 검사
- URL + 이메일 통합 모델

필요한 라이브러리 설치:
pip install nltk textblob pyspellchecker
python -m textblob.download_corpora
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import sys
import io
import time
import warnings
import re
from collections import Counter

# NLP 라이브러리들
import nltk
from textblob import TextBlob
from spellchecker import SpellChecker

warnings.filterwarnings('ignore')

class IntegratedPhishingDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # URL 특성 (기존)
        self.url_feature_columns = []
        
        # 텍스트 분석 도구들 (신규)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,  # 상위 100개 단어만
            stop_words='english',
            ngram_range=(1, 2),  # 1-gram, 2-gram
            lowercase=True,
            min_df=2,  # 최소 2번 이상 나타나는 단어만
            max_df=0.95  # 너무 자주 나타나는 단어 제외
        )
        
        self.spell_checker = SpellChecker()
        
        # 모델 관련
        self.best_model = None
        self.best_model_name = ""
        self.performance_metrics = {}
        
        # 데이터
        self.url_data = None
        self.email_data = None
        
        print("통합 AI 피싱 탐지 모델 학습기")
        
    def load_collected_data(self, data_dir=None, email_file=None):
        """URL 데이터와 이메일 데이터 로드"""
        print("\n 데이터 로드 중...")
        
        # 1. 기존 URL 데이터 로드
        if data_dir is None:
            current_dir = os.getcwd()
            phishing_dirs = [d for d in os.listdir(current_dir) 
                            if d.startswith('phishing_data') and os.path.isdir(d)]
            if phishing_dirs:
                data_dir = max(phishing_dirs)
                
        if data_dir:
            url_file = os.path.join(data_dir, 'url_dataset_with_features.csv')
            if os.path.exists(url_file):
                self.url_data = pd.read_csv(url_file)
                print(f"URL 데이터: {self.url_data.shape}")
            else:
                print("URL 데이터를 찾을 수 없습니다.")
        
        # 2. 이메일 데이터 로드
        if email_file is None:
            email_file = 'email_datasets/phishing_email_dataset.csv'
            
        if os.path.exists(email_file):
            self.email_data = pd.read_csv(email_file)
            print(f"이메일 데이터: {self.email_data.shape}")
            print(f"레이블 분포: {self.email_data['label'].value_counts().to_dict()}")
        else:
            print(f"이메일 데이터를 찾을 수 없습니다: {email_file}")
            print("먼저 phishing_email_collector.py를 실행하세요!")
            return False
            
        return True
    
    def extract_email_text_features(self, df):
        """이메일 텍스트에서 고급 NLP 특성 추출"""
        print("\n이메일 텍스트 특성 추출 중...")
        
        # 텍스트 결합 (제목 + 본문)
        df['combined_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
        df['combined_text'] = df['combined_text'].str.strip()
        
        features_list = []
        
        print("기본 텍스트 특성 추출...")
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"      진행률: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
                
            features = self.extract_single_email_features(
                row['subject'], row['body'], row['sender'], row['combined_text']
            )
            features_list.append(features)
        
        # DataFrame으로 변환
        features_df = pd.DataFrame(features_list)
        
        print(f"추출된 기본 특성: {len(features_df.columns)}개")
        return features_df
    
    def extract_single_email_features(self, subject, body, sender, combined_text):
        """단일 이메일에서 특성 추출"""
        features = {}
        
        # 안전한 텍스트 처리
        subject = str(subject) if pd.notna(subject) else ""
        body = str(body) if pd.notna(body) else ""
        sender = str(sender) if pd.notna(sender) else ""
        combined_text = str(combined_text) if pd.notna(combined_text) else ""
        
        # 1. 기본 길이 특성
        features['subject_length'] = len(subject)
        features['body_length'] = len(body)
        features['combined_length'] = len(combined_text)
        features['sender_length'] = len(sender)
        
        # 2. 단어 개수 특성
        subject_words = subject.split()
        body_words = body.split()
        combined_words = combined_text.split()
        
        features['subject_word_count'] = len(subject_words)
        features['body_word_count'] = len(body_words)
        features['combined_word_count'] = len(combined_words)
        
        # 3. 피싱 키워드 특성
        urgent_keywords = [
            'urgent', 'immediately', 'asap', 'expire', 'suspend', 'deadline',
            'action required', 'verify now', 'click now', 'act now', 'hurry'
        ]
        
        personal_info_keywords = [
            'verify', 'confirm', 'update', 'provide', 'enter your',
            'personal information', 'credit card', 'password', 'ssn',
            'social security', 'bank account', 'routing number'
        ]
        
        scam_keywords = [
            'congratulations', 'winner', 'prize', 'lottery', 'million',
            'inheritance', 'beneficiary', 'claim', 'reward', 'fortune'
        ]
        
        combined_lower = combined_text.lower()
        
        features['urgent_keyword_count'] = sum(1 for keyword in urgent_keywords if keyword in combined_lower)
        features['personal_info_keyword_count'] = sum(1 for keyword in personal_info_keywords if keyword in combined_lower)
        features['scam_keyword_count'] = sum(1 for keyword in scam_keywords if keyword in combined_lower)
        
        features['has_urgent_keywords'] = 1 if features['urgent_keyword_count'] > 0 else 0
        features['has_personal_info_keywords'] = 1 if features['personal_info_keyword_count'] > 0 else 0
        features['has_scam_keywords'] = 1 if features['scam_keyword_count'] > 0 else 0
        
        # 4. 특수문자 및 패턴 특성
        features['exclamation_count'] = combined_text.count('!')
        features['question_count'] = combined_text.count('?')
        features['dollar_count'] = combined_text.count('$')
        features['percent_count'] = combined_text.count('%')
        
        # 대문자 비율
        if len(combined_text) > 0:
            features['uppercase_ratio'] = sum(1 for c in combined_text if c.isupper()) / len(combined_text)
        else:
            features['uppercase_ratio'] = 0
        
        # 5. URL 특성 (이메일 내 링크)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, combined_text)
        
        features['url_count'] = len(urls)
        features['has_urls'] = 1 if len(urls) > 0 else 0
        
        # 단축 URL 탐지
        short_domains = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly']
        features['short_url_count'] = sum(1 for url in urls for domain in short_domains if domain in url)
        
        # 6. 발신자 분석
        features['sender_suspicious'] = self.analyze_sender_domain(sender)
        
        return features
    
    def analyze_sender_domain(self, sender):
        """발신자 도메인 의심도 분석"""
        if not sender or '@' not in sender:
            return 1
        
        try:
            domain = sender.split('@')[1].lower()
            
            # 신뢰할 수 있는 도메인들
            trusted_domains = [
                'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
                'apple.com', 'amazon.com', 'paypal.com', 'microsoft.com',
                'google.com', 'facebook.com', 'netflix.com'
            ]
            
            # 의심스러운 패턴들
            suspicious_patterns = [
                '.tk', '.ml', '.ga', '.cf',  # 무료 도메인
                'secure-', 'verify-', 'account-', 'update-',  # 의심스러운 접두사
                'temp', 'fake', 'scam', 'phish'  # 명백히 의심스러운 단어
            ]
            
            if domain in trusted_domains:
                return 0
            
            for pattern in suspicious_patterns:
                if pattern in domain:
                    return 1
                    
            return 0.5
            
        except:
            return 1
    
    def extract_tfidf_features(self, texts):
        """TF-IDF 특성 추출"""
        print("\nTF-IDF 특성 추출 중...")
        
        # 텍스트 전처리
        processed_texts = []
        for text in texts:
            if pd.isna(text):
                processed_texts.append("")
            else:
                # 기본 전처리
                text = str(text).lower()
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # 특수문자 제거
                text = re.sub(r'\s+', ' ', text).strip()  # 연속 공백 제거
                processed_texts.append(text)
        
        # TF-IDF 벡터화
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # 특성명 생성
        feature_names = [f'tfidf_{word}' for word in self.tfidf_vectorizer.get_feature_names_out()]
        
        # DataFrame으로 변환
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)
        
        print(f"TF-IDF 특성: {tfidf_df.shape[1]}개")
        print(f"주요 단어들: {list(self.tfidf_vectorizer.get_feature_names_out()[:10])}")
        
        return tfidf_df
    
    def extract_sentiment_features(self, texts):
        """감정 분석 특성 추출 (TextBlob 사용)"""
        print("\n감정 분석 특성 추출 중...")
        
        sentiment_features = []
        
        for idx, text in enumerate(texts):
            if idx % 1000 == 0:
                print(f"진행률: {idx}/{len(texts)} ({idx/len(texts)*100:.1f}%)")
                
            features = {}
            
            if pd.isna(text) or text == "":
                features['sentiment_polarity'] = 0
                features['sentiment_subjectivity'] = 0
                features['sentiment_positive'] = 0
                features['sentiment_negative'] = 0
                features['sentiment_neutral'] = 1
            else:
                try:
                    blob = TextBlob(str(text))
                    
                    # 극성 (-1: 부정적, +1: 긍정적)
                    features['sentiment_polarity'] = blob.sentiment.polarity
                    
                    # 주관성 (0: 객관적, 1: 주관적)
                    features['sentiment_subjectivity'] = blob.sentiment.subjectivity
                    
                    # 감정 카테고리
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
                    # 분석 실패 시 중립으로 설정
                    features['sentiment_polarity'] = 0
                    features['sentiment_subjectivity'] = 0
                    features['sentiment_positive'] = 0
                    features['sentiment_negative'] = 0
                    features['sentiment_neutral'] = 1
            
            sentiment_features.append(features)
        
        sentiment_df = pd.DataFrame(sentiment_features)
        print(f"감정 분석 특성: {sentiment_df.shape[1]}개")
        
        return sentiment_df
    
    def extract_grammar_features(self, texts):
        """문법 및 맞춤법 특성 추출"""
        print("\n문법/맞춤법 특성 추출 중...")
        
        grammar_features = []
        
        for idx, text in enumerate(texts):
            if idx % 1000 == 0:
                print(f"      진행률: {idx}/{len(texts)} ({idx/len(texts)*100:.1f}%)")
                
            features = {}
            
            if pd.isna(text) or text == "":
                features['spelling_error_count'] = 0
                features['spelling_error_ratio'] = 0
            else:
                try:
                    # 단어 추출 (알파벳만)
                    words = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
                    
                    if len(words) == 0:
                        features['spelling_error_count'] = 0
                        features['spelling_error_ratio'] = 0
                    else:
                        # 맞춤법 오류 개수 계산
                        misspelled = self.spell_checker.unknown(words)
                        
                        features['spelling_error_count'] = len(misspelled)
                        features['spelling_error_ratio'] = len(misspelled) / len(words)
                        
                except:
                    features['spelling_error_count'] = 0
                    features['spelling_error_ratio'] = 0
            
            grammar_features.append(features)
        
        grammar_df = pd.DataFrame(grammar_features)
        print(f"문법/맞춤법 특성: {grammar_df.shape[1]}개")
        
        return grammar_df
    
    def extract_ngram_features(self, texts, n=2, max_features=50):
        """N-gram 패턴 특성 추출"""
        print(f"\n{n}-gram 패턴 특성 추출 중...")
        
        # N-gram용 TF-IDF (별도)
        ngram_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(n, n),  # n-gram만
            lowercase=True,
            min_df=3,  # 최소 3번 이상
            max_df=0.8,
            stop_words='english'
        )
        
        # 텍스트 전처리
        processed_texts = []
        for text in texts:
            if pd.isna(text):
                processed_texts.append("")
            else:
                text = str(text).lower()
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                processed_texts.append(text)
        
        try:
            ngram_features = ngram_vectorizer.fit_transform(processed_texts)
            feature_names = [f'{n}gram_{gram}' for gram in ngram_vectorizer.get_feature_names_out()]
            ngram_df = pd.DataFrame(ngram_features.toarray(), columns=feature_names)
            
            print(f"  {n}-gram 특성: {ngram_df.shape[1]}개")
            if ngram_df.shape[1] > 0:
                print(f"   주요 {n}-gram: {list(ngram_vectorizer.get_feature_names_out()[:5])}")
            
            return ngram_df, ngram_vectorizer
            
        except ValueError as e:
            print(f"{n}-gram 추출 실패: {e}")
            # 빈 DataFrame 반환
            return pd.DataFrame(), None
    
    def create_integrated_dataset(self):
        """URL 데이터와 이메일 데이터를 통합하여 최종 데이터셋 생성"""
        print("\n통합 데이터셋 생성 중...")
        
        if self.email_data is None:
            print("  이메일 데이터가 없습니다.")
            return None, None
        
        # 1. 이메일 데이터 정리
        email_df = self.email_data.copy()
        
        # 레이블 통일 (phishing/legitimate)
        label_mapping = {
            'spam': 'phishing',
            'ham': 'legitimate',
            'malicious': 'phishing',
            'normal': 'legitimate',
            'phishing': 'phishing',
            'legitimate': 'legitimate'
        }
        
        email_df['label'] = email_df['label'].map(lambda x: label_mapping.get(str(x).lower(), x))
        
        # 2. 이메일 텍스트 특성 추출
        basic_features = self.extract_email_text_features(email_df)
        
        # 3. TF-IDF 특성 추출
        tfidf_features = self.extract_tfidf_features(email_df['combined_text'])
        
        # 4. 감정 분석 특성 추출
        sentiment_features = self.extract_sentiment_features(email_df['combined_text'])
        
        # 5. 문법 특성 추출
        grammar_features = self.extract_grammar_features(email_df['combined_text'])
        
        # 6. 2-gram 특성 추출
        bigram_features, self.bigram_vectorizer = self.extract_ngram_features(
            email_df['combined_text'], n=2, max_features=30
        )
        
        # 7. 모든 특성 결합
        print("\n모든 특성 결합 중...")
        
        feature_dataframes = [basic_features, tfidf_features, sentiment_features, grammar_features]
        
        if not bigram_features.empty:
            feature_dataframes.append(bigram_features)
        
        # 인덱스 재설정
        for df in feature_dataframes:
            df.reset_index(drop=True, inplace=True)
        
        # 특성 결합
        integrated_features = pd.concat(feature_dataframes, axis=1)
        
        # 8. 레이블 추가
        y = email_df['label'].reset_index(drop=True)
        
        print(f"\n통합 데이터셋 생성 완료:")
        print(f"   총 샘플 수: {len(integrated_features):,}개")
        print(f"   총 특성 수: {len(integrated_features.columns):,}개")
        print(f"   레이블 분포: {y.value_counts().to_dict()}")
        
        # 특성 타입별 개수 출력
        basic_count = len(basic_features.columns)
        tfidf_count = len(tfidf_features.columns)
        sentiment_count = len(sentiment_features.columns)
        grammar_count = len(grammar_features.columns)
        bigram_count = len(bigram_features.columns) if not bigram_features.empty else 0
        
        print(f"\n   특성 구성:")
        print(f"     기본 특성: {basic_count}개")
        print(f"     TF-IDF: {tfidf_count}개")
        print(f"     감정 분석: {sentiment_count}개")
        print(f"     문법 분석: {grammar_count}개")
        print(f"     2-gram: {bigram_count}개")
        
        return integrated_features, y
    
    def train_integrated_models(self, X, y):
        """통합 특성으로 다중 모델 학습"""
        print(f"\n 통합 모델 학습 시작...")
        print(f"   입력 데이터: {X.shape[0]:,}개 샘플, {X.shape[1]:,}개 특성")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 레이블 인코딩
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 데이터 스케일링 (일부 모델용)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 설정
        models_config = {
            'XGBoost': {
                'model': XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'use_scaling': False
            },
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaling': False
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'use_scaling': False
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    C=1.0
                ),
                'use_scaling': True
            }
        }
        
        best_score = 0
        
        for name, config in models_config.items():
            print(f"\n  {name} 학습 중...")
            
            model = config['model']
            use_scaling = config['use_scaling']
            
            # 모델 학습
            start_time = time.time()
            
            if use_scaling:
                model.fit(X_train_scaled, y_train_encoded)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=3, scoring='accuracy')
            else:
                model.fit(X_train, y_train_encoded)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=3, scoring='accuracy')
            
            training_time = time.time() - start_time
            
            # 성능 계산
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, average='weighted')
            recall = recall_score(y_test_encoded, y_pred, average='weighted')
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # 결과 저장
            self.models[name] = {
                'model': model,
                'use_scaling': use_scaling,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"학습 시간: {training_time:.1f}초")
            print(f"정확도: {accuracy:.4f}")
            print(f"CV 점수: {cv_mean:.4f} (±{cv_std:.4f})")
            print(f"F1 점수: {f1:.4f}")
            
            # 최고 성능 모델 업데이트
            if cv_mean > best_score:
                best_score = cv_mean
                self.best_model_name = name
                self.best_model = model
        
        # 테스트 데이터 저장
        self.X_test = X_test
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test_encoded
        
        print(f"\n최고 성능 모델: {self.best_model_name}")
        print(f" CV 점수: {self.models[self.best_model_name]['cv_mean']:.4f}")
        
        return self.models
    
    def analyze_feature_importance(self):
        """특성 중요도 분석"""
        print(f"\n특성 중요도 분석...")
        
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            print("특성 중요도를 분석할 수 없습니다.")
            return
        
        # 특성 중요도 추출
        feature_names = self.X_test.columns.tolist()
        importances = self.best_model.feature_importances_
        
        # 중요도 순으로 정렬
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n상위 20개 중요 특성:")
        for i, (feature, importance) in enumerate(feature_importance[:20], 1):
            print(f"   {i:2d}. {feature:<30} {importance:.4f}")
        
        # 특성 타입별 중요도 합계
        tfidf_importance = sum(imp for name, imp in feature_importance if name.startswith('tfidf_'))
        sentiment_importance = sum(imp for name, imp in feature_importance if name.startswith('sentiment_'))
        gram_importance = sum(imp for name, imp in feature_importance if name.startswith('2gram_'))
        basic_importance = sum(imp for name, imp in feature_importance 
                             if not any(name.startswith(prefix) for prefix in ['tfidf_', 'sentiment_', '2gram_']))
        
        print(f"\n특성 타입별 중요도:")
        print(f"   기본 특성: {basic_importance:.3f}")
        print(f"   TF-IDF: {tfidf_importance:.3f}")
        print(f"   감정 분석: {sentiment_importance:.3f}")
        print(f"   2-gram: {gram_importance:.3f}")
        
        return feature_importance
    
    def save_integrated_model(self, filename="integrated_phishing_detector.pkl"):
        """통합 모델 저장"""
        print(f"\n통합 모델 저장 중...")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'bigram_vectorizer': getattr(self, 'bigram_vectorizer', None),
            'spell_checker': self.spell_checker,
            'feature_columns': self.X_test.columns.tolist() if hasattr(self, 'X_test') else [],
            'performance_metrics': self.performance_metrics,
            'use_scaling': self.models[self.best_model_name]['use_scaling'],
            'training_timestamp': datetime.now().isoformat(),
            'model_version': '3.0_integrated_nlp'
        }
        
        joblib.dump(model_data, filename)
        
        print(f"모델 저장 완료: {filename}")
        print(f"모델 타입: {self.best_model_name}")
        print(f"특성 수: {len(model_data['feature_columns'])}개")
        
        return filename
    
    def run_integrated_training(self):
        """전체 통합 학습 파이프라인 실행"""
        print("통합 AI 피싱 탐지 모델 학습 시작!")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 데이터 로드
        print("1.데이터 로드...")
        if not self.load_collected_data():
            return False
        
        # 2. 통합 데이터셋 생성
        print("\n2. 통합 데이터셋 생성...")
        X, y = self.create_integrated_dataset()
        
        if X is None:
            print("❌ 데이터셋 생성 실패")
            return False
        
        # 3. 모델 학습
        print("\n3. 모델 학습...")
        self.train_integrated_models(X, y)
        
        # 4. 성능 평가
        print("\n4. 성능 평가...")
        self.evaluate_final_model()
        
        # 5. 특성 중요도 분석
        print("\n5. 특성 중요도 분석...")
        self.analyze_feature_importance()
        
        # 6. 모델 저장
        print("\n6. 모델 저장...")
        saved_file = self.save_integrated_model()
        
        # 완료 시간
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n통합 모델 학습 완료!")
        print("=" * 60)
        print(f"총 소요 시간: {duration:.1f}초 ({duration/60:.1f}분)")
        print(f"최고 성능 모델: {self.best_model_name}")
        print(f"최종 정확도: {self.models[self.best_model_name]['accuracy']:.4f}")
        print(f"저장된 모델: {saved_file}")
        print("=" * 60)
        print("다음 단계: Streamlit 앱에서 이 모델을 사용하세요!")
        
        return saved_file
    
    def evaluate_final_model(self):
        """최종 모델 성능 평가"""
        print(f"\n{self.best_model_name} 최종 성능 평가...")
        
        use_scaling = self.models[self.best_model_name]['use_scaling']
        
        if use_scaling:
            y_pred = self.best_model.predict(self.X_test_scaled)
        else:
            y_pred = self.best_model.predict(self.X_test)
        
        # 성능 메트릭 계산
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        self.performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"정확도: {accuracy:.4f}")
        print(f"정밀도: {precision:.4f}")
        print(f"재현율: {recall:.4f}")
        print(f"F1 점수: {f1:.4f}")
        
        # 클래스별 성능
        print(f"\n클래스별 성능:")
        report = classification_report(self.y_test, y_pred, 
                                     target_names=self.label_encoder.classes_,
                                     output_dict=True)
        
        for class_name in self.label_encoder.classes_:
            if class_name in report:
                metrics = report[class_name]
                print(f"     {class_name}:")
                print(f"       정밀도: {metrics['precision']:.4f}")
                print(f"       재현율: {metrics['recall']:.4f}")
                print(f"       F1 점수: {metrics['f1-score']:.4f}")
        
        return self.performance_metrics


def main():
    """메인 실행 함수"""
    print("통합 AI 피싱 탐지 모델 학습기 v3.0")
    print("이메일 텍스트 분석 + URL 분석 통합")
    
    # 필요 라이브러리 확인
    try:
        import nltk
        import textblob
        from spellchecker import SpellChecker
        print("모든 NLP 라이브러리가 설치되어 있습니다.")
    except ImportError as e:
        print(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치하세요:")
        print("pip install nltk textblob pyspellchecker")
        print("python -m textblob.download_corpora")
        return
    
    # NLTK 데이터 다운로드 확인
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK 데이터 다운로드 중...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("NLTK 데이터 다운로드 완료")
        except:
            print("NLTK 데이터 다운로드 실패 (계속 진행)")
    
    detector = IntegratedPhishingDetector()
    
    print("\n통합 AI 모델 학습을 시작하시겠습니까?")
    print("주의: 전체 학습에는 5-15분이 소요될 수 있습니다.")
    print("이메일 데이터셋이 필요합니다 (phishing_email_collector.py 실행 필요)")
    
    response = input("\n계속하시겠습니까? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        try:
            model_file = detector.run_integrated_training()
            
            if model_file:
                print(f"\n통합 AI 모델 학습 성공!")
                print(f"이제 Streamlit 앱에서 이 모델을 사용할 수 있습니다.")
            else:
                print(f"\n모델 학습에 실패했습니다.")
                
        except KeyboardInterrupt:
            print(f"\n\n사용자가 학습을 중단했습니다.")
        except Exception as e:
            print(f"\n예상치 못한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n모델 학습이 취소되었습니다.")


if __name__ == "__main__":
    main()