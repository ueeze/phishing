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

st.title(" AI 기반 피싱 탐지 시스템 ")
st.write("고도화된 AI 모델로 실제 URL의 피싱 여부를 정확하게 분석합니다!")

# URL 특성 추출 함수들
def extract_url_features(url):
    """URL에서 피싱 탐지를 위한 특성들을 추출"""
    try:
        # URL 파싱
        parsed_url = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path
        
        features = {}
        
        # 기본 특성들
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        
        # 추가 특성들
        features['has_https'] = 1 if url.startswith('https://') else 0
        features['num_subdomains'] = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
        features['has_suspicious_words'] = 1 if any(word in url.lower() for word in 
                                                  ['secure', 'account', 'update', 'confirm', 'verify', 'login']) else 0
        features['num_digits'] = len(re.findall(r'\d', url))
        features['num_params'] = len(parsed_url.query.split('&')) if parsed_url.query else 0
        features['path_length'] = len(path)
        features['has_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0
        
        # 도메인 의심도 체크
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        features['suspicious_tld'] = 1 if any(url.lower().endswith(tld) for tld in suspicious_tlds) else 0
        
        return features, None
        
    except Exception as e:
        return None, str(e)

def check_domain_reputation(domain):
    """도메인 평판 간단 체크"""
    try:
        socket.gethostbyname(domain)
        return " 도메인이 존재합니다"
    except:
        return " 도메인을 찾을 수 없습니다"

# 통합된 피싱 탐지 시스템 클래스
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
                    # 딕셔너리 형태로 저장된 경우
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    if model_data.get('feature_columns'):
                        self.feature_names = model_data['feature_columns']
                else:
                    # 모델만 저장된 경우
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
        
        # 정상 URL 데이터 (1500개)
        for i in range(1500):
            data.append({
                'url_length': np.random.randint(10, 50),
                'domain_length': np.random.randint(5, 20),
                'num_dots': np.random.randint(1, 3),
                'num_hyphens': np.random.randint(0, 2),
                'has_https': np.random.choice([0, 1], p=[0.1, 0.9]),  # 90% HTTPS
                'num_subdomains': np.random.randint(0, 2),
                'has_suspicious_words': np.random.choice([0, 1], p=[0.9, 0.1]),  # 10% 의심 단어
                'num_digits': np.random.randint(0, 3),
                'num_params': np.random.randint(0, 2),
                'path_length': np.random.randint(0, 20),
                'has_ip': 0,  # 정상 사이트는 IP 주소 사용 안함
                'suspicious_tld': 0,  # 정상 사이트는 의심스러운 TLD 사용 안함
                'label': 0  # 정상
            })
        
        # 피싱 URL 데이터 (1500개)
        for i in range(1500):
            data.append({
                'url_length': np.random.randint(50, 150),
                'domain_length': np.random.randint(20, 60),
                'num_dots': np.random.randint(3, 8),
                'num_hyphens': np.random.randint(2, 10),
                'has_https': np.random.choice([0, 1], p=[0.4, 0.6]),  # 60% HTTPS
                'num_subdomains': np.random.randint(2, 6),
                'has_suspicious_words': np.random.choice([0, 1], p=[0.2, 0.8]),  # 80% 의심 단어
                'num_digits': np.random.randint(3, 15),
                'num_params': np.random.randint(2, 8),
                'path_length': np.random.randint(15, 80),
                'has_ip': np.random.choice([0, 1], p=[0.7, 0.3]),  # 30% IP 사용
                'suspicious_tld': np.random.choice([0, 1], p=[0.6, 0.4]),  # 40% 의심 TLD
                'label': 1  # 피싱
            })
        
        return pd.DataFrame(data)
    
    def train_fallback_model(self):
        """외부 모델이 없을 경우 사용할 폴백 모델 학습"""
        df = self.create_enhanced_sample_data()
        
        X = df[self.feature_names]
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 스케일러 초기화
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 더 강력한 랜덤 포레스트 모델
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
        
        # 모든 특성이 있는지 확인하고 누락된 것은 0으로 채움
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        input_data = np.array([feature_vector])
        
        # 스케일링 적용 (스케일러가 있는 경우)
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

# 시스템 초기화
@st.cache_resource
def initialize_system():
    system = UnifiedPhishingDetectionSystem()
    
    # 1. 먼저 외부 고급 모델 로드 시도
    success, model_type = system.load_external_model('phishing_detector.pkl')
    
    if success:
        st.success(f"{model_type} 로드 완료!")
        return system, model_type
    else:
        # 2. 외부 모델이 없으면 폴백 모델 학습
        with st.spinner("내장 고성능 모델을 학습중입니다... (약 5초 소요)"):
            try:
                accuracy = system.train_fallback_model()
                st.success(f"내장 모델 학습 완료! 정확도: {accuracy:.2%}")
                return system, "내장 고성능 모델"
            except Exception as e:
                st.error(f"❌ 모델 초기화 실패: {e}")
                return None, None

# 시스템 초기화
if 'system' not in st.session_state:
    system, model_type = initialize_system()
    st.session_state.system = system
    st.session_state.model_type = model_type
else:
    system = st.session_state.system
    model_type = st.session_state.model_type

if system:
    # 메인 인터페이스
    st.header("🌐 실시간 URL 피싱 탐지")
    
    # URL 입력
    url_input = st.text_input(
        "URL을 입력하세요:",
        placeholder="예: https://www.google.com 또는 suspicious-site.com",
        help="http:// 또는 https://를 포함하여 입력하거나, 도메인만 입력해도 됩니다."
    )
    
    # 빠른 테스트 버튼들
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
            if st.button(label, key=f"test_{i}"):
                st.session_state.selected_url = url
                st.rerun()
    
    # 선택된 URL 처리
    if 'selected_url' in st.session_state:
        url_input = st.session_state.selected_url
        st.info(f"선택된 URL: {url_input}")
    
    # URL 분석 실행
    if url_input and st.button("URL 분석 실행", type="primary"):
        with st.spinner("AI가 URL을 분석중입니다..."):
            # URL 특성 추출
            features, error = extract_url_features(url_input)
            
            if error:
                st.error(f"URL 분석 중 오류 발생: {error}")
            else:
                # 도메인 평판 체크
                parsed_url = urlparse(url_input if url_input.startswith(('http://', 'https://')) else 'http://' + url_input)
                domain_status = check_domain_reputation(parsed_url.netloc)
                
                # 예측 수행
                result = system.predict_from_features(features)
                
                if result:
                    st.subheader(f" 분석 결과")
                    
                    # 메인 결과 표시
                    if result['prediction'] == 'Phishing':
                        st.error(" **피싱 사이트로 의심됩니다!**")
                        st.error(f"**피싱 확률: {result['phishing_probability']:.1%}**")
                        st.warning("이 사이트에 개인정보를 입력하지 마세요!")
                    else:
                        st.success(" **비교적 안전한 사이트입니다.**")
                        st.success(f"**안전 확률: {1-result['phishing_probability']:.1%}**")
                    
                    # 추가 정보
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("신뢰도", f"{result['confidence']:.1%}")
                    with col2:
                        st.info(f"🌐 {domain_status}")
                    
                    # 위험 요소 분석
                    risk_factors = []
                    if features['url_length'] > 75:
                        risk_factors.append(f"⚠️ URL이 매우 깁니다 ({features['url_length']}자)")
                    if features['domain_length'] > 30:
                        risk_factors.append(f"⚠️ 도메인이 비정상적으로 깁니다 ({features['domain_length']}자)") 
                    if features['num_dots'] > 5:
                        risk_factors.append(f"⚠️ 서브도메인이 많습니다 (점 {features['num_dots']}개)")
                    if features['num_hyphens'] > 4:
                        risk_factors.append(f"⚠️ 하이픈이 많습니다 ({features['num_hyphens']}개)")
                    if features['has_https'] == 0:
                        risk_factors.append("⚠️ HTTPS를 사용하지 않습니다")
                    if features['has_suspicious_words'] == 1:
                        risk_factors.append("⚠️ 의심스러운 단어가 포함되어 있습니다")
                    if features['has_ip'] == 1:
                        risk_factors.append("⚠️ IP 주소를 직접 사용합니다")
                    if features['suspicious_tld'] == 1:
                        risk_factors.append("⚠️ 의심스러운 최상위 도메인을 사용합니다") 
                    
                    if risk_factors:
                        st.subheader("🚨 발견된 위험 요소")
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.success("✅ **특별한 위험 요소가 발견되지 않았습니다.**")
                    
                    # 상세 특성 정보 (접을 수 있는 형태)
                    with st.expander("🔍 상세 분석 정보"):
                        feature_df = pd.DataFrame([
                            {"특성": "URL 길이", "값": features['url_length']},
                            {"특성": "도메인 길이", "값": features['domain_length']},
                            {"특성": "점(.) 개수", "값": features['num_dots']},
                            {"특성": "하이픈(-) 개수", "값": features['num_hyphens']},
                            {"특성": "HTTPS 사용", "값": "예" if features['has_https'] else "아니오"},
                            {"특성": "서브도메인 개수", "값": features['num_subdomains']},
                            {"특성": "의심 단어 포함", "값": "예" if features['has_suspicious_words'] else "아니오"},
                            {"특성": "숫자 개수", "값": features['num_digits']},
                            {"특성": "매개변수 개수", "값": features['num_params']},
                            {"특성": "경로 길이", "값": features['path_length']},
                            {"특성": "IP 주소 사용", "값": "예" if features['has_ip'] else "아니오"},
                            {"특성": "의심스러운 TLD", "값": "예" if features['suspicious_tld'] else "아니오"}
                        ])
                        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    # 사이드바 - 시스템 정보
    st.sidebar.title("🤖 시스템 정보")
    st.sidebar.success(f"**활성 모델**: {model_type}")
    
    if system:
        importance = system.get_feature_importance()
        if importance:
            st.sidebar.write("### 🎯 주요 탐지 특성")
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, imp in sorted_importance:
                st.sidebar.write(f"**{feature}**: {imp:.3f}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 피싱 탐지 가이드
    
    **위험 신호:**
    - 매우 긴 URL (75자 이상)
    - 많은 하이픈과 점
    - IP 주소 직접 사용
    - 'secure', 'verify' 등 급박함을 조성하는 단어
    - 의심스러운 도메인 (.tk, .ml 등)
    
    **안전 신호:**
    - HTTPS 사용
    - 짧고 단순한 도메인
    - 신뢰할 수 있는 최상위 도메인
    
    ### 주의사항
    이 도구는 참고용입니다. 의심스러운 사이트에는 개인정보를 입력하지 마세요.
    """)

else:
    st.error("❌ 시스템 초기화에 실패했습니다. 페이지를 새로고침해주세요.")

# 푸터
st.markdown("---")
st.markdown("🛡️ **AI 피싱 탐지 시스템 v4** | 고도화된 AI로 더욱 정확한 피싱 탐지")