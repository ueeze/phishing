# 7. new_streamlit_app.py — 세션 안전 버전 (이모지 없음)

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

# ==========================
# URL 정규화 및 특성 추출
# ==========================
TRUSTED_DOMAINS = [
    "google.com", "naver.com", "daum.net", "kakao.com", "youtube.com",
    "facebook.com", "twitter.com", "instagram.com", "amazon.com"
]
SUSPICIOUS_TLDS = [".tk", ".ml", ".ga", ".cf", ".click", ".download"]

def normalize_url(user_input: str):
    s = (user_input or "").strip()
    if not s:
        return "", 0, 0

    has_scheme = s.startswith(("http://", "https://"))
    domain_part = s.split("/")[0].lower() if not has_scheme else urlparse(s).netloc.lower()
    is_trusted = 1 if any(td in domain_part for td in TRUSTED_DOMAINS) else 0

    if not has_scheme:
        if is_trusted:
            normalized = "https://" + s
            https_inferred = 1
        else:
            normalized = "http://" + s
            https_inferred = 0
    else:
        normalized = s
        https_inferred = 0

    return normalized, https_inferred, is_trusted

def extract_url_features(user_input: str):
    try:
        normalized_url, https_inferred, is_trusted = normalize_url(user_input)
        if not normalized_url:
            return None, None, "빈 URL 입니다."

        parsed = urlparse(normalized_url)
        domain = parsed.netloc.lower()
        original = user_input.strip()

        features = {
            "url_length": len(original),
            "domain_length": len(domain),
            "num_dots": original.count("."),
            "num_hyphens": original.count("-"),
            "num_params": len(parsed.query.split("&")) if parsed.query else 0,
            "path_length": len(parsed.path),
            "num_digits": len(re.findall(r"\d", original)),
            "has_https": 1 if parsed.scheme == "https" else 0,
            "num_subdomains": len(domain.split(".")) - 2 if len(domain.split(".")) > 2 else 0,
            "has_suspicious_words": 1 if any(w in original.lower() for w in
                                             ["secure", "account", "update", "confirm", "verify", "login"]) else 0,
            "has_ip": 1 if re.match(r"^\d+\.\d+\.\d+\.\d+", domain) else 0,
            "suspicious_tld": 1 if any(original.lower().endswith(tld) for tld in SUSPICIOUS_TLDS) else 0,
            "is_trusted_domain": is_trusted,
            "https_inferred": https_inferred,
        }
        return features, normalized_url, None
    except Exception as e:
        return None, None, str(e)

def check_domain_reputation(domain: str):
    try:
        socket.gethostbyname(domain)
        return "도메인이 존재합니다"
    except:
        return "도메인을 찾을 수 없습니다"

# ===========================================
# URL용 간단 모델: 외부가 없으면 폴백 학습
# ===========================================
class UnifiedPhishingDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            "url_length","domain_length","num_dots","num_hyphens",
            "has_https","num_subdomains","has_suspicious_words",
            "num_digits","num_params","path_length","has_ip","suspicious_tld"
        ]
        self.model_loaded = False

    def load_external_model(self, model_path="phishing_detector.pkl"):
        try:
            if os.path.exists(model_path):
                data = joblib.load(model_path)
                if isinstance(data, dict):
                    self.model  = data.get("model", None)
                    self.scaler = data.get("scaler", None)
                    if data.get("feature_columns"):
                        self.feature_names = data["feature_columns"]
                else:
                    self.model = data
                self.model_loaded = True
                return True, "외부 모델"
            return False, None
        except Exception:
            return False, None

    def create_enhanced_sample_data(self, n=3000):
        np.random.seed(42)
        data = []

        for _ in range(n//2):  # Legitimate
            data.append({
                "url_length": np.random.randint(10, 50),
                "domain_length": np.random.randint(5, 20),
                "num_dots": np.random.randint(1, 3),
                "num_hyphens": np.random.randint(0, 2),
                "has_https": 1,
                "num_subdomains": np.random.randint(0, 2),
                "has_suspicious_words": 0,
                "num_digits": np.random.randint(0, 3),
                "num_params": np.random.randint(0, 2),
                "path_length": np.random.randint(0, 20),
                "has_ip": 0,
                "suspicious_tld": 0,
                "label": 0
            })

        for _ in range(n//2):  # Phishing
            data.append({
                "url_length": np.random.randint(50, 150),
                "domain_length": np.random.randint(20, 60),
                "num_dots": np.random.randint(3, 8),
                "num_hyphens": np.random.randint(2, 10),
                "has_https": np.random.choice([0,1], p=[0.6,0.4]),
                "num_subdomains": np.random.randint(2, 6),
                "has_suspicious_words": np.random.choice([0,1], p=[0.2,0.8]),
                "num_digits": np.random.randint(3, 15),
                "num_params": np.random.randint(2, 8),
                "path_length": np.random.randint(15, 80),
                "has_ip": np.random.choice([0,1], p=[0.7,0.3]),
                "suspicious_tld": np.random.choice([0,1], p=[0.6,0.4]),
                "label": 1
            })
        return pd.DataFrame(data)

    def train_fallback_model(self):
        df = self.create_enhanced_sample_data()
        X = df[self.feature_names]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=15,
            min_samples_split=3, min_samples_leaf=2, random_state=42
        )
        self.model.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test_s))
        return acc

    def predict_from_features(self, features: dict):
        if self.model is None:
            return None
        vec = [features.get(f, 0) for f in self.feature_names]
        X = np.array([vec], dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return {
            "prediction": "Phishing" if pred == 1 else "Legitimate",
            "phishing_probability": float(proba[1]) if len(proba) > 1 else float(1 - proba[0]),
            "confidence": float(np.max(proba))
        }

    def get_feature_importance(self):
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return None
        imp = self.model.feature_importances_
        return dict(zip(self.feature_names, imp))

# ======================
# 이메일 분석(간이)
# ======================
try:
    from textblob import TextBlob
    from spellchecker import SpellChecker
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# --- AIEmailAnalyzer: 통합 모델(pkl) 사용 시도 + 폴백 ---
class AIEmailAnalyzer:
    def __init__(self, model_path='integrated_phishing_detector.pkl'):
        self.ai_loaded = False
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.model_info = {"name": "규칙 기반", "version": "N/A"}

        # 통합 pkl 로딩 시도 (dict 또는 단일 모델 모두 허용)
        try:
            if os.path.exists(model_path):
                obj = joblib.load(model_path)
                if isinstance(obj, dict):
                    # 통합 trainer가 저장한 키들이 있다면 우선 사용
                    self.model = obj.get('email_model') or obj.get('model')
                    self.vectorizer = obj.get('email_vectorizer')
                    self.scaler = obj.get('scaler')
                    self.model_info = {
                        "name": obj.get('name', 'IntegratedDetector'),
                        "version": obj.get('version', 'unknown')
                    }
                else:
                    # 모델만 저장된 경우: 벡터라이저 없이 있으면 규칙 기반으로 폴백
                    self.model = obj
                    self.model_info = {"name": "IntegratedDetector", "version": "unknown"}

                # 모델이 실제로 이메일 텍스트를 처리할 수 있는지 최소 보장
                self.ai_loaded = self.model is not None
        except Exception:
            self.ai_loaded = False

    def _featurize_text_basic(self, subject, body, sender):
        # 벡터라이저가 없을 때 사용할 간단 피처(폴백)
        text = f"{subject or ''} {body or ''}"
        feats = {
            "len_subject": len(subject or ""),
            "len_body": len(body or ""),
            "num_links": text.count("http://") + text.count("https://"),
            "num_exclaim": text.count("!"),
            "has_suspicious_words": int(any(w in text.lower() for w in
                ["urgent","verify","confirm","password","click","winner","lottery","update","account","login"]))
        }
        return np.array([[feats[k] for k in feats]])

    def predict(self, subject, body, sender):
        # 1) 통합 모델 + 벡터라이저가 있으면 TF-IDF 변환
        if self.ai_loaded and self.vectorizer is not None:
            text = f"{subject or ''}\n{body or ''}"
            X = self.vectorizer.transform([text])
            if self.scaler is not None:
                X = self.scaler.transform(X)
            proba = self.model.predict_proba(X)[0]
            pred = self.model.predict(X)[0]
            return {
                "prediction": "Phishing" if int(pred) == 1 else "Legitimate",
                "phishing_probability": float(proba[1]) if len(proba) > 1 else float(1 - proba[0]),
                "confidence": float(np.max(proba))
            }

        # 2) 통합 모델만 있고 벡터라이저가 없으면 간단 피처로 시도(안 되면 폴백)
        if self.ai_loaded and self.vectorizer is None:
            try:
                X = self._featurize_text_basic(subject, body, sender)
                if self.scaler is not None:
                    X = self.scaler.transform(X)
                proba = self.model.predict_proba(X)[0]
                pred = self.model.predict(X)[0]
                return {
                    "prediction": "Phishing" if int(pred) == 1 else "Legitimate",
                    "phishing_probability": float(proba[1]) if len(proba) > 1 else float(1 - proba[0]),
                    "confidence": float(np.max(proba))
                }
            except Exception:
                pass  # 아래 규칙 기반으로 폴백

        # 3) 규칙 기반 폴백
        text = f"{subject or ''} {body or ''}".lower()
        score = sum(w in text for w in ["urgent","verify","confirm","password","click","winner","lottery","update","account","login"])
        label = 'Phishing' if score >= 2 else 'Legitimate'
        prob = min(0.2 + 0.15*score, 0.95) if label == 'Phishing' else max(0.8 - 0.1*score, 0.05)
        return {"prediction": label, "phishing_probability": prob, "confidence": max(prob, 1-prob)}
    

# ====== 여기 '바로 아래'에 추가 ======
# Email risk-factor explainer (for the Email tab)
EMAIL_SUS_KEYWORDS = [
    "urgent","verify","confirm","password","click","winner","lottery",
    "update","account","login","security","limited","suspend","invoice",
    "payment","reset","unlock","credential"
]
EMAIL_URGENCY_PATTERNS = [
    "urgent","immediately","within 24 hours","action required",
    "verify now","limited time","suspended","deactivated","deadline"
]
EMAIL_CRED_PATTERNS = [
    "password","otp","one-time code","credit card","ssn","security code",
    "account verification","confirm your identity","login to update"
]
EMAIL_BRANDS = [
    "paypal","bank","apple","amazon","microsoft","google","instagram","facebook","netflix"
]

def explain_email_risks(subject: str, body: str, sender: str):
    text = f"{subject or ''}\n{body or ''}"
    text_l = text.lower()
    factors = []

    matched_kw = sorted({kw for kw in EMAIL_SUS_KEYWORDS if kw in text_l})
    if matched_kw:
        factors.append(f"Suspicious keywords present: {', '.join(matched_kw[:8])}")

    links = re.findall(r'https?://[^\s)>\"]+', text_l)
    if links:
        factors.append(f"Contains {len(links)} link(s)")
    # HTTP/HTTPS 구분 경고는 이메일 위험요소에서 제외
    ip_links = re.findall(r'https?://(?:\d{1,3}\.){3}\d{1,3}', text_l)
    if ip_links:
        factors.append("Link uses direct IP address")

    exclaims = text.count("!")
    if exclaims >= 3:
        factors.append(f"Excessive exclamation marks ({exclaims})")

    sender_domain = ""
    m = re.search(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', sender or "")
    if m:
        sender_domain = m.group(1).lower()
    brands_mentioned = [b for b in EMAIL_BRANDS if b in text_l]
    if brands_mentioned and sender_domain and not any(b in sender_domain for b in brands_mentioned):
        factors.append(
            f"Sender domain '{sender_domain}' does not match claimed brand ({', '.join(sorted(set(brands_mentioned)))})"
        )

    urg = [p for p in EMAIL_URGENCY_PATTERNS if p in text_l]
    if urg:
        factors.append(f"Urgency language: {', '.join(urg)}")

    cred = [p for p in EMAIL_CRED_PATTERNS if p in text_l]
    if cred:
        factors.append(f"Requests sensitive info: {', '.join(cred[:6])}")

    if not factors:
        factors.append("No obvious phishing indicators found.")
    return factors
# ====== 추가 끝 ======


# --- 시스템 초기화: URL도 통합 pkl 먼저 시도 ---
@st.cache_resource
def initialize_systems():
    url_system = UnifiedPhishingDetectionSystem()
    # 통합 모델 파일명으로 교체
    ok, url_model_type = url_system.load_external_model('integrated_phishing_detector.pkl')
    if not ok:
        with st.spinner("내장 URL 모델을 학습 중입니다..."):
            acc = url_system.train_fallback_model()
        url_model_type = f"내장 URL 모델 (검증 정확도 {acc:.1%})"

    email_ai = AIEmailAnalyzer(model_path='integrated_phishing_detector.pkl')
    email_model_type = "AI 모델" if email_ai.ai_loaded else "규칙 기반"
    return url_system, url_model_type, email_ai, email_model_type


# ======================
# 시스템 초기화 (캐시)
# ======================
@st.cache_resource
def initialize_systems():
    url_system = UnifiedPhishingDetectionSystem()
    ok, url_model_type = url_system.load_external_model("phishing_detector.pkl")
    if not ok:
        with st.spinner("내장 URL 모델을 학습 중입니다..."):
            acc = url_system.train_fallback_model()
        url_model_type = f"내장 URL 모델 (검증 정확도 {acc:.1%})"
    email_ai = AIEmailAnalyzer()
    email_model_type = "규칙 기반" if not email_ai.ai_loaded else "AI 모델"
    return url_system, url_model_type, email_ai, email_model_type

# =================
# 메인 UI
# =================
st.title("AI 기반 피싱 탐지 시스템")
st.write("URL과 이메일을 종합 분석하여 피싱 위험을 탐지합니다.")

if "systems" not in st.session_state:
    url_system, url_model_type, email_ai, email_model_type = initialize_systems()
    st.session_state.systems = {
        "url_system": url_system,
        "url_model_type": url_model_type,
        "email_ai": email_ai,
        "email_model_type": email_model_type,
    }
else:
    s = st.session_state.systems
    url_system, url_model_type, email_ai, email_model_type = (
        s["url_system"], s["url_model_type"], s["email_ai"], s["email_model_type"]
    )

if url_system and email_ai:
    tab1, tab2 = st.tabs(["URL 분석", "이메일 분석"])

    # ---------------- URL 분석 탭 ----------------
    with tab1:
        st.header("실시간 URL 피싱 탐지")

        # 1) 빠른 테스트 버튼을 먼저 렌더링하고, 클릭 시 세션 상태를 갱신
        st.subheader("빠른 테스트")
        col1, col2, col3, col4 = st.columns(4)
        test_urls = {
            "정상 사이트": "https://www.google.com",
            "의심 사이트 1": "http://secure-bank-verify-account-12345.suspicious-domain.tk/login?confirm=true&urgent=yes",
            "의심 사이트 2": "https://192.168.1.100/paypal-security-update-urgent-verify-now.html?user=12345",
            "의심 사이트 3": "http://www-amazon-security-update-confirm-account.verify-login-details.ml/secure/update",
        }
        for i, (label, url) in enumerate(test_urls.items()):
            col = [col1, col2, col3, col4][i]
            with col:
                if st.button(label, key=f"url_test_{i}"):
                    # 텍스트 입력 위젯이 생성되기 전에 값을 세팅해야 함
                    st.session_state["url_input"] = url
                    st.rerun()  # ← experimental_rerun() 대신 rerun()

        # 2) 그 다음에 입력창을 생성하면, 위에서 세팅한 값이 초기값으로 들어감
        url_input = st.text_input(
            "URL을 입력하세요:",
            key="url_input",
            placeholder="예: https://www.google.com 또는 suspicious-site.com",
            help="http:// 또는 https://를 포함하여 입력하거나, 도메인만 입력해도 됩니다.",
        )

        if st.session_state.get("url_input"):
            st.info(f"선택된 URL: {st.session_state['url_input']}")

        # 실행/초기화 버튼을 같은 줄에 배치
        btn_col1, btn_col2 = st.columns([1, 1])

        def _reset_all():
            # ← 입력창까지 완전히 비우는 핵심
            st.session_state["url_input"] = ""
            st.session_state.pop("selected_url", None)
            # 분석 결과에 쓰던 임시 값들도 쓰신 게 있다면 같이 제거
            for k in ("normalized_url", "prediction_result", "domain_status"):
                st.session_state.pop(k, None)
        
        with btn_col1:
            analyze_clicked = st.button("URL 분석 실행", type="primary", key="analyze_btn")
        with btn_col2:
            st.button("입력/결과 초기화", key="reset_btn", on_click=_reset_all)  # ← on_click 사용

        # 분석 실행
        if analyze_clicked:
            url_to_analyze = (st.session_state.get("url_input") or "").strip()
            if not url_to_analyze:
                st.warning("URL을 입력하세요.")
            else:
                with st.spinner("URL을 분석중입니다."):
                    features, normalized_url, error = extract_url_features(url_to_analyze)
                    if error:
                        st.error(f"URL 분석 중 오류: {error}")
                    else:
                        if normalized_url and normalized_url != url_to_analyze:
                            st.info(f"분석된 URL: {normalized_url}")

                        parsed = urlparse(normalized_url)
                        domain_status = check_domain_reputation(parsed.netloc)

                        result = url_system.predict_from_features(features)
                        if result:
                            st.subheader("URL 분석 결과")

                            is_trusted = features.get("is_trusted_domain", 0)
                            has_https  = features.get("has_https", 0)
                            https_inferred = features.get("https_inferred", 0)

                            if result["prediction"] == "Phishing":
                                st.error("피싱 사이트로 의심됩니다.")
                                st.error(f"피싱 확률: {result['phishing_probability']:.1%}")
                                st.warning("이 사이트에 개인정보를 입력하지 마세요.")
                            else:
                                if is_trusted and has_https == 1:
                                    st.success("신뢰할 수 있는 안전한 사이트입니다.")
                                elif is_trusted and has_https == 0:
                                    st.warning("신뢰할 수 있는 도메인이지만, HTTPS가 없어 안전하지 않을 수 있습니다.")
                                elif has_https == 0:
                                    st.warning("HTTPS가 없어 안전하지 않을 수 있습니다.")
                                else:
                                    st.success("비교적 안전한 사이트입니다.")
                                st.success(f"안전 확률: {1 - result['phishing_probability']:.1%}")

                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("신뢰도", f"{result['confidence']:.1%}")
                            with c2:
                                st.info(domain_status)

                            risk_factors = []
                            if features["url_length"] > 75:
                                risk_factors.append(f"URL이 매우 깁니다 ({features['url_length']}자)")
                            if features["domain_length"] > 30:
                                risk_factors.append(f"도메인이 비정상적으로 깁니다 ({features['domain_length']}자)")
                            if features["num_dots"] > 5:
                                risk_factors.append(f"서브도메인이 많습니다 (점 {features['num_dots']}개)")
                            if features["num_hyphens"] > 4:
                                risk_factors.append(f"하이픈이 많습니다 ({features['num_hyphens']}개)")
                            if features["has_suspicious_words"] == 1:
                                risk_factors.append("의심스러운 단어가 포함되어 있습니다")
                            if features["has_ip"] == 1:
                                risk_factors.append("IP 주소를 직접 사용합니다")
                            if features["suspicious_tld"] == 1:
                                risk_factors.append("의심스러운 최상위 도메인을 사용합니다")

                            if risk_factors:
                                st.subheader("발견된 위험 요소")
                                for rf in risk_factors:
                                    st.write(f"- {rf}")
                            else:
                                if is_trusted:
                                    st.success("신뢰할 수 있는 도메인이며 별도의 위험 요소가 발견되지 않았습니다.")
                                else:
                                    st.success("특별한 위험 요소가 발견되지 않았습니다.")

                            if https_inferred == 1 and has_https == 1:
                                st.info("보안 팁: 접속 시 https:// 를 포함해 입력하면 더 안전합니다.")

                            with st.expander("상세 특성 정보"):
                                table = pd.DataFrame([{
                                    "URL 길이": features["url_length"],
                                    "도메인 길이": features["domain_length"],
                                    "점(.) 개수": features["num_dots"],
                                    "하이픈(-) 개수": features["num_hyphens"],
                                    "HTTPS 사용": "예" if has_https else "아니오",
                                    "서브도메인 개수": features["num_subdomains"],
                                    "의심 단어 포함": "예" if features["has_suspicious_words"] else "아니오",
                                    "숫자 개수": features["num_digits"],
                                    "매개변수 개수": features["num_params"],
                                    "경로 길이": features["path_length"],
                                    "IP 주소 사용": "예" if features["has_ip"] else "아니오",
                                    "의심스러운 TLD": "예" if features["suspicious_tld"] else "아니오",
                                    "신뢰 도메인": "예" if is_trusted else "아니오",
                                }])
                                st.dataframe(table, use_container_width=True, hide_index=True)

    # ---------------- 이메일 분석 탭 ----------------
    # ---------------- 이메일 분석 탭 ----------------
    with tab2:
        st.header("이메일 분석")

        # 2-1) 빠른 테스트 (입력 위젯보다 먼저)
        st.subheader("빠른 테스트")
        ec1, ec2, ec3, ec4 = st.columns(4)
        test_emails = {
            "정상 이메일": {
                "subject": "Team Meeting Reminder – Tomorrow at 2 PM",
                "sender": "manager@company.com",
                "body": (
                    "Hi team,\n\n"
                    "This is a reminder that our quarterly planning meeting is scheduled for tomorrow at 2 PM "
                    "in the main conference room. Please bring your latest status updates.\n\n"
                    "Best regards,\n"
                    "Operations"
                )
            },
            "의심 이메일 1": {
                "subject": "URGENT: Verify your bank account",
                "sender": "security@fake-bank.com",
                "body": (
                    "Urgent notice: Your bank account will be suspended within 24 hours unless you verify your identity.\n"
                    "Click the link below to confirm your password and update your account details:\n"
                    "http://verify-secure-login.bank-alerts.tk/confirm\n\n"
                    "Failure to confirm may result in permanent account closure."       
                )
            },
            "의심 이메일 2": {
                "subject": "You are our lucky winner!",
                "sender": "rewards@promo-gift.net",
                "body": (
                    "Congratulations! You are our lottery winner this month.\n"
                    "Click here to claim your prize. To complete the process, login and provide your account information.\n"
                    "Offer expires in 24 hours."
                )
            },
            "의심 이메일 3": {
                "subject": "PayPal Security Update – Action Required",
                "sender": "notice@paypal-security-support.co",
                "body": (
                    "Security update required for your PayPal account. We couldn't verify your recent activity.\n"
                    "To restore access, update your password and confirm your details at:\n"
                    "http://paypal-security-update-login.co/verify\n"
                    "If you do not login and verify, your account may be limited."
                )
            }
        }
        for i, (label, data) in enumerate(test_emails.items()):
            col = [ec1, ec2, ec3, ec4][i]
            with col:
                if st.button(label, key=f"email_test_{i}"):
                    st.session_state["email_subject"] = data["subject"]
                    st.session_state["email_sender"] = data["sender"]
                    st.session_state["email_body"] = data["body"]
                    st.rerun()

        # 2-2) 입력 위젯 (세션 키 사용)
        subj = st.text_input("제목", key="email_subject")
        send = st.text_input("발신자 이메일(선택)", key="email_sender")
        body = st.text_area("본문", key="email_body", height=200)

        # 2-3) 실행/초기화 버튼
        ebc1, ebc2 = st.columns([1, 1])
        with ebc1:
            email_analyze = st.button("이메일 분석 실행", key="email_analyze_btn")
        def _reset_email():
            st.session_state["email_subject"] = ""
            st.session_state["email_sender"] = ""
            st.session_state["email_body"] = ""
        with ebc2:
            st.button("입력 초기화", key="email_reset_btn", on_click=_reset_email)

        # 2-4) 분석 로직
        if email_analyze:
            res = email_ai.predict(subj, body, send)
            if res["prediction"] == "Phishing":
                st.error("피싱 이메일로 의심됩니다.")
                st.error(f"피싱 확률: {res['phishing_probability']:.1%}")
            else:
                st.success("비교적 정상적인 이메일로 보입니다.")
                st.success(f"정상 확률: {1 - res['phishing_probability']:.1%}")
            st.metric("신뢰도", f"{res['confidence']:.1%}")

            # ====== 여기부터 '위험 요소' 블록 추가 ======
            factors = explain_email_risks(subj, body, send)
            st.subheader("발견된 위험 요소")
            for f in factors:
                st.write(f"- {f}")

            with st.expander("이메일 다시보기 (raw)"):
                st.text(f"From: {send or '(unknown)'}")
                st.text(f"Subject: {subj or ''}")
                st.write(body or "")
            # ====== 추가 끝 ======


        # 사이드바
        st.sidebar.title("시스템 정보")
        st.sidebar.success(f"URL 분석: {url_model_type}")
        st.sidebar.success(f"이메일 분석: {email_model_type}")
        if url_system:
            imp = url_system.get_feature_importance()
            if imp:
                st.sidebar.write("주요 탐지 특성")
                for f, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.sidebar.write(f"- {f}: {v:.3f}")

else:
    st.error("시스템 초기화에 실패했습니다. 페이지를 새로고침하세요.")
