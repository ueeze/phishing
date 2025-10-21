"""
AI 피싱 탐지 시스템용 데이터셋 수집기
PyCharm에서 실행 가능한 웹 스크래핑 코드

필요한 라이브러리 설치:
pip install requests beautifulsoup4 pandas lxml kaggle openpyxl tqdm
"""

import requests 
import pandas as pd
import json
import time
import os
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PhishingDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
    def create_output_directory(self):
        """출력 디렉토리 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"phishing_data"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def collect_phishtank_data(self):
        """PhishTank 공개 데이터 수집"""
        print("📊 PhishTank 데이터 수집 중...")
        try:
            # PhishTank 공개 데이터베이스 URL
            url = "http://data.phishtank.com/data/online-valid.json"
            
            print("   다운로드 시작...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"   ✅ {len(data)}개의 피싱 URL 수집 완료")
            
            # DataFrame으로 변환
            df = pd.DataFrame(data)
            
            # 필요한 컬럼만 선택
            if not df.empty:
                df = df[['phish_id', 'url', 'phish_detail_url', 'submission_time', 'verified', 'online']]
                df['label'] = 'phishing'
                df['source'] = 'phishtank'
                
            return df
            
        except Exception as e:
            print(f"   ❌ PhishTank 데이터 수집 실패: {e}")
            return pd.DataFrame()
    
    def collect_openphish_data(self):
        """OpenPhish 데이터 수집"""
        print("📊 OpenPhish 데이터 수집 중...")
        try:
            url = "https://openphish.com/feed.txt"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            urls = response.text.strip().split('\n')
            urls = [url.strip() for url in urls if url.strip()]
            
            print(f"   ✅ {len(urls)}개의 피싱 URL 수집 완료")
            
            # DataFrame으로 변환
            df = pd.DataFrame({
                'url': urls,
                'label': 'phishing',
                'source': 'openphish',
                'collection_time': datetime.now().isoformat()
            })
            
            return df
            
        except Exception as e:
            print(f"   ❌ OpenPhish 데이터 수집 실패: {e}")
            return pd.DataFrame()
    
    def collect_urlhaus_data(self):
        """URLhaus 멀웨어 URL 데이터 수집"""
        print("📊 URLhaus 데이터 수집 중...")
        try:
            url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # CSV 데이터 파싱
            from io import StringIO
            csv_data = StringIO(response.text)
            
            # 주석 라인 제거
            lines = [line for line in csv_data.getvalue().split('\n') if not line.startswith('#')]
            clean_csv = '\n'.join(lines)
            
            df = pd.read_csv(StringIO(clean_csv))
            
            if not df.empty:
                # 필요한 컬럼만 선택
                df = df[['url', 'url_status', 'threat', 'tags']]
                df['label'] = 'malicious'
                df['source'] = 'urlhaus'
                
            print(f"   ✅ {len(df)}개의 악성 URL 수집 완료")
            return df
            
        except Exception as e:
            print(f"   ❌ URLhaus 데이터 수집 실패: {e}")
            return pd.DataFrame()
    
    def collect_alexa_top_sites(self, count=1000):
        """정상 웹사이트 URL 수집 (대체 방법)"""
        print("📊 정상 웹사이트 URL 수집 중...")
        try:
            # 공개된 인기 웹사이트 리스트 활용
            normal_sites = [
                'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
                'linkedin.com', 'reddit.com', 'wikipedia.org', 'amazon.com', 'netflix.com',
                'microsoft.com', 'apple.com', 'github.com', 'stackoverflow.com', 'medium.com',
                'naver.com', 'daum.net', 'kakao.com', 'coupang.com', '11st.co.kr',
                'gmarket.co.kr', 'interpark.com', 'auction.co.kr', 'yes24.com', 'aladin.co.kr',
                'bbc.com', 'cnn.com', 'nytimes.com', 'guardian.com', 'reuters.com'
            ]
            
            # HTTP/HTTPS 버전 생성
            urls = []
            for site in normal_sites:
                urls.append(f'https://{site}')
                urls.append(f'https://www.{site}')
            
            df = pd.DataFrame({
                'url': urls,
                'label': 'legitimate',
                'source': 'known_good',
                'collection_time': datetime.now().isoformat()
            })
            
            print(f"   ✅ {len(df)}개의 정상 URL 생성 완료")
            return df
            
        except Exception as e:
            print(f"   ❌ 정상 URL 수집 실패: {e}")
            return pd.DataFrame()
    
    def extract_url_features(self, df):
        """URL에서 피처 추출"""
        print("🔍 URL 피처 추출 중...")
        
        def get_url_features(url):
            try:
                parsed = urlparse(url)
                return {
                    'url_length': len(url),
                    'domain_length': len(parsed.netloc), 
                    'path_length': len(parsed.path), 
                    'num_dots': url.count('.'),
                    'num_hyphens': url.count('-'),
                    'num_underscores': url.count('_'),
                    'num_percent': url.count('%'), 
                    'num_query_params': len(parsed.query.split('&')) if parsed.query else 0, 
                    'has_ip': any(c.isdigit() for c in parsed.netloc.split('.')) and len(parsed.netloc.split('.')) == 4,
                    'has_https': parsed.scheme == 'https',
                    'subdomain_count': len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0 
                }
            except:
                return {}
        
        # URL 피처 추출
        features_list = [] 
        for url in tqdm(df['url'], desc="URL 분석"):
            features = get_url_features(url)
            features_list.append(features)
        
        # 피처를 DataFrame에 추가
        features_df = pd.DataFrame(features_list)
        result_df = pd.concat([df, features_df], axis=1)
        
        print(f"   ✅ {len(features_df.columns)}개의 URL 피처 추출 완료")
        return result_df
    
    def collect_spam_email_samples(self):
        """스팸 이메일 샘플 데이터 생성"""
        print("📧 스팸 이메일 샘플 데이터 생성 중...")
        
        # 실제 스팸 이메일 패턴 기반 샘플 생성
        spam_samples = [
            {
                'subject': 'URGENT: Verify Your Account Now!',
                'body': 'Dear Customer, Your account will be suspended unless you verify immediately. Click here: http://suspicious-bank-verify.com',
                'sender': 'security@fake-bank.com',
                'label': 'spam'
            },
            {
                'subject': 'You Won $1,000,000!',
                'body': 'Congratulations! You have won our lottery. Send your personal details to claim your prize.',
                'sender': 'winner@lottery-scam.net',
                'label': 'spam'
            },
            {
                'subject': 'Free iPhone - Limited Time!',
                'body': 'Get your free iPhone now! Just click this link and enter your credit card details.',
                'sender': 'offers@freestuff-scam.com',
                'label': 'spam'
            }
        ]
        
        # 정상 이메일 샘플
        legitimate_samples = [
            {
                'subject': 'Meeting Reminder - Tomorrow 2PM',
                'body': 'Hi team, Just a reminder about our meeting tomorrow at 2PM in the conference room.',
                'sender': 'manager@company.com',
                'label': 'legitimate'
            },
            {
                'subject': 'Your Order Confirmation #12345',
                'body': 'Thank you for your order. Your items will be shipped within 2 business days.',
                'sender': 'orders@legitimate-store.com',
                'label': 'legitimate'
            }
        ]
        
        all_samples = spam_samples + legitimate_samples
        df = pd.DataFrame(all_samples)
        df['source'] = 'generated_samples'
        
        print(f"   ✅ {len(df)}개의 이메일 샘플 생성 완료")
        return df
    
    def save_datasets(self, datasets, output_dir):
        """데이터셋 저장"""
        print(f"💾 데이터셋 저장 중... (경로: {output_dir})")
        
        for name, df in datasets.items():
            if not df.empty:
                # CSV 파일로 저장
                csv_path = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8')
                
                # Excel 파일로 저장
                excel_path = os.path.join(output_dir, f"{name}.xlsx")
                df.to_excel(excel_path, index=False)
                
                print(f"   ✅ {name}: {len(df)}행 저장 완료")
            else:
                print(f"   ⚠️ {name}: 데이터 없음")
    
    def run_collection(self):
        """전체 데이터 수집 프로세스 실행"""
        print("🚀 피싱 탐지 데이터셋 수집 시작")
        print("=" * 50)
        
        # 출력 디렉토리 생성
        output_dir = self.create_output_directory()
        
        datasets = {}
        
        # 1. PhishTank 데이터 수집
        phishtank_df = self.collect_phishtank_data()
        if not phishtank_df.empty:
            datasets['phishtank_urls'] = phishtank_df
        
        # 2. OpenPhish 데이터 수집
        openphish_df = self.collect_openphish_data()
        if not openphish_df.empty:
            datasets['openphish_urls'] = openphish_df
        
        # 3. URLhaus 데이터 수집
        urlhaus_df = self.collect_urlhaus_data()
        if not urlhaus_df.empty:
            datasets['urlhaus_malicious'] = urlhaus_df
        
        # 4. 정상 웹사이트 URL 수집
        legitimate_df = self.collect_alexa_top_sites()
        if not legitimate_df.empty:
            datasets['legitimate_urls'] = legitimate_df
        
        # 5. 스팸 이메일 샘플 생성
        email_df = self.collect_spam_email_samples()
        if not email_df.empty:
            datasets['email_samples'] = email_df
        
        # 6. URL 데이터 통합 및 피처 추출
        if 'phishtank_urls' in datasets or 'openphish_urls' in datasets:
            print("\n🔄 URL 데이터셋 통합 중...")
            combined_urls = []
            
            if 'phishtank_urls' in datasets:
                combined_urls.append(datasets['phishtank_urls'][['url', 'label', 'source']])
            if 'openphish_urls' in datasets:
                combined_urls.append(datasets['openphish_urls'][['url', 'label', 'source']])
            if 'legitimate_urls' in datasets:
                combined_urls.append(datasets['legitimate_urls'][['url', 'label', 'source']])
            
            if combined_urls:
                combined_df = pd.concat(combined_urls, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['url'])
                
                # URL 피처 추출
                featured_df = self.extract_url_features(combined_df)
                datasets['url_dataset_with_features'] = featured_df
        
        # 7. 데이터셋 저장
        if datasets:
            self.save_datasets(datasets, output_dir)
            
            # 요약 정보 출력
            print("\n📊 수집 완료 요약:")
            print("-" * 30)
            total_records = 0
            for name, df in datasets.items():
                count = len(df)
                total_records += count
                print(f"  {name}: {count:,}개")
            
            print(f"\n총 {total_records:,}개의 레코드 수집 완료!")
            print(f"저장 위치: {os.path.abspath(output_dir)}")
            
            # 데이터셋 사용 예시 코드 생성
            self.generate_usage_example(output_dir)
            
        else:
            print("❌ 수집된 데이터가 없습니다.")
        
        return output_dir
    
    def generate_usage_example(self, output_dir):
        """수집된 데이터 사용 예시 코드 생성"""
        example_code = '''
# 수집된 데이터셋 사용 예시
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
url_data = pd.read_csv('url_dataset_with_features.csv')
email_data = pd.read_csv('email_samples.csv')

# 기본 통계 확인
print("URL 데이터셋 정보:")
print(url_data.info())
print("\\n레이블 분포:")
print(url_data['label'].value_counts())

# 피처 분석
numerical_features = ['url_length', 'domain_length', 'num_dots', 'num_hyphens']
for feature in numerical_features:
    if feature in url_data.columns:
        print(f"\\n{feature} - 피싱 vs 정상:")
        print(url_data.groupby('label')[feature].describe())

# 시각화
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features[:4], 1):
    if feature in url_data.columns:
        plt.subplot(2, 2, i)
        sns.boxplot(data=url_data, x='label', y=feature)
        plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.show()

# 머신러닝 준비
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 피처와 타겟 분리
feature_columns = [col for col in url_data.columns 
                  if col not in ['url', 'label', 'source', 'phish_id', 
                               'phish_detail_url', 'submission_time', 
                               'verified', 'online', 'collection_time']]

X = url_data[feature_columns].fillna(0)
y = url_data['label']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = rf_model.predict(X_test)
print("\\n분류 성능:")
print(classification_report(y_test, y_pred))

# 중요 피처 확인
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n중요 피처 Top 10:")
print(feature_importance.head(10))
'''
        
        example_path = os.path.join(output_dir, "usage_example.py")
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        print(f"   📝 사용 예시 코드 생성: {example_path}")


def main():
    """메인 실행 함수"""
    collector = PhishingDataCollector()
    output_dir = collector.run_collection()
 
    print(f"\n🎉 모든 작업 완료!")
    print(f"📁 결과 확인: {os.path.abspath(output_dir)}")
    print("\n다음 단계:")
    print("1. 생성된 CSV/Excel 파일들을 확인하세요")
    print("2. usage_example.py를 실행하여 데이터 분석을 시작하세요")
    print("3. 필요에 따라 추가 데이터를 수집하세요")

if __name__ == "__main__":
    main()