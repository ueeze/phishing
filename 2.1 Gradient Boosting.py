import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime
import sys
import io
import warnings
warnings.filterwarnings('ignore')

class OptimizedPhishingDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.performance_metrics = {}

    def load_collected_data(self, data_dir=None):
        if data_dir is None:
            current_dir = os.getcwd()
            phishing_dirs = [d for d in os.listdir(current_dir) 
                             if d.startswith('phishing_data') and os.path.isdir(d)]
            if not phishing_dirs:
                print("X 수집된 데이터 폴더를 찾을 수 없습니다.")
                return False
            data_dir = max(phishing_dirs)
            print(f" 데이터 디렉토리: {data_dir}")

        try:
            url_file = os.path.join(data_dir, 'url_dataset_with_features.csv')
            if os.path.exists(url_file):
                self.url_data = pd.read_csv(url_file)
                print(f" URL 데이터 로드: {self.url_data.shape}")
            else:
                print("X URL 피처 데이터셋을 찾을 수 없습니다.")
                return False
            return True
        except Exception as e:
            print(f"X 데이터 로드 실패: {e}")
            return False

    def preprocess_data(self):
        print("\n 데이터 전처리 시작...")
        self.url_data = self.url_data.dropna().drop_duplicates(subset=['url'])

        exclude_columns = {
            'url', 'label', 'source', 'phish_id', 'phish_detail_url', 
            'submission_time', 'verified', 'online', 'collection_time',
            'url_status', 'threat', 'tags'
        }
        self.feature_columns = [col for col in self.url_data.columns 
                                if col not in exclude_columns and 
                                self.url_data[col].dtype in ['int64', 'float64', 'bool']]

        print(f"   선택된 피처: {len(self.feature_columns)}개")
        X = self.url_data[self.feature_columns].fillna(0)
        y = self.url_data['label']
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    def train_model(self, X, y):
        print("\n GradientBoosting 모델 학습 중...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        print(f"   정확도: {accuracy_score(y_test, y_pred):.4f}")
        print(f"   CV 평균 점수: {cv_scores.mean():.4f}")
        print(f"   F1 점수: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    def optimize_model(self, X, y):
        print("\n GradientBoosting 하이퍼파라미터 최적화...")
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 1.0]
        }

        grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"   최적 파라미터: {grid_search.best_params_}")
        print(f"   최적 점수: {grid_search.best_score_:.4f}")

    def evaluate_model(self):
        print("\n 최종 성능 평가...")
        y_pred = self.model.predict(self.X_test)

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

        print(f"   정확도: {accuracy:.4f}")
        print(f"   정밀도: {precision:.4f}")
        print(f"   재현율: {recall:.4f}")
        print(f"   F1 점수: {f1:.4f}")

    def save_model(self):
        filename = "phishing_detector_gb.pkl"
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics,
            'training_timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, filename)
        print(f"\n💾 모델 저장 완료: {filename}")

    def run_full_training(self, data_dir=None, optimize=True):
        print("AI 피싱 탐지 모델 학습 시작")
        print("=" * 50)
        if not self.load_collected_data(data_dir):
            return False

        X, y = self.preprocess_data()
        self.train_model(X, y)
        if optimize:
            self.optimize_model(X, y)
        self.evaluate_model()
        self.save_model()
        print("\n학습 완료!")
        return True

def main():
    detector = OptimizedPhishingDetector()
    detector.run_full_training(optimize=True)

if __name__ == "__main__":
    original_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    main()

    sys.stdout = original_stdout
    html_content = captured_output.getvalue().replace('\n', '<br>\n')
    with open("training_output_log2.html", "w", encoding="utf-8") as html_file:
        html_file.write("<html><body style='font-family:monospace;'>\n")
        html_file.write(html_content)
        html_file.write("\n</body></html>")
    print("✅ 터미널 출력이 'training_output_log2.html' 파일로 저장되었습니다.")

