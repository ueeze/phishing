import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import sys
import io
import time
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore')

class OptimizedPhishingDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.best_model = None
        self.best_model_name = ""
        self.performance_metrics = {}
        
    def load_collected_data(self, data_dir=None):
        """첫 번째 스크립트로 수집된 데이터 로드"""
        if data_dir is None:
            # 가장 최근 생성된 phishing_data 폴더 찾기
            current_dir = os.getcwd()
            phishing_dirs = [d for d in os.listdir(current_dir) 
                            if d.startswith('phishing_data') and os.path.isdir(d)]
            if not phishing_dirs:
                print("X 수집된 데이터 폴더를 찾을 수 없습니다.")
                print(f"현재 디렉토리: {current_dir}")
                print(f"사용 가능한 폴더들: {[d for d in os.listdir(current_dir) if os.path.isdir(d)]}")
                return False
            # 가장 최근 폴더 선택 (날짜 기준)
            data_dir = max(phishing_dirs)
                
            print(f" 데이터 디렉토리: {data_dir}")
            
            try:
                # URL 데이터셋 로드
                url_file = os.path.join(data_dir, 'url_dataset_with_features.csv')
                if os.path.exists(url_file):
                    self.url_data = pd.read_csv(url_file)
                    print(f" URL 데이터 로드: {self.url_data.shape}")
                else:
                    print("X URL 피처 데이터셋을 찾을 수 없습니다.")
                    return False
                
                # 이메일 데이터셋 로드 (선택사항)
                email_file = os.path.join(data_dir, 'email_samples.csv')
                if os.path.exists(email_file):
                    self.email_data = pd.read_csv(email_file)
                    print(f" 이메일 데이터 로드: {self.email_data.shape}")
                
                return True
                
            except Exception as e:
                print(f"X 데이터 로드 실패: {e}")
                return False
    
    def preprocess_data(self):
        """데이터 전처리 및 피처 준비"""
        print("\n 데이터 전처리 시작...")
        
        # 결측치 및 중복 제거
        initial_size = len(self.url_data)
        self.url_data = self.url_data.dropna()
        self.url_data = self.url_data.drop_duplicates(subset=['url'])
        
        print(f"   전처리 후: {len(self.url_data)}개 ({initial_size - len(self.url_data)}개 제거)")
        
        # 피처 컬럼 선택
        exclude_columns = {
            'url', 'label', 'source', 'phish_id', 'phish_detail_url', 
            'submission_time', 'verified', 'online', 'collection_time',
            'url_status', 'threat', 'tags'
        }
        
        self.feature_columns = [col for col in self.url_data.columns 
                               if col not in exclude_columns and 
                               self.url_data[col].dtype in ['int64', 'float64', 'bool']]
        
        print(f"   선택된 피처: {len(self.feature_columns)}개")
        print(f"   피처 목록: {self.feature_columns}")
        
        # 피처와 타겟 분리
        X = self.url_data[self.feature_columns].fillna(0)
        y = self.url_data['label']
        
        # 레이블 인코딩
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 클래스 분포 확인
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"   클래스 분포: {dict(zip(unique_labels, counts))}")
        
        return X, y_encoded
    
    def train_models(self, X, y):
        """다중 모델 학습 및 평가"""
        print("\n 모델 학습 시작...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 정의
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'use_scaling': False
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'use_scaling': False
            },
            'XGBoost': {  # 추가
                'model': XGBClassifier(random_state=42, 
                                       # use_label_encoder=False, 
                                       eval_metric='logloss'),
                'use_scaling': False
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=20000),
                'use_scaling': True
            },
            'NeuralNetwork': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
                'use_scaling': True
            }
        }
        
        best_score = 0
        
        # 명시적 학습 순서 지정
        model_order = ['GradientBoosting', 'RandomForest', 'NeuralNetwork', 'LogisticRegression']
        
        for name, config in models_config.items():
            print(f"\n   {name} 학습 중...")
            
            model = config['model']
            use_scaling = config['use_scaling']
            
            # 모델 학습
            if use_scaling:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # 성능 메트릭 계산
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
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
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"      정확도: {accuracy:.4f}")
            print(f"      CV 점수: {cv_mean:.4f} (±{cv_std:.4f})")
            print(f"      F1 점수: {f1:.4f}")
            
            # 최고 성능 모델 업데이트
            if cv_mean > best_score:
                best_score = cv_mean
                self.best_model_name = name
                self.best_model = model
        
        # 테스트 데이터 저장 (예측용)
        self.X_test = X_test
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        
        print(f"\n 최고 성능 모델: {self.best_model_name}")
        print(f"   CV 점수: {self.models[self.best_model_name]['cv_mean']:.4f}")
        
        return self.models
    
    def optimize_best_model(self):
        """최고 성능 모델 하이퍼파라미터 최적화"""
        print(f"\n {self.best_model_name} 하이퍼파라미터 최적화...")
        
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 모델별 하이퍼파라미터 그리드
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8, 1.0]
            },
            'XGBoost': {
                'n_estimators': [100, 300, 500,700],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'gamma': [0, 1],
                'reg_lambda': [1, 5],  # L2 정규화
                'reg_alpha': [0, 1]    # L1 정규화
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'NeuralNetwork': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            base_model = self.models[self.best_model_name]['model']
            use_scaling = self.models[self.best_model_name]['use_scaling']
            
            # GridSearchCV 실행
            grid_search = GridSearchCV(
                base_model.__class__(**base_model.get_params()),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2
            )
            
            if use_scaling:
                X_train_scaled = self.scaler.fit_transform(X_train)
                grid_search.fit(X_train_scaled, y_train)
            else:
                grid_search.fit(X_train, y_train)
            
            # 최적화된 모델로 업데이트
            self.best_model = grid_search.best_estimator_
            self.models[self.best_model_name]['model'] = self.best_model
            
            print(f"   최적 파라미터: {grid_search.best_params_}")
            print(f"   최적 점수: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        
        return self.best_model
    
    def evaluate_final_model(self):
        """최종 모델 성능 평가"""
        print(f"\n {self.best_model_name} 최종 성능 평가...")
        
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
        
        print(f"   정확도: {accuracy:.4f}")
        print(f"   정밀도: {precision:.4f}")
        print(f"   재현율: {recall:.4f}")
        print(f"   F1 점수: {f1:.4f}")
        
        # 클래스별 성능
        print("\n   클래스별 성능:")
        report = classification_report(self.y_test, y_pred, 
                                     target_names=self.label_encoder.classes_,
                                     output_dict=True)
        
        for class_name in self.label_encoder.classes_:
            if class_name in report:
                metrics = report[class_name]
                print(f"   {class_name}:")
                print(f"     정밀도: {metrics['precision']:.4f}")
                print(f"     재현율: {metrics['recall']:.4f}")
                print(f"     F1 점수: {metrics['f1-score']:.4f}")
        
        return self.performance_metrics
    
    def save_model(self, filename=None):
        """모델 및 관련 데이터 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phishing_detector.pkl"
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics,
            'use_scaling': self.models[self.best_model_name]['use_scaling'],
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filename)
        print(f"\n💾 모델 저장 완료: {filename}")
        print(f"   모델: {self.best_model_name}")
        print(f"   성능: {self.performance_metrics.get('accuracy', 0):.4f}")
        
        return filename
    
    def load_model(self, filename):
        """저장된 모델 로드"""
        try:
            model_data = joblib.load(filename)
            
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.performance_metrics = model_data.get('performance_metrics', {})
            
            print(f"✅ 모델 로드 완료: {filename}")
            print(f"   모델: {self.best_model_name}")
            
            return True
            
        except Exception as e:
            print(f" 모델 로드 실패: {e}")
            return False
    
    def predict_single_url(self, url_features):
        """단일 URL 예측"""
        if self.best_model is None:
            print(" 학습된 모델이 없습니다.")
            return None
        
        try:
            # 피처 DataFrame 생성
            features_df = pd.DataFrame([url_features])
            features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
            
            # 예측 실행
            use_scaling = self.models.get(self.best_model_name, {}).get('use_scaling', False)
            
            if use_scaling:
                features_scaled = self.scaler.transform(features_df)
                prediction = self.best_model.predict(features_scaled)[0]
                probability = self.best_model.predict_proba(features_scaled)[0]
            else:
                prediction = self.best_model.predict(features_df)[0]
                probability = self.best_model.predict_proba(features_df)[0]
            
            # 결과 변환
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probability)
            
            # 피싱 확률 계산 (클래스 순서에 따라)
            phishing_prob = probability[1] if len(probability) > 1 else (
                probability[0] if self.label_encoder.classes_[0] == 'phishing' else 1 - probability[0]
            )
            
            result = {
                'prediction': predicted_label,
                'confidence': confidence,
                'phishing_probability': phishing_prob,
                'legitimate_probability': 1 - phishing_prob
            }
            
            return result
            
        except Exception as e:
            print(f" 예측 실패: {e}")
            return None
    
    def batch_predict(self, urls_features):
        """배치 URL 예측"""
        if self.best_model is None:
            print(" 학습된 모델이 없습니다.")
            return None
        
        try:
            # 피처 DataFrame 생성
            features_df = pd.DataFrame(urls_features)
            features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
            
            # 예측 실행
            use_scaling = self.models.get(self.best_model_name, {}).get('use_scaling', False)
            
            if use_scaling:
                features_scaled = self.scaler.transform(features_df)
                predictions = self.best_model.predict(features_scaled)
                probabilities = self.best_model.predict_proba(features_scaled)
            else:
                predictions = self.best_model.predict(features_df)
                probabilities = self.best_model.predict_proba(features_df)
            
            # 결과 변환
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predicted_labels, probabilities)):
                confidence = max(prob)
                phishing_prob = prob[1] if len(prob) > 1 else (
                    prob[0] if self.label_encoder.classes_[0] == 'phishing' else 1 - prob[0]
                )
                
                results.append({
                    'prediction': pred,
                    'confidence': confidence,
                    'phishing_probability': phishing_prob,
                    'legitimate_probability': 1 - phishing_prob
                })
            
            return results
            
        except Exception as e:
            print(f" 배치 예측 실패: {e}")
            return None
    
    def run_full_training(self, data_dir=None, optimize=True):

        start_time = time.time()  # 이 줄이 없으면 NameError 발생

        # 전체 학습 파이프라인 실행
        print(" AI 피싱 탐지 모델 학습 시작")
        print("=" * 50)
           
        # 1. 데이터 로드
        if not self.load_collected_data(data_dir):
            return False
            
        # 2. 데이터 전처리
        X, y = self.preprocess_data()
        
        # 3. 모델 학습
        self.train_models(X, y)

        # 자꾸만 선형회귀를 선택하길래 XGBoost를 강제로 선택
        self.best_model_name = 'XGBoost'
        self.best_model = self.models['XGBoost']['model']

        # 4. 하이퍼파라미터 최적화 (선택사항)
        if optimize:
            self.optimize_best_model()
            
        # 5. 최종 평가
        self.evaluate_final_model()
            
        # 6. 모델 저장
        model_file = self.save_model()

        end_time = time.time()
        duration = end_time - start_time
            
        print("\n 🎉 학습 완료!")
        print(f" 📁 모델 파일: {model_file}")
        print(f" ⏰ 총 학습 시간: {duration:.2f}초")
        print(f" 🕓 완료 시각: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

            
        return model_file


def main():
    """메인 실행 함수"""
    detector = OptimizedPhishingDetector()
    
    # 전체 학습 실행
    model_file = detector.run_full_training(optimize=True)
    
    if model_file:
        print(f"\n✨ 사용 가능한 모델: {model_file}")
        
        # 예측 예시
        print("\n 예측 테스트:")
        sample_features = {
            'url_length': 85,
            'domain_length': 15,
            'num_dots': 3,
            'num_hyphens': 1,
            'num_underscores': 0,
            'num_percent': 0,
            'num_query_params': 2,
            'has_ip': False,
            'has_https': True,
            'subdomain_count': 2
        }
        
        result = detector.predict_single_url(sample_features)
        if result:
            print(f"   예측: {result['prediction']}")
            print(f"   신뢰도: {result['confidence']:.3f}")
            print(f"   피싱 확률: {result['phishing_probability']:.3f}")

"""
import joblib
model_data = joblib.load("phishing_detector.pkl")
print("최종 모델:", model_data['model_name'])
"""

if __name__ == "__main__":

    # 출력 캡처를 위한 스트림 생성
    original_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # main 실행
    main()

    # 캡처된 출력 내용을 HTML로 저장
    sys.stdout = original_stdout
    html_content = captured_output.getvalue().replace('\n', '<br>\n')

    with open("training_output_log.html", "w", encoding="utf-8") as html_file:
        html_file.write("<html><body style='font-family:monospace;'>\n")
        html_file.write(html_content)
        html_file.write("\n</body></html>")

    print("✅ 터미널 출력이 'training_output_log.html' 파일로 저장되었습니다.")


#if __name__ == "__main__":
#    main()