import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd
import joblib
import numpy as np
import matplotlib.font_manager as fm
import sys
import io
import datetime
import os
import shutil

class ModelEvaluator:
    def __init__(self, model_file):
        self.model_file = model_file
        self.model_data = joblib.load(model_file)

        # 구성요소 로드
        self.model = self.model_data['model']
        self.model_name = self.model_data['model_name']
        self.scaler = self.model_data['scaler']
        self.label_encoder = self.model_data['label_encoder']
        self.feature_columns = self.model_data['feature_columns']
        self.performance_metrics = self.model_data.get('performance_metrics', {})
        self.use_scaling = self.model_data['use_scaling']

        # 평가용 데이터는 외부에서 주입 필요
        self.X_test = None
        self.y_test = None
        self.y_test_encoded = None

    def load_test_data(self, test_df, label_column=None):
        """
        테스트 데이터 로드 (개선된 버전)
        
        Args:
            test_df: 테스트 데이터프레임
            label_column: 레이블 컬럼명 (None이면 자동 감지)
        """
        print("🔍 테스트 데이터 분석 중...")
        print(f"   데이터 크기: {test_df.shape[0]}행 x {test_df.shape[1]}열")
        print(f"   컬럼 목록: {list(test_df.columns)}")
        
        # 레이블 컬럼 찾기
        label_col = self._find_label_column(test_df, label_column)
        if label_col is None:
            print("❌ 레이블 컬럼을 찾을 수 없습니다.")
            return False
        
        print(f" 레이블 컬럼 발견: '{label_col}'")
        
        # 레이블 분포 확인
        if label_col in test_df.columns:
            label_counts = test_df[label_col].value_counts()
            print(f"   레이블 분포: {label_counts.to_dict()}")
        
        # 피처 컬럼 준비
        print(f"\n 피처 데이터 준비 중...")
        print(f"   모델 요구 피처: {self.feature_columns}")
        
        # 누락된 피처 컬럼 확인 및 추가
        missing_features = []
        for feature in self.feature_columns:
            if feature not in test_df.columns:
                missing_features.append(feature)
                test_df[feature] = 0  # 기본값으로 0 설정
        
        if missing_features:
            print(f"   ⚠ 누락된 피처 ({len(missing_features)}개): {missing_features}")
            print(f"      → 기본값 0으로 설정됨")
        
        # 피처 데이터 추출
        test_df_reindexed = test_df.reindex(columns=self.feature_columns, fill_value=0)
        self.X_test = test_df_reindexed[self.feature_columns].fillna(0)
        
        # 레이블 데이터 처리
        self.y_test = test_df[label_col]
        
        # 레이블 인코딩
        try:
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
            print(f" 레이블 인코딩 완료")
        except ValueError as e:
            print(f" 레이블 인코딩 실패: {e}")
            print(f"   모델이 학습한 레이블: {list(self.label_encoder.classes_)}")
            print(f"   테스트 데이터 레이블: {list(self.y_test.unique())}")
            
            # 알려지지 않은 레이블 처리
            unknown_labels = set(self.y_test.unique()) - set(self.label_encoder.classes_)
            if unknown_labels:
                print(f"   알려지지 않은 레이블: {unknown_labels}")
                print("   → 해당 레이블은 제거됩니다.")
                
                # 알려진 레이블만 필터링
                valid_mask = self.y_test.isin(self.label_encoder.classes_)
                self.X_test = self.X_test[valid_mask]
                self.y_test = self.y_test[valid_mask]
                self.y_test_encoded = self.label_encoder.transform(self.y_test)
                
                print(f"   최종 테스트 데이터: {len(self.X_test)}개")
        
        # 스케일링 적용
        if self.use_scaling:
            print(" 데이터 스케일링 적용 중...")
            self.X_test = self.scaler.transform(self.X_test)
            print(" 스케일링 완료")
        
        print(f"\n 최종 테스트 데이터:")
        print(f"   피처 데이터: {self.X_test.shape}")
        print(f"   레이블 데이터: {len(self.y_test)}개")
        print(f"   레이블 분포: {pd.Series(self.y_test).value_counts().to_dict()}")
        
        return True

    def _find_label_column(self, test_df, label_column=None):
        """레이블 컬럼 자동 감지"""
        if label_column and label_column in test_df.columns:
            return label_column
        
        # 가능한 레이블 컬럼명들
        possible_labels = ['label', 'target', 'class', 'y', 'category', 'type']
        
        for col in possible_labels:
            if col in test_df.columns:
                return col
        
        # 문자열 데이터가 있는 컬럼 찾기 (레이블일 가능성)
        for col in test_df.columns:
            if test_df[col].dtype == 'object':
                unique_vals = test_df[col].nunique()
                if 2 <= unique_vals <= 10:  # 2-10개의 고유값 (분류용)
                    print(f"    '{col}' 컬럼이 레이블로 보입니다 (고유값: {unique_vals}개)")
                    return col
        
        return None

    def evaluate_confusion_matrix(self):
        """혼동 행렬 시각화"""
        if self.X_test is None or self.y_test_encoded is None:
            print("테스트 데이터가 로드되지 않았습니다.")
            return

        print("\n 혼동 행렬 생성 중...")
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test_encoded, y_pred)
        labels = self.label_encoder.classes_

        # 한글 폰트 설정 
        font_path = "C:/Windows/Fonts/malgun.ttf"
        font_prop = fm.FontProperties(fname=font_path)

        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues')
        plt.title(f'혼동 행렬 - {self.model_name}', fontproperties=font_prop)  # ← 이모지 제거
        plt.grid(False)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'confusion_matrix_{self.model_name}.png')
        logger.track_image(f'confusion_matrix_{self.model_name}.png')

        # 혼동 행렬 출력
        print(" 혼동 행렬 결과:")
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                print(f"   실제 {true_label} → 예측 {pred_label}: {cm[i][j]}개")

    def evaluate_roc_curve(self):
        """ROC 곡선 시각화"""
        if self.X_test is None or self.y_test_encoded is None:
            print("❌ 테스트 데이터가 로드되지 않았습니다.")
            return
            
        if not hasattr(self.model, "predict_proba"):
            print("❌ 이 모델은 ROC Curve를 지원하지 않습니다.")
            return

        print("\n ROC 곡선 생성 중...")
        y_score = self.model.predict_proba(self.X_test)
        n_classes = y_score.shape[1]

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(self.y_test_encoded, y_score[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(f'roc_curve_{self.model_name}.png')
            logger.track_image(f'roc_curve_{self.model_name}.png')
            
            print(f"AUC Score: {roc_auc:.4f}")
        else:
            print("다중 클래스 ROC Curve는 아직 지원하지 않습니다.")

    def report_summary(self):
        """모델 성능 요약 보고서"""
        print("\n" + "="*50)
        print(" 모델 성능 요약 보고서")
        print("="*50)
        
        print(f" 모델명: {self.model_name}")
        print(f" 모델 타입: {type(self.model).__name__}")
        print(f" 스케일링 사용: {'예' if self.use_scaling else '아니오'}")
        print(f" 피처 개수: {len(self.feature_columns)}")
        print(f" 클래스: {list(self.label_encoder.classes_)}")
        
        if self.X_test is not None:
            print(f" 테스트 샘플 수: {len(self.X_test)}")
            
        print(f"\n 학습 시 성능 지표:")
        if self.performance_metrics:
            for key, value in self.performance_metrics.items():
                print(f"   {key}: {value:.4f}")
        else:
            print("   성능 지표가 저장되지 않았습니다.")
        
        print("="*50)

    def predict_single(self, url_features):
        """단일 URL 예측"""
        if isinstance(url_features, dict):
            # 딕셔너리 형태의 피처를 DataFrame으로 변환
            feature_df = pd.DataFrame([url_features])
        else:
            # 리스트나 배열 형태의 피처
            feature_df = pd.DataFrame([url_features], columns=self.feature_columns)
        
        # 누락된 피처 채우기
        for feature in self.feature_columns:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        # 피처 순서 맞추기
        X = feature_df[self.feature_columns].fillna(0)
        
        # 스케일링 적용
        if self.use_scaling:
            X = self.scaler.transform(X)
        
        # 예측
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0] if hasattr(self.model, 'predict_proba') else None
        
        # 결과 디코딩
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'prediction': predicted_label,
            'confidence': max(probability) if probability is not None else None,
            'probabilities': dict(zip(self.label_encoder.classes_, probability)) if probability is not None else None
        }
        
        return result


def main():
    """사용 예시"""
    print(" 모델 평가 도구 실행")
    
    # 모델 파일 확인
    import os
    model_file = 'phishing_detector.pkl'
    
    if not os.path.exists(model_file):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_file}")
        return
    
    # 테스트 데이터 파일 확인
    test_files = ['test_data.csv', 'phishing_data.csv', 'url_data.csv']
    test_file = None
    
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print(" 테스트 데이터 파일을 찾을 수 없습니다.")
        print("   다음 파일 중 하나를 준비하세요:", test_files)
        return
    
    try:
        # 모델 로드
        print(f" 모델 로드: {model_file}")
        evaluator = ModelEvaluator(model_file)
        
        # 테스트 데이터 로드
        print(f" 테스트 데이터 로드: {test_file}")
        test_data = pd.read_csv(test_file)
        
        # 평가 실행
        if evaluator.load_test_data(test_data):
            evaluator.report_summary()
            evaluator.evaluate_confusion_matrix()
            evaluator.evaluate_roc_curve()
        else:
            print(" 테스트 데이터 로드 실패")
            
    except Exception as e:
        print(f" 오류 발생: {e}")
        import traceback
        traceback.print_exc()


class HTMLLogger:
    def __init__(self, filename='model_evaluation_output.html', image_dir='output_images'):
        self.terminal = sys.stdout
        self.log = io.StringIO()
        self.filename = filename
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_files = []

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

    def track_image(self, filepath):
        """생성된 이미지 경로를 추적"""
        if os.path.exists(filepath):
            target_path = os.path.join(self.image_dir, os.path.basename(filepath))
            shutil.copy(filepath, target_path)
            self.image_files.append(target_path)

    def save_html(self):
        content = self.log.getvalue()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        img_tags = ""
        for img in self.image_files:
            img_tags += f'<h3>{os.path.basename(img)}</h3>\n<img src="{img}" style="max-width:100%; border:1px solid #ccc; margin-bottom:20px;"><br>\n'

        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>모델 평가 결과</title>
            <style>
                body {{ font-family: Consolas, monospace; background: #f9f9f9; padding: 20px; }}
                pre {{ background: #fff; padding: 20px; border: 1px solid #ccc; white-space: pre-wrap; word-wrap: break-word; }}
                img {{ display: block; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h2> 모델 평가 로그</h2>
            <p><strong>생성 시각:</strong> {timestamp}</p>
            <pre>{content}</pre>
            <hr>
            <h2> 생성된 그래프</h2>
            {img_tags}
        </body>
        </html>
        """
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.terminal.write(f"\n HTML 로그와 이미지가 저장되었습니다: {self.filename}\n")

# main 실행을 감싸서 자동 저장 처리
if __name__ == "__main__":
    logger = HTMLLogger()
    sys.stdout = logger  # stdout 가로채기
    try:
        main()
    finally:
        sys.stdout = logger.terminal  # 복원
        logger.save_html()

#if __name__ == "__main__":
#    main()