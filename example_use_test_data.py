# test_data.csv 사용 예시
import pandas as pd
from Model_Evaluation_Tool import ModelEvaluator

# 1. 테스트 데이터 로드
test_data = pd.read_csv('test_data.csv')

print("테스트 데이터 정보:")
print(f"크기: {test_data.shape[0]}행 x {test_data.shape[1]}열")
print(f"레이블 분포: {test_data['label'].value_counts().to_dict()}")

# 2. 모델 평가
model_file = 'phishing_detector.pkl'  # 실제 모델 파일명으로 변경
evaluator = ModelEvaluator(model_file)

# 3. 테스트 데이터 로드 및 평가
if evaluator.load_test_data(test_data):
    evaluator.report_summary()
    evaluator.evaluate_confusion_matrix()
    evaluator.evaluate_roc_curve()
else:
    print("테스트 데이터 로드 실패")
