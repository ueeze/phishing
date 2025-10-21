# 피싱 탐지기에서 수집한 데이터로 평가용 test_data 생성
import pandas as pd
import os
from datetime import datetime

class TestDataGenerator:
    def __init__(self):
        self.base_dir = os.getcwd()
    
    def find_latest_data_dir(self):
        """가장 최근 생성된 phishing_data 폴더 찾기"""
        try:
            # 폴더만 필터링 (파일 제외)
            data_dirs = [d for d in os.listdir(self.base_dir) 
                        if d.startswith("phishing_data") and os.path.isdir(os.path.join(self.base_dir, d))]
            
            if not data_dirs:
                print("❌ phishing_data 폴더를 찾을 수 없습니다.")
                print("   먼저 setup_and_kaggle_downloader.py를 실행하세요.")
                return None
            
            latest_dir = sorted(data_dirs)[-1]
            print(f" 최신 데이터 폴더 발견: {latest_dir}")
            return latest_dir
        except Exception as e:
            print(f" 폴더 검색 실패: {e}")
            return None
    
    def load_feature_data(self, data_dir):
        """피처가 포함된 URL 데이터셋 로드"""
        feature_file = os.path.join(data_dir, 'url_dataset_with_features.csv')
        
        if not os.path.exists(feature_file):
            print(f" {feature_file} 파일이 없습니다.")
            
            # 대안 파일들 찾기
            alternative_files = [
                'phishtank_urls.csv',
                'openphish_urls.csv', 
                'legitimate_urls.csv',
                'email_samples.csv'
            ]
            
            print("\n 대안 파일들 확인 중...")
            for alt_file in alternative_files:
                alt_path = os.path.join(data_dir, alt_file)
                if os.path.exists(alt_path):
                    print(f"    {alt_file} 발견")
                    try:
                        df = pd.read_csv(alt_path)
                        print(f"      크기: {df.shape[0]}행 x {df.shape[1]}열")
                        print(f"      컬럼: {list(df.columns)}")
                        
                        # label 컬럼 확인 및 생성
                        if 'label' in df.columns:
                            print(f"      레이블 분포: {df['label'].value_counts().to_dict()}")
                        else:
                            # label 컬럼이 없으면 파일명 기반으로 생성
                            if 'phish' in alt_file.lower():
                                df['label'] = 'phishing'
                                print(f"      레이블 생성: 'phishing' ({len(df)}개)")
                            elif 'legitimate' in alt_file.lower():
                                df['label'] = 'legitimate'
                                print(f"      레이블 생성: 'legitimate' ({len(df)}개)")
                            else:
                                print(f"       레이블을 결정할 수 없습니다. 수동으로 설정 필요")
                        
                        return df, alt_file
                    except Exception as e:
                        print(f"       로드 실패: {e}")
            
            return None, None
        
        try:
            df = pd.read_csv(feature_file)
            print(f" 피처 데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
            print(f"   컬럼 개수: {len(df.columns)}")
            
            # 컬럼 정보 출력
            feature_cols = [col for col in df.columns if col not in ['url', 'label', 'source', 'phish_id', 'phish_detail_url', 'submission_time', 'verified', 'online', 'collection_time']]
            print(f"   피처 컬럼 수: {len(feature_cols)}")
            
            if 'label' in df.columns:
                print(f"   레이블 분포: {df['label'].value_counts().to_dict()}")
            
            return df, 'url_dataset_with_features.csv'
        except Exception as e:
            print(f" 피처 데이터 로드 실패: {e}")
            return None, None
    
    def create_balanced_test_data(self, df, total_samples=500, balance_ratio=None):
        """균형잡힌 테스트 데이터 생성"""
        if 'label' not in df.columns:
            print("❌ 'label' 컬럼이 없습니다.")
            print("   사용 가능한 컬럼:", list(df.columns))
            
            # 자동으로 레이블 컬럼 추가 시도
            if 'url' in df.columns:
                print("   ️ 'url' 컬럼만 있어서 모든 데이터를 'unknown'으로 레이블링합니다.")
                df['label'] = 'unknown'
            else:
                return None
        
        print(f"\n 균형잡힌 테스트 데이터 생성 중... (총 {total_samples}개)")
        
        # 레이블별 데이터 개수 확인
        label_counts = df['label'].value_counts()
        print(f"   원본 데이터 레이블 분포: {label_counts.to_dict()}")
        
        # 균형 비율 설정
        if balance_ratio is None:
            # 자동 균형 (각 레이블당 동일한 수)
            n_labels = len(label_counts)
            samples_per_label = total_samples // n_labels
            balance_ratio = {label: samples_per_label for label in label_counts.index}
            
            # 나머지 샘플들을 가장 많은 클래스에 추가
            remainder = total_samples % n_labels
            if remainder > 0:
                max_label = label_counts.index[0]
                balance_ratio[max_label] += remainder
        
        print(f"   목표 레이블 분포: {balance_ratio}")
        
        # 각 레이블별로 샘플링
        sampled_dfs = []
        for label, target_count in balance_ratio.items():
            label_df = df[df['label'] == label]
            
            if len(label_df) == 0:
                print(f"   ️ '{label}' 레이블 데이터가 없습니다.")
                continue
            
            if len(label_df) < target_count:
                print(f"    '{label}' 레이블: 요청 {target_count}개, 사용가능 {len(label_df)}개")
                sampled = label_df.sample(n=len(label_df), random_state=42)
            else:
                sampled = label_df.sample(n=target_count, random_state=42)
            
            sampled_dfs.append(sampled)
            print(f"    '{label}' 레이블: {len(sampled)}개 샘플링")
        
        # 결합 및 셔플
        if sampled_dfs:
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 셰플
            
            print(f"\n 최종 테스트 데이터: {len(result_df)}행")
            print(f"   최종 레이블 분포: {result_df['label'].value_counts().to_dict()}")
            
            return result_df
        else:
            print("❌ 샘플링된 데이터가 없습니다.")
            return None
    
    def save_test_data(self, test_df, filename="test_data.csv"):
        """테스트 데이터 저장"""
        try:
            test_df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\n 테스트 데이터 저장 완료: {filename}")
            print(f"   파일 크기: {os.path.getsize(filename):,} bytes")
            print(f"   저장 위치: {os.path.abspath(filename)}")
            
            # 저장된 파일 검증
            verify_df = pd.read_csv(filename)
            print(f"   검증: {verify_df.shape[0]}행 x {verify_df.shape[1]}열")
            
            return True
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            return False
    
    def generate_test_data(self, sample_size=500, output_filename="test_data.csv", custom_balance=None):
        """전체 테스트 데이터 생성 프로세스"""
        print(" 피싱 탐지 평가용 테스트 데이터 생성 시작")
        print("=" * 60)
        
        # 1. 최신 데이터 폴더 찾기
        latest_dir = self.find_latest_data_dir()
        if not latest_dir:
            return False
        
        # 2. 피처 데이터 로드
        df, source_file = self.load_feature_data(latest_dir)
        if df is None:
            return False
        
        print(f" 사용할 데이터 파일: {source_file}")
        
        # 3. 균형잡힌 테스트 데이터 생성
        test_df = self.create_balanced_test_data(df, total_samples=sample_size, balance_ratio=custom_balance)
        if test_df is None:
            return False
        
        # 4. 테스트 데이터 저장
        success = self.save_test_data(test_df, output_filename)
        
        if success:
            print(f"\n 테스트 데이터 생성 완료!")
            print(f" 다음 단계:")
            print(f"   1. {output_filename} 파일 확인")
            print(f"   2. Model_Evaluation_Tool.py에서 다음과 같이 사용:")
            print(f"      test_data = pd.read_csv('{output_filename}')")
            
            # 사용 예시 코드 생성
            self.generate_usage_example(output_filename)
            
        return success
    
    def generate_usage_example(self, test_filename):
        """사용 예시 코드 생성"""
        example_code = f'''# {test_filename} 사용 예시
import pandas as pd
from Model_Evaluation_Tool import ModelEvaluator

# 1. 테스트 데이터 로드
test_data = pd.read_csv('{test_filename}')

print("테스트 데이터 정보:")
print(f"크기: {{test_data.shape[0]}}행 x {{test_data.shape[1]}}열")
print(f"레이블 분포: {{test_data['label'].value_counts().to_dict()}}")

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
'''
        
        example_filename = f"example_use_{test_filename.replace('.csv', '')}.py"
        try:
            with open(example_filename, 'w', encoding='utf-8') as f:
                f.write(example_code)
            print(f"    사용 예시 파일 생성: {example_filename}")
        except Exception as e:
            print(f"   ️ 예시 파일 생성 실패: {e}")


def main():
    """메인 실행 함수"""
    generator = TestDataGenerator()
    
    print(" 테스트 데이터 생성 옵션:")
    print("1. 기본 설정 (500개 샘플, 균형잡힌 분포)")
    print("2. 사용자 정의 설정")
    
    try:
        choice = input("\n선택하세요 (1 또는 2, 기본값: 1): ").strip()
        
        if choice == "2":
            # 사용자 정의 설정
            sample_size = int(input("총 샘플 수 (기본값: 500): ") or "500")
            filename = input("출력 파일명 (기본값: test_data.csv): ") or "test_data.csv"
            
            # 사용자 정의 균형 비율 (고급 사용자용)
            print("\n고급 설정: 레이블별 샘플 수를 직접 지정하시겠습니까? (y/n)")
            advanced = input("기본값: n): ").strip().lower()
            
            custom_balance = None
            if advanced == 'y':
                print("레이블별 샘플 수를 입력하세요 (예: phishing:200,legitimate:300)")
                balance_input = input("입력: ").strip()
                if balance_input:
                    try:
                        custom_balance = {}
                        for item in balance_input.split(','):
                            label, count = item.split(':')
                            custom_balance[label.strip()] = int(count.strip())
                    except:
                        print(" 잘못된 형식입니다. 기본 균형을 사용합니다.")
                        custom_balance = None
            
            generator.generate_test_data(sample_size, filename, custom_balance)
        else:
            # 기본 설정
            generator.generate_test_data()
            
    except KeyboardInterrupt:
        print("\n\n 작업이 취소되었습니다.")
    except Exception as e:
        print(f"\n 오류 발생: {e}")


if __name__ == "__main__":
    main()