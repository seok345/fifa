import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 데이터 로드
try:
    fifa_df = pd.read_csv('fifa1.csv', encoding='utf-8-sig', low_memory=False)
    print("✅ utf-8-sig 인코딩으로 파일 읽기 성공!")
except UnicodeDecodeError:
    fifa_df = pd.read_csv('fifa1.csv', encoding='latin1', low_memory=False)
    print("✅ latin1 인코딩으로 파일 읽기 성공!")

# 🧠 전처리 함수: 특수문자 제거 + 소문자 통일 + 불용어 제거
def clean_text(text):
    text = str(text).lower()                      # 소문자 변환
    text = re.sub(r'[^a-z\s]', '', text)          # 영어 제외 특수문자 제거
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # 불용어 제거
    return text

# content 컬럼 전처리 적용
fifa_df['clean_content'] = fifa_df['content'].fillna('').apply(clean_text)

# 전처리된 샘플 확인
print(fifa_df[['content', 'clean_content']].head(10))

# 전처리된 파일 저장 (옵션)
fifa_df.to_csv('fifa_preprocessed.csv', index=False, encoding='utf-8-sig')
print("\n✅ 전처리 완료! 'fifa_preprocessed.csv'로 저장되었습니다.")
