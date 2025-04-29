import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 파일 읽기
fifa_df = pd.read_excel('fifa2.xlsx')
print("✅ 파일 읽기 성공!")

# 2. 점수 숫자형 변환
fifa_df['score'] = pd.to_numeric(fifa_df['score'], errors='coerce')

# 3. 라벨링 (긍정 1, 부정 -1)
def label_score(score):
    if pd.isna(score):
        return None
    if score >= 4:
        return 1  # 긍정
    elif score <= 2:
        return -1  # 부정
    else:
        return None  # 중립 제거

fifa_df['label'] = fifa_df['score'].apply(label_score)

# 중립 제거
fifa_df = fifa_df.dropna(subset=['label'])
fifa_df['label'] = fifa_df['label'].astype(int)

print("\n💡 라벨 분포 (중립 제거 후):")
print(fifa_df['label'].value_counts().rename({1: '긍정', -1: '부정'}))

# 4. 텍스트 전처리 (깨진 글자 제거)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # 영어/숫자/공백만 남김
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS and len(word) > 1]
    return ' '.join(tokens)

fifa_df['content_clean'] = fifa_df['content'].fillna('').apply(clean_text)

# 5. 긍정 데이터 언더샘플링 (부정과 수 맞추기)
pos_df = fifa_df[fifa_df['label'] == 1].sample(34000, random_state=42)
neg_df = fifa_df[fifa_df['label'] == -1]

balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42)  # 섞기

print("\n🧹 언더샘플링 후 라벨 분포:")
print(balanced_df['label'].value_counts().rename({1: '긍정', -1: '부정'}))

# 6. 학습/테스트 분할
X = balanced_df['content_clean']
y = balanced_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. 텍스트 벡터화 (TF-IDF)
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 8. 모델 학습 (로지스틱 회귀)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 9. 예측 및 평가
y_pred = model.predict(X_test_tfidf)

print("\n✅ 모델 정확도:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\n📊 분류 상세 평가:")
print(classification_report(y_test, y_pred, target_names=['부정', '긍정']))

# 10. 최종 라벨링 데이터 저장
fifa_df[['content', 'score', 'label']].to_csv('fifa_labeled_binary.csv', index=False, encoding='utf-8-sig')
print("\n🎉 라벨링된 데이터가 'fifa_labeled_binary.csv'로 저장되었습니다!")
