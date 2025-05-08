import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 한글 폰트 설정 (경고 방지용)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 로딩
try:
    fifa_df = pd.read_excel('fifa2.xlsx')
    print("✅ 엑셀 파일 로드 성공!")
except Exception as e:
    print(f"❌ 파일 로드 실패: {e}")
    exit()

# 2. 텍스트 전처리 함수
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

fifa_df['clean_content'] = fifa_df['content'].fillna('').apply(clean_text)

# 3. 감성 레이블링
positive_keywords = ['good', 'great', 'fun', 'awesome', 'love', 'best', 'amazing', 'cool']
negative_keywords = ['bad', 'worst', 'bug', 'error', 'hate', 'boring', 'trash', 'crash']

def label_sentiment(text):
    if any(word in text for word in positive_keywords):
        return 'positive'
    elif any(word in text for word in negative_keywords):
        return 'negative'
    else:
        return 'neutral'

fifa_df['label'] = fifa_df['clean_content'].apply(label_sentiment)
fifa_df = fifa_df[fifa_df['label'].isin(['positive', 'negative'])]  # 중립 제거

# 4. 모델 학습
X = fifa_df['clean_content']
y = fifa_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. 평가
y_pred = model.predict(X_test_tfidf)
print("\n🎯 정확도:", accuracy_score(y_test, y_pred))
print("\n📊 분류 성능:\n", classification_report(y_test, y_pred))

# 6. 날짜 처리 및 예측 적용
fifa_df['at'] = pd.to_datetime(fifa_df['at'], errors='coerce')
fifa_df = fifa_df.dropna(subset=['at'])  # 날짜가 없는 행 제거
fifa_df['year_month'] = fifa_df['at'].dt.to_period('M')
fifa_df['predicted_label'] = model.predict(vectorizer.transform(fifa_df['clean_content']))

# 7. 부정 리뷰 월별 분석
monthly_negative = fifa_df[fifa_df['predicted_label'] == 'negative'].groupby('year_month').size()
max_month = monthly_negative.idxmax()

print(f"\n📅 부정 리뷰가 가장 많은 월: {max_month}")
print(monthly_negative)

# 8. 시각화
plt.figure(figsize=(12, 5))
monthly_negative.plot(kind='bar', color='crimson')
plt.title('월별 부정 리뷰 수')
plt.xlabel('월')
plt.ylabel('부정 리뷰 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. 저장
fifa_df.to_csv('fifa_labeled.csv', index=False, encoding='utf-8-sig')
print("\n✅ 결과 저장: 'fifa_labeled.csv'")
