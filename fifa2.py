import pandas as pd

# 데이터 로드
try:
    fifa_df = pd.read_csv('fifa1.csv', encoding='utf-8-sig', low_memory=False)
    print("✅ utf-8-sig 인코딩으로 파일 읽기 성공!")
except UnicodeDecodeError:
    fifa_df = pd.read_csv('fifa1.csv', encoding='latin1', low_memory=False)
    print("✅ latin1 인코딩으로 파일 읽기 성공!")

# 점수 문자열을 숫자형으로 변환
fifa_df['score'] = pd.to_numeric(fifa_df['score'], errors='coerce')

# 날짜 데이터 변환
fifa_df['at'] = pd.to_datetime(fifa_df['at'], errors='coerce')

# 부정 리뷰 필터링 (점수 <= 2)
negative_reviews = fifa_df[fifa_df['score'] <= 2]

# 연-월 단위로 그룹화
negative_reviews['year_month'] = negative_reviews['at'].dt.to_period('M')

# 월별 부정 리뷰 수 집계
result = negative_reviews.groupby('year_month').size().reset_index(name='부정 리뷰 수')

# 결과 확인
print("\n📊 년/월별 부정 리뷰 수 TOP:")
print(result.sort_values('부정 리뷰 수', ascending=False).head(10))

# CSV 저장
result.to_csv('negative_review_trend.csv', index=False, encoding='utf-8-sig')
print("\n✅ 분석 결과가 'negative_review_trend.csv'로 저장되었습니다!")
