import pandas as pd

# CSV 파일 경로
input_path = "fifa3.csv"
output_path = "fifa_cleaned.csv"

# CSV 불러오기
df = pd.read_csv(input_path, encoding="utf-8-sig")

# 컬럼 이름 정리
df.columns = df.columns.str.strip().str.lower()

# content와 score가 존재하는 행만 필터링
df = df.dropna(subset=["content", "score"])
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])

# score가 3점이 아닌 리뷰만 선택하고, content가 10글자 이상인 경우만 유지
df = df[(df["score"] != 3) & (df["content"].str.len() >= 10)]

# 감정 레이블 추가 (1~2점은 0: 부정, 4~5점은 1: 긍정)
df["as"] = df["score"].apply(lambda x: 0 if x <= 2 else 1)

# 저장
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {output_path}")
