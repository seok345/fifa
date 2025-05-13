import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 로딩
data_path = "fifa_cleaned.csv"  # 전처리된 FIFA 리뷰 CSV (예: 10자 이상, 중립점수 제거 등)
df = pd.read_csv(data_path, encoding="utf-8-sig")

# 감정 라벨 생성 (1~2점 -> 0, 4~5점 -> 1) / 3점 제거
df = df[df['score'] != 3]
df['Sentiment'] = df['score'].apply(lambda x: 0 if x <= 2 else 1)

# 입력 텍스트와 라벨 분리
data_X = df['content'].astype(str).tolist()
labels = df['Sentiment'].astype(int).values

print("리뷰 샘플 개수:", len(data_X))
print("감성 라벨 샘플:", labels[:5])

# 토크나이저 로딩 및 토큰화
tokenizer = MobileBertTokenizer.from_pretrained("mobilebert_fifa_model")
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("✅ 토큰화 완료")

# DataLoader 생성
batch_size = 8
test_inputs = input_ids
test_labels = torch.tensor(labels).long()
test_masks = attention_mask
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = torch.utils.data.SequentialSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
print("✅ DataLoader 구축 완료")

# 모델 로딩
model = MobileBertForSequenceClassification.from_pretrained("mobilebert_fifa_model")
model.to(device)
model.eval()

# 예측 수행
test_pred = []
test_true = []

for batch in tqdm(test_dataloader, desc="Evaluating"):
    batch_ids, batch_mask, batch_labels = [b.to(device) for b in batch]

    with torch.no_grad():
        outputs = model(batch_ids, attention_mask=batch_mask)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)

    test_pred.extend(preds.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

# 정확도 계산
test_accuracy = np.mean(np.array(test_pred) == np.array(test_true))
print(f"\n🎯 전체 FIFA 리뷰 데이터에 대한 분류 정확도: {test_accuracy:.4f}")
