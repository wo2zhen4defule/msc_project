import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# ======================
# 路径
# ======================
train_path = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\train.jsonl"
val_path = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\validation.jsonl"
test_path = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\test.jsonl"

# ======================
# 数据清洗
# ======================


def load_and_clean(path):
    data = []
    bad = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)

                # 必须有字段
                if "text" not in obj or "sentiment" not in obj:
                    bad += 1
                    continue

                # 去空文本
                text = str(obj["text"]).strip()
                if not text:
                    bad += 1
                    continue

                data.append({
                    "text": text,
                    "sentiment": obj["sentiment"]
                })

            except:
                bad += 1

    print(f"{path} 加载完成: {len(data)} 条, 过滤: {bad}")
    return pd.DataFrame(data)

train_df = load_and_clean(train_path)
val_df = load_and_clean(val_path)
test_df = load_and_clean(test_path)

# ======================
# 标签
# ======================
le = LabelEncoder()
y_train = le.fit_transform(train_df["sentiment"])
y_val = le.transform(val_df["sentiment"])
y_test = le.transform(test_df["sentiment"])

X_train = train_df["text"].tolist()
X_val = val_df["text"].tolist()
X_test = test_df["text"].tolist()



# ======================
# BERT
# ======================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def get_embeddings(texts):
    embs = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embs.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embs)

print("提取特征...")
X_train_emb = get_embeddings(X_train)
X_val_emb = get_embeddings(X_val)
X_test_emb = get_embeddings(X_test)

# ======================
# SVM + 调参
# ======================
best_model = None
best_f1 = 0

for C in [0.1, 1, 5]:
    svm = LinearSVC(C=C)
    svm.fit(X_train_emb, y_train)

    y_val_pred = svm.predict(X_val_emb)

    print(f"\n=== 验证集 C={C} ===")
    print(classification_report(y_val, y_val_pred, digits=4))

    f1 = f1_score(y_val, y_val_pred, average='weighted')

    if f1 > best_f1:
        best_f1 = f1
        best_model = svm

# ======================
# 测试
# ======================
y_pred = best_model.predict(X_test_emb)

print("\n=== BERT + SVM 测试集结果 ===")
print(classification_report(y_test, y_pred, digits=4))