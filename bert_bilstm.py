import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ======================
# 路径
# ======================
train_path = r"train.jsonl"
val_path = r"validation.jsonl"
test_path = r"test.jsonl"

# ======================
# 数据清洗
# ======================
# def load_and_clean(path):
#     data = []
#     for line in open(path, encoding="utf-8"):
#         try:
#             obj = json.loads(line)
#             if "text" in obj and "sentiment" in obj:
#                 text = obj["text"].strip()
#                 if text:
#                     data.append(obj)
#         except:
#             continue
#     return pd.DataFrame(data)


def load_and_clean(path):
    data = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text" not in obj or "sentiment" not in obj:
                    bad += 1
                    continue
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

# ======================
# Dataset
# ======================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }

# ======================
# 模型
# ======================
class BertBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        return self.fc(out)

# ======================
# 训练
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_loader = DataLoader(TextDataset(train_df["text"].tolist(), y_train, tokenizer), batch_size=16, shuffle=True)
val_loader = DataLoader(TextDataset(val_df["text"].tolist(), y_val, tokenizer), batch_size=16)
test_loader = DataLoader(TextDataset(test_df["text"].tolist(), y_test, tokenizer), batch_size=16)

model = BertBiLSTM(len(le.classes_)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

best_f1 = 0

for epoch in range(3):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # ===== 验证 =====
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, mask)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    f1 = f1_score(labels_list, preds, average='weighted')
    print(f"\nEpoch {epoch+1} 验证 F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pt")

# ======================
# 测试
# ======================
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

preds, labels_list = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, mask)
        pred = torch.argmax(outputs, dim=1)

        preds.extend(pred.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

print("\n=== BERT + BiLSTM 测试集结果 ===")
print(classification_report(labels_list, preds, digits=4))