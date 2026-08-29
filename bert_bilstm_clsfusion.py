import json
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm


# ======================
# 路径
# ======================
train_path = r"train.jsonl"
val_path = r"validation.jsonl"
test_path = r"test.jsonl"

model_path = "./bert-base-uncased"


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

                text = str(obj.get("text", "")).strip()
                sentiment = str(obj.get("sentiment", "")).strip()

                if not text or not sentiment:
                    bad += 1
                    continue

                data.append({
                    "text": text,
                    "sentiment": sentiment
                })

            except Exception as e:
                bad += 1
                if bad <= 5:
                    print("错误原因:", e)
                    print("错误行:", line[:300])

    print(f"{path} 加载完成: {len(data)} 条, 过滤: {bad}")
    return pd.DataFrame(data, columns=["text", "sentiment"])


train_df = load_and_clean(train_path)
val_df = load_and_clean(val_path)
test_df = load_and_clean(test_path)


# ======================
# 标签编码
# ======================
le = LabelEncoder()

y_train = le.fit_transform(train_df["sentiment"])
y_val = le.transform(val_df["sentiment"])
y_test = le.transform(test_df["sentiment"])

print("标签类别:", list(le.classes_))


# ======================
# Dataset
# ======================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ======================
# 模型：BERT + BiLSTM + Attention + CLS Fusion
# ======================
class BertBiLSTMAttentionCLSFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_path)

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.3)

        # attention_vector = 512
        # CLS vector = 768
        self.fc = nn.Linear(512 + 768, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = bert_output.last_hidden_state

        # CLS token representation
        cls_output = sequence_output[:, 0, :]

        lstm_output, _ = self.lstm(sequence_output)

        attn_scores = self.attention(lstm_output).squeeze(-1)

        attn_scores = attn_scores.masked_fill(
            attention_mask == 0,
            -1e9
        )

        attn_weights = torch.softmax(attn_scores, dim=1)

        attention_vector = torch.sum(
            lstm_output * attn_weights.unsqueeze(-1),
            dim=1
        )

        combined = torch.cat(
            [attention_vector, cls_output],
            dim=1
        )

        combined = self.dropout(combined)

        logits = self.fc(combined)

        return logits


# ======================
# 训练准备
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)

tokenizer = BertTokenizer.from_pretrained(model_path)

batch_size = 16
max_length = 128
epochs = 3

train_dataset = TextDataset(
    train_df["text"].tolist(),
    y_train,
    tokenizer,
    max_length=max_length
)

val_dataset = TextDataset(
    val_df["text"].tolist(),
    y_val,
    tokenizer,
    max_length=max_length
)

test_dataset = TextDataset(
    test_df["text"].tolist(),
    y_test,
    tokenizer,
    max_length=max_length
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


model = BertBiLSTMAttentionCLSFusion(
    num_classes=len(le.classes_)
).to(device)


# ======================
# Loss：不使用 class_weight
# 目标是提升 Accuracy
# ======================
criterion = nn.CrossEntropyLoss()


# ======================
# Optimizer：BERT 用较小学习率
# ======================
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": 1e-5},
            {"params": model.lstm.parameters(), "lr": 2e-5},
            {"params": model.attention.parameters(), "lr": 2e-5},
            {"params": model.fc.parameters(), "lr": 2e-5}
        ],
        weight_decay=0.01
    )


# ======================
# 训练 + 验证
# ======================
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ======================
    # 验证
    # ======================
    model.eval()

    preds = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    val_acc = accuracy_score(labels_list, preds)
    val_f1 = f1_score(labels_list, preds, average="weighted")

    print(f"\nEpoch {epoch + 1}")
    print(f"训练 Loss: {avg_train_loss:.4f}")
    print(f"验证 Accuracy: {val_acc:.4f}")
    print(f"验证 Weighted F1: {val_f1:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            model.state_dict(),
            "best_bert_bilstm_attention_cls.pt"
        )
        print("保存最佳模型")


# ======================
# 测试
# ======================
model.load_state_dict(
    torch.load(
        "best_bert_bilstm_attention_cls.pt",
        map_location=device
    )
)

model.eval()

preds = []
labels_list = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pred = torch.argmax(outputs, dim=1)

        preds.extend(pred.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())


print("\n=== BERT + BiLSTM + Attention + CLS Fusion 测试集结果 ===")
print(
    classification_report(
        labels_list,
        preds,
        target_names=le.classes_,
        digits=4
    )
)