import json
import re
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
# ======================
# 路径
# ======================
train_path = r"train.jsonl"
val_path = r"validation.jsonl"
test_path = r"test.jsonl"

# train_path = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\train.jsonl"
# val_path = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\validation.jsonl"
# test_path = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\test.jsonl"
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
# 情感词典 Lexicon
# ======================
positive_words = {
    "good", "great", "excellent", "amazing", "perfect",
    "love", "loved", "wonderful", "fantastic", "best",
    "nice", "happy", "satisfied", "beautiful", "awesome",
    "recommend", "recommended", "worth", "easy", "comfortable"
}

negative_words = {
    "bad", "terrible", "awful", "poor", "worst",
    "hate", "hated", "broken", "waste", "refund",
    "disappointed", "useless", "cheap", "problem", "problems",
    "defective", "damage", "damaged", "difficult", "uncomfortable"
}

negation_words = {
    "not", "no", "never", "none", "nothing",
    "n't", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't"
}


def extract_lexicon_features(text):
    """
    提取人工情感特征：
    1. positive word count
    2. negative word count
    3. negation word count
    4. exclamation mark count
    5. text length
    """

    text_lower = text.lower()
    words = re.findall(r"\b\w+\b|n't", text_lower)

    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    negation_count = sum(1 for w in words if w in negation_words)
    exclamation_count = text.count("!")
    text_length = len(words)

    # 简单归一化，防止数值过大
    features = np.array([
        pos_count / max(text_length, 1),
        neg_count / max(text_length, 1),
        negation_count / max(text_length, 1),
        min(exclamation_count, 5) / 5,
        min(text_length, 300) / 300
    ], dtype=np.float32)

    return features


# ======================
# Dataset
# ======================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.lexicon_features = [
            extract_lexicon_features(text)
            for text in texts
        ]

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
            "lexicon_features": torch.tensor(
                self.lexicon_features[idx],
                dtype=torch.float
            ),
            "label": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }


# ======================
# 模型：BERT + BiLSTM + Lexicon Features
# ======================
class BertBiLSTMLexicon(nn.Module):
    def __init__(self, num_classes, lexicon_dim=5):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_path)

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # BiLSTM 输出是 256 * 2 = 512
        # Lexicon features 是 5 维
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 + lexicon_dim, num_classes)

    def forward(self, input_ids, attention_mask, lexicon_features):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = bert_output.last_hidden_state

        lstm_output, _ = self.lstm(sequence_output)

        # 注意：这里仍然取最后一个时间步
        # 后续你也可以改成 Attention Pooling
        sentence_vector = lstm_output[:, -1, :]

        combined = torch.cat(
            [sentence_vector, lexicon_features],
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

model_path = "./bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_path)

batch_size = 16
max_length = 128

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


model = BertBiLSTMLexicon(
    num_classes=len(le.classes_),
    lexicon_dim=5
).to(device)


# ======================
# Class Weight
# ======================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)


# ======================
# 训练 + 验证
# ======================
best_f1 = 0.0
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lexicon_features = batch["lexicon_features"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lexicon_features=lexicon_features
        )

        loss = criterion(outputs, labels)

        loss.backward()
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
            lexicon_features = batch["lexicon_features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lexicon_features=lexicon_features
            )

            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    val_f1 = f1_score(
        labels_list,
        preds,
        average="weighted"
    )

    print(f"\nEpoch {epoch + 1}")
    print(f"训练 Loss: {avg_train_loss:.4f}")
    print(f"验证 Weighted F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_bert_bilstm_lexicon.pt")
        print("保存最佳模型")


# ======================
# 测试
# ======================
model.load_state_dict(
    torch.load(
        "best_bert_bilstm_lexicon.pt",
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
        lexicon_features = batch["lexicon_features"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lexicon_features=lexicon_features
        )

        pred = torch.argmax(outputs, dim=1)

        preds.extend(pred.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())


print("\n=== BERT + BiLSTM + Lexicon Features 测试集结果 ===")
print(
    classification_report(
        labels_list,
        preds,
        target_names=le.classes_,
        digits=4
    )
)




# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#
# from huggingface_hub import snapshot_download
#
# snapshot_download(
#     repo_id="bert-base-uncased",
#     local_dir="./bert-base-uncased",
#     local_dir_use_symlinks=False
# )


# %pip install transformers
# %pip install scikit-learn
# %pip install -U huggingface_hub