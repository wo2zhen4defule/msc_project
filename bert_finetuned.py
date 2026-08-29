import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time

# 1. 数据准备
df = pd.read_csv(r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\sample.csv")
df['Text'] = df['Text'].fillna("")

le = LabelEncoder()
df['Sentiment_Label'] = le.fit_transform(df['Sentiment'])
target_names = le.classes_.tolist()

# 划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Text'].tolist(),
    df['Sentiment_Label'].tolist(),
    test_size=0.8,
    random_state=42,
    stratify=df['Sentiment_Label']
)



# 2. Dataset and DataLoader

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader = DataLoader(ReviewDataset(train_texts, train_labels, tokenizer), batch_size=4, shuffle=True)
val_loader = DataLoader(ReviewDataset(val_texts, val_labels, tokenizer), batch_size=4)


# 3. 初始化模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)


# 4. 训练循环 (1个 Epoch)
print(f"开始训练 (设备: {device})... need some time.")
model.train()
start_time = time.time()

for batch_idx, batch in enumerate(train_loader):
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    if (batch_idx + 1) % 10 == 0:
        print(f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

print(f"训练完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")


# 5. Assessment Module

print("\n正在验证集上评估模型性能...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n" + "=" * 20 + " BERT 分类报告 " + "=" * 20)
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))