import json
import random

input_file = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\reviews_with_sentiment.jsonl"

train_file = "../train.jsonl"
val_file = "../validation.jsonl"
test_file = "../test.jsonl"

# 读取所有数据
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# 打乱数据
random.shuffle(data)

total = len(data)
print(total)
train_end = int(total * 0.7)
val_end = int(total * 0.8)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# 保存函数
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(train_file, train_data)
save_jsonl(val_file, val_data)
save_jsonl(test_file, test_data)

print("数据划分完成")
print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))