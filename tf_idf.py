import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
import json
# 过滤
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


# 读取训练集
df_train =  load_and_clean(r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\train.jsonl")
X_train = df_train["text"]
y_train = df_train["sentiment"]

# 读取验证集
df_val =  load_and_clean(r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\validation.jsonl")
X_val = df_val["text"]
y_val = df_val["sentiment"]

# 读取测试集
df_test =  load_and_clean(r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\test.jsonl")
X_test = df_test["text"]
y_test = df_test["sentiment"]
# 模型
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        min_df=5,
        max_df=0.9,
        max_features=20000,
        ngram_range=(1,2),
        stop_words='english'
    )),
    ('clf', LinearSVC(class_weight='balanced'))
])


# ===== 记录训练时间 =====
start_time = time.time()

# 训练
model.fit(X_train, y_train)

end_time = time.time()
train_time = end_time - start_time

print(f"Training time: {train_time:.2f} seconds")



# 预测
y_pred = model.predict(X_test)

print("=== result ===")
print(classification_report(y_test, y_pred, digits=4))