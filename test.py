import pandas as pd

# lines=True 是读取 jsonl 的关键参数
df = pd.read_json(r'C:\Users\24224\Desktop\毕业设计\TF-IDFcode\meta_Handmade_Products.jsonl', lines=True)

# 查看前几行
print(df.head())