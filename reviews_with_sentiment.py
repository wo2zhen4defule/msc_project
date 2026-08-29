import json

input_file = r"C:\Users\24224\Desktop\毕业设计\TF-IDFcode\Handmade_Products.jsonl"  # 原始文件
output_file = "../reviews_with_sentiment.jsonl"  # 输出文件


def get_sentiment(rating):
    if rating > 3.0:
        return "Positive"
    elif rating == 3.0:
        return "Neutral"
    else:
        return "Negative"


with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line.strip())

        rating = data.get("rating", 0)
        data["sentiment"] = get_sentiment(rating)  # 插入 sentiment 字段

        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print("处理完成，已保存到:", output_file)