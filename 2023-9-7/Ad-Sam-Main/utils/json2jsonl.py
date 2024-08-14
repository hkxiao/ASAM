import json

# 读取 JSON 文件
with open('../sam-1b/sa_000138-controlnet-validation.json', 'r') as infile:
    data = json.load(infile)

# 写入 JSONL 文件
with open('../sam-1b/sa_000138-controlnet-validation.jsonl', 'w') as outfile:
    if isinstance(data, dict):
        # 如果是字典，包裹在列表中
        data = [data]
    for item in data:
        outfile.write(json.dumps(item) + '\n')
