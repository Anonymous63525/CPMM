import pandas as pd
import csv
import random

# 设置随机种子
random_seed = 42
random.seed(random_seed)

dataset = 'Toys'
csv_file = f'../data/{dataset}.txt'

# 读取CSV文件
input_data = []
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        # 将行转换为字符串，并使用空格连接
        input_data.append(' '.join(row))

# 存储结果的列表
train_data = []
val_data = []
test_data = []

# 解析并切分数据
for line in input_data:
    parts = line.split()
    user_id = parts[0]
    items = parts[1:]

    # 打乱点击项顺序
    random.shuffle(items)

    # 处理点击数少于10的用户
    if len(items) < 10:
        if len(items) == 1:
            train_data.append((user_id, items[0]))
        elif len(items) == 2:
            train_data.append((user_id, items[0]))
            val_data.append((user_id, items[1]))
        else:
            # 确保每个集合至少有一个样本
            train_items = items[:-2]
            val_item = items[-2]
            test_item = items[-1]

            train_data.extend([(user_id, item) for item in train_items])
            val_data.append((user_id, val_item))
            test_data.append((user_id, test_item))
        continue


    # 计算划分点
    num_items = len(items)

    train_end = int(num_items * 0.8)
    if (num_items-train_end) % 2 == 0:
        pass
    else:
        train_end -= 1

    val_end = train_end + int((num_items - train_end) / 2)

    # 切分数据集
    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]

    # 保存到相应的数据集中
    train_data.extend([(user_id, item) for item in train_items])
    val_data.extend([(user_id, item) for item in val_items])
    test_data.extend([(user_id, item) for item in test_items])

# 保存到文件
def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['user_id', 'item_id'])
    df.to_csv(filename, index=False)

save_to_csv(train_data, f'dataset/{dataset}_train.csv')
save_to_csv(val_data, f'dataset/{dataset}_val.csv')
save_to_csv(test_data, f'dataset/{dataset}_test.csv')

print("数据切分完成并保存到文件中。")