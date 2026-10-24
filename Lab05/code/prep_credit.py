# Tiền xử lý dữ liệu gốc Kaggle/UCI "Default of Credit Card Clients" -> data.csv
# (tạo các cột PAY_1, EDUCATION_CAT và các cột one-hot như file data.csv mà bài lab sử dụng)
import pandas as pd

df = pd.read_csv('data/default_of_credit_card_clients.csv')
df = df.rename(columns={'PAY_0': 'PAY_1'})
df = df.drop_duplicates(subset='ID')                 # ID là duy nhất
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})   # gộp các giá trị không có trong mô tả vào "others"
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
edu_map = {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'}
df['EDUCATION_CAT'] = df['EDUCATION'].map(edu_map)
for name in ['graduate school', 'high school', 'none', 'others', 'university']:
    df[name] = (df['EDUCATION_CAT'] == name).astype(int)
resp = 'default payment next month'
cols = [c for c in df.columns if c != resp] + [resp]
df[cols].to_csv('data.csv', index=False)
print(df.shape, df[resp].mean())
