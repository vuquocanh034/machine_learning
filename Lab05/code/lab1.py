import numpy as np
import pandas as pd
from scipy import stats
# Load the .csv into a dataframe using read_csv
covid_data = pd.read_csv("data/covid-data.csv")
covid_data = covid_data[['iso_code','continent',
 'location','date','total_cases','new_cases']]

# Take a quick look at the data
covid_data.head(5)

covid_data.dtypes

covid_data.shape

# Get the mean of the data
data_mean = np.mean(covid_data["new_cases"])
# Get the median of the data
data_median = np.median(covid_data["new_cases"])
# Get the mode of the data
data_mode = stats.mode(covid_data["new_cases"])
# Obtain the variance of the data
data_variance = np.var(covid_data["new_cases"])
# Obtain the standard deviation of the data
data_sd = np.std(covid_data["new_cases"])
# Compute the maximum and minimum values of the data
data_max = np.max(covid_data["new_cases"])
data_min = np.min(covid_data["new_cases"])
# Obtain the 60th percentile of the data
data_percentile = np.percentile(covid_data["new_cases"],60)
# Obtain the quartiles of the data
data_quartile = np.quantile(covid_data["new_cases"],0.75)
# Get the IQR of the data
data_IQR = stats.iqr(covid_data["new_cases"])

print("mean      :", data_mean)
print("median    :", data_median)
print("mode      :", data_mode)
print("variance  :", data_variance)
print("std       :", data_sd)
print("max / min :", data_max, "/", data_min)
print("P60       :", data_percentile)
print("Q3        :", data_quartile)
print("IQR       :", data_IQR)

# [Mở rộng] bỏ giá trị thiếu trước khi tính thống kê
nc = covid_data["new_cases"].dropna()
print("Số dòng thiếu new_cases:", covid_data["new_cases"].isna().sum())
print("mean      :", np.mean(nc))
print("median    :", np.median(nc))
print("mode      :", stats.mode(nc).mode)
print("variance  :", np.var(nc))
print("std       :", np.std(nc))
print("max / min :", np.max(nc), "/", np.min(nc))
print("P60       :", np.percentile(nc,60))
print("Q3        :", np.quantile(nc,0.75))
print("IQR       :", stats.iqr(nc))

import pandas as pd
# [Chỉnh] file Marketing Campaign (Kaggle) phân cách bằng dấu Tab nên cần sep='\t'
marketing_data = pd.read_csv("data/marketing_campaign.csv", sep='\t', encoding='utf-8-sig')
# [Chỉnh] tên cột đúng là 'Dt_Customer' (đề in nhầm 'Dt_ Customer')
marketing_data = marketing_data[['ID','Year_Birth', 'Education',
 'Marital_Status','Income','Kidhome', 'Teenhome',
 'Dt_Customer', 'Recency','NumStorePurchases',
 'NumWebVisitsMonth']]

marketing_data.head()

# Remove duplicates across the columns in our dataset:
marketing_data_duplicate = marketing_data.drop_duplicates()
print(marketing_data.shape, '->', marketing_data_duplicate.shape)

# Delete a specified row at index value 1:
marketing_data.drop(labels=[1], axis=0)

# Delete a single column
marketing_data.drop(labels=['Year_Birth'], axis=1)

# Replace the values in Teenhome with has teen and has no teen
marketing_data['Teenhome_replaced'] = marketing_data['Teenhome'].replace([0,1,2],['has no teen','has teen','has teen'])
marketing_data[['Teenhome','Teenhome_replaced']].head()

# Fill NAs in the Income column
marketing_data['Income'] = marketing_data['Income'].fillna(0)
# Change the data type of the Income column from float to int
marketing_data['Income_changed'] = marketing_data['Income'].astype(int)
marketing_data[['Income','Income_changed']].dtypes

# Check for missing values using the isnull and sum methods
marketing_data.isnull().sum()

# Drop missing values using the dropna method
marketing_data_withoutna = marketing_data.dropna(how = 'any')
marketing_data_withoutna.shape

import numpy as np
import pandas as pd
from scipy import stats
wine = pd.read_csv("data/winequality-red.csv", sep=';')
print(wine.shape)
wine.head()

wine.dtypes

wine.isnull().sum()

wine.describe().T

# Các thước đo thống kê mô tả cho một số cột (ví dụ: alcohol)
col = wine['alcohol']
print("mean      :", np.mean(col))
print("median    :", np.median(col))
print("mode      :", stats.mode(col).mode)
print("variance  :", np.var(col))
print("std       :", np.std(col))
print("range     :", np.max(col) - np.min(col))
print("P60       :", np.percentile(col, 60))
print("Q1,Q2,Q3  :", np.quantile(col, [0.25, 0.5, 0.75]))
print("IQR       :", stats.iqr(col))
print("skewness  :", stats.skew(col))

# Phân bố của biến mục tiêu quality và độ lệch (skewness) của các biến
print(wine['quality'].value_counts().sort_index())
wine.skew().sort_values(ascending=False)

# Phát hiện ngoại lai bằng quy tắc 1.5*IQR
Q1 = wine.quantile(0.25); Q3 = wine.quantile(0.75); IQR = Q3 - Q1
outliers = ((wine < (Q1 - 1.5*IQR)) | (wine > (Q3 + 1.5*IQR))).sum()
outliers.sort_values(ascending=False)

diabetes = pd.read_csv("data/diabetes.csv")
print(diabetes.shape)
diabetes.head()

diabetes.describe().T

# Trong dữ liệu Pima, giá trị 0 ở các cột dưới đây thực chất là dữ liệu thiếu
cols0 = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
print((diabetes[cols0] == 0).sum())

# Xử lý dữ liệu thiếu: thay 0 bằng NaN rồi điền bằng median của từng cột
diabetes_clean = diabetes.copy()
diabetes_clean[cols0] = diabetes_clean[cols0].replace(0, np.nan)
diabetes_clean[cols0] = diabetes_clean[cols0].fillna(diabetes_clean[cols0].median())
diabetes_clean.describe().T

print("Phân bố nhãn Outcome:"); print(diabetes['Outcome'].value_counts())
print()
print("Trung bình theo nhóm Outcome:")
diabetes_clean.groupby('Outcome').mean().T

import os
HAS_HOUSE = os.path.exists("data/HousingPricesData.csv")
print("Có file HousingPricesData.csv:", HAS_HOUSE)

import pandas as pd
import matplotlib.pyplot as plt  # [Chỉnh] bổ sung import (đề chưa import ở mục này)
houseprices_data = pd.read_csv("data/HousingPricesData.csv")
houseprices_data = houseprices_data[['Zip', 'Price', 'Area', 'Room']]
# Create a PriceperSqm variable based on the Price and Area variables:
houseprices_data['PriceperSqm'] = houseprices_data['Price']/houseprices_data['Area']

houseprices_sorted = houseprices_data.sort_values('Price', ascending = False)
houseprices_sorted.head()

# case 1: basic
plt.figure(figsize= (12,6))
x = houseprices_sorted['Zip'][0:10]
y = houseprices_sorted['Price'][0:10]
plt.bar(x,y)
plt.show()

# case 2: advanced 1
plt.figure(figsize= (12,6))
plt.bar(x,y)
plt.title('Top 10 Areas with the highest house prices', fontsize=15)
plt.xlabel('Zip code', fontsize = 12)
plt.xticks(fontsize=10)
plt.ylabel('House prices in millions', fontsize=12)
plt.yticks(fontsize=10)
plt.show()

# case 3: advanced 2
fig, ax = plt.subplots(figsize=(40,18))
x = houseprices_sorted['Zip'][0:10]
y = houseprices_sorted['Price'][0:10]
y1 = houseprices_sorted['PriceperSqm'][0:10]
plt.subplot(1,2,1)
plt.bar(x,y)
plt.xticks(fontsize=17)
plt.ylabel('House prices in millions', fontsize=25)
plt.yticks(fontsize=20)
plt.title('Top 10 Areas with the highest house prices', fontsize=25)
plt.subplot(1,2,2)
plt.bar(x,y1)
plt.xticks(fontsize=17)
plt.ylabel('House prices per sqm', fontsize=25)
plt.yticks(fontsize=20)
plt.title('Top 10 Areas with the highest house prices per sqm', fontsize=25)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
# case 1: basic
plt.figure(figsize= (12,6))
data = houseprices_sorted[0:10]
sns.barplot(data= data, x= 'Zip',y = 'Price')

# case 2: advanced 1
plt.figure(figsize= (12,6))
data = houseprices_sorted[0:10]
ax = sns.barplot(data= data, x= 'Zip',y = 'Price')
ax.set_xlabel('Zip code',fontsize = 15)
ax.set_ylabel('House prices in millions', fontsize = 15)
ax.set_title('Top 10 Areas with the highest house prices', fontsize= 20)

# case 3: view multiple perspectives at once
fig, ax = plt.subplots(1, 2,figsize=(40,18))
data = houseprices_sorted[0:10]
sns.set(font_scale = 3)
ax1 = sns.barplot(data= data, x= 'Zip',y = 'Price', ax = ax[0])
ax1.set_xlabel('Zip code')
ax1.set_ylabel('House prices in millions')
ax1.set_title('Top 10 Areas with the highest house prices')
ax2 = sns.barplot(data= data, x= 'Zip',y = 'PriceperSqm',
ax=ax[1])
ax2.set_xlabel('Zip code')
ax2.set_ylabel('House prices per sqm')
ax2.set_title('Top 10 Areas with the highest price per sqm')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()   # đưa style về mặc định sau khi đặt font_scale ở trên
wine = pd.read_csv("data/winequality-red.csv", sep=';')

# 1. Biểu đồ cột: phân bố chất lượng rượu
plt.figure(figsize=(8,5))
ax = sns.countplot(data=wine, x='quality')
ax.set_title('Distribution of wine quality', fontsize=15)
plt.show()

# 2. Histogram của nồng độ cồn
plt.figure(figsize=(8,5))
ax = sns.histplot(data=wine, x='alcohol', bins=20, kde=True)
ax.set_title('Histogram of alcohol', fontsize=15)
plt.show()

# 3. Boxplot alcohol theo quality (biến số vs biến phân loại)
plt.figure(figsize=(8,5))
ax = sns.boxplot(data=wine, x='quality', y='alcohol')
ax.set_title('Alcohol by quality', fontsize=15)
plt.show()

# 4. Heatmap tương quan giữa các biến
plt.figure(figsize=(11,8))
ax = sns.heatmap(wine.corr(), annot=True, fmt='.2f', cmap='coolwarm')
ax.set_title('Correlation heatmap - red wine', fontsize=15)
plt.show()

diabetes = pd.read_csv("data/diabetes.csv")
cols0 = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
diabetes[cols0] = diabetes[cols0].replace(0, np.nan)
diabetes[cols0] = diabetes[cols0].fillna(diabetes[cols0].median())

# 1. Số lượng người bệnh / không bệnh
plt.figure(figsize=(6,4))
ax = sns.countplot(data=diabetes, x='Outcome')
ax.set_title('Outcome distribution', fontsize=15)
plt.show()

# 2. Histogram Glucose theo Outcome
plt.figure(figsize=(8,5))
ax = sns.histplot(data=diabetes, x='Glucose', hue='Outcome', kde=True)
ax.set_title('Glucose by Outcome', fontsize=15)
plt.show()

# 3. Boxplot BMI theo Outcome và scatter Glucose - BMI
fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.boxplot(data=diabetes, x='Outcome', y='BMI', ax=axes[0]).set_title('BMI by Outcome')
sns.scatterplot(data=diabetes, x='Glucose', y='BMI', hue='Outcome', ax=axes[1]).set_title('Glucose vs BMI')
plt.show()

# 4. Heatmap tương quan
plt.figure(figsize=(9,7))
ax = sns.heatmap(diabetes.corr(), annot=True, fmt='.2f', cmap='coolwarm')
ax.set_title('Correlation heatmap - diabetes', fontsize=15)
plt.show()

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set()
retail = pd.read_excel("data/Online_Retail.xlsx")
print(retail.shape)
retail.head()

retail.info()

print(retail.isnull().sum())
print("Số dòng trùng lặp:", retail.duplicated().sum())
retail.describe()

print("Đơn bị hủy:", retail['InvoiceNo'].astype(str).str.startswith('C').sum())
print("Quantity <= 0:", (retail['Quantity'] <= 0).sum(), "| UnitPrice <= 0:", (retail['UnitPrice'] <= 0).sum())
clean = retail[(retail['Quantity'] > 0) & (retail['UnitPrice'] > 0)].drop_duplicates().copy()
clean['Revenue'] = clean['Quantity'] * clean['UnitPrice']
clean['Month'] = clean['InvoiceDate'].dt.to_period('M').astype(str)
print(clean.shape)

# 10 quốc gia có doanh thu cao nhất
top_country = clean.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
print(top_country)
plt.figure(figsize=(10,5))
ax = sns.barplot(x=top_country.values, y=top_country.index)
ax.set_title('Top 10 countries by revenue', fontsize=15)
plt.show()

# 10 sản phẩm bán nhiều nhất và doanh thu theo tháng
top_prod = clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print(top_prod)
monthly = clean.groupby('Month')['Revenue'].sum()
plt.figure(figsize=(11,4))
ax = sns.lineplot(x=monthly.index, y=monthly.values, marker='o')
ax.set_title('Monthly revenue', fontsize=15)
plt.xticks(rotation=45)
plt.show()

# Phân bố số lượng và đơn giá (thang log do lệch phải mạnh)
fig, axes = plt.subplots(1, 2, figsize=(12,4))
sns.histplot(clean['Quantity'].clip(upper=100), bins=40, ax=axes[0]).set_title('Quantity (clipped at 100)')
sns.histplot(clean['UnitPrice'].clip(upper=30), bins=40, ax=axes[1]).set_title('UnitPrice (clipped at 30)')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
penguins_data = pd.read_csv("data/penguins_size.csv")
penguins_data = penguins_data[['species','culmen_length_mm']]

# case 1: basic
sns.histplot( data = penguins_data, x= penguins_data["culmen_length_mm"])

# case 2: advanced
plt.figure(figsize= (12,6))
ax = sns.histplot( data = penguins_data, x= penguins_data["culmen_length_mm"])
ax.set_xlabel('Culmen Length in mm',fontsize = 15)
ax.set_ylabel('Count of records', fontsize = 15)
ax.set_title('Univariate analysis of Culmen Length',fontsize= 20)

# case 1: basic
sns.countplot(data = penguins_data, x= penguins_data['species'])

# case 2: advanced
plt.figure(figsize= (12,6))
ax = sns.countplot(data = penguins_data, x= penguins_data['species'])
ax.set_xlabel('Penguin Species',fontsize = 15)
ax.set_ylabel('Count of records',fontsize = 15)
ax.set_title('Univariate analysis of Penguin Species', fontsize= 20)

penguins_group = penguins_data.groupby('species').count()
penguins_group = penguins_group.reset_index()
# case 1: basic
plt.pie(penguins_group["culmen_length_mm"], labels =
                      penguins_group['species'])
plt.show()

# case 2: advanced
cols = ['g', 'b', 'r']
plt.pie(penguins_group["culmen_length_mm"], labels =
                          penguins_group['species'],colors = cols)
plt.title('Univariate Analysis of Species', fontsize=15)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
houseprices_data = pd.read_csv("data/HousingPricesData.csv")
houseprices_data = houseprices_data[['Zip','Price','Area','Room']]
# case 1: Create a boxplot using the boxplot method
sns.boxplot(data = houseprices_data, x= houseprices_data["Price"])

# case 2: advanced
plt.figure(figsize= (12,6))
ax = sns.boxplot(data = houseprices_data, x= houseprices_data["Price"])
ax.set_xlabel('House Prices in millions',fontsize = 15)
ax.set_title('Univariate analysis of House Prices', fontsize= 20)
plt.ticklabel_format(style='plain', axis='x')

# case 1: basic
sns.violinplot(data = houseprices_data, x= houseprices_data["Price"])

# case 2: advanced
plt.figure(figsize= (12,6))
ax = sns.violinplot(data = houseprices_data, x=houseprices_data["Price"])
ax.set_xlabel('House Prices in millions',fontsize = 15)
ax.set_title('Univariate analysis of House Prices', fontsize= 20)
plt.ticklabel_format(style='plain', axis='x')

houseprices_data.describe()

import pandas as pd
import numpy as np   # [Chỉnh] cần cho np.mean ở pivot_table
import matplotlib.pyplot as plt
import seaborn as sns
penguins_data = pd.read_csv("data/penguins_size.csv")
penguins_data = penguins_data[['species','culmen_length_mm', 'body_mass_g']]

# case 1: basic
sns.scatterplot(data = penguins_data, x= penguins_data["culmen_length_mm"],
                                      y= penguins_data['body_mass_g'])

# case 2: advanced
plt.figure(figsize= (12,6))
ax = sns.scatterplot(data = penguins_data,
        x= penguins_data["culmen_length_mm"],
        y= penguins_data['body_mass_g'])
ax.set_title('Bivariate analysis of Culmen Length and body mass', fontsize=
20)

# [Chỉnh] cột 'sex' không được chọn ở bước 1 của đề nên nạp lại từ file gốc
penguins_full = pd.read_csv("data/penguins_size.csv")
pd.crosstab(index= penguins_full['species'], columns= penguins_full['sex'])

pd.pivot_table(penguins_data, values='culmen_length_mm',
                       index='species', aggfunc=np.mean)

import seaborn as sns
sns.pairplot( data = penguins_data)

import pandas as pd
from ydata_profiling import ProfileReport
# [Chỉnh] sep='\t' cho đúng định dạng file Marketing Campaign
marketing_data = pd.read_csv("data/marketing_campaign.csv", sep='\t', encoding='utf-8-sig')
# Create an Automated EDA report using the ProfileReport class in the
# ydata_profiling library. Use the to_file method to output the report to an HTML file
profile = ProfileReport(marketing_data)
profile.to_file("Reports/profile_output.html")

import pandas as pd
import dtale
marketing_data = pd.read_csv("data/marketing_campaign.csv", sep='\t', encoding='utf-8-sig')
dtale.show(marketing_data).open_browser()

import sweetviz as sv
marketing_data = pd.read_csv("data/marketing_campaign.csv", sep='\t', encoding='utf-8-sig')
report = sv.analyze(marketing_data)
report.show_html("Reports/sweetviz_report.html", open_browser=False)

from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
dft = AV.AutoViz("data/marketing_campaign.csv", sep='\t')
