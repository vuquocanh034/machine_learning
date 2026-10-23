import numpy as np #numerical computation
import pandas as pd #data wrangling
import matplotlib.pyplot as plt #plotting package
#Next line helps with rendering plots
%matplotlib inline
import matplotlib as mpl #add'l plotting functionality
mpl.rcParams['figure.dpi'] = 400 #high res figures
import graphviz #to visualize decision trees

df = pd.read_csv('data.csv') #Load the cleaned data
features_response = df.columns.tolist() #Get a list of column names
#Make a list of columns to remove that aren't features or the response variable
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3',\
                   'PAY_4', 'PAY_5', 'PAY_6',\
                   'EDUCATION_CAT', 'graduate school',\
                   'high school', 'none',\
                   'others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]
features_response

from sklearn.model_selection import train_test_split
from sklearn import tree
#Split the data into training and testing sets using the same random seed
X_train, X_test, y_train, y_test = \
    train_test_split(df[features_response[:-1]].values,
    df['default payment next month'].values,
    test_size=0.2, random_state=24)

# the tree will grow to a depth of at most 2
dt = tree.DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)

dot_data = tree.export_graphviz(dt,
                out_file=None,
                filled=True,
                rounded=True,
                feature_names=\
                  features_response[:-1],
                proportion=True,
                class_names=['Not defaulted', 'Defaulted'])
graph = graphviz.Source(dot_data)
graph

# (lưu thêm ảnh PNG để đưa vào báo cáo)
graph.render('Reports/decision_tree_depth2', format='png', cleanup=True)

from sklearn.model_selection import GridSearchCV
params = {'max_depth':[1, 2, 4, 6, 8, 10, 12]} #parameters
dt = tree.DecisionTreeClassifier() #tree modal
cv = GridSearchCV(dt, param_grid=params, scoring='roc_auc',
                  n_jobs=None, refit=True, cv=4, verbose=1,
                  error_score=np.nan,
                  return_train_score=True) # cv is the best model.
cv.fit(X_train, y_train)

print("Tham số tốt nhất:", cv.best_params_)
print("ROC AUC (CV) tốt nhất:", cv.best_score_)

cv_results_df = pd.DataFrame(cv.cv_results_)
#View the names of the remaining columns in the results DataFrame
cv_results_df.columns

mpl.rcParams['figure.dpi'] = 100   # [Chỉnh] hạ dpi để ảnh nhẹ hơn (đề dùng 400)
ax = plt.axes()
ax.errorbar(cv_results_df['param_max_depth'],
            cv_results_df['mean_train_score'],
            yerr=cv_results_df['std_train_score']/np.sqrt(4),
            label='Mean $\pm$ 1 SE training scores')
ax.errorbar(cv_results_df['param_max_depth'],
            cv_results_df['mean_test_score'],
            yerr=cv_results_df['std_test_score']/np.sqrt(4),
            label='Mean $\pm$ 1 SE testing scores')
ax.legend()
plt.xlabel('max_depth')
plt.ylabel('ROC AUC')

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier\
  (n_estimators=10, criterion='gini', max_depth=3,
   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
   max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
   bootstrap=True, oob_score=False, n_jobs=None,
   random_state=4, verbose=0, warm_start=False, class_weight=None)

#a parameter grid for this exercise in order to search the numbers of trees, ranging from 10 to 100 by 10s
rf_params_ex = {'n_estimators':list(range(10,110,10))}
# [Chỉnh] error_score=np.nan (đề viết chuỗi 'np.nan' - scikit-learn bản mới không chấp nhận chuỗi này)
cv_rf_ex = GridSearchCV(rf, param_grid=rf_params_ex,
                        scoring='roc_auc', n_jobs=None,
                        refit=True, cv=4, verbose=1,
                        error_score=np.nan,
                        return_train_score=True)
cv_rf_ex.fit(X_train, y_train)

cv_rf_ex_results_df = pd.DataFrame(cv_rf_ex.cv_results_)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
axs[0].plot(cv_rf_ex_results_df['param_n_estimators'],
            cv_rf_ex_results_df['mean_fit_time'],
            '-o')
axs[0].set_xlabel('Number of trees')
axs[0].set_ylabel('Mean fit time (seconds)')
axs[1].errorbar(cv_rf_ex_results_df['param_n_estimators'],
                cv_rf_ex_results_df['mean_test_score'],
                yerr=cv_rf_ex_results_df['std_test_score']/np.sqrt(4))
axs[1].set_xlabel('Number of trees')
axs[1].set_ylabel('Mean testing ROC AUC $\pm$ 1 SE ')
plt.tight_layout()

cv_rf_ex.best_params_

# the feature names and importances
feat_imp_df = pd.DataFrame({
    'Importance':cv_rf_ex.best_estimator_.feature_importances_},
    index=features_response[:-1])
feat_imp_df.sort_values('Importance', ascending=True).plot.barh()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
titanic = pd.read_csv('data/train.csv')
print(titanic.shape)
print(titanic.isnull().sum()[lambda s: s>0])
# Tiền xử lý: điền tuổi thiếu bằng median, mã hóa giới tính và cảng lên tàu
t = titanic[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].copy()
t['Age'] = t['Age'].fillna(t['Age'].median())
t['Embarked'] = t['Embarked'].fillna(t['Embarked'].mode()[0])
t['Sex'] = t['Sex'].map({'male':0,'female':1})
t = pd.get_dummies(t, columns=['Embarked'], drop_first=True)
X = t.drop(columns='Survived'); y = t['Survived']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)
t.head()

# Cây quyết định: chọn max_depth bằng GridSearchCV
gs = GridSearchCV(tree.DecisionTreeClassifier(random_state=24),
                  {'max_depth':[1,2,3,4,5,6,8,10]}, scoring='roc_auc', cv=4, return_train_score=True)
gs.fit(Xtr, ytr)
print("Cây quyết định - tham số tốt nhất:", gs.best_params_, " AUC(CV) = %.4f" % gs.best_score_)
best_dt = gs.best_estimator_
print("Accuracy trên tập test: %.4f" % accuracy_score(yte, best_dt.predict(Xte)))
dot = tree.export_graphviz(best_dt, out_file=None, filled=True, rounded=True, feature_names=list(X.columns),
                           proportion=True, class_names=['Died','Survived'])
graphviz.Source(dot).render('Reports/titanic_tree', format='png', cleanup=True)
graphviz.Source(dot)

# Rừng cây
gs_rf = GridSearchCV(RandomForestClassifier(max_depth=5, random_state=4),
                     {'n_estimators':list(range(10,110,10))}, scoring='roc_auc', cv=4)
gs_rf.fit(Xtr, ytr)
print("Random Forest - tham số tốt nhất:", gs_rf.best_params_, " AUC(CV) = %.4f" % gs_rf.best_score_)
pred = gs_rf.best_estimator_.predict(Xte)
print("Accuracy trên tập test: %.4f" % accuracy_score(yte, pred))
print(classification_report(yte, pred, target_names=['Died','Survived']))
pd.Series(gs_rf.best_estimator_.feature_importances_, index=X.columns).sort_values().plot.barh(title='Feature importance - Titanic')

diab = pd.read_csv('data/diabetes_prediction_dataset.csv')
print(diab.shape)
print("Giá trị thiếu:", diab.isnull().sum().sum(), "| dòng trùng:", diab.duplicated().sum())
print(diab['diabetes'].value_counts())
# Mã hóa biến phân loại (gender, smoking_history)
diab = pd.get_dummies(diab, columns=['gender','smoking_history'], drop_first=True)
Xd = diab.drop(columns='diabetes'); yd = diab['diabetes']
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=24, stratify=yd)
Xd.head()

gs_d = GridSearchCV(tree.DecisionTreeClassifier(random_state=24),
                    {'max_depth':[1,2,3,4,5,6,8,10]}, scoring='roc_auc', cv=4, return_train_score=True)
gs_d.fit(Xd_tr, yd_tr)
print("Cây quyết định - tham số tốt nhất:", gs_d.best_params_, " AUC(CV) = %.4f" % gs_d.best_score_)
print("Accuracy trên tập test: %.4f" % accuracy_score(yd_te, gs_d.best_estimator_.predict(Xd_te)))

gs_drf = GridSearchCV(RandomForestClassifier(random_state=4, n_jobs=-1),
                      {'n_estimators':[50,100,200], 'max_depth':[5,8,None]}, scoring='roc_auc', cv=4)
gs_drf.fit(Xd_tr, yd_tr)
print("Random Forest - tham số tốt nhất:", gs_drf.best_params_, " AUC(CV) = %.4f" % gs_drf.best_score_)
pred = gs_drf.best_estimator_.predict(Xd_te)
print("Accuracy trên tập test: %.4f" % accuracy_score(yd_te, pred))
print(classification_report(yd_te, pred))
pd.Series(gs_drf.best_estimator_.feature_importances_, index=Xd.columns).sort_values().plot.barh(title='Feature importance - Diabetes')

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
# Download&Load dữ liệu iris từ datasets của scikit-learn
iris = datasets.load_iris()
# Hiển thị mô ta dữ liệu, chỉ có trong các bộ dữ liệu chuẩn và mở để học tập và nghiên cứu
print(iris.DESCR)

# Từ tập dữ liệu ban đầu, tách lấy ma trận biểu diễn các đặc trưng và nhãn.
data = iris.data
target = iris.target
# TODO: Chia dữ liệu và nhãn thành 2 tập dữ liệu huấn luyện và dữ liệu kiểm tra theo tỉ lệ 80:20
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state=101)

from sklearn import svm
# khởi tạo mô hình phân lớp
clf = svm.SVC()
# Sử dụng phương thức 'fit' để huấn luyện mô hình với dữ liệu huấn luyện và nhãn huấn luyện
# fit (X,Y) với X là tập các đối tượng, Y là tập nhãn tương ứng của đối tượng.
clf.fit(X_train, y_train)

# Tính độ chính xác trên tập huấn luyện và tập kiểm tra
train_acc = clf.score(X_train,y_train)
val_acc = clf.score(X_test,y_test)
print('Training accuracy: {}'.format(train_acc))
print('Validation accuracy: {}'.format(val_acc))

# best_svm, best_val_acc và best_kernel lần lượt là các biến lưu mô hình tốt nhất,
# độ chính xác cao nhất trên tập kiểm tra và kernel tốt nhất
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
best_svm = None
best_val_acc = -1
best_kernel = None
# Huấn luyện các mô hình dựa trên dữ liệu huấn luyện và tham số kernel
# Tính toán độ chính xác trên tập huấn luyện và tập kiểm tra để tìm được mô hình tốt nhất
for i in range(4):
    clf = svm.SVC(kernel=kernels[i], probability=True)
    clf.fit(X_train, y_train)
    tmp_val_acc = clf.score(X_test, y_test)
    print("kernel = {:8s} validation accuracy = {:.4f}".format(kernels[i], tmp_val_acc))
    if (tmp_val_acc > best_val_acc):
        best_val_acc = tmp_val_acc
        best_svm = clf
        best_kernel = kernels[i]
# Hiển thị mô hình tốt nhất cùng với độ chính xác
print("Best validation accuracy : {} with kernel: {}".format(best_val_acc, best_kernel))

from sklearn.model_selection import GridSearchCV
gs_svm = GridSearchCV(svm.SVC(), {'kernel':['linear','poly','rbf','sigmoid'], 'C':[0.1,1,10,100]}, cv=5)
gs_svm.fit(X_train, y_train)
print(gs_svm.best_params_, "CV accuracy = %.4f" % gs_svm.best_score_, " test accuracy = %.4f" % gs_svm.score(X_test, y_test))

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
mpl.rcParams['figure.dpi'] = 100
digits = load_digits(n_class=10)

#thông tin toàn bộ dữ liệu đã tải về
digits

#xem thông tin của một hình dưới dạng ma trận 8 x 8
digits['data'][0].reshape(8,8)

#xem thông tin của một hình dưới dạng mảng
digits['data'][0]

#xem thông tin 9 nhãn đầu tiên
digits['target'][0:9]

# Each Digit is represented in digits.images as a matrix of 8x8 = 64 pixels. Each of the 64 values represent
# a greyscale. The Greyscale are then plotted in the right scale by the imshow method.
fig, ax = plt.subplots(8,8, figsize=(10,10))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])

# Hàm vẽ 1 ảnh có kích thước 8 x 8 (ảnh lấy từ ma images)
def view_digit(index):
    plt.imshow(digits.images[index] , cmap = plt.cm.gray_r)
    plt.title('Orignal it is: '+ str(digits.target[index]))
    plt.show()
# vẽ ảnh ở vị trí thứ 4
view_digit(4)

# Thực hiện import các thư viện cần thiết để xây dựng mô hình SVM
# Thực hiện bước 1 của nhiệm vụ 1
from sklearn import svm
main_data = digits['data']
targets = digits['target']
svc = svm.SVC(gamma=0.001 , C = 100)
# GAMMA is a parameter for non linear hyperplanes.
# The higher the gamma value it tries to exactly fit the training data set
# C is the penalty parameter of the error term.
# It controls the trade off between smooth decision boundary and classifying the training points correctly.
svc.fit(main_data[:1500] , targets[:1500])
predictions = svc.predict(main_data[1501:])
# list(zip(predictions , targets[1501:]))

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(predictions, targets[1501:])
conf_matrix = pd.DataFrame(data = cm)
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

from sklearn.metrics import classification_report
print(classification_report(predictions, targets[1501:]))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# dùng lại Xd_tr, Xd_te, yd_tr, yd_te (đã xử lý ở bài 2.1.4)
# SVM huấn luyện chậm với 80.000 mẫu nên chỉ lấy ngẫu nhiên 10.000 mẫu của tập train để tìm tham số; đánh giá trên toàn bộ tập test
Xs = Xd_tr.sample(10000, random_state=24); ys = yd_tr.loc[Xs.index]
# SVM nhạy với thang đo nên chuẩn hóa bằng StandardScaler
pipe = make_pipeline(StandardScaler(), svm.SVC())
param = {'svc__kernel':['linear','rbf'], 'svc__C':[0.1,1,10], 'svc__gamma':['scale']}
gs_sd = GridSearchCV(pipe, param, cv=3, scoring='accuracy')
gs_sd.fit(Xs, ys)
print("Tham số tốt nhất:", gs_sd.best_params_)
print("Accuracy CV: %.4f" % gs_sd.best_score_)
pred = gs_sd.predict(Xd_te)
print("Accuracy test: %.4f" % accuracy_score(yd_te, pred))
print(confusion_matrix(yd_te, pred))
print(classification_report(yd_te, pred))

from sklearn.preprocessing import OneHotEncoder
animal = pd.read_csv('data/Animal-Condition-Predict.csv')
print(animal.shape)
print(animal.isnull().sum())
print(animal['Dangerous'].value_counts())
animal = animal.dropna(subset=['Dangerous'])      # bỏ 2 dòng thiếu nhãn
Xa = animal.drop(columns='Dangerous'); ya = animal['Dangerous']
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.2, random_state=42, stratify=ya)
# Tất cả đặc trưng là biến phân loại nên mã hóa one-hot trước khi đưa vào SVM
pipe_a = make_pipeline(OneHotEncoder(handle_unknown='ignore'), svm.SVC())
param_a = {'svc__kernel':['linear','rbf'], 'svc__C':[0.1,1,10,100]}
gs_a = GridSearchCV(pipe_a, param_a, cv=5, scoring='accuracy')
gs_a.fit(Xa_tr, ya_tr)
print("Tham số tốt nhất:", gs_a.best_params_, " Accuracy CV: %.4f" % gs_a.best_score_)
pred = gs_a.predict(Xa_te)
print("Accuracy test: %.4f" % accuracy_score(ya_te, pred))
print(confusion_matrix(ya_te, pred))
print(classification_report(ya_te, pred, zero_division=0))

#Importing the Necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv('data/spam.csv', encoding='latin-1')   # [Chỉnh] đọc từ thư mục data/
#display the first 5 rows
data.head()

# Drop the columns with NaN values
# [Chỉnh] file spam.csv tải về chỉ có 2 cột (Category, Message), không có các cột 'Unnamed' như bản trong đề
data = data.drop(columns=[c for c in data.columns if c.startswith('Unnamed')], axis=1)
# Rename columns for clarity:
data.columns = ['label', 'text']
# Separate features (X) and target labels (y)
X = data.drop('label', axis=1)
y = data['label']
# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
# Fit and transform the training data (X_train)
X_train_vectorized = vectorizer.fit_transform(X_train['text'])
# Transform the test data (X_test)
X_test_vectorized = vectorizer.transform(X_test['text'])

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Make predictions on the test data
y_pred = classifier.predict(X_test_vectorized)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
cust = pd.read_csv('data/Customer_Behaviour.csv')
print(cust.shape)
print(cust.isnull().sum().sum(), "giá trị thiếu")
print(cust['Purchased'].value_counts())
cust.head()

# Đặc trưng: Age, EstimatedSalary; nhãn: Purchased (khách có mua hàng hay không)
Xc = cust[['Age','EstimatedSalary']].values
yc = cust['Purchased'].values
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
# Chuẩn hóa đặc trưng (fit trên tập train)
sc = StandardScaler()
Xc_tr = sc.fit_transform(Xc_tr); Xc_te = sc.transform(Xc_te)
gnb_c = GaussianNB()
gnb_c.fit(Xc_tr, yc_tr)
pred = gnb_c.predict(Xc_te)
print("Accuracy test: %.4f" % accuracy_score(yc_te, pred))
print(confusion_matrix(yc_te, pred))
print(classification_report(yc_te, pred, target_names=['Not purchased','Purchased']))

# Biên quyết định của Gaussian Naïve Bayes trên dữ liệu test (2 đặc trưng đã chuẩn hóa)
import matplotlib.pyplot as plt
xx, yy = np.meshgrid(np.linspace(Xc_te[:,0].min()-1, Xc_te[:,0].max()+1, 300),
                     np.linspace(Xc_te[:,1].min()-1, Xc_te[:,1].max()+1, 300))
zz = gnb_c.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(7,5))
plt.contourf(xx, yy, zz, alpha=0.3, cmap='coolwarm')
plt.scatter(Xc_te[:,0], Xc_te[:,1], c=yc_te, cmap='coolwarm', edgecolor='k')
plt.xlabel('Age (scaled)'); plt.ylabel('EstimatedSalary (scaled)')
plt.title('GaussianNB decision boundary - test set')
plt.show()

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
mush = pd.read_csv('data/mushrooms.csv')      # cột class: e = ăn được, p = độc
print(mush.shape); print(mush['class'].value_counts())
Xm = mush.drop(columns='class'); ym = mush['class']
enc = OrdinalEncoder()
Xm_enc = enc.fit_transform(Xm).astype(int)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm_enc, ym, test_size=0.2, random_state=42, stratify=ym)
cnb = CategoricalNB(min_categories=[len(c) for c in enc.categories_])
cnb.fit(Xm_tr, ym_tr)
pred = cnb.predict(Xm_te)
print("Accuracy test: %.4f" % accuracy_score(ym_te, pred))
print(confusion_matrix(ym_te, pred))
print(classification_report(ym_te, pred, target_names=['edible (e)','poisonous (p)']))
